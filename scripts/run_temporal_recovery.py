from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.video import VideoReader
from module1.data.sampler import FrameSampler
from module1.detection.base import Detection
from module1.tracking.inputs import load_fused_frames
from module1.recovery import RecoveryConfig, TemporalRecovery
from module1.recovery.inputs import load_tracker_frames, load_pose_quality
from module1.recovery.metrics import summarize_recovery
from module1.logging_utils import setup_logging


def annotate(image, row):
    h = image.shape[0]
    canvas = cv2.copyMakeBorder(image, 0, 82, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    box = row['final_infant_bbox']
    if box is not None:
        points = tuple(int(round(v)) for v in box)
        color = (50, 200, 60) if row['final_status'] == 'DETECTED' else (220, 170, 30)
        cv2.rectangle(canvas, points[:2], points[2:], color, 2)
    lines = [f"{row['video_id']} | frame={row['frame_index']} | t={row['timestamp_sec']:.3f}s",
             f"detector={row['detector_status']} | tracker={row['tracker_status']}",
             f"{row['final_status']} | confidence={row['final_infant_confidence']:.3f} | gap={row['gap_age']}",
             row['rejection_reason'] or ('RECOVERY ACCEPTED' if row['final_status'] == 'RECOVERED' else 'DIRECT DETECTION')]
    for i, line in enumerate(lines):
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]
        scale = min(.48, .48 * max(1, image.shape[1]-8)/max(1, width))
        cv2.putText(canvas, line, (4, h+16+i*20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bounded temporal recovery from saved fusion and tracking outputs.')
    parser.add_argument('--config', default=str(ROOT / 'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--fusion')
    parser.add_argument('--tracker', choices=('bytetrack', 'botsort'))
    parser.add_argument('--tracking-results')
    parser.add_argument('--strategy')
    parser.add_argument('--pose-quality')
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('recovery', {})
    params = RecoveryConfig(**{k: settings[k] for k in RecoveryConfig.__dataclass_fields__ if k in settings})
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    backend = args.tracker or settings.get('tracker', 'bytetrack')
    if backend not in ('bytetrack', 'botsort'):
        raise ValueError('Unsupported tracker')
    strategy = args.strategy or settings.get('fusion_strategy', cfg.get('fusion', {}).get('strategy', 'or'))
    fusion_path = project_path(cfg, args.fusion or settings.get('fusion_path', 'outputs/fusion/detection_fusion.csv'))
    track_path = project_path(cfg, args.tracking_results or f'outputs/tracking/{backend}/tracking_results.csv')
    manifest_path = project_path(cfg, args.manifest)
    pose_value = args.pose_quality or settings.get('pose_quality_path')
    pose_path = project_path(cfg, pose_value) if pose_value else None
    sources = {'fusion': fusion_path, 'tracking': track_path, 'manifest': manifest_path}
    if pose_path:
        sources['pose_quality'] = pose_path
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    fused = load_fused_frames(fusion_path, strategy)
    tracked = load_tracker_frames(track_path, fused, strategy, backend)
    poses = load_pose_quality(pose_path, fused)
    tracking_metadata = None
    provenance = track_path.parent / 'experiment_metadata.json'
    if provenance.exists():
        tracking_metadata = json.loads(provenance.read_text(encoding='utf-8'))
        if tracking_metadata.get('fusion_sha256') != hashes['fusion'] or tracking_metadata.get('fusion_strategy') != strategy:
            raise ValueError('Tracking provenance does not match fusion input')
    manifest = DatasetManifest.load(manifest_path)
    if manifest.df.video_id.duplicated().any() or set(manifest.df.video_id) != {v for v, _ in fused}:
        raise ValueError('Manifest/fusion video set mismatch or duplicates')
    videos = []
    fps = float(cfg['processing']['target_fps'])
    for _, rec in manifest.df.iterrows():
        video_id = str(rec.video_id)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', video_id) or video_id in ('.', '..'):
            raise ValueError('Invalid video ID')
        group = {i: row for (v, i), row in fused.items() if v == video_id}
        if str(rec.split) not in ('train', 'validation', 'test') or any(row['split'] != rec.split for row in group.values()):
            raise ValueError('Manifest split mismatch')
        path = manifest.resolve_video_path(root, rec)
        reader = VideoReader(path)
        meta = reader.metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'{video_id}: unreadable video')
        if set(group) != set(FrameSampler(reader, fps).selected_indices(meta)) or any(
                abs(row['timestamp_sec'] - i/meta.native_fps) > 1e-8 for i, row in group.items()):
            raise ValueError('Recovery inputs differ from original sampling timeline')
        videos.append((video_id, str(rec.split), path, meta, group))
    logger.info('Recovery validation PASS: %d videos, %d frames, tracker=%s', len(videos), len(fused), backend)
    output = project_path(cfg, settings.get('output_dir', 'outputs/recovery'))
    vis = output / 'visualizations'
    vis.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0
    for video_id, split, path, meta, group in videos:
        recovery = TemporalRecovery(params)
        local = []
        for index in sorted(group):
            row = group[index]
            selected = next((c['fused'] for c in json.loads(row['candidates_json']) if c['selected']), None)
            detection = Detection(tuple(selected['bbox']), selected['confidence'], selected['class_id'], selected['class_name'],
                                  index, row['timestamp_sec']) if selected else None
            result = recovery.update(detection, tracked[(video_id, index)], index, row['timestamp_sec'],
                                     (meta.width, meta.height), poses.get((video_id, index)))
            result.update(video_id=video_id, split=split, tracker_backend=backend, fusion_strategy=strategy)
            local.append(result)
        if settings.get('save_annotated_frames', True) and not args.no_visualizations:
            wanted = {r['frame_index']: r for r in local}
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f'Cannot decode {path}')
                index = 0
                while wanted:
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'{video_id}: early decoding failure')
                    if index in wanted:
                        row = wanted.pop(index)
                        target = vis / f'{video_id}_frame_{index:08d}.jpg'
                        if not cv2.imwrite(str(target), annotate(image, row)):
                            raise OSError(f'Cannot write {target}')
                        written += 1
                    index += 1
            finally:
                capture.release()
        rows.extend(local)
        logger.info('%s: accepted %d of %d detector misses', video_id,
                    sum(r['recovered_status'] == 'ACCEPTED' for r in local),
                    sum(r['detector_status'] == 'MISSED' for r in local))
    metrics = summarize_recovery(rows)
    if any(sha256_file(path) != hashes[name] for name, path in sources.items()):
        raise ValueError('Recovery inputs changed during execution')
    table = pd.DataFrame(rows)
    for column in ('final_infant_bbox', 'candidate_bbox'):
        table[column] = table[column].map(lambda box: json.dumps(list(box)) if box is not None else None)
    table.to_csv(output / 'recovery_results.csv', index=False)
    pd.DataFrame(metrics).to_csv(output / 'recovery_metrics.csv', index=False)
    metadata = dict(phase='PHASE_6_TEMPORAL_RECOVERY', configuration=cfg, recovery_parameters=asdict(params),
                    tracker=backend, fusion_strategy=strategy, total_frames=len(rows), total_videos=len(videos),
                    visualization_count=written, input_validation='PASS',
                    sources={name: dict(path=str(path), sha256=hashes[name]) for name, path in sources.items()},
                    tracking_provenance_status='AVAILABLE' if tracking_metadata else 'UNAVAILABLE',
                    source_tracking_experiment=tracking_metadata, optical_flow_used=False,
                    pose_quality_frames=len(poses), pose_validation_status='AVAILABLE' if poses else 'UNAVAILABLE',
                    ground_truth_status='UNAVAILABLE', false_recovery_rate_status='NOT_AVAILABLE',
                    confidence_interpretation='uncalibrated_temporal_consistency_score',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                    opencv_version=cv2.__version__, native_video_metadata=[dict(video_id=v, split=s, **asdict(m)) for v,s,_,m,_ in videos])
    (output / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    overall = next(m for m in metrics if m['scope'] == 'overall')
    logger.info('Recovery complete: %d accepted; miss rate %.4f -> %.4f; false recovery rate NOT_AVAILABLE',
                overall['recovered_frames'], overall['detector_miss_rate'], overall['recovered_miss_rate'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
