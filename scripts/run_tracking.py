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
import numpy as np
import pandas as pd
import ultralytics

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.video import VideoReader
from module1.data.sampler import FrameSampler
from module1.logging_utils import setup_logging
from module1.tracking import ByteTrackAdapter, BoTSORTAdapter, TrackingConfig, TrackingFrame
from module1.tracking.inputs import load_fused_frames
from module1.tracking.metrics import summarize_tracking


def track_row(track, video_id, split):
    row = asdict(track)
    row.update(video_id=video_id, split=split, timestamp_sec=track.timestamp,
               velocity_x=track.velocity[0], velocity_y=track.velocity[1],
               bbox_source='matched_track' if track.observed else 'prediction_only',
               ground_truth_status='UNAVAILABLE')
    row['bbox'] = json.dumps(row['bbox'])
    row['velocity'] = json.dumps(row['velocity'])
    return row


def annotate(image, primary, video_id, frame_index, name):
    height = image.shape[0]
    canvas = cv2.copyMakeBorder(image, 0, 64, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    if primary:
        box = tuple(int(round(v)) for v in primary.bbox)
        color = (40, 210, 70) if primary.observed else (30, 160, 240)
        cv2.rectangle(canvas, box[:2], box[2:], color, 2)
    lines = [f'{name} | {video_id} | frame={frame_index}',
             f'ID={primary.track_id} | {primary.track_state} | conf={primary.confidence:.3f}' if primary else 'NO CONFIRMED PRIMARY TRACK',
             'OBSERVED' if primary and primary.observed else 'MISSING - predictions are not recovered detections']
    for i, line in enumerate(lines):
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
        scale = min(.5, .5 * max(1, image.shape[1] - 8) / max(1, width))
        cv2.putText(canvas, line, (4, height + 17 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Track saved fusion candidates with ByteTrack and BoT-SORT.')
    parser.add_argument('--config', default=str(ROOT / 'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--fusion')
    parser.add_argument('--strategy')
    parser.add_argument('--tracker', choices=('both', 'bytetrack', 'botsort'), default='both')
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('tracking', {})
    parameters = TrackingConfig(**{k: settings[k] for k in TrackingConfig.__dataclass_fields__ if k in settings})
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    root = Path(cfg['_project_root'])
    source = project_path(cfg, args.fusion or settings.get('fusion_path', 'outputs/fusion/detection_fusion.csv'))
    strategy = args.strategy or settings.get('fusion_strategy', cfg.get('fusion', {}).get('strategy', 'or'))
    source_hash = sha256_file(source)
    frames = load_fused_frames(source, strategy)
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    if manifest.df.video_id.duplicated().any() or set(manifest.df.video_id) != {v for v, _ in frames}:
        raise ValueError('Fusion and manifest video IDs differ or contain duplicates')
    fps = float(cfg['processing']['target_fps'])
    videos = []
    for _, record in manifest.df.iterrows():
        video_id = str(record.video_id)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', video_id) or video_id in ('.', '..'):
            raise ValueError('Invalid video ID')
        group = {i: row for (v, i), row in frames.items() if v == video_id}
        if str(record.split) not in ('train', 'validation', 'test') or any(row['split'] != record.split for row in group.values()):
            raise ValueError(f'{video_id}: split mismatch')
        path = manifest.resolve_video_path(root, record)
        reader = VideoReader(path)
        meta = reader.metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'{video_id}: unreadable video')
        if set(group) != set(FrameSampler(reader, fps).selected_indices(meta)) or any(
                abs(row['timestamp_sec'] - i / meta.native_fps) > 1e-8 for i, row in group.items()):
            raise ValueError(f'{video_id}: fusion frames differ from original sampling timeline')
        videos.append((video_id, str(record.split), path, meta, group))
    logger.info('Tracking validation PASS: %d videos; %d shared frames; strategy=%s', len(videos), len(frames), strategy)
    backends = {'bytetrack': ByteTrackAdapter, 'botsort': BoTSORTAdapter}
    names = list(backends) if args.tracker == 'both' else [args.tracker]
    for name in names:
        output = project_path(cfg, settings.get('output_dir', 'outputs/tracking')) / name
        vis = output / 'visualizations'
        vis.mkdir(parents=True, exist_ok=True)
        frame_rows, audit_rows = [], []
        written = 0
        for video_id, split, path, meta, group in videos:
            seed = int(cfg.get('project', {}).get('seed', 42))
            np.random.seed(seed)
            cv2.setRNGSeed(seed)
            tracker = backends[name](parameters)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f'Cannot decode {path}')
            index = processed = missing = 0
            try:
                while processed < len(group):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'{video_id}: video ended before all requested frames were decoded')
                    if index in group:
                        row = group[index]
                        frame = TrackingFrame(image, index, row['timestamp_sec'])
                        tracks = tracker.update(row['detections'], frame)
                        audit_rows.extend(dict(track_row(t, video_id, split), tracker=name,
                                               fusion_strategy=strategy) for t in tracks)
                        primary = next((t for t in tracks if t.is_primary), None)
                        observed = primary is not None and primary.observed and primary.track_state == 'CONFIRMED'
                        missing = 0 if observed else missing + 1
                        result = track_row(primary, video_id, split) if primary else dict(
                            video_id=video_id, split=split, track_id=None, bbox=None, confidence=0.,
                            frame_index=index, timestamp=frame.timestamp, timestamp_sec=frame.timestamp,
                            velocity=None, velocity_x=None, velocity_y=None, age=0, number_of_hits=0,
                            number_of_misses=0, track_state='MISSING', consecutive_misses=0,
                            is_primary=False, observed=False, reliability=0., class_id=parameters.class_id,
                            bbox_source='none', ground_truth_status='UNAVAILABLE')
                        result.update(observed=observed, consecutive_missing_frames=missing,
                                      tracker=name, fusion_strategy=strategy, num_input_candidates=len(row['detections']),
                                      num_active_tracks=sum(t.observed for t in tracks))
                        frame_rows.append(result)
                        if settings.get('save_annotated_frames', True) and not args.no_visualizations:
                            target = vis / f'{video_id}_frame_{index:08d}.jpg'
                            if not cv2.imwrite(str(target), annotate(image, primary, video_id, index, name)):
                                raise OSError(f'Cannot write {target}')
                            written += 1
                        processed += 1
                    index += 1
            finally:
                capture.release()
            logger.info('%s %s: processed %d frames', name, video_id, processed)
        metrics = summarize_tracking(frame_rows)
        if sha256_file(source) != source_hash:
            raise ValueError('Fusion inputs changed during tracking')
        pd.DataFrame(frame_rows).to_csv(output / 'tracking_results.csv', index=False)
        audit_columns = list(frame_rows[0]) if frame_rows else []
        pd.DataFrame(audit_rows, columns=audit_columns).to_csv(output / 'tracks.csv', index=False)
        pd.DataFrame(metrics).to_csv(output / 'tracking_metrics.csv', index=False)
        metadata = dict(phase='PHASE_5_TRACKING', tracker=name, configuration=cfg, tracker_parameters=asdict(parameters),
                        fusion_path=str(source), fusion_sha256=source_hash, fusion_strategy=strategy,
                        manifest_path=str(manifest_path), manifest_sha256=sha256_file(manifest_path),
                        total_frames=len(frame_rows), total_videos=len(videos), visualization_count=written,
                        frame_alignment='PASS', processing_fps=fps, velocity_units='original_pixels_per_second',
                        age_and_buffer_units='processed_frames', with_reid=False,
                        ground_truth_status='UNAVAILABLE', id_switches_status='NOT_AVAILABLE',
                        temporal_recovery_implemented=False, ultralytics_version=ultralytics.__version__,
                        python_version=platform.python_version(), opencv_version=cv2.__version__,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        native_video_metadata=[dict(video_id=v, split=s, **asdict(m)) for v, s, _, m, _ in videos])
        source_metadata = source.parent / 'experiment_metadata.json'
        if source_metadata.exists():
            metadata['source_fusion_experiment'] = json.loads(source_metadata.read_text(encoding='utf-8'))
        (output / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        overall = next(m for m in metrics if m['scope'] == 'overall')
        logger.info('%s complete: coverage %.4f; fragments %d; observed primary ID changes %d',
                    name, overall['track_coverage'], overall['number_of_fragments'], overall['observed_primary_id_changes'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
