from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.detection.fusion import DetectorFusion, FusionConfig, STRATEGIES
from module1.detection.fusion_io import (
    ablation_metrics, fusion_row, observation_row, read_detection_csv, validate_alignment,
)
from module1.logging_utils import setup_logging


def frame_filename(video_id, index):
    return f'{video_id}_frame_{index:08d}.jpg'


def annotate(frame, row):
    height = frame.shape[0]
    canvas = cv2.copyMakeBorder(frame, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    legends = []
    # Draw source evidence first, with the selected fused box last.
    for key, label, color in (
        ('yolo_bbox', 'YOLO', (230, 150, 20)),
        ('rtdetr_bbox', 'RT-DETR', (20, 180, 20)),
        ('fused_bbox', 'Fused', (30, 30, 230)),
    ):
        if row[key] is not None:
            x1, y1, x2, y2 = map(lambda v: int(round(v)), json.loads(row[key]))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            conf = row['fused_confidence' if key == 'fused_bbox' else key.replace('_bbox', '_confidence')]
            legends.append((f'{label}: {conf:.3f}', color))
        else:
            legends.append((f'{label}: MISSED', color))
    labels = [
        (f"{row['video_id']} | frame={row['frame_index']} | t={row['timestamp_sec']:.3f}s", (255, 255, 255)),
        (f"{row['fusion_strategy']} | {row['detection_status']} | {row['selected_source']}", (255, 255, 255)),
        *legends,
    ]
    # Put labels outside the original image, preserving source-coordinate geometry.
    width = max(1, canvas.shape[1] - 8)
    for index, (label, color) in enumerate(labels):
        scale = min(.48, .48 * width / max(1, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]))
        cv2.putText(canvas, label, (4, height + 16 + index * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    return canvas


def save_visualizations(video_path, video_id, rows, destination):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'Cannot open {video_path}')
    wanted = {r['frame_index']: r for r in rows}
    saved = 0
    try:
        index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'{video_id}: video ended before {len(wanted)} required frames were decoded')
            if index in wanted:
                row = wanted.pop(index)
                path = destination / frame_filename(video_id, index)
                if not cv2.imwrite(str(path), annotate(frame, row)):
                    raise OSError(f'Could not write {path}')
                saved += 1
            index += 1
    finally:
        capture.release()
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fuse saved YOLO and RT-DETR candidates, without inference or tracking.')
    parser.add_argument('--config', default=str(ROOT / 'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--yolo-detections')
    parser.add_argument('--rtdetr-detections')
    parser.add_argument('--strategy', choices=STRATEGIES)
    parser.add_argument('--iou', type=float)
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('fusion', {})
    fusion = FusionConfig(**{k: settings[k] for k in FusionConfig.__dataclass_fields__ if k in settings})
    if args.strategy:
        fusion = replace(fusion, strategy=args.strategy)
    if args.iou is not None:
        fusion = replace(fusion, iou_threshold=args.iou)
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    sources = {
        'yolo': project_path(cfg, args.yolo_detections or settings.get('yolo_detections', 'outputs/yolo/yolo_detections.csv')),
        'rtdetr': project_path(cfg, args.rtdetr_detections or settings.get('rtdetr_detections', 'outputs/rtdetr/rtdetr_detections.csv')),
    }
    yolo, rtdetr = (read_detection_csv(sources[name]) for name in ('yolo', 'rtdetr'))
    validate_alignment(yolo, rtdetr)
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    if manifest.df.video_id.duplicated().any():
        raise ValueError('Manifest contains duplicate video IDs')
    if set(manifest.df.video_id) != {k[0] for k in yolo}:
        raise ValueError('Manifest and detector video sets differ')
    paths, native = {}, []
    fps = float(cfg['processing']['target_fps'])
    for _, record in manifest.df.iterrows():
        video_id = str(record.video_id)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', video_id) or video_id in ('.', '..'):
            raise ValueError(f'Video ID cannot be used as a deterministic filename: {video_id}')
        frames = [f for (v, _), f in yolo.items() if v == video_id]
        if any(f.split != str(record.split) for f in frames):
            raise ValueError(f'{video_id}: manifest split differs from detector outputs')
        path = manifest.resolve_video_path(root, record)
        reader = VideoReader(path)
        meta = reader.metadata()
        if not meta.opened_successfully or meta.native_fps <= 0:
            raise ValueError(f'{video_id}: unreadable video or invalid native FPS')
        expected = set(FrameSampler(reader, fps).selected_indices(meta))
        if expected != {f.frame_index for f in frames} or any(
                abs(f.timestamp - f.frame_index / meta.native_fps) > 1e-8 for f in frames):
            raise ValueError(f'{video_id}: saved detections do not match the configured native-timeline sampling plan')
        paths[video_id] = path
        native.append(dict(video_id=video_id, split=str(record.split), **asdict(meta)))
    logger.info('Fusion input validation PASS: %d frames across %d videos', len(yolo), len(paths))
    engines = {s: DetectorFusion(replace(fusion, strategy=s)) for s in STRATEGIES}
    methods = {'YOLO': [], 'RT-DETR': []}
    for strategy in STRATEGIES:
        methods['YOLO+RT-DETR:' + strategy] = []
    rows = []
    for key in sorted(yolo):
        y, r = yolo[key], rtdetr[key]
        methods['YOLO'].append(observation_row(y, y.detections[0] if y.detections else None))
        methods['RT-DETR'].append(observation_row(r, r.detections[0] if r.detections else None))
        for strategy, engine in engines.items():
            result = engine.fuse(list(y.detections), list(r.detections))
            row = fusion_row(y, r, result, strategy, fusion.strategy)
            rows.append(row)
            methods['YOLO+RT-DETR:' + strategy].append(row)
    metrics = ablation_metrics(methods)
    output = project_path(cfg, settings.get('output_dir', 'outputs/fusion'))
    visualizations = output / 'visualizations'
    visualizations.mkdir(parents=True, exist_ok=True)
    saved = 0
    if settings.get('save_annotated_frames', True) and not args.no_visualizations:
        default_rows = methods['YOLO+RT-DETR:' + fusion.strategy]
        for video_id, path in paths.items():
            group = [r for r in default_rows if r['video_id'] == video_id]
            count = save_visualizations(path, video_id, group, visualizations)
            saved += count
            logger.info('%s: wrote %d fusion visualizations (%s)', video_id, count, fusion.strategy)
    pd.DataFrame(rows).to_csv(output / 'detection_fusion.csv', index=False)
    pd.DataFrame(metrics).to_csv(output / 'fusion_metrics.csv', index=False)
    metadata = dict(
        phase='PHASE_4_DETECTION_FUSION', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path), manifest_sha256=sha256_file(manifest_path),
        configuration=cfg, fusion_configuration=asdict(fusion), evaluated_strategies=list(STRATEGIES),
        source_csvs={name: dict(path=str(path), sha256=sha256_file(path)) for name, path in sources.items()},
        input_frame_alignment='PASS', sampling_plan_validation='PASS', total_frames=len(yolo),
        total_videos=len(paths), visualization_strategy=fusion.strategy, visualization_count=saved,
        ground_truth_status='UNAVAILABLE', metric_interpretation='detector_observation_not_ground_truth_accuracy',
        selection_rule='highest confidence plus configurable agreement bonus; deterministic ties',
        matching_rule='class-aware greedy highest IoU; one candidate per detector per pair',
        weighted_box_fusion_variant='two-model one-to-one pairs; confidence penalized by missing model weight',
        native_video_metadata=native, python_version=platform.python_version(), opencv_version=cv2.__version__,
    )
    source_metadata = {}
    for name, path in sources.items():
        meta_path = path.parent / 'metadata/experiment_metadata.json'
        if meta_path.exists():
            source_metadata[name] = json.loads(meta_path.read_text(encoding='utf-8'))
    metadata['source_experiments'] = source_metadata
    (output / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    for metric in metrics:
        if metric['scope'] == 'overall':
            logger.info('%s: detection rate %.4f; longest miss %d', metric['method'], metric['detection_rate'], metric['longest_consecutive_miss'])
    logger.info('Phase 4 complete: %s', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
