from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.sampler import FrameSampler, SampledFrame
from module1.data.video import VideoReader
from module1.logging_utils import setup_logging
from module1.pose import Sapiens2PoseEstimator, PoseQualityConfig
from module1.pose.inputs import load_recovery_frames
from module1.segmentation.inputs import load_mask, load_sam2_table


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')


def runtime_versions():
    versions = {}
    for package in ('numpy', 'pandas', 'opencv-python', 'torch', 'safetensors', 'sapiens'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'UNAVAILABLE'
    return versions


def annotate(image, row, result, skeleton, threshold):
    canvas = image.copy()
    if row['bbox'] is not None:
        x1, y1, x2, y2 = (round(v) for v in row['bbox'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 200, 60), 2)
    if result is not None:
        points = {p.index: (round(p.x), round(p.y)) for p in result.keypoints
                  if math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.confidence)
                  and p.confidence >= threshold and 0 <= p.x < image.shape[1] and 0 <= p.y < image.shape[0]}
        for a, b in skeleton:
            if a in points and b in points:
                cv2.line(canvas, points[a], points[b], (220, 180, 40), 1)
        for point in points.values():
            cv2.circle(canvas, point, 2, (60, 70, 245), -1)
    h = canvas.shape[0]
    canvas = cv2.copyMakeBorder(canvas, 0, 64, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    lines = [f"{row['video_id']} | frame={row['frame_index']} | t={row['timestamp_sec']:.3f}s",
             f"source={row['final_status']} | pose={result.status if result else 'NO_BBOX'}",
             f"quality={result.pose_quality:.3f} | confidence={result.confidence:.3f}" if result else 'Pose not estimated']
    for i, line in enumerate(lines):
        width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
        scale = min(.45, .45 * max(1, image.shape[1]-8) / max(1, width))
        cv2.putText(canvas, line, (4, h+17+i*20), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sapiens2 308-keypoint poses from immutable Phase 6 boxes.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--recovery')
    parser.add_argument('--sam2-dir', help='Use reliable Phase 12 masks as an optional pose input')
    parser.add_argument('--without-sam2', action='store_true', help='Force the original direct Phase 6-to-pose baseline')
    parser.add_argument('--source-root')
    parser.add_argument('--model-config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'))
    parser.add_argument('--output-dir')
    parser.add_argument('--max-frames', type=int, help='Explicit partial smoke run; use a separate output directory')
    parser.add_argument('--no-visualizations', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate Phase 6 timeline without loading a model')
    args = parser.parse_args(argv)
    if args.sam2_dir and args.without_sam2:
        raise ValueError('--sam2-dir and --without-sam2 are mutually exclusive')
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    cfg = load_config(args.config)
    settings = cfg.get('pose', {})
    quality = PoseQualityConfig(**settings.get('quality', {}))
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    recovery_path = project_path(cfg, args.recovery or settings.get('recovery_path', 'outputs/recovery/recovery_results.csv'))
    manifest_path = project_path(cfg, args.manifest)
    input_hashes = {str(path): sha256_file(path) for path in (recovery_path, manifest_path)}
    frames = load_recovery_frames(recovery_path)
    sam2_value = None if args.without_sam2 else (args.sam2_dir or settings.get('sam2_dir') or
        (cfg.get('sam2', {}).get('output_dir', 'outputs/sam2') if cfg.get('sam2', {}).get('enabled') else None))
    sam2_dir = project_path(cfg, sam2_value) if sam2_value else None
    sam2_frames = None
    sam2_metadata = None
    if sam2_dir is not None:
        sam2_frames, sam2_metadata = load_sam2_table(sam2_dir)
        if set(sam2_frames) != set(frames):
            raise ValueError('SAM 2 and Temporal Recovery frame sets differ')
        if sam2_metadata.get('source_hashes', {}).get('recovery') != input_hashes[str(recovery_path)]:
            raise ValueError('SAM 2 used a different Temporal Recovery input')
        for key, row in sam2_frames.items():
            recovery = frames[key]
            if (row['split'] != recovery['split'] or
                    abs(row['timestamp_sec']-recovery['timestamp_sec']) > 1e-8 or
                    (json.loads(row['final_infant_bbox']) if row['final_infant_bbox'] else None) !=
                    (list(recovery['bbox']) if recovery['bbox'] is not None else None)):
                raise ValueError('SAM 2 and Temporal Recovery localization metadata disagree')
        for path in (sam2_dir/'sam2_masks.csv', sam2_dir/'metadata/experiment_metadata.json'):
            input_hashes[str(path)] = sha256_file(path)
    manifest = DatasetManifest.load(manifest_path)
    if manifest.df.video_id.duplicated().any() or set(manifest.df.video_id) != {key[0] for key in frames}:
        raise ValueError('Recovery/manifest video set mismatch or duplicate videos')
    videos = []
    for _, record in manifest.df.iterrows():
        group = {i: row for (v, i), row in frames.items() if v == record.video_id}
        if record.split not in ('train', 'validation', 'test') or any(row['split'] != record.split for row in group.values()):
            raise ValueError('Recovery/manifest split mismatch')
        path = manifest.resolve_video_path(root, record)
        reader = VideoReader(path)
        meta = reader.metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'Unreadable video: {path}')
        if set(group) != set(FrameSampler(reader, float(cfg['processing']['target_fps'])).selected_indices(meta)):
            raise ValueError('Recovery frames differ from original sampling timeline')
        if any(abs(row['timestamp_sec']-i/meta.native_fps) > 1e-8 for i, row in group.items()):
            raise ValueError('Recovery timestamps differ from original sampling timeline')
        for row in group.values():
            box = row['bbox']
            if box is not None and (box[2] <= 0 or box[3] <= 0 or box[0] >= meta.width or box[1] >= meta.height):
                raise ValueError('Final infant box does not intersect the original frame')
        videos.append((str(record.video_id), path, meta, group))
    logger.info('Pose input validation PASS: %d videos, %d frames, %d final boxes',
                len(videos), len(frames), sum(row['bbox'] is not None for row in frames.values()))
    if args.validate_only:
        return 0
    source = project_path(cfg, args.source_root or settings.get('source_root', 'external/sapiens2'))
    model_config = project_path(cfg, args.model_config or settings.get('model_config',
        'external/sapiens2/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py'))
    checkpoint = project_path(cfg, args.checkpoint or settings.get('checkpoint', 'models/sapiens2_0.4b_pose.safetensors'))
    output = project_path(cfg, args.output_dir or settings.get('output_dir', 'outputs/pose'))
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError('Pose output directory is not empty; use --output-dir for a new experiment')
    logger.info('Loading Sapiens2 checkpoint: %s; device=%s', checkpoint, args.device or settings.get('device', 'cpu'))
    estimator = Sapiens2PoseEstimator(source, model_config, checkpoint,
                                    args.device or settings.get('device', 'cpu'), quality,
                                    dtype=args.dtype or settings.get('dtype', 'float32'))
    logger.info('Sapiens2 ready: 308 keypoints; starting frame inference')
    for path in (model_config, checkpoint, source/'sapiens/pose/configs/_base_/keypoints308.py'):
        input_hashes[str(path)] = sha256_file(path)
    try:
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'UNAVAILABLE'
    (output/'keypoints').mkdir(parents=True)
    (output/'visualizations').mkdir()
    metadata = dict(phase='PHASE_7_SAPIENS2_POSE', run_status='RUNNING', total_input_frames=len(frames),
                    requested_frame_limit=args.max_frames, configuration=cfg, quality_parameters=asdict(quality),
                    input_hashes=input_hashes, sapiens2_revision=revision, model_config=str(model_config),
                    checkpoint=str(checkpoint), device=getattr(estimator.backend, 'device', 'UNAVAILABLE'),
                    dtype=args.dtype or settings.get('dtype', 'float32'), checkpoint_loading='strict_streamed_pread',
                    coordinate_system='original_frame_xy_pixels', bbox_source='phase6_final_unchanged',
                    keypoint_count=308, keypoint_names=estimator.names, skeleton=estimator.skeleton,
                    confidence_interpretation='raw_heatmap_peak_not_calibrated_probability',
                    pose_quality_interpretation='heuristic_not_neonatal_validation_or_ROI_score',
                    visibility_status='UNAVAILABLE_FROM_OFFICIAL_BACKEND', ground_truth_status='UNAVAILABLE',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                    runtime_versions=runtime_versions(),
                    sam2_mode='MASK_CONSTRAINED' if sam2_frames is not None else 'DIRECT_RECOVERY_BOX',
                    sam2_source_dir=str(sam2_dir) if sam2_dir is not None else None,
                    sam2_source_run_status=sam2_metadata.get('run_status') if sam2_metadata else None,
                    sam2_mask_application='neutral ImageNet-mean BGR outside reliable original-frame mask; unchanged Phase 6 bbox',
                    native_video_metadata=[dict(video_id=v, **asdict(m)) for v, _, m, _ in videos])
    write_json(output/'experiment_metadata.json', metadata)
    rows, vis_count = [], 0
    try:
        for video, path, meta, group in videos:
            estimator.reset()
            previous_track = None
            capture = cv2.VideoCapture(str(path))
            local_count = 0
            try:
                if not capture.isOpened():
                    raise ValueError(f'Cannot decode {path}')
                for index in range(max(group)+1):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'{video}: early decoding failure at frame {index}')
                    if index not in group:
                        continue
                    row = group[index]
                    if image.shape[:2] != (meta.height, meta.width):
                        raise ValueError('Decoded frame dimensions differ from video metadata')
                    if row['track_id'] != previous_track:
                        estimator.reset()
                    previous_track = row['track_id']
                    sam2_row = sam2_frames[(video, index)] if sam2_frames is not None else None
                    mask, mask_error = None, None
                    if sam2_row is not None and row['bbox'] is None:
                        mask_error = 'NO_PHASE6_BBOX_FOR_POSE'
                    elif sam2_row is not None:
                        try:
                            mask = load_mask(sam2_dir, sam2_row, expected_shape=image.shape[:2])
                        except (OSError, ValueError) as exc:
                            mask_error = f'{type(exc).__name__}: {exc}'
                            logger.warning('%s frame %d: invalid SAM 2 mask; using Phase 6 bbox: %s',
                                           video, index, mask_error)
                        if mask is None and mask_error is None:
                            mask_error = sam2_row.get('failure_reason') or sam2_row['segmentation_status']
                    if row['bbox'] is None:
                        result = None
                    elif mask is not None:
                        result = estimator.estimate(SampledFrame(index, row['timestamp_sec'], image),
                                                    row['bbox'], mask=mask)
                    else:
                        result = estimator.estimate(SampledFrame(index, row['timestamp_sec'], image), row['bbox'])
                    name = f'{video}_frame_{index:08d}'
                    payload = dict(video_id=video, split=row['split'], frame_index=index, timestamp=row['timestamp_sec'],
                                   source_final_status=row['final_status'], bbox=row['bbox'],
                                   source_final_confidence=row['final_infant_confidence'], status='NO_BBOX',
                                   keypoints=[], confidence=None, pose_quality=None, checks={},
                                   sam2_segmentation_status=sam2_row['segmentation_status'] if sam2_row else 'DISABLED',
                                   sam2_mask_path=sam2_row['mask_path'] if mask is not None else None,
                                   sam2_mask_quality=float(sam2_row['mask_quality']) if mask is not None else None,
                                   sam2_mask_used=mask is not None, sam2_fallback_reason=mask_error)
                    if result is not None:
                        if result.bbox != row['bbox']:
                            raise ValueError('Pose estimator changed the final infant bbox')
                        payload.update(asdict(result))
                    write_json(output/'keypoints'/f'{name}.json', payload)
                    rows.append(dict(video_id=video, split=row['split'], frame_index=index, timestamp=row['timestamp_sec'],
                                     timestamp_sec=row['timestamp_sec'], final_status=row['final_status'],
                                     final_infant_bbox=json.dumps(row['bbox']) if row['bbox'] is not None else None,
                                     final_infant_confidence=row['final_infant_confidence'], track_id=row['track_id'],
                                     pose_status=payload['status'], confidence=payload['confidence'],
                                     pose_quality=payload['pose_quality'], keypoint_count=len(payload['keypoints']),
                                     keypoints_path=f'keypoints/{name}.json', checks_json=json.dumps(payload['checks']),
                                     sam2_segmentation_status=payload['sam2_segmentation_status'],
                                     sam2_mask_path=payload['sam2_mask_path'],
                                     sam2_mask_quality=payload['sam2_mask_quality'],
                                     sam2_mask_used=payload['sam2_mask_used'],
                                     sam2_fallback_reason=payload['sam2_fallback_reason'],
                                     visibility_status='UNAVAILABLE' if result is None or all(p.visibility is None for p in result.keypoints) else 'AVAILABLE'))
                    if settings.get('save_annotated_frames', True) and not args.no_visualizations:
                        if not cv2.imwrite(str(output/'visualizations'/f'{name}.jpg'),
                                           annotate(image, row, result, estimator.skeleton, quality.confidence_threshold)):
                            raise OSError('Cannot write pose visualization')
                        vis_count += 1
                    local_count += 1
                    if local_count == 1 or local_count % 25 == 0:
                        logger.info('%s: %d pose-frame records written', video, local_count)
                    if args.max_frames is not None and len(rows) >= args.max_frames:
                        break
            finally:
                capture.release()
            logger.info('%s: completed %d frames', video, local_count)
            if args.max_frames is not None and len(rows) >= args.max_frames:
                break
        if any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
            raise ValueError('Pose inputs changed during execution')
        pd.DataFrame(rows).to_csv(output/'pose_results.csv', index=False)
        metadata.update(run_status='COMPLETE' if len(rows) == len(frames) else 'PARTIAL',
                        processed_frames=len(rows), visualization_count=vis_count,
                        status_counts=dict(Counter(row['pose_status'] for row in rows)))
        write_json(output/'experiment_metadata.json', metadata)
    except Exception as exc:
        metadata.update(run_status='FAILED', processed_frames=len(rows), error=str(exc))
        write_json(output/'experiment_metadata.json', metadata)
        raise
    logger.info('Pose run %s: %d frames; %s', metadata['run_status'], len(rows), metadata['status_counts'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
