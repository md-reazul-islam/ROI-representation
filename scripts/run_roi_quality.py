from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import pandas as pd
import numpy as np

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.logging_utils import setup_logging
from module1.roi import ROI_NAMES
from module1.roi_quality import ROIQualityCalculator, ROIQualityConfig
from module1.roi_quality.inputs import load_roi_table, pose_record_for_roi


def draw_quality(image, rows):
    canvas = image.copy()
    colors = [(40, 210, 60), (210, 150, 40), (200, 70, 180), (70, 180, 220)]
    for idx, row in enumerate(rows):
        if row['roi_status'] != 'AVAILABLE' or not row.get('bbox'):
            continue
        bbox = json.loads(row['bbox'])
        polygon = json.loads(row['polygon'])
        points = np.rint(np.asarray(polygon, dtype=float)).astype(np.int32)
        color = colors[idx % len(colors)]
        if len(points) >= 3:
            cv2.polylines(canvas, [points], True, color, 1)
        else:
            cv2.rectangle(canvas, tuple(np.rint(bbox[:2]).astype(int)), tuple(np.rint(bbox[2:]).astype(int)), color, 1)
        centroid = json.loads(row['centroid'])
        # First frames have no motion; unavailable quality values are None.
        skin, motion = row.get('visible_skin_proportion'), row.get('motion_magnitude')
        text = (f"{row['roi_name']} s={'NA' if skin is None else f'{skin:.2f}'}"
                f" m={'NA' if motion is None else f'{motion:.2f}'}")
        cv2.putText(canvas, text, tuple(np.rint(centroid).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .28, color, 1, cv2.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Characterize candidate ROI image, pose and temporal quality features.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--roi-dir')
    parser.add_argument('--pose-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('roi_quality', {})
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    roi_dir = project_path(cfg, args.roi_dir or settings.get('roi_dir', 'outputs/roi'))
    pose_dir = project_path(cfg, args.pose_dir or settings.get('pose_dir', 'outputs/pose'))
    output = project_path(cfg, args.output_dir or settings.get('output_dir', 'outputs/roi_quality'))
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError('ROI quality output directory is not empty; use --output-dir for a new experiment')
    roi_records, roi_metadata = load_roi_table(roi_dir)
    pose_metadata = json.loads((pose_dir/'experiment_metadata.json').read_text(encoding='utf-8'))
    names = pose_metadata.get('keypoint_names')
    if not isinstance(names, list) or len(names) != 308:
        raise ValueError('Pose metadata must identify 308 keypoint names')
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    if set(manifest.df.video_id) != {video for video, _, _ in roi_records}:
        raise ValueError('ROI/manifest video set mismatch')
    quality = ROIQualityConfig(**settings.get('parameters', {}))
    calculator = ROIQualityCalculator(names, quality)
    sources = {'manifest': manifest_path, 'roi_results': roi_dir/'candidate_rois.csv',
               'roi_metadata': roi_dir/'experiment_metadata.json', 'pose_metadata': pose_dir/'experiment_metadata.json'}
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    videos = {}
    for _, record in manifest.df.iterrows():
        video = str(record.video_id)
        path = manifest.resolve_video_path(root, record)
        meta = VideoReader(path).metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'Unreadable video: {video}')
        frames = sorted({frame for current_video, frame, _ in roi_records if current_video == video})
        expected = set(FrameSampler(VideoReader(path), float(cfg['processing']['target_fps'])).selected_indices(meta))
        if not set(frames).issubset(expected):
            raise ValueError(f'ROI frame index outside sampling timeline: {video}')
        videos[video] = (path, meta, frames)
    logger.info('ROI quality input validation PASS: %d frames across %d videos', len({(v, f) for v, f, _ in roi_records}), len(videos))
    (output/'visualizations').mkdir(parents=True)
    rows = []
    visualizations = 0
    for video, (path, meta, frames) in videos.items():
        by_frame = {frame: [roi_records[(video, frame, name)] for name in ROI_NAMES] for frame in frames}
        previous_image, previous_pose, previous_rows, previous_timestamp = None, None, None, None
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f'Cannot decode video: {path}')
        current_index = 0
        try:
            for frame_index in frames:
                while current_index <= frame_index:
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'{video}: early decoding failure at frame {frame_index}')
                    read_index = current_index
                    current_index += 1
                    if read_index == frame_index:
                        current_image = image
                        break
                frame_rows = by_frame[frame_index]
                pose = pose_record_for_roi(pose_dir, frame_rows[0])
                current_rows = []
                for row in frame_rows:
                    features = calculator.calculate(current_image, row, previous_image, previous_rows.get(row['roi_name']) if previous_rows else None,
                                                     pose['keypoints'], previous_pose['keypoints'] if previous_pose else None,
                                                     float(row['timestamp_sec']), previous_timestamp)
                    features['pose_confidence'] = pose.get('confidence')
                    current_rows.append(dict(row, **features))
                    rows.append(current_rows[-1])
                if not args.no_visualizations and settings.get('save_annotated_frames', True):
                    target = output/'visualizations'/f'{video}_frame_{frame_index:08d}.jpg'
                    if not cv2.imwrite(str(target), draw_quality(current_image, current_rows)):
                        raise OSError(f'Cannot write {target}')
                    visualizations += 1
                previous_image, previous_pose, previous_rows, previous_timestamp = current_image, pose, {r['roi_name']: r for r in frame_rows}, float(frame_rows[0]['timestamp_sec'])
        finally:
            capture.release()
        logger.info('%s: %d frames, %d ROI records', video, len(frames), len(frames)*17)
    if any(sha256_file(path) != digest for key, digest in hashes.items() for path in [sources[key]]):
        raise ValueError('ROI quality inputs changed during execution')
    table = pd.DataFrame(rows)
    table.to_csv(output/'roi_quality_features.csv', index=False)
    metadata = dict(phase='PHASE_9_ROI_QUALITY_CHARACTERIZATION', run_status='COMPLETE',
                    total_frames=len({(v, f) for v, f, _ in roi_records}), roi_records=len(rows),
                    visualization_count=visualizations, quality_parameters=asdict(quality),
                    source_hashes={key: dict(path=str(sources[key]), sha256=hashes[key]) for key in sources},
                    normalized_quality_factors=[
                        'pose_confidence', 'visible_skin_proportion', 'occlusion_proxy', 'blur_score',
                        'illumination_score', 'saturation_score', 'motion_magnitude', 'landmark_velocity',
                        'roi_centroid_displacement', 'roi_area_variation', 'consecutive_roi_iou',
                        'temporal_consistency', 'swaddling_proxy'],
                    sam2_features_preserved=['SAM2_mask_support', 'SAM2_mask_quality',
                        'roi_inside_mask_ratio', 'roi_mask_overlap', 'visible_infant_pixel_ratio',
                        'landmark_inside_mask_ratio', 'mask_supported_roi_status'],
                    sam2_mode=roi_metadata.get('sam2_mode', 'BASELINE'),
                    unavailable_temporal_feature_note='First frame per video has no preceding frame; temporal values are null.',
                    skin_note='visible_skin_proportion and occlusion_proxy are color-based proxies, not segmentation or clinical occlusion labels',
                    confidence_scoring_implemented=False, final_roi_confidence_implemented=False,
                    ground_truth_status='UNAVAILABLE', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), opencv_version=cv2.__version__,
                    roi_input_run_status=roi_metadata.get('run_status'), pose_input_run_status=pose_metadata.get('run_status'))
    (output/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('ROI quality complete: %d records, %d visualizations', len(rows), visualizations)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
