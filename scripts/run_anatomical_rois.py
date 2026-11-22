from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.logging_utils import setup_logging
from module1.roi import AnatomicalROIMapper
from module1.roi.inputs import load_pose_table
from module1.segmentation.inputs import load_mask, load_sam2_table
from module1.segmentation.support import landmark_mask_support, roi_mask_support


def draw_rois(image, row_group):
    canvas = image.copy()
    colors = [(40, 210, 60), (210, 150, 40), (200, 70, 180), (70, 180, 220)]
    for idx, row in enumerate(row_group):
        if row['roi_status'] != 'AVAILABLE' or not row['bbox']:
            continue
        box = json.loads(row['bbox'])
        polygon = json.loads(row['polygon'])
        points = [(round(point[0]), round(point[1])) for point in polygon]
        if len(points) >= 3:
            cv2.polylines(canvas, [__import__('numpy').array(points, dtype='int32')], True, colors[idx % len(colors)], 1)
        else:
            cv2.rectangle(canvas, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), colors[idx % len(colors)], 1)
        cx, cy = json.loads(row['centroid'])
        cv2.putText(canvas, row['roi_name'], (round(cx), round(cy)), cv2.FONT_HERSHEY_SIMPLEX, .32,
                    colors[idx % len(colors)], 1, cv2.LINE_AA)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate dynamic anatomical candidate ROIs from Sapiens2 keypoints.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--pose-dir')
    parser.add_argument('--sam2-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--min-keypoint-confidence', type=float)
    parser.add_argument('--min-support-points', type=int)
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('roi', {})
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    pose_dir = project_path(cfg, args.pose_dir or settings.get('pose_dir', 'outputs/pose'))
    output = project_path(cfg, args.output_dir or settings.get('output_dir', 'outputs/roi'))
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError('ROI output directory is not empty; use --output-dir for a new experiment')
    records, names, pose_metadata = load_pose_table(pose_dir)
    sam2_value = args.sam2_dir or pose_metadata.get('sam2_source_dir')
    sam2_dir = project_path(cfg, sam2_value) if sam2_value else None
    sam2_records = None
    if sam2_dir is not None:
        declared = pose_metadata.get('sam2_source_dir')
        if (pose_metadata.get('sam2_mode') != 'MASK_CONSTRAINED' or not declared
                or project_path(cfg, declared).resolve() != sam2_dir.resolve()):
            raise ValueError('ROI SAM 2 source must match the mask-constrained pose run')
        sam2_csv = (sam2_dir/'sam2_masks.csv').resolve()
        if not any(Path(path).resolve() == sam2_csv and digest == sha256_file(sam2_csv)
                   for path, digest in pose_metadata.get('input_hashes', {}).items()):
            raise ValueError('ROI SAM 2 masks changed since pose inference')
        sam2_records, _ = load_sam2_table(sam2_dir)
        if not set(records).issubset(sam2_records):
            raise ValueError('SAM 2 masks do not cover all pose frames')
        for key, pose_row in records.items():
            mask_row = sam2_records[key]
            if (mask_row['split'] != pose_row['split'] or
                    abs(mask_row['timestamp_sec']-pose_row['timestamp_sec']) > 1e-8):
                raise ValueError('SAM 2 and pose timelines disagree')
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    if manifest.df.video_id.duplicated().any() or set(manifest.df.video_id) != {video for video, _ in records}:
        raise ValueError('Pose/manifest video set mismatch or duplicate videos')
    threshold = args.min_keypoint_confidence if args.min_keypoint_confidence is not None else settings.get('min_keypoint_confidence', .5)
    min_support = args.min_support_points if args.min_support_points is not None else settings.get('min_support_points', 2)
    mapper = AnatomicalROIMapper(names, threshold, min_support)
    sources = {'manifest': manifest_path, 'pose_results': pose_dir/'pose_results.csv', 'pose_metadata': pose_dir/'experiment_metadata.json'}
    if sam2_dir is not None:
        sources.update(sam2_results=sam2_dir/'sam2_masks.csv',
                       sam2_metadata=sam2_dir/'metadata/experiment_metadata.json')
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    videos = {}
    for _, record in manifest.df.iterrows():
        video = str(record.video_id)
        path = manifest.resolve_video_path(root, record)
        meta = VideoReader(path).metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'Unreadable video: {video}')
        group = {index: value for (v, index), value in records.items() if v == video}
        expected = set(FrameSampler(VideoReader(path), float(cfg['processing']['target_fps'])).selected_indices(meta))
        if not set(group).issubset(expected):
            raise ValueError(f'Pose frame index outside original sampling timeline: {video}')
        if any(abs(value['timestamp_sec']-index/meta.native_fps) > 1e-8 for index, value in group.items()):
            raise ValueError(f'Pose timestamp mismatch: {video}')
        videos[video] = (path, meta, group)
    logger.info('ROI input validation PASS: %d pose frames across %d videos', len(records), len(videos))
    (output/'visualizations').mkdir(parents=True)
    rows = []
    visualization_count = 0
    landmark_file = (output/'landmark_mask_support.csv').open('w', newline='', encoding='utf-8') if sam2_dir is not None else None
    landmark_writer = None
    if landmark_file is not None:
        landmark_writer = csv.DictWriter(landmark_file, fieldnames=[
            'video_id', 'split', 'frame_index', 'timestamp_sec', 'keypoint_index',
            'keypoint_name', 'x', 'y', 'confidence', 'inside_sam2_mask', 'segmentation_status'])
        landmark_writer.writeheader()
    for video, (path, meta, group) in videos.items():
        if not args.no_visualizations and settings.get('save_annotated_frames', True):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f'Cannot decode video: {path}')
        else:
            capture = None
        wanted = set(group)
        try:
            for index in sorted(group):
                record = group[index]
                pose = record['json']
                mapped = mapper.map_pose(pose['bbox'], pose['keypoints']) if pose['bbox'] is not None else mapper.map_pose(
                    (0, 0, 1, 1), [])
                sam2_row = sam2_records[(video, index)] if sam2_records is not None else None
                mask = None
                if sam2_row is not None:
                    try:
                        mask = load_mask(sam2_dir, sam2_row, expected_shape=(meta.height, meta.width))
                    except (OSError, ValueError) as exc:
                        logger.warning('%s frame %d: unusable SAM 2 mask; ROI falls back: %s', video, index, exc)
                roi_rows = [result.as_record(video, record['split'], index, record['timestamp_sec'], pose['status'])
                            for result in mapped]
                for roi_row in roi_rows:
                    roi_row['keypoints_path'] = record['keypoints_path']
                    roi_row['pose_confidence'] = pose.get('confidence')
                    roi_row.update(roi_mask_support(mask, roi_row, pose['keypoints'],
                        min_support_ratio=float(settings.get('min_mask_support_ratio', .5))))
                    roi_row['SAM2_mask_quality'] = float(sam2_row['mask_quality']) if mask is not None else None
                    roi_row['sam2_segmentation_status'] = sam2_row['segmentation_status'] if sam2_row else 'DISABLED'
                if landmark_writer is not None:
                    support = dict(landmark_mask_support(mask, pose['keypoints']))
                    for point_index in range(308):
                        point = pose['keypoints'][point_index] if pose['keypoints'] else {}
                        landmark_writer.writerow(dict(video_id=video, split=record['split'], frame_index=index,
                            timestamp_sec=record['timestamp_sec'], keypoint_index=point_index,
                            keypoint_name=names[point_index], x=point.get('x'), y=point.get('y'),
                            confidence=point.get('confidence'), inside_sam2_mask=support.get(point_index),
                            segmentation_status=sam2_row['segmentation_status']))
                rows.extend(roi_rows)
                if capture is not None:
                    while wanted:
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError(f'{video}: early decoding failure at frame {index}')
                        current = capture.get(cv2.CAP_PROP_POS_FRAMES)-1
                        if int(round(current)) == index:
                            target = output/'visualizations'/f'{video}_frame_{index:08d}.jpg'
                            if not cv2.imwrite(str(target), draw_rois(image, roi_rows)):
                                raise OSError(f'Cannot write {target}')
                            visualization_count += 1
                            wanted.remove(index)
                            break
        finally:
            if capture is not None:
                capture.release()
        logger.info('%s: %d frames, %d ROI records', video, len(group), len(group)*17)
    if landmark_file is not None:
        landmark_file.close()
    if any(sha256_file(path) != digest for key, digest in hashes.items() for path in [sources[key]]):
        raise ValueError('ROI inputs changed during execution')
    table = pd.DataFrame(rows)
    table.to_csv(output/'candidate_rois.csv', index=False)
    metadata = dict(phase='PHASE_8_ANATOMICAL_MAPPING', run_status='COMPLETE', total_pose_frames=len(records),
                    roi_records=len(rows), visualization_count=visualization_count,
                    roi_names=list(__import__('module1.roi', fromlist=['ROI_NAMES']).ROI_NAMES),
                    keypoint_count=308, min_keypoint_confidence=threshold, min_support_points=min_support,
                    source_hashes={key: dict(path=str(path), sha256=digest) for key, (path, digest) in
                                   ((name, (sources[name], hashes[name])) for name in sources)},
                    pose_run_status=pose_metadata.get('run_status'),
                    sam2_mode='ENHANCED' if sam2_dir is not None else 'BASELINE',
                    sam2_source_dir=str(sam2_dir) if sam2_dir is not None else None,
                    sam2_overlap_definitions=dict(roi_inside_mask_ratio='intersection / ROI pixels',
                        roi_mask_overlap='intersection / union of ROI and infant mask',
                        visible_infant_pixel_ratio='ROI-mask intersection / infant mask pixels',
                        SAM2_mask_support='same as ROI inside-mask ratio'),
                    anatomical_confidence_definition='mean source keypoint confidence for supporting points; not physiological ROI confidence',
                    confidence_scoring_implemented=False, roi_scoring_implemented=False,
                    torso_normalization='trapezoid coordinates from left/right shoulder and hip anchors when all four are valid',
                    ground_truth_status='UNAVAILABLE', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), opencv_version=cv2.__version__)
    (output/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('ROI mapping complete: %d records, %d visualizations', len(rows), visualization_count)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
