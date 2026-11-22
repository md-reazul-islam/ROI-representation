import json
from pathlib import Path

import pandas as pd

from module1.roi import ROI_NAMES
from module1.roi.inputs import load_pose_record


def load_roi_table(roi_dir):
    roi_dir = Path(roi_dir)
    csv_path, meta_path = roi_dir/'candidate_rois.csv', roi_dir/'experiment_metadata.json'
    if not csv_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError('ROI directory requires candidate_rois.csv and experiment_metadata.json')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    table = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'roi_name', 'roi_status',
                'polygon', 'bbox', 'centroid', 'support_keypoint_indices'}
    if table.empty or not required.issubset(table.columns):
        raise ValueError('ROI CSV is empty or missing required columns')
    grouped = {}
    for row in table.to_dict('records'):
        key = (str(row['video_id']), int(row['frame_index']), str(row['roi_name']))
        if key in grouped:
            raise ValueError('Duplicate ROI record')
        if key[2] not in ROI_NAMES or int(float(row['frame_index'])) != key[1] or float(row['timestamp_sec']) < 0:
            raise ValueError('Invalid ROI frame or region metadata')
        if row['roi_status'] not in ('AVAILABLE', 'UNAVAILABLE'):
            raise ValueError('Invalid ROI status')
        grouped[key] = row
    for frame_key in {(video, frame) for video, frame, _ in grouped}:
        names = {roi for video, frame, roi in grouped if (video, frame) == frame_key}
        if names != set(ROI_NAMES):
            raise ValueError('Every ROI frame must contain all 17 region names')
    return grouped, metadata


def pose_record_for_roi(pose_dir, row):
    relative = row.get('keypoints_path') or f"keypoints/{row['video_id']}_frame_{int(row['frame_index']):08d}.json"
    path = Path(pose_dir) / str(relative)
    if Path(pose_dir).resolve() not in path.resolve().parents:
        raise ValueError('ROI keypoint path escapes pose directory')
    payload = load_pose_record(path)
    if payload['video_id'] != str(row['video_id']) or payload['frame_index'] != int(row['frame_index']):
        raise ValueError('ROI and pose frame metadata disagree')
    if abs(payload['timestamp']-float(row['timestamp_sec'])) > 1e-8:
        raise ValueError('ROI and pose timestamps disagree')
    return payload
