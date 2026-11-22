import json
import math
from pathlib import Path


def load_pose_record(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    required = {'video_id', 'frame_index', 'timestamp', 'bbox', 'keypoints', 'status'}
    if not required.issubset(payload):
        raise ValueError(f'Pose record missing required fields: {path.name}')
    frame_index = payload['frame_index']
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError('Pose frame_index must be a nonnegative integer')
    timestamp = float(payload['timestamp'])
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Pose timestamp must be finite and nonnegative')
    keypoints = payload['keypoints']
    if not isinstance(keypoints, list) or (keypoints and len(keypoints) != 308):
        raise ValueError('Pose record must contain exactly 308 keypoints when present')
    for expected, point in enumerate(keypoints):
        if not isinstance(point, dict) or point.get('index') != expected:
            raise ValueError('Pose keypoints must be ordered by index 0..307')
        for field in ('x', 'y', 'confidence'):
            value = point.get(field)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid pose keypoint {field}')
    bbox = payload['bbox']
    if bbox is not None:
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError('Invalid pose bbox')
    if payload['status'] not in ('ESTIMATED', 'FLAGGED', 'INSUFFICIENT_EVIDENCE', 'NO_BBOX'):
        raise ValueError('Invalid pose status')
    if payload['status'] == 'NO_BBOX' and (bbox is not None or keypoints):
        raise ValueError('NO_BBOX pose record must not contain a box or keypoints')
    return payload


def load_pose_table(pose_dir):
    pose_dir = Path(pose_dir)
    metadata_path = pose_dir / 'experiment_metadata.json'
    results_path = pose_dir / 'pose_results.csv'
    if not metadata_path.is_file() or not results_path.is_file():
        raise FileNotFoundError('Pose directory requires experiment_metadata.json and pose_results.csv')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    names = metadata.get('keypoint_names')
    if metadata.get('keypoint_count') not in (None, 308) or not isinstance(names, list) or len(names) != 308:
        raise ValueError('Pose metadata must identify the official 308 keypoints')
    import pandas as pd
    table = pd.read_csv(results_path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'keypoints_path', 'final_infant_bbox', 'pose_status'}
    if table.empty or not required.issubset(table.columns):
        raise ValueError('Pose results are empty or missing required columns')
    records = {}
    for row in table.to_dict('records'):
        key = (str(row['video_id']), int(row['frame_index']))
        if key in records:
            raise ValueError('Duplicate pose frame')
        if float(row['frame_index']) != key[1] or float(row['timestamp_sec']) < 0:
            raise ValueError('Invalid pose results frame metadata')
        record_path = pose_dir / str(row['keypoints_path'])
        if pose_dir.resolve() not in record_path.resolve().parents:
            raise ValueError('Keypoint path escapes pose directory')
        payload = load_pose_record(record_path)
        if payload['video_id'] != key[0] or payload['frame_index'] != key[1] or abs(payload['timestamp']-float(row['timestamp_sec'])) > 1e-8:
            raise ValueError('Pose CSV and JSON metadata disagree')
        records[key] = dict(row, json=payload, frame_index=key[1], timestamp_sec=float(row['timestamp_sec']))
    return records, tuple(names), metadata
