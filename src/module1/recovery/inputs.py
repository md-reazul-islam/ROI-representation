import json
import math

import pandas as pd

from .temporal import TrackerObservation, valid_box


def boolean(value):
    if str(value).lower() not in ('true', 'false'):
        raise ValueError('Expected explicit True/False flag')
    return str(value).lower() == 'true'


def frame_key(row):
    value = float(row['frame_index'])
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError('Invalid frame index')
    return str(row['video_id']), int(value)


def load_tracker_frames(path, fused, strategy, backend):
    table = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'track_id', 'bbox', 'velocity',
                'track_state', 'observed', 'confidence', 'reliability', 'fusion_strategy', 'tracker', 'is_primary'}
    if required - set(table.columns):
        raise ValueError(f'Missing tracker columns: {sorted(required - set(table.columns))}')
    frames = {}
    for row in table.to_dict('records'):
        key = frame_key(row)
        if key in frames or key not in fused:
            raise ValueError('Duplicate or unexpected tracker frame')
        source = fused[key]
        timestamp = float(row['timestamp_sec'])
        if not math.isfinite(timestamp) or abs(timestamp-source['timestamp_sec']) > 1e-8 or row['split'] != source['split']:
            raise ValueError('Tracker/fusion timestamp or split mismatch')
        if row['tracker'] != backend or row['fusion_strategy'] != strategy:
            raise ValueError('Tracker backend or fusion strategy mismatch')
        tid = None
        if row['track_id'] != '':
            value = float(row['track_id'])
            if not math.isfinite(value) or value < 1 or not value.is_integer():
                raise ValueError('Invalid track ID')
            tid = int(value)
        box = tuple(json.loads(row['bbox'])) if row['bbox'] else None
        velocity = tuple(json.loads(row['velocity'])) if row['velocity'] else None
        observed = boolean(row['observed'])
        primary = boolean(row['is_primary'])
        state = row['track_state']
        if state not in ('CONFIRMED', 'LOST', 'MISSING', 'REMOVED', 'TENTATIVE'):
            raise ValueError('Unknown tracker state')
        if (state == 'CONFIRMED' and not observed) or (state in ('LOST', 'REMOVED', 'MISSING') and observed):
            raise ValueError('Contradictory tracker observation/state')
        if tid is None:
            if state != 'MISSING' or box is not None or velocity is not None or primary:
                raise ValueError('Missing tracker record has contradictory fields')
        elif not primary or not valid_box(box) or velocity is None or len(velocity) != 2 or not all(math.isfinite(v) for v in velocity):
            raise ValueError('Invalid primary tracker geometry')
        confidence, reliability = float(row['confidence']), float(row['reliability'])
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (confidence, reliability)):
            raise ValueError('Invalid tracker score')
        frames[key] = TrackerObservation(tid, box, velocity, state, observed, confidence, reliability)
    if frames.keys() != fused.keys():
        raise ValueError('Tracker and fusion frame sets differ')
    return frames


def load_pose_quality(path, fused):
    if path is None:
        return {}
    table = pd.read_csv(path)
    required = {'video_id', 'frame_index', 'timestamp_sec', 'pose_quality'}
    if required - set(table.columns):
        raise ValueError('Pose validation CSV requires video_id, frame_index, timestamp_sec, pose_quality')
    output = {}
    for row in table.to_dict('records'):
        key = frame_key(row)
        quality, timestamp = float(row['pose_quality']), float(row['timestamp_sec'])
        if key not in fused or key in output or not math.isfinite(timestamp) or abs(timestamp-fused[key]['timestamp_sec']) > 1e-8:
            raise ValueError('Pose validation frames do not align with fusion')
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('Invalid pose quality')
        output[key] = quality
    return output
