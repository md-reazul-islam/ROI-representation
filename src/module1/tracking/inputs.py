import json
import math

import pandas as pd

from module1.detection.base import Detection
from module1.detection.fusion import validate_detection


def load_fused_frames(path, strategy):
    table = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'fusion_strategy',
                'detection_status', 'candidates_json'}
    if required - set(table.columns):
        raise ValueError(f'Missing fusion columns: {sorted(required - set(table.columns))}')
    table = table[table.fusion_strategy == strategy].copy()
    if table.empty:
        raise ValueError(f'No fusion frames for strategy {strategy}')
    if table.duplicated(['video_id', 'frame_index']).any():
        raise ValueError('Duplicate fusion frame records')
    frames = {}
    for row in table.to_dict('records'):
        index, timestamp = float(row['frame_index']), float(row['timestamp_sec'])
        if not math.isfinite(index) or not index.is_integer() or index < 0 or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid fusion frame metadata')
        row['frame_index'], row['timestamp_sec'] = int(index), timestamp
        detections = []
        candidates = json.loads(row['candidates_json'])
        if not isinstance(candidates, list):
            raise ValueError('Expected fusion candidate list')
        selected_count = 0
        for candidate in candidates:
            accepted, selected = candidate['accepted'], candidate['selected']
            if not isinstance(accepted, bool) or not isinstance(selected, bool) or (selected and not accepted):
                raise ValueError('Invalid fusion candidate flags')
            selected_count += selected
            if accepted:
                d = candidate['fused']
                item = Detection(tuple(d['bbox']), float(d['confidence']), d['class_id'], d['class_name'], int(index), timestamp)
                validate_detection(item)
                detections.append(item)
        if row['detection_status'] not in ('DETECTED', 'MISSED') or selected_count != int(row['detection_status'] == 'DETECTED'):
            raise ValueError('Fusion selected-candidate/status mismatch')
        if bool(detections) != (row['detection_status'] == 'DETECTED'):
            raise ValueError('Fusion candidate/status mismatch')
        row['detections'] = detections
        frames[(str(row['video_id']), int(index))] = row
    return frames
