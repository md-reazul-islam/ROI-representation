import json
from pathlib import Path

import pandas as pd

from module1.roi import ROI_NAMES


def load_quality_table(directory):
    directory = Path(directory)
    csv_path, metadata_path = directory/'roi_quality_features.csv', directory/'experiment_metadata.json'
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError('Quality directory requires roi_quality_features.csv and experiment_metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    table = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'roi_name', 'roi_status',
                'pose_confidence', 'visible_skin_proportion', 'motion_magnitude', 'temporal_consistency'}
    if table.empty or not required.issubset(table.columns):
        raise ValueError('ROI quality table is empty or missing required columns')
    records = []
    seen = set()
    for row in table.to_dict('records'):
        key = (str(row['video_id']), int(float(row['frame_index'])), str(row['roi_name']))
        if key in seen:
            raise ValueError('Duplicate ROI quality record')
        if key[2] not in ROI_NAMES or float(row['frame_index']) != key[1] or float(row['timestamp_sec']) < 0:
            raise ValueError('Invalid ROI quality frame or region metadata')
        if row['roi_status'] not in ('AVAILABLE', 'UNAVAILABLE'):
            raise ValueError('Invalid ROI quality ROI status')
        seen.add(key)
        records.append(dict(row, video_id=key[0], frame_index=key[1], roi_name=key[2],
                             timestamp_sec=float(row['timestamp_sec'])))
    for frame_key in {(row['video_id'], row['frame_index']) for row in records}:
        names = {row['roi_name'] for row in records if (row['video_id'], row['frame_index']) == frame_key}
        if names != set(ROI_NAMES):
            raise ValueError('Every frame must contain all 17 ROI candidates')
    return records, metadata
