import json
import math
import re

import pandas as pd


def load_recovery_frames(path):
    table = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'final_infant_bbox',
                'final_infant_confidence', 'final_status', 'detector_status', 'recovered_status', 'track_id'}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError('Recovery CSV is empty or missing required columns')
    frames = {}
    for row in table.to_dict('records'):
        video = str(row['video_id'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', video) or video in ('.', '..'):
            raise ValueError('Invalid video ID')
        index, timestamp = float(row['frame_index']), float(row['timestamp_sec'])
        if not math.isfinite(index) or index < 0 or not index.is_integer() or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid recovery frame metadata')
        key = (video, int(index))
        if key in frames:
            raise ValueError('Duplicate recovery frame')
        confidence = float(row['final_infant_confidence'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid final infant confidence')
        status = row['final_status']
        if status not in ('DETECTED', 'RECOVERED', 'MISSED'):
            raise ValueError('Invalid final status')
        if status == 'DETECTED' and (row['detector_status'] != 'DETECTED' or row['recovered_status'] != 'NOT_NEEDED'):
            raise ValueError('Contradictory detection status')
        if status == 'RECOVERED' and (row['detector_status'] != 'MISSED' or row['recovered_status'] != 'ACCEPTED'):
            raise ValueError('Contradictory recovery status')
        if status == 'MISSED' and (row['detector_status'] != 'MISSED' or row['recovered_status'] != 'REJECTED'):
            raise ValueError('Contradictory missing status')
        box = json.loads(row['final_infant_bbox']) if row['final_infant_bbox'] else None
        if status == 'MISSED':
            if box is not None or confidence != 0:
                raise ValueError('Missing frame has a final box or confidence')
        elif (not isinstance(box, list) or len(box) != 4 or
              not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
              or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('Invalid final infant bbox')
        track_id = None
        if row['track_id'] != '':
            value = float(row['track_id'])
            if not math.isfinite(value) or value <= 0 or not value.is_integer():
                raise ValueError('Invalid primary track ID')
            track_id = int(value)
        row.update(video_id=video, frame_index=int(index), timestamp_sec=timestamp,
                   bbox=tuple(box) if box is not None else None, track_id=track_id,
                   final_infant_confidence=confidence)
        frames[key] = row
    return frames
