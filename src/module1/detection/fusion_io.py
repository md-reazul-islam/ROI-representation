"""CSV validation, auditable frame records, and detector-observation ablations."""
from dataclasses import dataclass
import json
import math

import pandas as pd

from .base import Detection
from .fusion import box_iou, detection_key, validate_detection
from .metrics import calculate_detection_metrics
from .rtdetr_metrics import aggregate_metrics


@dataclass(frozen=True)
class DetectionFrame:
    video_id: str
    split: str
    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...]


def read_detection_csv(path):
    table = pd.read_csv(path, keep_default_na=False)
    required = {'video_id', 'split', 'frame_index', 'timestamp_sec', 'detection_status',
                'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'}
    missing = required - set(table.columns)
    if missing or table.empty:
        raise ValueError(f'{path}: empty detections or missing columns: {sorted(missing)}')
    frames = {}
    splits = {}
    for (video_id, frame_index), group in table.groupby(['video_id', 'frame_index'], sort=True):
        index = float(frame_index)
        if not math.isfinite(index) or index < 0 or not index.is_integer():
            raise ValueError(f'{path}: invalid frame index {frame_index}')
        frame_index = int(index)
        video_id = str(video_id)
        split = str(group.iloc[0]['split'])
        timestamps = [float(v) for v in group.timestamp_sec]
        timestamp = timestamps[0]
        if not video_id or split not in ('train', 'validation', 'test') or group.split.nunique() != 1:
            raise ValueError(f'{path}: invalid video or inconsistent split')
        if video_id in splits and splits[video_id] != split:
            raise ValueError(f'{path}: video appears in multiple splits: {video_id}')
        splits[video_id] = split
        if any(not math.isfinite(t) or t < 0 or abs(t - timestamp) > 1e-8 for t in timestamps):
            raise ValueError(f'{path}: invalid or inconsistent timestamps')
        statuses = set(group.detection_status)
        if statuses not in ({'DETECTED'}, {'MISSED'}) or (statuses == {'MISSED'} and len(group) != 1):
            raise ValueError(f'{path}: inconsistent detection status for {video_id}/{frame_index}')
        detections = []
        if statuses == {'DETECTED'}:
            for row in group.to_dict('records'):
                class_id = float(row['class_id'])
                if not math.isfinite(class_id) or not class_id.is_integer() or class_id < 0:
                    raise ValueError(f'{path}: invalid class ID')
                d = Detection(tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')),
                              float(row['confidence']), int(class_id), str(row['class_name']),
                              frame_index, timestamp)
                validate_detection(d)
                detections.append(d)
            if len(set(detections)) != len(detections):
                raise ValueError(f'{path}: duplicate candidate in {video_id}/{frame_index}')
        frames[(video_id, frame_index)] = DetectionFrame(
            video_id, split, frame_index, timestamp, tuple(sorted(detections, key=detection_key)))
    return frames


def validate_alignment(yolo, rtdetr):
    if yolo.keys() != rtdetr.keys():
        raise ValueError('Detector frame sets differ; missing output rows are not detector misses')
    for key, y in yolo.items():
        r = rtdetr[key]
        if y.split != r.split or abs(y.timestamp - r.timestamp) > 1e-8:
            raise ValueError(f'Detector split/timestamp mismatch: {key}')


def detection_payload(d):
    if d is None:
        return None
    return dict(bbox=list(d.bbox), confidence=d.confidence, class_id=d.class_id, class_name=d.class_name)


def observation_row(frame, detection):
    return dict(video_id=frame.video_id, split=frame.split, frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp, detection_status='DETECTED' if detection else 'MISSED',
                selected_detection=detection is not None, confidence=detection.confidence if detection else None,
                ground_truth_status='UNAVAILABLE')


def fusion_row(yolo, rtdetr, result, strategy, default_strategy):
    selected = result.selected
    detection = selected.detection if selected else None
    row = observation_row(yolo, detection)
    row.update(fusion_strategy=strategy, is_default_strategy=strategy == default_strategy,
               source_iou=box_iou(yolo.detections[0].bbox, rtdetr.detections[0].bbox)
               if yolo.detections and rtdetr.detections else None,
               fused_bbox=json.dumps(list(detection.bbox)) if detection else None,
               fused_confidence=detection.confidence if detection else None,
               class_id=detection.class_id if detection else None,
               class_name=detection.class_name if detection else None,
               iou=selected.iou if selected else None,
               agreement_flag=selected.agreement if selected else False,
               frame_agreement_flag=any(c.agreement for c in result.candidates),
               selected_source=selected.source if selected else 'none',
               selection_score=selected.selection_score if selected else None,
               num_yolo_candidates=len(yolo.detections), num_rtdetr_candidates=len(rtdetr.detections),
               num_agreeing_pairs=sum(c.agreement for c in result.candidates),
               num_yolo_only=sum(c.source == 'yolo_only' for c in result.candidates),
               num_rtdetr_only=sum(c.source == 'rtdetr_only' for c in result.candidates))
    for name, frame in (('yolo', yolo), ('rtdetr', rtdetr)):
        top = frame.detections[0] if frame.detections else None
        evidence = getattr(selected, name) if selected else None
        row[name + '_bbox'] = json.dumps(list(top.bbox)) if top else None
        row[name + '_confidence'] = top.confidence if top else None
        row['selected_' + name + '_bbox'] = json.dumps(list(evidence.bbox)) if evidence else None
        row['selected_' + name + '_confidence'] = evidence.confidence if evidence else None
    for k, value in zip(('x1', 'y1', 'x2', 'y2'), detection.bbox if detection else (None,) * 4):
        row[k] = value
    row['candidates_json'] = json.dumps([
        dict(yolo=detection_payload(c.yolo), rtdetr=detection_payload(c.rtdetr), iou=c.iou,
             source=c.source, fused=detection_payload(c.detection), accepted=c.detection is not None,
             selected=c is selected, selection_score=c.selection_score)
        for c in result.candidates], allow_nan=False)
    return row


def ablation_metrics(methods):
    output = []
    for method, rows in methods.items():
        videos = {}
        for row in rows:
            videos.setdefault(row['video_id'], []).append(row)
        for video_id, group in videos.items():
            output.append(dict(method=method, scope='video', video_id=video_id,
                               split=group[0]['split'], **calculate_detection_metrics(group)))
        for metric in aggregate_metrics(rows, 'split'):
            output.append(dict(method=method, scope='split', split=metric.pop('group'), **metric))
        for metric in aggregate_metrics(rows):
            output.append(dict(method=method, scope='overall', **metric))
    return output
