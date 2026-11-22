from dataclasses import replace
import json

import pandas as pd
import pytest

from module1.detection.base import Detection
from module1.detection.fusion import DetectorFusion
from module1.detection.fusion_io import (
    ablation_metrics, DetectionFrame, fusion_row, observation_row, read_detection_csv, validate_alignment,
)


def csv_row(**kwargs):
    row = dict(video_id='v', split='train', frame_index=0, timestamp_sec=0, detection_status='DETECTED',
               class_id=0, class_name='person', confidence=.8, x1=0, y1=0, x2=10, y2=10)
    return {**row, **kwargs}


def test_csv_preserves_all_candidates_and_explicit_misses(tmp_path):
    path = tmp_path / 'detections.csv'
    pd.DataFrame([csv_row(), csv_row(confidence=.9, x1=1),
                  csv_row(frame_index=1, timestamp_sec=.05, detection_status='MISSED', confidence=None)]).to_csv(path, index=False)
    frames = read_detection_csv(path)
    assert len(frames) == 2
    assert [d.confidence for d in frames[('v', 0)].detections] == [.9, .8]
    assert frames[('v', 1)].detections == ()


@pytest.mark.parametrize('rows', [
    [csv_row(), csv_row()],
    [csv_row(), csv_row(detection_status='MISSED')],
    [csv_row(), csv_row(confidence=.9, timestamp_sec=1)],
    [csv_row(), csv_row(frame_index=1, split='test')],
    [csv_row(frame_index=-1)], [csv_row(frame_index=.5)], [csv_row(class_id=.5)],
    [csv_row(confidence=1.2)], [csv_row(detection_status='UNKNOWN')],
    [csv_row(timestamp_sec=float('inf'))],
])
def test_invalid_input_fails(tmp_path, rows):
    path = tmp_path / 'bad.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_detection_csv(path)


def test_alignment_checks_missing_frames_splits_and_timestamps():
    frame = DetectionFrame('v', 'train', 0, 0, ())
    with pytest.raises(ValueError, match='frame sets'):
        validate_alignment({('v', 0): frame}, {})
    for changed in (replace(frame, split='test'), replace(frame, timestamp=1)):
        with pytest.raises(ValueError, match='mismatch'):
            validate_alignment({('v', 0): frame}, {('v', 0): changed})


def test_output_preserves_evidence_and_missed_frame():
    d = Detection((0, 0, 10, 10), .8, 0, 'person', 0, 0)
    y, r = DetectionFrame('v', 'train', 0, 0, (d,)), DetectionFrame('v', 'train', 0, 0, ())
    row = fusion_row(y, r, DetectorFusion().fuse([d], []), 'or', 'or')
    assert row['fused_confidence'] == .8
    assert row['rtdetr_bbox'] is None
    assert not row['agreement_flag']
    assert row['num_yolo_only'] == 1
    assert json.loads(row['candidates_json'])[0]['source'] == 'yolo_only'
    missed = fusion_row(r, r, DetectorFusion().fuse([], []), 'or', 'or')
    assert missed['detection_status'] == 'MISSED'
    assert missed['ground_truth_status'] == 'UNAVAILABLE'
    assert missed['fused_bbox'] is None


def test_ablation_counts_frames_once_and_respects_video_gap_boundaries():
    rows = []
    for v, statuses in [('a', [False, False]), ('b', [True, False])]:
        for index, positive in enumerate(statuses):
            frame = DetectionFrame(v, 'train', index, index / 20, ())
            d = Detection((0, 0, 1, 1), .8, 0, 'person', index, index/20) if positive else None
            rows.append(observation_row(frame, d))
    metrics = ablation_metrics({'YOLO': rows, 'RT-DETR': rows, 'YOLO+RT-DETR:or': rows})
    overall = [m for m in metrics if m['scope'] == 'overall']
    assert len(overall) == 3
    for m in overall:
        assert m['total_processed_frames'] == 4
        assert m['detection_rate'] == .25
        assert m['miss_rate'] == .75
        assert m['longest_consecutive_miss'] == 2
        assert m['number_of_detection_gaps'] == 2
        assert m['average_gap_length'] == 1.5


def test_disagreement_retains_source_iou():
    y = DetectionFrame('v', 'train', 0, 0, (Detection((0, 0, 10, 10), .8, 0, 'person', 0, 0),))
    r = DetectionFrame('v', 'train', 0, 0, (Detection((20, 0, 30, 10), .9, 0, 'person', 0, 0),))
    result = DetectorFusion().fuse(list(y.detections), list(r.detections))
    row = fusion_row(y, r, result, 'or', 'or')
    assert row['source_iou'] == 0
    assert row['iou'] is None
    assert not row['frame_agreement_flag']
