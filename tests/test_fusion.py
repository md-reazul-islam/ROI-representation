from dataclasses import replace
import itertools

import pytest

from module1.detection.base import Detection
from module1.detection.fusion import box_iou, DetectorFusion, FusionConfig, STRATEGIES


def det(box=(0, 0, 10, 10), confidence=.8, class_id=0):
    return Detection(box, confidence, class_id, 'person', 7, .35)


def test_iou_geometry():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_empty_inputs(strategy):
    result = DetectorFusion(FusionConfig(strategy=strategy)).fuse([], [])
    assert result.selected is None
    assert result.candidates == ()


@pytest.mark.parametrize('strategy,expected', [('or', .8), ('and', None),
                                             ('confidence_weighted', .8), ('weighted_box_fusion', .4)])
def test_single_detector_support(strategy, expected):
    engine = DetectorFusion(FusionConfig(strategy=strategy))
    for y, r in (([det()], []), ([], [det()])):
        result = engine.fuse(y, r)
        assert len(result.candidates) == 1
        assert result.candidates[0].agreement is False
        if expected is None:
            assert result.selected is None
        else:
            assert result.selected.detection.confidence == pytest.approx(expected)


def test_matched_fusion_numeric_rules_and_metadata():
    y, r = det(confidence=.8), det((2, 0, 12, 10), .6)
    for strategy in STRATEGIES:
        result = DetectorFusion(FusionConfig(strategy=strategy, yolo_weight=2, rtdetr_weight=1)).fuse([y], [r])
        assert len(result.candidates) == 1
        assert result.selected.agreement
        assert result.selected.iou == pytest.approx(2 / 3)
        d = result.selected.detection
        assert (d.frame_index, d.timestamp, d.class_id) == (7, .35, 0)
        if strategy in ('confidence_weighted', 'weighted_box_fusion'):
            assert d.bbox[0] == pytest.approx(1.2 / 2.2)
            assert d.confidence == pytest.approx(2.2 / 3)
        else:
            assert d.bbox == y.bbox
            assert d.confidence == pytest.approx(.6 if strategy == 'and' else .8)


def test_matching_is_one_to_one_and_class_aware():
    result = DetectorFusion().fuse([det(), det((1, 0, 11, 10), .7)], [det(confidence=.9)])
    assert sum(c.agreement for c in result.candidates) == 1
    assert sum(c.source == 'yolo_only' for c in result.candidates) == 1
    result = DetectorFusion(FusionConfig(strategy='and')).fuse([det()], [det(class_id=1)])
    assert result.selected is None
    assert {c.source for c in result.candidates} == {'yolo_only', 'rtdetr_only'}


def test_disagreeing_boxes_remain_visible_and_and_rejects_them():
    y, r = det(), det((20, 20, 30, 30), .9)
    result = DetectorFusion().fuse([y], [r])
    assert len(result.candidates) == 2
    assert result.selected.source == 'rtdetr_only'
    assert DetectorFusion(FusionConfig(strategy='and')).fuse([y], [r]).selected is None


def test_threshold_boundary_and_output_confidence_threshold():
    y, r = det(), det((5, 0, 15, 10))
    assert DetectorFusion(FusionConfig(iou_threshold=1/3)).fuse([y], [r]).selected.agreement
    assert not DetectorFusion(FusionConfig(iou_threshold=.34)).fuse([y], [r]).selected.agreement
    result = DetectorFusion(FusionConfig(strategy='weighted_box_fusion', min_confidence=.5)).fuse([y], [])
    assert result.selected is None
    assert len(result.candidates) == 1


def test_input_permutations_cannot_change_selection_or_matches():
    y = [det(), det((1, 0, 11, 10)), det((20, 20, 30, 30))]
    r = [det((0, 0, 10, 10), .9), det((21, 20, 31, 30), .8)]
    engine = DetectorFusion()
    expected = engine.fuse(y, r)
    for a, b in itertools.product(itertools.permutations(y), itertools.permutations(r)):
        assert engine.fuse(list(a), list(b)) == expected


def test_agreement_bonus_is_a_selection_score_not_confidence():
    y = [det(confidence=.6), det((20, 20, 30, 30), .65)]
    result = DetectorFusion(FusionConfig(agreement_bonus=.1)).fuse(y, [det(confidence=.6)])
    assert result.selected.agreement
    assert result.selected.selection_score == pytest.approx(.7)
    assert result.selected.detection.confidence == .6


@pytest.mark.parametrize('kwargs', [dict(strategy='bad'), dict(iou_threshold=0), dict(iou_threshold=1.1),
                                  dict(yolo_weight=0), dict(rtdetr_weight=float('nan')),
                                  dict(min_confidence=-1), dict(agreement_bonus=2)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FusionConfig(**kwargs)


@pytest.mark.parametrize('d', [det((0, 0, 0, 10)), det(confidence=float('nan')),
                             replace(det(), timestamp=float('nan')), replace(det(), frame_index=-1)])
def test_invalid_detection(d):
    with pytest.raises(ValueError):
        DetectorFusion().fuse([d], [])


def test_frame_and_timestamp_mismatch_rejected():
    for d in (replace(det(), frame_index=8), replace(det(), timestamp=.4)):
        with pytest.raises(ValueError, match='different frames'):
            DetectorFusion().fuse([det()], [d])


def test_zero_confidence_pair_has_finite_valid_box():
    result = DetectorFusion(FusionConfig(strategy='weighted_box_fusion')).fuse(
        [det(confidence=0)], [det(confidence=0)])
    assert result.selected.detection.confidence == 0
    assert result.selected.detection.bbox == (0, 0, 10, 10)
