from dataclasses import replace
import math

import pytest

from module1.detection.base import Detection
from module1.recovery import RecoveryConfig, TemporalRecovery, TrackerObservation
from module1.recovery.metrics import per_video_metrics, summarize_recovery


BOX = (20., 20., 60., 60.)


def observation(state='LOST', **kwargs):
    return replace(TrackerObservation(1, BOX, (0., 0.), state, state == 'CONFIRMED',
                                      .9 if state == 'CONFIRMED' else 0., .9), **kwargs)


def detection(index=0):
    return Detection(BOX, .9, 0, 'person', index, index * .05)


def seeded(config=None, tracker=None):
    engine = TemporalRecovery(config)
    engine.update(detection(), tracker or observation('CONFIRMED'), 0, 0., (100, 100))
    return engine


def update(engine, i, tracker=None, pose=None):
    return engine.update(None, tracker or observation(), i, i*.05, (100, 100), pose)


def test_direct_detections_preserved_even_without_tracking():
    engine = TemporalRecovery()
    missing = TrackerObservation(None, None, None, 'MISSING', False, 0., 0.)
    row = engine.update(detection(), missing, 0, 0., (100, 100))
    assert row['final_infant_bbox'] == BOX and row['final_infant_confidence'] == .9
    assert row['recovered_status'] == 'NOT_NEEDED'
    assert update(engine, 1)['rejection_reason'] == 'NO_CONFIRMED_DETECTION_ANCHOR'


def test_stationary_prediction_confidence_and_expiry():
    engine = seeded(RecoveryConfig(max_recovery_frames=3, min_recovery_confidence=0))
    for i in range(1, 4):
        row = update(engine, i)
        assert row['final_status'] == 'RECOVERED'
        assert row['recovery_confidence'] == pytest.approx(.9 * .9**i)
        assert row['anchor_frame_index'] == 0 and row['gap_age'] == i
    for i in range(4, 20):
        row = update(engine, i)
        assert row['rejection_reason'] == 'RECOVERY_LIMIT_EXCEEDED'
        assert row['final_infant_bbox'] is None
        assert row['final_infant_confidence'] == 0


def test_rejection_does_not_reset_budget_and_detection_does():
    engine = seeded(RecoveryConfig(max_recovery_frames=2, min_recovery_confidence=0))
    assert update(engine, 1, observation(track_id=2))['recovered_status'] == 'REJECTED'
    assert update(engine, 2)['recovered_status'] == 'ACCEPTED'
    assert update(engine, 3)['recovered_status'] == 'REJECTED'
    engine.update(detection(4), observation('CONFIRMED'), 4, .2, (100, 100))
    assert update(engine, 5)['gap_age'] == 1
    assert update(engine, 5+1)['recovered_status'] == 'ACCEPTED'


@pytest.mark.parametrize('tracker,reason', [
    (observation(track_id=2), 'TRACK_ID_CHANGED_OR_MISSING'),
    (observation('CONFIRMED'), 'NO_LOST_TRACK_PREDICTION'),
    (observation(state='REMOVED'), 'NO_LOST_TRACK_PREDICTION'),
    (observation(bbox=(0, 0, 0, 10)), 'INVALID_PREDICTION'),
    (observation(bbox=(0, 0, math.nan, 10)), 'INVALID_PREDICTION'),
    (observation(bbox=(-1, 20, 39, 60)), 'PREDICTION_OUTSIDE_IMAGE'),
    (observation(bbox=(60, 60, 100, 100)), 'CENTER_DISPLACEMENT'),
    (observation(bbox=(10, 10, 70, 70)), 'BOX_SCALE_CHANGE'),
    (observation(bbox=(35, 20, 75, 60)), 'TRAJECTORY_INCONSISTENCY'),
])
def test_rejection_gates(tracker, reason):
    row = update(seeded(), 1, tracker)
    assert row['rejection_reason'] == reason
    assert row['recovered_status'] == 'REJECTED'
    assert row['final_infant_bbox'] is None


def test_low_iou_gate_independent_of_center_gate():
    config = RecoveryConfig(max_center_displacement=2, max_trajectory_error=2)
    assert update(seeded(config), 1, observation(bbox=(50, 20, 90, 60)))['rejection_reason'] == 'LOW_PREVIOUS_IOU'


def test_velocity_uses_original_timestamps_not_frame_step():
    engine = seeded(tracker=observation('CONFIRMED', velocity=(100., 0.)))
    row = engine.update(None, observation(bbox=(30, 20, 70, 60)), 3, .1, (100, 100))
    assert row['trajectory_error'] == pytest.approx(0)
    assert row['final_status'] == 'RECOVERED'


def test_anchor_disagreement_and_new_unconfirmed_detection_clear_state():
    engine = seeded(tracker=observation('CONFIRMED', bbox=(70, 70, 90, 90)))
    assert update(engine, 1)['rejection_reason'] == 'NO_CONFIRMED_DETECTION_ANCHOR'
    engine = seeded()
    engine.update(detection(1), observation(state='TENTATIVE'), 1, .05, (100, 100))
    assert update(engine, 2)['rejection_reason'] == 'NO_CONFIRMED_DETECTION_ANCHOR'


def test_pose_validation_and_unavailable_pose():
    assert update(seeded(), 1, pose=.2)['rejection_reason'] == 'POSE_VALIDATION_FAILED'
    good = update(seeded(), 1, pose=.8)
    assert good['recovery_confidence'] == pytest.approx(.9*.9*.8)
    assert good['pose_validation_status'] == 'AVAILABLE'
    assert update(seeded(), 1)['pose_validation_status'] == 'UNAVAILABLE'


def test_low_confidence_and_disabled_recovery():
    assert update(seeded(RecoveryConfig(min_recovery_confidence=.99)), 1)['rejection_reason'] == 'LOW_RECOVERY_CONFIDENCE'
    assert update(seeded(RecoveryConfig(max_recovery_frames=0)), 1)['rejection_reason'] == 'RECOVERY_LIMIT_EXCEEDED'


def test_reset_and_temporal_validation():
    engine = seeded()
    with pytest.raises(ValueError, match='increase'):
        update(engine, 0)
    with pytest.raises(ValueError, match='pose_quality'):
        update(engine, 1, pose=math.nan)
    engine.reset()
    assert update(engine, 0)['rejection_reason'] == 'NO_CONFIRMED_DETECTION_ANCHOR'


@pytest.mark.parametrize('kwargs', [dict(max_recovery_frames=-1), dict(max_recovery_frames=1.5),
                                  dict(min_recovery_confidence=2), dict(max_box_change=0),
                                  dict(max_center_displacement=math.nan), dict(confidence_decay=1)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): RecoveryConfig(**kwargs)


def test_metrics_and_video_boundaries():
    rows = [dict(video_id='a', split='train', frame_index=i, detector_status='MISSED' if i else 'DETECTED',
                 recovered_status='ACCEPTED' if i in (1, 2) else 'REJECTED') for i in range(4)]
    m = per_video_metrics(rows)
    assert m['detector_miss_rate'] == .75
    assert m['recovered_miss_rate'] == .25
    assert m['temporal_recovery_rate'] == pytest.approx(2/3)
    assert m['longest_recovered_sequence'] == 2
    assert m['false_recovery_rate'] is None
    other = [dict(video_id='b', split='test', frame_index=0, detector_status='MISSED', recovered_status='ACCEPTED')]
    overall = next(r for r in summarize_recovery(rows+other) if r['scope'] == 'overall')
    assert overall['longest_recovered_sequence'] == 2
    assert overall['temporal_recovery_rate'] == .75
    no_misses = [dict(frame_index=0, detector_status='DETECTED', recovered_status='NOT_NEEDED')]
    assert math.isnan(per_video_metrics(no_misses)['temporal_recovery_rate'])
