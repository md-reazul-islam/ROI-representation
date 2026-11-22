from dataclasses import replace
import numpy as np
import pytest

from module1.detection.base import Detection
from module1.tracking import ByteTrackAdapter, BoTSORTAdapter, TrackingConfig, TrackingFrame, BaseTracker
from module1.tracking.metrics import tracking_metrics, summarize_tracking


@pytest.fixture(params=[ByteTrackAdapter, BoTSORTAdapter])
def tracker(request):
    return request.param(TrackingConfig(track_buffer=3, primary_hold_frames=2, gmc_method='none'))


def step(tracker, index, boxes=((10, 10, 50, 60),), scores=None):
    scores = scores or [.9] * len(boxes)
    frame = TrackingFrame(np.zeros((100, 100, 3), dtype=np.uint8), index * 2, index * .1)
    ds = [Detection(box, score, 0, 'person', frame.frame_index, frame.timestamp) for box, score in zip(boxes, scores)]
    return tracker.update(ds, frame)


def primary(tracks):
    return next((t for t in tracks if t.is_primary), None)


def test_common_interface_confirmation_and_velocity(tracker):
    assert isinstance(tracker, BaseTracker)
    assert primary(step(tracker, 0)) is None
    assert primary(step(tracker, 1)) is None
    t = primary(step(tracker, 2, ((12, 10, 52, 60),)))
    assert t.track_id == 1 and t.track_state == 'CONFIRMED'
    assert (t.frame_index, t.timestamp) == (4, .2)
    assert t.age == t.number_of_hits == 3 and t.number_of_misses == 0
    assert t.velocity[0] > 0 and t.velocity[1] == pytest.approx(0)
    assert t.observed


def test_single_frame_false_track_never_becomes_primary(tracker):
    assert primary(step(tracker, 0)) is None
    for i in range(1, 7):
        assert primary(step(tracker, i, ())) is None


def test_short_gap_preserves_id_and_counts_misses(tracker):
    for i in range(3):
        tracks = step(tracker, i)
    identity = primary(tracks).track_id
    t = primary(step(tracker, 3, ()))
    assert t.track_id == identity and t.track_state == 'LOST'
    assert t.confidence == 0 and not t.observed
    assert t.number_of_misses == t.consecutive_misses == 1
    recovered = primary(step(tracker, 4))
    assert recovered.track_id == identity and recovered.observed
    assert recovered.number_of_hits == 4 and recovered.number_of_misses == 1
    assert recovered.consecutive_misses == 0


def test_lost_track_expires_and_new_id_not_fabricated(tracker):
    for i in range(3):
        step(tracker, i)
    for i in range(3, 10):
        tracks = step(tracker, i, ())
    assert primary(tracks) is None
    assert not any(t.track_state == 'LOST' for t in tracks)
    for i in range(10, 13):
        tracks = step(tracker, i)
    assert primary(tracks).track_id != 1


def test_primary_does_not_switch_to_high_confidence_competitor(tracker):
    for i in range(3):
        step(tracker, i)
    for i in range(3, 8):
        tracks = step(tracker, i, ((10, 10, 50, 60), (65, 10, 95, 60)), [.6, .99])
        assert primary(tracks).track_id == 1


def test_low_score_second_stage_association(tracker):
    for i in range(3):
        step(tracker, i)
    t = primary(step(tracker, 3, scores=[.2]))
    assert t is not None and t.observed and t.track_id == 1


def test_missing_primary_is_reserved_before_replacement(tracker):
    for i in range(5):
        tracks = step(tracker, i, ((10, 10, 50, 60), (65, 10, 95, 60)), [.95, .8])
    identity = primary(tracks).track_id
    for i in (5, 6):
        held = primary(step(tracker, i, ((65, 10, 95, 60),), [.8]))
        assert held.track_id == identity and not held.observed
    replacement = primary(step(tracker, 7, ((65, 10, 95, 60),), [.8]))
    assert replacement.observed and replacement.track_id != identity


def test_reset_isolates_video_ids_and_timestamps(tracker):
    for i in range(3):
        step(tracker, i)
    tracker.reset()
    for i in range(3):
        tracks = step(tracker, i)
    assert primary(tracks).track_id == 1
    assert primary(tracks).age == 3


def test_temporal_metadata_errors_fail_without_advancing(tracker):
    step(tracker, 0)
    with pytest.raises(ValueError, match='increase'):
        step(tracker, 0)
    f = TrackingFrame(np.zeros((10, 10, 3), dtype=np.uint8), 2, .1)
    d = Detection((0, 0, 5, 5), .9, 0, 'person', 3, .1)
    with pytest.raises(ValueError, match='metadata'):
        tracker.update([d], f)
    assert tracker._step == 1


def test_nms_and_class_filter(tracker):
    f = TrackingFrame(np.zeros((100, 100, 3), dtype=np.uint8), 0, 0.)
    d = Detection((10, 10, 50, 60), .9, 0, 'person', 0, 0.)
    tracks = tracker.update([d, replace(d, confidence=.8), replace(d, class_id=1)], f)
    assert len(tracks) == 1 and tracks[0].number_of_hits == 1


def test_two_live_adapters_do_not_collide_on_backend_ids():
    a = ByteTrackAdapter(TrackingConfig(min_hits=1))
    step(a, 0)
    b = BoTSORTAdapter(TrackingConfig(gmc_method='none'))
    step(b, 0)
    tracks = step(a, 1, ((10, 10, 50, 60), (65, 10, 95, 60)))
    assert len({t.track_id for t in tracks}) == 2
    assert a.engine.tracked_stracks[0].track_id != a.engine.tracked_stracks[1].track_id


@pytest.mark.parametrize('kwargs', [dict(min_hits=0), dict(track_buffer=-1), dict(primary_hold_frames=31),
                                  dict(track_high_thresh=.05), dict(nms_iou=2), dict(gmc_method='bad')])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TrackingConfig(**kwargs)


def test_metrics_count_observations_and_do_not_join_video_timelines():
    rows = [dict(video_id='a', split='train', frame_index=i, track_id=tid, observed=tid is not None, age=i+1)
            for i, tid in enumerate([None, 1, 1, None, 1, 2, None])]
    m = tracking_metrics(rows)
    assert m['track_coverage'] == pytest.approx(4/7)
    assert m['track_continuity'] == pytest.approx(1/6)
    assert m['longest_track'] == 2
    assert m['number_of_fragments'] == 2
    assert m['observed_primary_id_changes'] == 1
    assert m['id_switches'] is None and m['id_switches_status'] == 'NOT_AVAILABLE'
    other = [dict(video_id='b', split='test', frame_index=i, track_id=None, observed=False, age=0) for i in range(3)]
    overall = next(m for m in summarize_tracking(rows+other) if m['scope'] == 'overall')
    assert overall['longest_consecutive_missing_frames'] == 3
    assert overall['number_of_fragments'] == 2
    assert overall['track_coverage'] == .4
