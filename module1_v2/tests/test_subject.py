import numpy as np

from module1_v2 import subject as sj

NAMES = list(sj.PARTS) + ['nose']


def body(head, shoulders, arm, torso):
    """Upright figure: ears `head` apart, shoulders `shoulders` apart, arms of length `arm`, torso `torso`."""
    p = dict(left_ear=(-head / 2, 0), right_ear=(head / 2, 0), left_shoulder=(-shoulders / 2, 10),
             right_shoulder=(shoulders / 2, 10), left_elbow=(-shoulders / 2, 10 + arm / 2),
             right_elbow=(shoulders / 2, 10 + arm / 2), left_wrist=(-shoulders / 2, 10 + arm),
             right_wrist=(shoulders / 2, 10 + arm), left_hip=(-shoulders / 3, 10 + torso), right_hip=(shoulders / 3, 10 + torso))
    return {i: (*p[name], .9) for i, name in enumerate(NAMES) if name in p}


def test_infant_and_adult_proportions():
    parts = sj.part_indices(NAMES)
    infant = sj.proportions(body(head=9.5, shoulders=13, arm=16, torso=20), parts)
    adult = sj.proportions(body(head=15, shoulders=39, arm=60, torso=52), parts)
    assert sj.frame_decision(*infant) == 'INFANT'
    assert sj.frame_decision(*adult) == 'ADULT'


def test_unknown_when_keypoints_not_confident_and_fallback_cue():
    parts = sj.part_indices(NAMES)
    points = body(15, 39, 60, 52)
    points[0] = (*points[0][:2], .1)                       # left ear not confident -> r1 unknown
    r1, r2 = sj.proportions(points, parts)
    assert np.isnan(r1) and sj.frame_decision(r1, r2) == 'ADULT'
    assert sj.frame_decision(np.nan, np.nan) == 'UNKNOWN'


def test_track_majority_unknown_never_rejects_and_reset_on_track_change():
    d = ['ADULT', 'ADULT', 'UNKNOWN', 'INFANT', 'ADULT']
    out = sj.track_decisions(d, [1, 1, 1, 2, 2], [0, .1, .2, .3, .4], 2.)
    assert out == ['REJECT', 'REJECT', 'REJECT', 'ACCEPT', 'ACCEPT']
    assert sj.track_decisions(['UNKNOWN'] * 3, [1] * 3, [0, 1, 2]) == ['ACCEPT'] * 3
    assert sj.episodes([False, True, True, False, True]) == [(1, 3), (4, 5)]
