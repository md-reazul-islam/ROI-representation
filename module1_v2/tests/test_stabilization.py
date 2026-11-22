import math

import numpy as np

from module1_v2 import stabilization as st

NAMES = [f'R{i:02d}' for i in range(6)]


def test_ema_identity_when_beta_zero():
    Q = np.random.default_rng(0).random((5, 6))
    assert np.array_equal(st.ema(Q, 0.), Q)


def test_ema_carries_memory_and_resets_on_empty_frame():
    Q = np.array([[1., 0.], [np.nan, 0.], [0., 0.], [np.nan, np.nan], [0., 1.]])
    S = st.ema(Q, .5)
    assert math.isnan(S[1, 0])                     # unavailable -> not selectable
    assert S[2, 0] == .5                           # memory 1.0 carried over the gap
    assert S[4, 0] == 0. and S[4, 1] == 1.         # reset after a frame with no region


def test_topk_matches_v1_lexsort_order():
    rng = np.random.default_rng(1)
    S = rng.random((20, 6))
    S[rng.random(S.shape) < .2] = np.nan
    for row, chosen in zip(S, st.topk_sets(S, 3, NAMES)):
        valid = np.flatnonzero(np.isfinite(row))
        order = valid[np.lexsort((np.array(NAMES)[valid], -row[valid]))][:3]
        assert list(order) == chosen


def test_hysteresis_waits_n_frames_and_margin():
    # region 1 overtakes region 0 slightly from frame 1 on: with delta 0.1 never; with delta 0 after 3 frames
    Q = np.array([[.9, .5, .1]] + [[.6, .65, .1]] * 5)
    names = ['A', 'B', 'C']
    assert [s[0] for s in st.hysteresis_sets(Q, 1, names, .1, 3)] == [0] * 6
    tops = [s[0] for s in st.hysteresis_sets(Q, 1, names, 0., 3)]
    assert tops == [0, 0, 0, 1, 1, 1]


def test_hysteresis_fills_vacancy_immediately():
    Q = np.array([[.9, .5], [np.nan, .5]])
    assert st.hysteresis_sets(Q, 1, ['A', 'B'], .5, 10) == [[0], [1]]


def test_hysteresis_without_margin_equals_topk():
    rng = np.random.default_rng(2)
    S = rng.random((50, 6))
    assert st.hysteresis_sets(S, 3, NAMES, 0., 1) == st.topk_sets(S, 3, NAMES)


def test_viterbi_penalty_zero_is_argmax_and_large_penalty_is_constant():
    rng = np.random.default_rng(3)
    Q = rng.random((40, 5))
    assert st.viterbi_anchor(Q, 0.) == list(np.argmax(Q, 1))
    assert len(set(st.viterbi_anchor(Q, 1e6))) == 1


def test_viterbi_fixed_lag_equals_offline_with_full_lag_and_splits_segments():
    rng = np.random.default_rng(4)
    Q = rng.random((30, 4))
    Q[10] = np.nan
    offline = st.viterbi_anchor(Q, .3)
    assert offline[10] is None
    assert st.viterbi_anchor(Q, .3, lag=100) == offline


def test_fusion_weights_v1_and_crossfade_limit():
    S = np.array([[1., 0., np.nan], [0., 1., np.nan], [0., 1., np.nan]])
    sets = [[0], [1], [1]]
    W = st.fusion_weights(S, sets)
    assert np.allclose(W, [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    W = st.fusion_weights(S, sets, max_step=.25)
    assert np.all(np.abs(np.diff(W, axis=0)) <= .25 + 1e-9)
    assert np.allclose(W.sum(1), 1.)
