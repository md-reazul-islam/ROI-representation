import numpy as np

from module1_v2 import baselines as bl


def test_grid_tiles_cover_box_and_columns():
    tiles = bl.grid_tiles((0, 0, 40, 20), rows=2, cols=4)
    assert len(tiles) == 8
    assert [c for _, c in tiles] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert tiles[0][0][0] == [0, 0] and tiles[-1][0][2] == [40, 20]


def test_causal_selection_uses_only_earlier_window():
    scores = np.array([[1, 5, 3], [9, 0, 0], [0, 0, 9]], float)
    sel = bl.causal_selection(scores, lag=1, k=2)
    assert sel[0] is None
    assert list(sel[1]) == [1, 2]          # from window 0, not window 1's own scores
    assert list(sel[2]) == [0, 1]          # from window 1 (ties broken by index)


def test_oracle_uses_same_window():
    sel = bl.oracle_selection(np.array([[1, np.nan, 3], [np.nan, np.nan, np.nan]]))
    assert list(sel[0]) == [2] and sel[1] is None


def test_window_mean_ignores_nan_and_subsets():
    p = np.array([[1, 3, np.nan], [2, np.nan, np.nan]], float)
    assert np.allclose(bl.window_mean(p, 0, 2, None), [2, 2])
    assert np.isnan(bl.window_mean(p, 0, 2, np.array([2]))).all()


def test_phase_surrogate_keeps_spectrum_breaks_coherence_and_gaps():
    rng = np.random.default_rng(0)
    t = np.arange(600) / 30.
    s = np.sin(2 * np.pi * 2 * t)
    rgb = np.column_stack([s, s, s]) + 100
    rgb[10:12] = np.nan
    out = bl.phase_surrogate(rgb, rng)
    assert np.isnan(out[10:12]).all() and np.isfinite(np.delete(out, [10, 11], 0)).all()
    filled, _ = bl.fill_gaps(rgb)
    clean = bl.phase_surrogate(filled, np.random.default_rng(2))          # no gaps: spectrum exactly kept
    assert np.allclose(np.abs(np.fft.rfft(clean[:, 0])), np.abs(np.fft.rfft(filled[:, 0])), atol=1e-6)
    a, b = bl.fill_gaps(out)[0][:, 0], bl.fill_gaps(out)[0][:, 1]
    assert abs(np.corrcoef(a, b)[0, 1]) < .9                 # channels no longer coherent
    assert np.isclose(np.nanmean(out[:, 0]), filled[:, 0].mean())


def test_shuffle_surrogate_is_permutation():
    rng = np.random.default_rng(1)
    rgb = np.arange(30, dtype=float).reshape(10, 3)
    out = bl.shuffle_surrogate(rgb, rng, block=2)
    assert sorted(out[:, 0]) == sorted(rgb[:, 0])
    assert (np.diff(out[:, 0].reshape(-1, 2), axis=1) == 3).all()   # blocks kept intact


def test_present_and_empirical_p():
    assert list(bl.present([1., 10., np.nan], np.arange(100.))) == [False, False, False]
    assert list(bl.present([99.5], np.arange(100.))) == [True]
    assert bl.empirical_p(5., [1, 2, 3]) == .25
    assert bl.empirical_p(0., [1, 2, 3]) == 1.


def test_agree():
    assert bl.agree(120., 125.) is True
    assert bl.agree(120., 130.) is False
    assert bl.agree(None, 120.) is None


def test_clip_half_splits_square_disjointly():
    square = [[0, 0], [10, 0], [10, 4], [0, 4]]
    left, right = bl.clip_half(square, 'L'), bl.clip_half(square, 'R')
    assert left[:, 0].max() == 5 and right[:, 0].min() == 5
    assert left[:, 0].min() == 0 and right[:, 0].max() == 10
    assert bl.clip_half([[0, 0], [1, 1]], 'L') is None


def test_phase5_stride_keeps_time_constants():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / 'scripts/run_phase5.py'
    spec = importlib.util.spec_from_file_location('run_phase5', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flags = dict(ema_beta=.9, hysteresis_frames=8, selection='viterbi')
    assert module.strided(flags, 1) == flags
    two = module.strided(flags, 2)
    assert np.isclose(two['ema_beta'], .81) and two['hysteresis_frames'] == 4 and two['selection'] == 'viterbi'
    assert module.strided(dict(ema_beta=0., hysteresis_frames=1), 4) == dict(ema_beta=0., hysteresis_frames=1)


def test_units_or_all_never_leaves_its_kind():
    tiles = [40, 41, 42]
    assert bl.units_or_all(None, tiles) == tiles
    assert bl.units_or_all(np.array([2, 0]), tiles) == [42, 40]
    pulses = np.zeros((4, 50))
    pulses[:, 40:43] = 1.
    assert np.allclose(bl.window_mean(pulses, 0, 4, bl.units_or_all(None, tiles)), 1.)   # not diluted by other kinds
