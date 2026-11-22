import numpy as np

from module1_v2 import motion as mo


def textured(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    import cv2
    return cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (5, 5), 0)


def test_median_global_motion_removes_camera_translation():
    base = textured(seed=1)
    shifted = np.roll(base, 3, axis=1)                    # whole scene (camera) moves 3 px right
    flow = mo.dense_flow(base, shifted)
    infant = np.zeros(base.shape, bool)
    infant[40:80, 60:100] = True
    field, status = mo.global_flow(flow, mo.background_mask(base.shape, infant, dilation=3))
    total, residual, _ = mo.region_motion(flow, flow - field, infant)
    assert status == 'MEDIAN'
    assert total > 2 and residual < .5 * total


def test_infant_motion_survives_global_compensation():
    base = textured(seed=2)
    moved = base.copy()
    moved[40:80, 62:102] = base[40:80, 60:100]            # only the 'infant' patch moves
    flow = mo.dense_flow(base, moved)
    infant = np.zeros(base.shape, bool)
    infant[40:80, 60:100] = True
    field, _ = mo.global_flow(flow, mo.background_mask(base.shape, infant, dilation=3))
    _, residual, _ = mo.region_motion(flow, flow - field, infant)
    assert np.abs(field).max() < .5 and residual > .5


def test_homography_mode_runs_and_no_background_is_flagged():
    base = textured(seed=3)
    flow = mo.dense_flow(base, np.roll(base, 2, axis=0))
    field, status = mo.global_flow(flow, np.ones(base.shape, bool), 'homography')
    assert status == 'HOMOGRAPHY' and abs(np.median(field[..., 1]) - 2) < 1
    _, status = mo.global_flow(flow, np.zeros(base.shape, bool))
    assert status == 'NO_BACKGROUND'


def test_periodic_ratio_separates_breathing_from_noise():
    fs = 10.
    t = np.arange(0, 8, 1 / fs)
    breathing = np.stack([np.sin(2 * np.pi * .8 * t), .3 * np.sin(2 * np.pi * .8 * t)], 1)
    noise = np.random.default_rng(5).normal(size=(len(t), 2))
    assert mo.periodic_ratio(breathing, fs) > .8
    assert mo.periodic_ratio(noise, fs) < .4
    assert np.isnan(mo.periodic_ratio(breathing[:5], fs))


def test_nuisance_penalizes_only_aperiodic_motion_on_respiratory_regions():
    assert mo.nuisance(.6, .9, True) == .6 * (1 - .9)
    assert mo.nuisance(.6, .9, False) == .6
    assert mo.nuisance(.6, np.nan, True) == .6
