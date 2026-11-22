"""Synthetic checks of the D8 rules (no real video, no results): run with pytest module1_v2_d8/tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d8 import rules  # noqa: E402

FPS, T, R = 20.0, 600, 6          # 30 s, 6 regions
NAMES = ['FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE', 'UPPER_CHEST']


def base(seed=0):
    rng = np.random.default_rng(seed)
    s = np.full((T, R), 0.9) + rng.normal(0, .01, (T, R))
    A = np.ones((T, R), bool)
    C = np.empty((T, R, 3))
    C[..., 0] = .42 + rng.normal(0, .002, (T, R))
    C[..., 1] = .31 + rng.normal(0, .002, (T, R))
    C[..., 2] = 120 + rng.normal(0, 1, (T, R))
    return s, A, C


def test_nothing_happens():
    s, A, C = base()
    cov, occ, _ = rules.flag_regions(s, A, C, FPS)
    assert not cov.any() and not occ.any()


def test_global_lighting_change_not_flagged():
    s, A, C = base()
    C[300:, :, 2] *= 1.4                       # all regions 40 % brighter
    C[300:, :, 0] += .01                       # and warmer
    cov, occ, _ = rules.flag_regions(s, A, C, FPS)
    assert not occ.any()


def test_local_occluder_flagged_and_released():
    s, A, C = base()
    C[300:400, 2, 0] += .03                    # a hand-like colour change on LEFT_CHEEK only
    C[300:400, 2, 2] *= 1.25
    cov, occ, _ = rules.flag_regions(s, A, C, FPS)
    assert occ[300:400, 2].mean() > .8 and not occ[:300].any()
    assert not occ[450:, 2].any()              # released after the occluder leaves
    assert not occ[:, [0, 1, 3, 4, 5]].any()


def test_coverage_flagged():
    s, A, C = base()
    s[300:400, 5] = .2                         # cloth over UPPER_CHEST
    cov, occ, _ = rules.flag_regions(s, A, C, FPS)
    assert cov[300:400, 5].all() and not cov[:300].any() and not cov[420:, 5].any()


def test_redundancy_switches_to_all_clean():
    s, A, C = base()
    flagged = np.zeros((T, R), bool)
    flagged[300:320, 2] = True
    area = np.full((T, R), 1000.)
    affected, W = rules.redundancy(flagged, A, s, area, NAMES, FPS)
    assert affected[300:359].all() and not affected[359] and not affected[:300].any() and not affected[400:].any()
    assert W[310, 2] == 0 and abs(W[310].sum() - 1) < 1e-9 and (W[310] > 0).sum() == 5
