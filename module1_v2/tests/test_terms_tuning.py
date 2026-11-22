import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module1_v2 import terms, tuning

ROOT = Path(__file__).resolve().parents[2]
POSE_META = ROOT/'runs/pilot24_smoke/outputs/pose_sam2/experiment_metadata.json'


@pytest.mark.skipif(not POSE_META.is_file(), reason='needs the v1 smoke run for the 308 keypoint names')
def test_keypoint_visibility_uses_template_not_only_confident_support():
    from module1.roi.anatomical import AnatomicalROIMapper
    names = json.loads(POSE_META.read_text())['keypoint_names']
    mapper = AnatomicalROIMapper(names)
    index = {n: i for i, n in enumerate(names)}
    arm = ['left_shoulder', 'left_acromion', 'left_elbow', 'left_wrist']
    points = {index[n]: (10., 10., .9) for n in arm if n in index}
    assert terms.keypoint_visibility(mapper, 'LEFT_ARM', points) == 1.
    points[index['left_wrist']] = (10., 10., .2)          # one of the template points not confident
    present = [n for n in arm if n in index]
    assert terms.keypoint_visibility(mapper, 'LEFT_ARM', points) == pytest.approx((len(present) - 1) / len(present))
    assert terms.keypoint_visibility(mapper, 'LEFT_ARM', {}) == 0.


def test_correlation_audit_flags_double_counted_skin():
    rng = np.random.default_rng(0)
    s = rng.random(200)
    table = pd.DataFrame(dict(pose_term=rng.random(200), visibility_term=s, skin_term=s,
                              visibility_keypoint=rng.random(200)))
    _, _, flags = terms.correlation_audit(table)
    assert {tuple(x) for x in flags[['term_a', 'term_b']].values} == {('V', 'S')}


def test_choose_respects_retention_floor_then_minimizes_switches():
    results = [dict(params=1, switches_per_sec=1., retention=.97), dict(params=2, switches_per_sec=3., retention=.99),
               dict(params=3, switches_per_sec=2., retention=.985)]
    assert tuning.choose(results, .98)['params'] == 3
    fallback = tuning.choose(results, .999)
    assert fallback['params'] == 2 and fallback['feasible'] is False
