import numpy as np
import pandas as pd
import pytest

from module2.ablation import (HR_COMPONENTS, HR_VERSIONS, bootstrap_ci, compare, coverage_accuracy, friedman, holm,
                              pair_reference, paired_test, per_video_quality)
from module2.neural import TTA_VARIANTS, NeuralConfig, Sampler
from test_neural_rppg import clips


def windows(video, hr, reliable, sqi=.8):
    return pd.DataFrame(dict(dataset='d', video_id=video, method='POS', strategy='dynamic_fusion',
                             window_start_sec=np.arange(len(hr)), window_end_sec=np.arange(len(hr)) + 8., hr_bpm=hr,
                             reliable=reliable, sqi=sqi, snr_db=3.))


def test_per_video_quality_metrics():
    table = windows('a', [150., 152., np.nan, 80.], [True, True, False, False])
    row = per_video_quality(table, 'hr_bpm', (100., 200.)).iloc[0]
    assert row.windows == 4 and row.coverage_percent == 75. and row.reliable_percent == 50.
    assert row.plausible_percent == 50. and row.jump_median == 2. and row.rate_median_reliable == 151.


def test_paired_test_direction_effect_size_and_ties():
    a = np.arange(10.)
    result = paired_test(a, a + np.linspace(1, 2, 10))
    assert result['n'] == 10 and result['rank_biserial'] == 1. and result['p_value'] < .01
    assert paired_test(a, a - 1)['rank_biserial'] == -1.
    tie = paired_test(a, a)
    assert tie['p_value'] == 1. and tie['rank_biserial'] == 0.
    assert paired_test([1., np.nan], [2., 3.])['n'] == 1


def test_holm_and_friedman():
    assert np.allclose(holm([.01, .04, .03, np.nan])[:3], [.03, .06, .06]) and np.isnan(holm([np.nan])[0])
    table = pd.DataFrame(dict(v1=np.arange(8.), v2=np.arange(8.) + 1, v3=np.arange(8.) + 2))
    assert friedman(table)['p_value'] < .01 and friedman(table.iloc[:1])['n'] == 1


def test_bootstrap_ci_contains_the_mean():
    estimate, low, high = bootstrap_ci(np.random.default_rng(0).normal(5, 1, 50))
    assert low < estimate < high and abs(estimate - 5) < .5
    assert np.isnan(bootstrap_ci([np.nan])[0])


def test_compare_pairs_by_video():
    per_video = pd.DataFrame(dict(video_key=list('abcdefgh') * 2, version=['v1'] * 8 + ['v2'] * 8,
                                  sqi_median=list(np.linspace(.2, .5, 8)) + list(np.linspace(.4, .8, 8))))
    result = compare(per_video, 'v1', ['v2'], ['sqi_median']).iloc[0]
    assert result.n == 8 and result.mean_difference > 0 and result.p_holm == result.p_value


def test_reference_pairing_and_reliability_accuracy_curve():
    table = windows('a', [150., 160.], [True, False]).assign(sqi=[.9, .2])
    reference = pd.DataFrame(dict(video_id='a', timestamp_sec=np.arange(10.), hr_bpm=150.))
    pairs = pair_reference(table, reference, 'hr_bpm')
    assert list(pairs.reference) == [150., 150.]
    curve = coverage_accuracy(pairs, 'hr_bpm', thresholds=[0., .5])
    assert list(curve.mae) == [5., 0.] and list(curve.coverage_percent) == [100., 50.]


def test_version_definitions_are_consistent():
    keys = {k for k, *_ in HR_VERSIONS}
    assert keys == {'v1', 'v2', 'v3'} and all(reference in keys for _, _, _, reference, _ in HR_COMPONENTS)
    assert set(TTA_VARIANTS) >= {m for _, m, _, _, _ in HR_COMPONENTS if m.startswith('NEURAL_TTA')}


@pytest.mark.parametrize('switch, sampling_uniform, pairs', [('guide_sampling', True, True), ('guide_sites', False, False)])
def test_guidance_components_switch_off_independently(switch, sampling_uniform, pairs):
    cfg = NeuralConfig(batch_size=16, site_pairs=4, **{switch: False})
    sampler = Sampler([clips(150.)], cfg, True, np.random.default_rng(0))
    assert np.allclose(sampler.probabilities, sampler.probabilities[0]) == sampling_uniform
    assert (sampler.batch()[2] is not None) == pairs
