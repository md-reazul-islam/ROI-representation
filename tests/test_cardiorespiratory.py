import numpy as np
import pandas as pd
import pytest

from module2.cardiorespiratory import (CardioRespConfig, consensus_hr, near_harmonic, nearest_rr, roi_spectrum,
                                       select_hr_peak, shared_axis_displacements, site_of, thoraco_abdominal_phase)
from module2.respiration import RRConfig
from module2.rppg import HRConfig, estimate_hr


FS = 30.
HR = HRConfig()
CR = CardioRespConfig()


def tone(bpm, seconds=8., amplitude=1., phase=0.):
    t = np.arange(int(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * bpm / 60 * t + phase)


def noise(seconds=8., scale=.3, seed=0):
    return scale * np.random.default_rng(seed).standard_normal(int(seconds * FS))


def test_sites_group_the_module1_regions():
    assert site_of('FOREHEAD') == 'HEAD' and site_of('ABDOMEN') == 'TRUNK'
    assert site_of('LEFT_ARM') == 'ARMS' and site_of('RIGHT_LEG') == 'LEGS' and site_of('INFANT_BOX') is None


def test_consensus_rejects_a_local_artefact_that_dominates_the_weighted_pulse():
    """True HR 140 bpm at head, trunk and arm; a strong local artefact at 200 bpm on the most trusted ROI."""
    pulses = {'FOREHEAD': tone(140, amplitude=.5) + tone(200, amplitude=1.5) + noise(seed=1),
              'UPPER_CHEST': tone(140, amplitude=.5) + noise(seed=2),
              'LEFT_ARM': tone(140, amplitude=.5) + noise(seed=3)}
    names, weights = list(pulses), np.array([.8, .1, .1])
    weighted = sum(w * pulses[name] for name, w in zip(names, weights))
    assert abs(estimate_hr(weighted, FS, HR)[0] - 200) < 3            # weighted waveform follows the artefact
    spectra = [roi_spectrum(pulses[name], FS, HR) for name in names]
    result = consensus_hr(spectra, names, weights, FS, HR, CR)
    assert abs(result['hr_bpm'] - 140) < 3
    assert result['consensus_sites'] == 3 and result['consensus_support'] == 2


def test_consensus_with_one_site_equals_that_sites_peak():
    pulses = [tone(120) + noise(seed=4), tone(120) + noise(seed=5)]
    spectra = [roi_spectrum(x, FS, HR) for x in pulses]
    result = consensus_hr(spectra, ['FOREHEAD', 'LEFT_CHEEK'], np.array([.5, .5]), FS, HR, CR)
    assert abs(result['hr_bpm'] - 120) < 3 and result['consensus_sites'] == 1 and result['consensus_support'] == 1
    assert consensus_hr([None, None], ['FOREHEAD', 'NECK'], np.array([.5, .5]), FS, HR, CR)['hr_bpm'] is None


def test_respiratory_harmonic_is_rejected_when_a_comparable_pulse_peak_exists():
    """RR 55 /min: its 2nd harmonic (110) leaks into the pulse trace more strongly than the 150 bpm pulse."""
    x = tone(150, amplitude=1.) + tone(110, amplitude=1.5) + noise(seed=6, scale=.1)
    spectra = [roi_spectrum(x, FS, HR)]
    plain = consensus_hr(spectra, ['FOREHEAD'], np.array([1.]), FS, HR, CR)
    aware = consensus_hr(spectra, ['FOREHEAD'], np.array([1.]), FS, HR, CR, rr_bpm=55.)
    assert abs(plain['hr_bpm'] - 110) < 3 and not plain['harmonic_ambiguous']
    assert abs(aware['hr_bpm'] - 150) < 3 and not aware['harmonic_ambiguous']


def test_isolated_peak_on_a_harmonic_is_kept_but_flagged():
    """HR exactly 2 x RR (120 vs 60) is physiologically possible: keep it, flag it."""
    frequencies, power = roi_spectrum(tone(120) + noise(seed=7, scale=.1), FS, HR)
    band = (frequencies >= 1.5) & (frequencies <= 4.5)
    bpm, ambiguous = select_hr_peak(frequencies, power, band, 60., CR)
    assert abs(bpm - 120) < 3 and ambiguous
    assert near_harmonic(121., 60., CR) and not near_harmonic(150., 60., CR)


def test_nearest_rr_respects_the_offset_limit():
    table = pd.DataFrame(dict(window_center_sec=[5., 15.], rr_brpm=[40., 50.]))
    assert nearest_rr(table, 13., CR) == 50. and nearest_rr(table, 40., CR) is None
    assert nearest_rr(table.iloc[:0], 5., CR) is None


def breathing_velocity(rr_brpm, phase, seconds=15., amplitude=1.):
    t = np.arange(int(seconds * FS)) / FS
    displacement = amplitude * np.sin(2 * np.pi * rr_brpm / 60 * t + phase)
    velocity = np.diff(np.r_[0., displacement])
    return np.column_stack([.2 * velocity, velocity])


@pytest.mark.parametrize('phase_deg', [0., 90., 180.])
def test_thoraco_abdominal_phase_recovers_known_asynchrony(phase_deg):
    rr = RRConfig()
    chest = breathing_velocity(45., 0.)
    abdomen = breathing_velocity(45., np.radians(phase_deg), amplitude=.7)
    chest += .002 * np.random.default_rng(8).standard_normal(chest.shape)
    pair = shared_axis_displacements(chest, abdomen, FS, rr, CR)
    phase, coherence = thoraco_abdominal_phase(*pair, FS, 45., CR)
    assert abs(phase - phase_deg) < 15 and coherence > .8


def test_thoraco_abdominal_phase_needs_both_regions_observed():
    rr = RRConfig()
    chest, abdomen = breathing_velocity(45., 0.), breathing_velocity(45., 0.)
    abdomen[:len(abdomen) // 3] = np.nan                                   # abdomen hidden for 5 s
    assert shared_axis_displacements(chest, abdomen, FS, rr, CR) is None
    assert thoraco_abdominal_phase(np.ones(10), np.ones(10), FS, 45., CR) == (None, None)
