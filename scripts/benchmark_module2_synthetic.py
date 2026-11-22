"""Controlled synthetic comparison of existing vs proposed Module 2 techniques (300 random trials per scenario).

Each scenario isolates one failure mode with known ground truth; the numbers show whether a method handles
that failure mode, not its accuracy on real NICU video (use evaluate_hr.py / evaluate_rr.py with reference data).
  A  local artefact on the most trusted region: top1 / pool_mean / dynamic-weighted waveform vs site consensus
  B  respiratory-harmonic leakage into the pulse: plain FFT peak vs harmonic-aware selection (B3: HR = 2 x RR)
  C  paradoxical breathing: waveform averaging vs spectral fusion of chest and abdomen
  D  thoraco-abdominal phase recovery from 2-D flow
Usage: python scripts/benchmark_module2_synthetic.py [output.json]   (default reports/module2_synthetic_benchmark.json)
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from module2.cardiorespiratory import (CardioRespConfig, consensus_hr, near_harmonic, roi_spectrum,
                                       shared_axis_displacements, thoraco_abdominal_phase)
from module2.respiration import RRConfig, estimate_rr, fuse_spectra, window_spectrum
from module2.rppg import HRConfig, estimate_hr

FS, SECONDS, TRIALS = 30., 8., 300
HR, CR, RR = HRConfig(), CardioRespConfig(), RRConfig()
rng = np.random.default_rng(42)
t = np.arange(int(SECONDS * FS)) / FS


def tone(bpm, amplitude=1.):
    return amplitude * np.sin(2*np.pi*bpm/60*t + rng.uniform(0, 2*np.pi))


def noise(scale):
    return scale * rng.standard_normal(len(t))


def score(errors):
    errors = np.abs(np.asarray(errors, float))
    return dict(mae=round(float(np.mean(errors)), 1), within5=round(float(100*np.mean(errors <= 5)), 1))


def fft_hr(x):
    return estimate_hr(x, FS, HR, estimator='fft')[0]


results = {}

# A. Local artefact on the most trusted region (head), pulse visible at head, trunk and arm.
for label, artefact in (('A0_no_artefact', 0.), ('A1_artefact_1x', .5), ('A2_artefact_3x', 1.5)):
    errs = {k: [] for k in ('top1', 'pool_mean', 'dynamic_weighted', 'consensus')}
    for _ in range(TRIALS):
        hr = rng.uniform(100, 200)
        art = rng.uniform(90, 270)
        while abs(art - hr) < 15:
            art = rng.uniform(90, 270)
        rois = {'FOREHEAD': tone(hr, .5) + (tone(art, artefact) if artefact else 0) + noise(.3),
                'UPPER_CHEST': tone(hr, .5) + noise(.3), 'LEFT_ARM': tone(hr, .5) + noise(.3)}
        names, w = list(rois), np.array([.8, .1, .1])
        errs['top1'].append(fft_hr(rois['FOREHEAD']) - hr)
        errs['pool_mean'].append(fft_hr(sum(rois.values()) / 3) - hr)
        errs['dynamic_weighted'].append(fft_hr(sum(wi * rois[n] for n, wi in zip(names, w))) - hr)
        spectra = [roi_spectrum(rois[n], FS, HR) for n in names]
        errs['consensus'].append(consensus_hr(spectra, names, w, FS, HR, CR)['hr_bpm'] - hr)
    results[label] = {k: score(v) for k, v in errs.items()}

# B. Respiratory-harmonic leakage into the pulse trace (neonatal RR 30-70 /min).
for label, leak in (('B1_leak_0.7x', .7), ('B2_leak_1.5x', 1.5)):
    plain, aware, n = [], [], 0
    while n < TRIALS:
        hr, rr = rng.uniform(100, 200), rng.uniform(30, 70)
        if near_harmonic(hr, rr, CR) or any(abs(hr - k*rr) < 12 for k in CR.rr_harmonics):
            continue  # clearly separated cases; ambiguous ones are tested in B3
        x = tone(hr) + tone(2*rr, leak) + tone(3*rr, .5*leak) + noise(.3)
        plain.append(fft_hr(x) - hr)
        aware.append(consensus_hr([roi_spectrum(x, FS, HR)], ['FOREHEAD'], np.array([1.]), FS, HR, CR, rr_bpm=rr)['hr_bpm'] - hr)
        n += 1
    results[label] = dict(plain_fft=score(plain), harmonic_aware=score(aware))

# B3. HR truly equal to 2 x RR (physiological coincidence): must not be moved, should be flagged.
kept, flagged = [], 0
for _ in range(TRIALS):
    rr = rng.uniform(45, 70)
    hr = 2 * rr
    x = tone(hr) + noise(.3)
    out = consensus_hr([roi_spectrum(x, FS, HR)], ['FOREHEAD'], np.array([1.]), FS, HR, CR, rr_bpm=rr)
    kept.append(out['hr_bpm'] - hr)
    flagged += out['harmonic_ambiguous']
results['B3_hr_equals_2rr'] = dict(harmonic_aware=score(kept), flagged_percent=round(100*flagged/TRIALS, 1))

# C. Paradoxical breathing: waveform averaging vs spectral fusion of chest + abdomen (15 s windows).
t15 = np.arange(int(15 * FS)) / FS
wave_err, spec_err = [], []
for _ in range(TRIALS):
    rr = rng.uniform(30, 70)
    phase = np.pi + rng.normal(0, .15)
    chest = np.sin(2*np.pi*rr/60*t15) + .3*rng.standard_normal(len(t15))
    abdomen = np.sin(2*np.pi*rr/60*t15 + phase) + .3*rng.standard_normal(len(t15))
    wave = window_spectrum((chest + abdomen) / 2, FS, RR)
    wave_err.append((estimate_rr(*wave, 15., RR)[0] if wave else np.nan) - rr)
    fused = fuse_spectra([window_spectrum(chest, FS, RR), window_spectrum(abdomen, FS, RR)], [.5, .5])
    spec_err.append(estimate_rr(*fused, 15., RR)[0] - rr)
results['C_paradoxical_rr'] = dict(waveform_average=score(np.nan_to_num(wave_err, nan=99)), spectral_fusion=score(spec_err))

# D. Thoraco-abdominal phase recovery from 2-D flow (random phase 0-180 deg, RR 30-70).
errors = []
for _ in range(TRIALS):
    rr, phase = rng.uniform(30, 70), rng.uniform(0, 180)
    def velocity(ph, amp):
        d = amp * np.sin(2*np.pi*rr/60*t15 + np.radians(ph))
        v = np.diff(np.r_[0., d])
        return np.column_stack([.2*v, v]) + .003*rng.standard_normal((len(t15), 2))
    pair = shared_axis_displacements(velocity(0, 1.), velocity(phase, rng.uniform(.5, 1.)), FS, RR, CR)
    est, _ = thoraco_abdominal_phase(*pair, FS, rr, CR)
    errors.append(est - phase)
results['D_taa_phase'] = dict(mae_deg=round(float(np.mean(np.abs(errors))), 1),
                              within15_percent=round(float(100*np.mean(np.abs(errors) <= 15)), 1))
print(json.dumps(results, indent=1))
output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'reports/module2_synthetic_benchmark.json'
output.write_text(json.dumps(results, indent=1))
print('written', output)
