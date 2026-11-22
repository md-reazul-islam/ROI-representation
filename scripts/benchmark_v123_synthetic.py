"""Controlled synthetic benchmarks with known ground truth for the v1 / v2 / v3 ablation study.

  S1 mechanisms  scripts/benchmark_module2_synthetic.py (v2 components: site consensus, harmonic-aware HR,
                 spectral RR fusion under paradoxical breathing, thoraco-abdominal phase).
  S2 neural      domain-shifted neonates: the source infants (HR 120-160 bpm, light skin, low noise) differ from
                 the targets (HR 150-220 bpm with variability, darker skin, weaker pulse, more noise, two
                 low-confidence ROIs with an in-band colour artefact). Window HR error of POS, GREEN and
                 the neural variants (source, guided / unguided adaptation, each guidance component removed), all
                 fused with the same Module 1 prior.
  S3 events      random recordings with apnoeas (6-30 s), linked bradycardias (60-95 bpm) and desaturations,
                 and per recording at random: a colour flicker coherent over all sites at 60-85 bpm, breathing-
                 induced shading at 2 x RR on all sites, or head-only visibility with a local face artefact. Event-level sensitivity, PPV, F1 and false alarms per hour of the full detector and of the
                 detector with one safeguard removed.
The numbers show how each component handles a known failure mode under controlled conditions, not accuracy on
real NICU video. Usage: python scripts/benchmark_v123_synthetic.py --output-dir DIR [--targets N] [--recordings N]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd

from module2.events import EventConfig, detect_events, match_events, timeline
from module2.neural import TTA_VARIANTS, NeuralConfig, VideoClips, adapted_copy, build_model, finalize, infer, train
from module2.rppg import HRConfig, estimate_hr, extract_pulse

FS = 30.
NAMES = ['FOREHEAD', 'LEFT_CHEEK', 'UPPER_CHEST', 'LEFT_ARM', 'LEFT_LEG']
EVENT_TYPES = ('apnoea', 'bradycardia', 'desaturation')


# ----------------------------------------------------------------------------------------------- S2 neural
def neonate(rng, domain, seconds):
    """(grids (n, m, 4, 4, 3), Q (n, m), true HR per frame) of one synthetic neonate."""
    t = np.arange(int(seconds * FS)) / FS
    if domain == 'source':
        hr = rng.uniform(120, 160) + 0 * t
        base, direction, amplitude, noise = np.array([200., 140., 110.]), np.array([.33, .77, .53]), .004, .002
        q = np.array([.8, .75, .7, .65, .6])
        artefact_rois = ()
    else:
        hr = rng.uniform(150, 220) + 6 * np.sin(2 * np.pi * t / rng.uniform(10, 20) + rng.uniform(0, 6.3))
        base, direction, amplitude, noise = np.array([120., 80., 60.]), np.array([.33, .77, .53]), .003, .003
        q = np.array([.8, .75, .7, .25, .2])
        artefact_rois = (3, 4)
    phase = 2 * np.pi * np.cumsum(hr / 60.) / FS
    grids = np.empty((len(t), len(NAMES) + 1, 4, 4, 3))
    for r in range(len(NAMES) + 1):
        colour = 1 + amplitude * (1. if r < 2 else .6) * np.sin(phase)[:, None] * direction
        colour += (.02 * np.sin(2 * np.pi * .1 * t + rng.uniform(0, 6.3)))[:, None]          # illumination drift
        if r in artefact_rois:
            bpm = rng.uniform(90, 270)
            colour += 3 * amplitude * np.sin(2 * np.pi * bpm / 60 * t + rng.uniform(0, 6.3))[:, None] * \
                rng.uniform(.2, 1., 3)
        rgb = base * (colour + noise * rng.standard_normal((len(t), 3)))
        grids[:, r] = rgb[:, None, None, :] * (1 + .01 * rng.standard_normal((1, 4, 4, 3)))
    quality = np.c_[np.tile(q, (len(t), 1)) + .02 * rng.standard_normal((len(t), len(q))), np.full(len(t), np.nan)]
    return grids, quality, hr


def window_errors(pulse, weights, hr_true, config):
    fused = (np.nan_to_num(pulse[:, :len(weights)]) * weights).sum(axis=1) / weights.sum()
    length = int(config.estimation_window_sec * FS)
    errors = []
    for start in range(0, len(fused) - length + 1, int(FS)):
        estimate, _ = estimate_hr(fused[start:start + length], FS, config)
        errors.append(np.nan if estimate is None else estimate - hr_true[start:start + length].mean())
    return errors


def neural_benchmark(targets, sources, tta_steps, pretrain_steps, device, seed):
    rng = np.random.default_rng(seed)
    hr_config, cfg = HRConfig(), NeuralConfig(tta_steps=tta_steps)
    source_videos = []
    for i in range(sources):
        grids, q, _ = neonate(rng, 'source', 30.)
        source_videos.append(VideoClips(f's{i}', grids, q, NAMES + ['INFANT_BOX'], FS, cfg))
    model = build_model(cfg)
    start = time.time()
    train(model, source_videos, cfg, hr_config.band_hz, pretrain_steps, cfg.pretrain_lr, True, device)
    print(f'S2: source model on {sources} synthetic infants in {time.time() - start:.0f} s', flush=True)
    rows = []
    for i in range(targets):
        grids, q, hr_true = neonate(rng, 'target', 30.)
        clips = VideoClips(f't{i}', grids, q, NAMES + ['INFANT_BOX'], FS, cfg)
        prior = np.exp((q[0, :len(NAMES)] - q[0, :len(NAMES)].max()) / .1)
        rgb = grids.mean(axis=(2, 3))
        pulses = {m: np.stack([extract_pulse(rgb[:, r], FS, m, hr_config) for r in range(len(NAMES))], axis=1)
                  for m in ('POS', 'GREEN')}
        pulses['NEURAL_SRC'] = finalize(infer(model, clips, cfg, device), FS, hr_config)
        for name, (guided, overrides) in TTA_VARIANTS.items():
            variant = replace(cfg, **overrides)
            adapted, _ = adapted_copy(model, clips, variant, hr_config.band_hz, guided, device)
            pulses[name] = finalize(infer(adapted, clips, variant, device), FS, hr_config)
        for method, pulse in pulses.items():
            for w, error in enumerate(window_errors(pulse, prior, hr_true, hr_config)):
                rows.append(dict(target=i, window=w, method=method, error_bpm=error))
        print(f'S2: target {i + 1}/{targets} done', flush=True)
    windows = pd.DataFrame(rows)
    per_target = (windows.assign(abs_error=windows.error_bpm.abs())
                  .groupby(['method', 'target']).abs_error.mean().rename('mae_bpm').reset_index())
    summary = windows.assign(abs_error=windows.error_bpm.abs()).groupby('method').agg(
        mae_bpm=('abs_error', 'mean'), rmse_bpm=('error_bpm', lambda e: float(np.sqrt(np.nanmean(e ** 2)))),
        within5_percent=('abs_error', lambda e: 100 * float(np.mean(e <= 5))), windows=('abs_error', 'size')).reset_index()
    return windows, per_target, summary


# ----------------------------------------------------------------------------------------------- S3 events
def recording(rng, seconds):
    """Synthetic recording with random ABD events and artefacts; returns inputs of events.timeline + truth."""
    t = np.arange(int(seconds * FS)) / FS
    hr_base, rr_base = rng.uniform(130, 160), rng.uniform(35, 55)
    hr = np.full(len(t), hr_base)
    breathing = np.ones(len(t))
    red = np.ones(len(t))
    truth = []
    slots = np.arange(10., seconds - 45., 45.)
    for slot in rng.permutation(slots)[:rng.integers(1, len(slots) + 1)]:
        start = slot + rng.uniform(0, 8)
        duration = rng.choice([6., 12., 15., 22., 28.])
        breathing[(t >= start) & (t < start + duration)] = .02
        linked = False
        if rng.random() < .6:
            b0, b1 = start + rng.uniform(3, 8), rng.uniform(6, 14)
            hr[(t >= b0) & (t < b0 + b1)] = rng.uniform(60, 95)
            truth.append(dict(event_type='bradycardia', start_sec=b0, end_sec=b0 + b1))
            linked = True
        if rng.random() < .5:
            d0, d1 = start + rng.uniform(4, 10), rng.uniform(12, 25)
            red[(t >= d0) & (t < d0 + d1)] = rng.uniform(1.4, 1.8)
            truth.append(dict(event_type='desaturation', start_sec=d0, end_sec=d0 + d1))
            linked = True
        if duration >= 20 or (duration >= 10 and linked):
            truth.append(dict(event_type='apnoea', start_sec=start, end_sec=start + duration))
    phase = 2 * np.pi * np.cumsum(hr / 60.) / FS
    direction = np.tile([.33, .77, .53], (len(t), 1))
    direction[:, 0] *= red
    breath = np.sin(2 * np.pi * rr_base / 60 * t)
    scenario = rng.choice(['clean', 'flicker', 'shading', 'head_only'])
    flicker = .006 * np.sin(2 * np.pi * rng.uniform(60, 85) / 60 * t + rng.uniform(0, 6.3))
    artefact_start = rng.uniform(0, seconds - 20)
    local = ((t >= artefact_start) & (t < artefact_start + 15)) * .012 * np.sin(2 * np.pi * rng.uniform(60, 80) / 60 * t)
    rgb = np.empty((len(t), len(NAMES), 3))
    for r, name in enumerate(NAMES):
        colour = .004 * np.sin(phase)[:, None] * direction
        if scenario == 'flicker':                                   # e.g. phototherapy / monitor light
            colour += flicker[:, None] * np.array([.6, .8, .7])
        if scenario == 'shading':                                   # breathing-induced shading at 2 x RR
            colour += .008 * ((breath * breathing) ** 2 - .5)[:, None] * np.array([.3, .7, .5])
        if scenario == 'head_only' and name == 'LEFT_CHEEK':        # local twitch / glare on the only visible site
            colour += local[:, None] * np.array([.6, .8, .7])
        rgb[:, r] = np.array([200., 140., 110.]) * (1 + colour + .0006 * rng.standard_normal((len(t), 3)))
    weights = np.full((len(t), len(NAMES)), 1. / len(NAMES))
    if scenario == 'head_only':
        weights[:, 2:] = 0.
        weights /= weights.sum(axis=1, keepdims=True)
    waveform = breathing * breath + .01 * rng.standard_normal(len(t))
    centres = np.arange(4., seconds - 3.)
    paused = np.interp(centres, t, breathing) < .5
    rr = pd.DataFrame(dict(window_center_sec=centres, rr_brpm=np.where(paused, np.nan, rr_base),
                           taa_phase_deg=np.where(paused, np.nan, 20.), taa_coherence=np.where(paused, np.nan, .9)))
    return (rgb, weights, NAMES, FS, waveform, np.full(len(t), .01), rr), truth, scenario


EVENT_ABLATIONS = dict(full={}, minus_multisite_confirmation=dict(min_agreeing_sites=1),
                       minus_baseline_plausibility=dict(min_baseline_bpm=0.),
                       minus_harmonic_exclusion=dict(rr_harmonics=()),
                       minus_relative_drop=dict(brady_rel_drop=1.),
                       minus_extended_band=dict(hr_min_bpm=90.))


def event_benchmark(recordings, seconds, seed):
    rng = np.random.default_rng(seed)
    data = [recording(rng, seconds) for _ in range(recordings)]
    print('S3 scenarios:', pd.Series([d[2] for d in data]).value_counts().to_dict(), flush=True)
    rows = []
    for name, overrides in EVENT_ABLATIONS.items():
        cfg = replace(EventConfig(), **overrides)
        detected, reference = [], []
        for i, (inputs, truth, _) in enumerate(data):
            found = detect_events(timeline(*inputs, cfg), cfg)
            detected += [dict(e, video_id=i) for e in found if e['event_type'] in EVENT_TYPES and e['clinically_significant']]
            reference += [dict(e, video_id=i) for e in truth]
        hours = recordings * seconds / 3600.
        for kind in EVENT_TYPES:
            result = match_events(detected, reference, kind)
            false = sum(not any(d['video_id'] == r['video_id'] and d['start_sec'] < r['end_sec'] and r['start_sec'] < d['end_sec']
                                for r in reference if r['event_type'] == kind)
                        for d in detected if d['event_type'] == kind)
            s, p = result['sensitivity'], result['ppv']
            rows.append(dict(configuration=name, **result, f1=2 * s * p / (s + p) if s and p else 0.,
                             false_alarms_per_hour=false / hours))
        print(f'S3: {name} done', flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Synthetic benchmarks (known ground truth) for the v1/v2/v3 ablation.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--targets', type=int, default=12, help='S2 target neonates')
    parser.add_argument('--sources', type=int, default=8, help='S2 source neonates')
    parser.add_argument('--tta-steps', type=int, default=300)
    parser.add_argument('--pretrain-steps', type=int, default=1500)
    parser.add_argument('--recordings', type=int, default=24, help='S3 recordings')
    parser.add_argument('--seconds', type=float, default=150.)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--skip', nargs='*', default=[], choices=('S1', 'S2', 'S3'))
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(seed=args.seed, args=vars(args))
    if 'S1' not in args.skip:
        subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_module2_synthetic.py'), str(output/'S1_mechanisms.json')],
                       check=True, stdout=subprocess.DEVNULL)
        print('S1: mechanisms done', flush=True)
    if 'S2' not in args.skip:
        import torch
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        windows, per_target, summary = neural_benchmark(args.targets, args.sources, args.tta_steps,
                                                        args.pretrain_steps, device, args.seed)
        windows.to_csv(output/'S2_neural_windows.csv', index=False)
        per_target.to_csv(output/'S2_neural_per_target.csv', index=False)
        summary.to_csv(output/'S2_neural_summary.csv', index=False)
        meta['S2_device'] = device
        print(summary.round(2).to_string(index=False))
    if 'S3' not in args.skip:
        events = event_benchmark(args.recordings, args.seconds, args.seed)
        events.to_csv(output/'S3_events.csv', index=False)
        print(events.round(2).to_string(index=False))
    (output/'synthetic_metadata.json').write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
