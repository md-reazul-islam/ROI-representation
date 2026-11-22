"""Tune the Phase 1a stabilization parameters on TRAINING videos only (objective: module1_v2/tuning.py).

Only the stored per-frame confidences of training videos are read (no pixels, no pulse, nothing from
validation/test). Writes module1_v2/runs/<run>/tuning_log.json with the grid, every result, the chosen
values and the training video IDs + SHA-256.

Usage: python module1_v2/scripts/tune_phase1.py --run runs/dataset5_paper
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for path in (V2/'src', ROOT/'src'):
    sys.path.insert(0, str(path))

from module1_v2 import selection as sel, tuning                # noqa: E402

CFG = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())


def evaluate(videos, flags):
    switches, kept = [], []
    for frames, cand, fps in videos:
        _, _, Q, sets = sel.select_video(cand, frames, flags, fps, CFG['phase1']['top_k'], CFG['phase1']['temperature'])
        switches.append(sel.switches_per_sec(sets, fps))
        kept.append(sel.retention(Q, sets, CFG['phase1']['top_k']))
    return dict(switches_per_sec=float(np.nanmean(switches)), retention=float(np.nanmean(kept)),
                per_video_switches=switches, per_video_retention=kept)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tune Phase 1a stabilization on training videos only.')
    parser.add_argument('--run', required=True)
    args = parser.parse_args(argv)
    run = (ROOT/args.run).resolve()
    p1 = CFG['phase1']
    manifest = pd.read_csv(run/'manifest.csv')
    train = manifest[manifest.split == 'train']
    cand = pd.read_csv(run/'outputs/confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=['video_id', 'split', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'q'])
    assert set(cand[cand.video_id.isin(train.video_id)].split) <= {'train'}, 'split mismatch'
    videos = []
    for r in train.itertuples():
        rows = cand[cand.video_id == r.video_id]
        frames = sorted(rows.frame_index.unique())
        videos.append((frames, rows, len(frames) / r.duration_sec))
    started = time.perf_counter()
    base = dict(CFG['v1_equivalent'])
    log = dict(run=str(run), objective=tuning.__doc__.strip(), retention_min=p1['quality_retention'],
               training_videos=train.video_id.tolist(), training_sha256=train.sha256.tolist(),
               not_used=manifest[manifest.split != 'train'].video_id.tolist(), results={})

    log['v1_reference'] = evaluate(videos, base)
    grid = []
    for beta, delta, frames_n in itertools.product(p1['ema_beta_grid'], p1['hysteresis_delta_grid'], p1['hysteresis_frames_grid']):
        flags = dict(base, ema_beta=beta, selection='hysteresis', hysteresis_delta=delta, hysteresis_frames=frames_n)
        grid.append(dict(params=dict(ema_beta=beta, hysteresis_delta=delta, hysteresis_frames=frames_n), **evaluate(videos, flags)))
    d2 = tuning.choose(grid, p1['quality_retention'])
    log['results']['D2'] = grid
    beta_only = [dict(params=dict(ema_beta=b), **evaluate(videos, dict(base, ema_beta=b))) for b in p1['ema_beta_grid']]
    log['results']['D1_beta_only'] = beta_only

    chosen = dict(d2['params'])
    vit = []
    for penalty, lag in itertools.product(p1['viterbi_penalty_grid'], (None, p1['viterbi_lag_sec'])):
        flags = dict(base, **chosen, selection='viterbi', viterbi_penalty=penalty, viterbi_lag=lag)
        vit.append(dict(params=dict(viterbi_penalty=penalty, viterbi_lag=lag), **evaluate(videos, flags)))
    log['results']['D3'] = vit
    d3 = tuning.choose([v for v in vit if v['params']['viterbi_lag'] is None], p1['quality_retention'])
    d3l = tuning.choose([v for v in vit if v['params']['viterbi_lag'] is not None], p1['quality_retention'])
    best = tuning.choose([dict(d2, name='D2'), dict(d3, name='D3'), dict(d3l, name='D3L')], p1['quality_retention'])
    log['chosen'] = dict(D1=dict(ema_beta=chosen['ema_beta']), D2=chosen, D3=dict(chosen, **d3['params']),
                         D3L=dict(chosen, **d3l['params']), D4_base=best['name'],
                         feasible=dict(D2=d2['feasible'], D3=d3['feasible'], D3L=d3l['feasible']),
                         training_means={k: dict(switches_per_sec=v['switches_per_sec'], retention=v['retention'])
                                         for k, v in (('D2', d2), ('D3', d3), ('D3L', d3l))})
    log['wall_time_sec'] = time.perf_counter() - started
    target = V2/'runs'/run.name
    target.mkdir(parents=True, exist_ok=True)
    (target/'tuning_log.json').write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log['chosen'], indent=2, default=str))
    print('v1 reference (training):', {k: log['v1_reference'][k] for k in ('switches_per_sec', 'retention')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
