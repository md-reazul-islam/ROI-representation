"""Option A: does smoothed, cross-faded fusion (alpha EMA + cross-fade, the D4 addition) help where regions switch?

Pairs (identical selection, only the fusion weights differ):
  Full (D0) vs F-Full = Full + fusion smoothing      (~10 set changes / s)
  D2        vs F-D2   = D2   + fusion smoothing      (hysteresis only)
  D3        vs D4     = D3   + fusion smoothing      (the configuration in the paper)
Metrics per video: weight jitter (mean L1 change of the fusion weights per second), median window pulse SNR,
informative-window share, and median SNR over TRANSITION windows (windows of the base configuration that contain
at least one change of the selected set; the same windows are used for both members of a pair).
Usage: python module1_v2/scripts/run_fusion_test.py --run runs/final34_paper --v1-report reports/paper_final34_paper
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
import run_v2_evaluation as ev                                   # noqa: E402
from module2.ablation import holm, paired_test                   # noqa: E402
from module2.rppg import analysis_windows                        # noqa: E402
from module1_v2 import selection as sel                          # noqa: E402

PAIRS = [('D0', 'F-Full'), ('D2', 'F-D2'), ('D3', 'D4')]


def configs(chosen):
    c = ev.configurations(chosen)
    p1 = ev.CFG['phase1']
    fusion = dict(alpha_ema=p1['alpha_ema'], crossfade_sec=p1['crossfade_sec'])
    return {'D0': c['D0'], 'F-Full': dict(c['D0'], **fusion), 'D2': c['D2'], 'F-D2': dict(c['D2'], **fusion),
            'D3': c['D3'], 'D4': c['D4']}


def dynamics(cand, frames, flags, fps_sampled, names):
    """Per sampled frame: fusion-weight vector and selected set."""
    geometry, fusion, _, _ = sel.select_video(cand, frames, flags, fps_sampled, 3, 1., None)
    col = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(frames), len(names)))
    sets = []
    for i, f in enumerate(frames):
        for key, weight, _, _ in fusion.get(f) or []:
            W[i, col[key]] += weight
        sets.append(frozenset(r[0] for r in geometry.get(f) or []))
    return W, sets


def job(args):
    base_job, cfg = args
    video, record, out, cand, frames, recovery, sam2_rows, subjects, _ = base_job
    per_video, windows, _, seconds = ev.evaluate_video((video, record, out, cand, frames, recovery, sam2_rows,
                                                        subjects.iloc[0:0], cfg))
    sel_frames = sorted(cand.frame_index.unique())
    fps_sampled = len(frames) / float(record['duration_sec'])
    n, fps = int(record['frame_count']), float(record['fps'])
    wins = analysis_windows(n, fps, ev.HR)
    extra = {}
    for cid, flags in cfg.items():
        W, sets = dynamics(cand, sel_frames, flags, fps_sampled, sel.NAMES)
        jitter = np.abs(np.diff(W, axis=0)).sum(1).mean() * fps_sampled if len(W) > 1 else np.nan
        changes = [sel_frames[i] for i in range(1, len(sets)) if sets[i] and sets[i - 1] and sets[i] != sets[i - 1]]
        extra[cid] = dict(jitter=jitter, transition=[any(a <= f < b for f in changes) for a, b in wins])
    rows = []
    for base, test in PAIRS:
        flags_t = extra[base]['transition']          # transition windows defined by the BASE configuration
        for cid in (base, test):
            snr = np.array([w[f'snr_{cid}'] if w[f'snr_{cid}'] is not None else np.nan for w in windows], float)
            tr = snr[np.array(flags_t, bool)] if len(snr) else snr
            rows.append(dict(video_id=video, split=record['split'], pair=f'{base} vs {test}', config=cid,
                             weight_jitter=extra[cid]['jitter'],
                             pulse_snr_db=float(np.nanmedian(snr)) if np.isfinite(snr).any() else np.nan,
                             transition_snr_db=float(np.nanmedian(tr)) if len(tr) and np.isfinite(tr).any() else np.nan,
                             transition_windows=int(np.sum(flags_t)), windows=len(wins), snr_windows=snr.tolist()))
    return rows, seconds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--v1-report', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-videos', type=int, default=0)
    a = ap.parse_args(argv)
    run, v1 = (ROOT/a.run).resolve(), (ROOT/a.v1_report).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    target = V2/'reports'/run.name/'fusion_test'
    target.mkdir(parents=True, exist_ok=True)
    cfg = configs(json.loads((work/'tuning_log.json').read_text())['chosen'])
    tau = json.loads((v1/'evaluation_metadata.json').read_text())['thresholds']['informative_snr_db']
    manifest = pd.read_csv(run/'manifest.csv')
    if a.max_videos:
        manifest = manifest.head(a.max_videos)
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False, usecols=ev.CAND_COLUMNS)
    feat = pd.read_csv(work/'features/roi_features_median.csv')
    cand = cand.merge(feat[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                            'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    subjects = pd.read_csv(work/'features/subject_frames.csv')
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'], low_memory=False)
    rec = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = []
    for r in manifest.itertuples():
        c = cand[cand.video_id == r.video_id]
        frames = sorted(set(frames0[frames0.video_id == r.video_id].frame_index) | set(c.frame_index))
        jobs.append(((r.video_id, r._asdict(), str(out), c, frames, rec[rec.video_id == r.video_id],
                      sam[sam.video_id == r.video_id], subjects[subjects.video_id == r.video_id], cfg), cfg))
    started = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(job, jobs))
    pv = pd.DataFrame([row for r in results for row in r[0]])
    pv['informative_window_percent'] = [100. * np.mean(np.nan_to_num(np.array(w, float), nan=-np.inf) >= tau)
                                        if len(w) else np.nan for w in pv.snr_windows]
    pv.drop(columns='snr_windows').to_csv(target/'per_video.csv', index=False)

    # regression: D0 / D2 / D3 / D4 must reproduce Phase 1 SNR
    p1 = pd.read_csv(V2/'reports'/run.name/'phase1/per_video_metrics.csv')
    reg = {}
    for cid in ('D0', 'D2', 'D3', 'D4'):
        a_ = pv[pv.config == cid].drop_duplicates('video_id').set_index('video_id').pulse_snr_db
        b_ = p1[p1.config == cid].set_index('video_id').pulse_snr_db.reindex(a_.index)
        reg[cid] = float(np.nanmax(np.abs(a_ - b_)))
    metrics = ['weight_jitter', 'pulse_snr_db', 'transition_snr_db', 'informative_window_percent']
    rows, tests = [], []
    for base, test in PAIRS:
        part = pv[pv.pair == f'{base} vs {test}']
        row = dict(pair=f'{base} vs {test}', videos=int(part.video_id.nunique()),
                   transition_window_share=100. * part[part.config == base].transition_windows.sum() /
                   max(1, part[part.config == base].windows.sum()))
        block = []
        for m in metrics:
            wide = part.pivot_table(index='video_id', columns='config', values=m)
            row[f'{m}_{base}'] = float(wide[base].mean())
            row[f'{m}_{test}'] = float(wide[test].mean())
            block.append(dict(pair=f'{base} vs {test}', metric=m, **paired_test(wide[base], wide[test])))
        for b, adj in zip(block, holm([b['p_value'] for b in block])):
            b['p_holm'] = adj
        tests += block
        rows.append(row)
    pd.DataFrame(rows).to_csv(target/'table_fusion_test.csv', index=False)
    pd.DataFrame(tests).to_csv(target/'paired_fusion_test.csv', index=False)
    (target/'fusion_test_metadata.json').write_text(json.dumps(dict(
        pairs=PAIRS, tau_db=tau, regression_vs_phase1_max_abs_snr_diff=reg, wall_time_sec=time.perf_counter() - started,
        fusion=dict(alpha_ema=ev.CFG['phase1']['alpha_ema'], crossfade_sec=ev.CFG['phase1']['crossfade_sec'])), indent=2))
    print('Regression vs Phase 1 (max |SNR diff|):', reg)
    print(pd.DataFrame(rows).round(3).T.to_string())
    print(pd.DataFrame(tests)[['pair', 'metric', 'n', 'mean_difference', 'rank_biserial', 'p_value', 'p_holm']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
