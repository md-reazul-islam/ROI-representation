"""Tracked face box (FACE_TRACK) baseline report.

    python face_track_baseline/scripts/face_track_report.py

A. UBFC robustness (PPG reference): MAE / no-HR of FACE_TRACK next to the fixed box (B0) and NeoSTAR (D5), and paired
   contrasts FACE_TRACK - fixed and D5 - FACE_TRACK (moving-block bootstrap, block 8, 5000, seed 42, within video).
   B0 is re-run in the same job and checked against the earlier B0 numbers (must be identical).
B. final34 (label-free, infant band): per-video coverage, pulse-present, validated, SNR, split-half agreement (mean over
   videos), and geometry of the face box with the paper's own metric (src/module1/paper_eval.geometry_metrics).
Read-only on existing outputs; writes face_track_baseline/report/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'module1_v2/scripts'))
from refmatch import nearest_reference  # noqa: E402

HRR = ROOT/'HR RR using module1 v2/runs'
REF = ROOT/'face_track_baseline/reference_hr_ppg.csv'
OUT = ROOT/'face_track_baseline/report'
BLOCK, N_BOOT = 8, 5000
RUNS = ['ubfc_flat_38d8f553', 'ubfc_aug', 'ubfc_aug_fast', 'ubfc_aug_fast_xl', 'ubfc_swaddle_test',
        'ubfc_swaddle_motion', 'ubfc_occdemo']
LEVEL = {'xlarge': 'extra-large'}


def condition(v):
    s = v.replace('NEW_', '')
    parts = s.split('_')
    subj = parts[0]
    if len(parts) == 1:
        return 'Undisturbed', 'none', subj
    rest = '_'.join(parts[1:])
    for key, name in (('motion_fast_', 'Fast, large motion'), ('motion_', 'Slow camera motion'), ('lighting_', 'Lighting change'),
                      ('occlusion_', 'Hand + blanket occlusion'), ('swaddle_', 'Swaddling only')):
        if rest.startswith(key):
            lev = rest[len(key):]
            return name, LEVEL.get(lev, lev), subj
    if rest.startswith('swmo_'):
        return 'Swaddling + fast motion', {'medium': 'to neck', 'large': 'to mouth'}[rest[5:]], subj
    if rest == 'democentre':
        return 'Hand over face centre', 'hand present', subj
    return rest, '', subj


def strat_ci(parts, rng):
    parts = [p for p in parts if len(p)]
    if sum(len(p) for p in parts) < 2 * BLOCK:
        return np.nan, np.nan
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        tot = cnt = 0
        for d in parts:
            n = len(d)
            idx = rng.integers(0, n, n) if n <= BLOCK else \
                (rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))[:, None] + np.arange(BLOCK)).ravel()[:n]
            tot += d[idx].sum()
            cnt += n
        means[b] = tot / cnt
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrast(g, a, b):
    h = g[g.hr_ref.notna() & g[f'hr_{a}'].notna() & g[f'hr_{b}'].notna()].sort_values(['video_id', 'window'])
    parts = [((x[f'hr_{a}'] - x.hr_ref).abs() - (x[f'hr_{b}'] - x.hr_ref).abs()).to_numpy(float) for _, x in h.groupby('video_id')]
    d = np.concatenate(parts) if parts else np.array([])
    lo, hi = strat_ci(parts, np.random.default_rng(42))
    return dict(contrast=f'{a} - {b}', windows=len(d), mean_diff_bpm=d.mean() if len(d) else np.nan, ci95_low=lo,
                ci95_high=hi, verdict=f'{a} better' if hi < 0 else f'{b} better' if lo > 0 else 'no clear difference')


def ubfc():
    frames, check = [], []
    for run in RUNS:
        new = pd.read_csv(HRR/run/'facetrack'/'window_results.csv')
        old = pd.read_csv(HRR/run/'aug_eval_D5'/'window_results.csv')
        m = new[['video_id', 'window', 'window_center_sec', 'hr_bpm_B0', 'label_B0', 'hr_bpm_FACE_TRACK', 'label_FACE_TRACK']].merge(
            old[['video_id', 'window', 'hr_bpm_B0', 'hr_bpm_D5', 'label_D5']].rename(columns={'hr_bpm_B0': 'hr_B0_old'}),
            on=['video_id', 'window'], how='left')
        diff = (m.hr_bpm_B0 - m.hr_B0_old).abs()
        check.append(dict(run=run, windows=len(m), b0_max_abs_diff=float(np.nanmax(diff)) if diff.notna().any() else 0.0,
                          b0_nan_mismatch=int((m.hr_bpm_B0.isna() != m.hr_B0_old.isna()).sum())))
        m['run'] = run
        frames.append(m)
    w = pd.concat(frames, ignore_index=True).rename(columns={'hr_bpm_B0': 'hr_fixed', 'hr_bpm_FACE_TRACK': 'hr_FACE',
                                                              'hr_bpm_D5': 'hr_D5', 'label_B0': 'lab_fixed',
                                                              'label_FACE_TRACK': 'lab_FACE', 'label_D5': 'lab_D5'})
    w['hr_ref'] = np.asarray(nearest_reference(w.video_id, w.window_center_sec, REF))
    cond = w.video_id.map(condition)
    w['condition'], w['level'], w['subject'] = [c[0] for c in cond], [c[1] for c in cond], [c[2] for c in cond]
    occ = pd.read_csv(ROOT/'occlusion_position_test/demo_centre/report/windows.csv').set_index('window').hand_visibility
    demo = w.condition == 'Hand over face centre'
    w = w[~demo | (w.window.map(occ) >= 0.5)]
    rows, pairs = [], []
    order = ['Undisturbed', 'Slow camera motion', 'Fast, large motion', 'Lighting change', 'Hand + blanket occlusion',
             'Swaddling only', 'Swaddling + fast motion', 'Hand over face centre']
    for (c, lev), g in w.groupby(['condition', 'level'], sort=False):
        r = dict(condition=c, level=lev, subjects=', '.join(sorted(g.subject.str.replace('subject', '').unique())), windows=len(g))
        for k in ('fixed', 'FACE', 'D5'):
            e = (g[f'hr_{k}'] - g.hr_ref).abs()
            r[f'mae_{k}'], r[f'nohr_{k}'] = e.mean(), int(g[f'hr_{k}'].isna().sum())
            r[f'validated_pct_{k}'] = 100 * g[f'lab_{k}'].isin(['validated', 'validated_one_side']).mean()
        rows.append(r)
        for a, b in (('FACE', 'fixed'), ('D5', 'FACE'), ('D5', 'fixed')):
            pairs.append(dict(condition=c, level=lev, **contrast(g, a, b)))
    key = lambda df: df.assign(_o=df.condition.map(order.index)).sort_values(['_o'], kind='stable').drop(columns='_o')  # noqa: E731
    return key(pd.DataFrame(rows)), key(pd.DataFrame(pairs)), pd.DataFrame(check), w


def final34():
    w = pd.read_csv(HRR/'final34_paper'/'facetrack'/'window_results.csv')
    rows = []
    for k in ('B0', 'FACE_TRACK', 'D5', 'BG'):
        g = w.groupby('video_id')
        per = pd.DataFrame({
            'coverage_pct': g[f'hr_bpm_{k}'].apply(lambda s: 100 * s.notna().mean()),
            'pulse_present_pct': g[f'pulse_present_{k}'].apply(lambda s: 100 * s.astype(bool).mean()),
            'validated_pct': g[f'label_{k}'].apply(lambda s: 100 * s.isin(['validated', 'validated_one_side']).mean()),
            'snr_median_db': g[f'snr_db_{k}'].median(),
            'agreement_pct': g[f'halves_agree_{k}'].apply(lambda s: 100 * s.astype(bool).mean()),
        })
        rows.append(dict(method=k, videos=len(per), **{c: per[c].mean() for c in per}))
    return pd.DataFrame(rows)


def face_geometry():
    from module1.paper_eval import geometry_metrics
    d = pd.read_csv(ROOT/'module1_v2/runs/final34_paper/dynamic_roi_v2_D5/dynamic_roi_v2.csv',
                    usecols=['video_id', 'frame_index', 'fps_sampled', 'roi_name', 'available', 'pixel_polygon', 'infant_box'])
    out = []
    for v, g in d.groupby('video_id'):
        frames = sorted(g.frame_index.unique())
        face = g[(g.roi_name == 'FACE') & g.available & g.pixel_polygon.notna()].set_index('frame_index').pixel_polygon
        sel = {f: ([('FACE', 1., json.loads(face[f]), np.nan)] if f in face.index else []) for f in frames}
        ib = g.drop_duplicates('frame_index').set_index('frame_index').infant_box.dropna()
        boxes = {}
        for f, b in ib.items():
            try:
                b = json.loads(b) if isinstance(b, str) else b
                boxes[f] = tuple(float(x) for x in b[:4])
            except (ValueError, TypeError):
                pass
        m = geometry_metrics(sel, boxes, float(g.fps_sampled.iloc[0]))
        out.append(dict(video_id=v, **{k: m[k] for k in ('coverage_percent', 'temporal_iou', 'centroid_jitter')}))
    face = pd.DataFrame(out)
    pv = pd.read_csv(ROOT/'module1_v2/reports/final34_paper/phase1/per_video_metrics.csv')
    rows = [dict(method='FACE_TRACK (face box)', videos=len(face), **face[['coverage_percent', 'temporal_iou', 'centroid_jitter']].mean())]
    for c in ('B0', 'Full', 'D5'):
        p = pv[pv.config == c]
        rows.append(dict(method=c, videos=len(p), **p[['coverage_percent', 'temporal_iou', 'centroid_jitter']].mean()))
    return pd.DataFrame(rows), face


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    met, pair, check, w = ubfc()
    met.to_csv(OUT/'ubfc_metrics.csv', index=False)
    pair.to_csv(OUT/'ubfc_paired.csv', index=False)
    check.to_csv(OUT/'b0_consistency.csv', index=False)
    w.to_csv(OUT/'ubfc_windows.csv', index=False)
    pd.set_option('display.width', 250)
    print(check.to_string(index=False))
    print(met[['condition', 'level', 'windows', 'mae_fixed', 'nohr_fixed', 'mae_FACE', 'nohr_FACE', 'mae_D5', 'nohr_D5']].round(2).to_string(index=False))
    print(pair.round(2).to_string(index=False))
    try:
        f34 = final34()
        f34.to_csv(OUT/'final34_labelfree.csv', index=False)
        print(f34.round(2).to_string(index=False))
    except FileNotFoundError as e:
        print('final34 not available yet:', e)
    geo, per = face_geometry()
    geo.to_csv(OUT/'final34_geometry.csv', index=False)
    per.to_csv(OUT/'final34_geometry_face_per_video.csv', index=False)
    print(geo.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
