"""Occlusion-position test report with the pre-registered claim rules applied automatically (../PREREGISTRATION.md).

    python occlusion_position_test/scripts/occ_report.py

Reads (produced by run_occ_eval.sh with the unchanged comparison tool, strict gate 2, band 45-180 bpm):
  HR RR using module1 v2/runs/ubfc_occpos/aug_eval_D5   -> fixed box (B0), D5, BG
  HR RR using module1 v2/runs/ubfc_occpos/aug_eval_D0   -> v1 (method slot "D5" carries D0)
  HR RR using module1 v2/runs/ubfc_occpos/aug_eval_D8   -> D8 (secondary)
Writes occlusion_position_test/report/: occ_report.html, metrics.csv, per_subject.csv, paired.csv, claims.csv,
coverage.csv, d5_weight_on_hand.csv, windows.csv, figures.
"""
import base64
import datetime as dt
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import occluder as oc  # noqa: E402

ROOT = oc.ROOT
TEST = HERE.parent
sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
from refmatch import nearest_reference  # noqa: E402

RUN = 'ubfc_occpos'
M2 = ROOT/'HR RR using module1 v2/runs'/RUN
REF = TEST/'reference_hr_ppg.csv'
DATA = TEST/'data'
OUT = TEST/'report'
POS = list(oc.POSITIONS)
SIZES = list(oc.SIZES)
METHODS = {'fixed': ('Fixed box', '#2a78d6', 's'), 'v1': ('Module 1 v1', '#eb6834', '^'),
           'D5': ('Module 1 v2 D5', '#1baf7a', 'o'), 'D8': ('D8 (secondary)', '#7a4fd6', 'D')}
CONTRASTS = [('D5', 'fixed'), ('v1', 'fixed'), ('D8', 'fixed'), ('D5', 'v1')]
BLOCK, N_BOOT, OCC_VIS = 8, 5000, 0.5


def load():
    d = {k: M2/f'aug_eval_{k}'/'window_results_with_conditions.csv' for k in ('D5', 'D0', 'D8')}
    a = pd.read_csv(d['D5'])
    w = a[['video_id', 'window', 'window_start_sec', 'window_end_sec', 'window_center_sec', 'hr_bpm_B0', 'hr_bpm_D5']].rename(
        columns={'hr_bpm_B0': 'hr_fixed', 'hr_bpm_D5': 'hr_D5'})
    for k, key in (('D0', 'v1'), ('D8', 'D8')):
        if d[k].is_file():
            w = w.merge(pd.read_csv(d[k])[['video_id', 'window', 'hr_bpm_D5']].rename(columns={'hr_bpm_D5': f'hr_{key}'}),
                        on=['video_id', 'window'], how='left')
        else:
            w[f'hr_{key}'] = np.nan
    w['hr_ref'] = nearest_reference(w.video_id, w.window_center_sec, REF)
    tag = w.video_id.str.replace('NEW_subject', '', regex=False).str.split('_occpos_', expand=True)
    w['subject'] = tag[0].astype(int)
    w['position'] = tag[1].str.rsplit('_', n=1).str[0]
    w['size'] = tag[1].str.rsplit('_', n=1).str[1]
    vis = []
    for v, g in w.groupby('video_id', sort=False):
        fr = pd.read_csv(DATA/f"{v.replace('NEW_', '')}_frames.csv")
        for i, r in g.iterrows():
            m = (fr.time_sec >= r.window_start_sec) & (fr.time_sec < r.window_end_sec)
            vis.append((i, float(fr.visibility[m].mean()) if m.any() else np.nan))
    w['hand_visibility'] = pd.Series(dict(vis))
    w['occluded'] = w.hand_visibility >= OCC_VIS
    return w


def strat_ci(parts, rng):
    """Moving-block bootstrap within each video, pooled mean."""
    parts = [p for p in parts if len(p)]
    if not parts or sum(len(p) for p in parts) < 2 * BLOCK:
        return np.nan, np.nan
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        tot, cnt = 0., 0
        for d in parts:
            n = len(d)
            if n <= BLOCK:
                idx = rng.integers(0, n, n)
            else:
                k = int(np.ceil(n / BLOCK))
                idx = (rng.integers(0, n - BLOCK + 1, k)[:, None] + np.arange(BLOCK)).ravel()[:n]
            tot += d[idx].sum()
            cnt += n
        means[b] = tot / cnt
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics(w):
    rows, subj = [], []
    for occ_name, sub in (('occluded', w[w.occluded]), ('all', w)):
        for (p, s), g in sub.groupby(['position', 'size']):
            g = g[g.hr_ref.notna()]
            for m, (name, *_) in METHODS.items():
                err = (g[f'hr_{m}'] - g.hr_ref).abs()
                rows.append(dict(windows_set=occ_name, position=p, size=s, method=name, method_key=m, windows=len(g),
                                 no_hr=int(err.isna().sum()), mae=err.mean(), within5_pct=100 * (err <= 5).mean()))
                if occ_name == 'occluded':
                    for k, gg in g.groupby('subject'):
                        e = (gg[f'hr_{m}'] - gg.hr_ref).abs()
                        subj.append(dict(position=p, size=s, subject=k, method_key=m, windows=len(gg),
                                         no_hr=int(e.isna().sum()), mae=e.mean()))
    return pd.DataFrame(rows), pd.DataFrame(subj)


def paired(w):
    rng = np.random.default_rng(42)
    rows = []
    for occ_name, sub in (('occluded', w[w.occluded]), ('all', w)):
        for p in POS:
            for s in SIZES:
                g = sub[(sub.position == p) & (sub['size'] == s) & sub.hr_ref.notna()]
                for a, b in CONTRASTS:
                    h = g[g[f'hr_{a}'].notna() & g[f'hr_{b}'].notna()].sort_values(['video_id', 'window'])
                    parts = [((x[f'hr_{a}'] - x.hr_ref).abs() - (x[f'hr_{b}'] - x.hr_ref).abs()).to_numpy(float)
                             for _, x in h.groupby('video_id')]
                    d = np.concatenate(parts) if parts else np.array([])
                    lo, hi = strat_ci(parts, rng)
                    rows.append(dict(windows_set=occ_name, position=p, size=s, contrast=f'{a} - {b}', windows=len(d),
                                     mean_diff_bpm=d.mean() if len(d) else np.nan, ci95_low=lo, ci95_high=hi,
                                     verdict=f'{a} better' if hi < 0 else f'{b} better' if lo > 0 else 'no clear difference'))
    return pd.DataFrame(rows)


def claims(pair, subj):
    def row(p, s):
        r = pair[(pair.windows_set == 'occluded') & (pair.position == p) & (pair['size'] == s) & (pair.contrast == 'D5 - fixed')]
        return r.iloc[0] if len(r) else None

    def subjects_won(p, s):
        t = subj[(subj.position == p) & (subj['size'] == s)].pivot(index='subject', columns='method_key', values='mae')
        return int((t['D5'] < t['fixed']).sum()) if {'D5', 'fixed'} <= set(t.columns) else 0

    out = []
    for cid, s, label in (('C1 (primary)', 'large', 'large'), ('C2 (secondary)', 'small', 'small')):
        r, k = row('centre', s), subjects_won('centre', s)
        ok = r is not None and r.ci95_high < 0 and k >= 2
        out.append(dict(claim=cid, statement=f'D5 recovers HR when the hand covers the centre of the face ({label})',
                        rule='D5 - fixed CI upper < 0 (occluded windows) and D5 better in >= 2 of 3 subjects',
                        result='SUPPORTED' if ok else 'NOT supported',
                        detail=f'diff {r.mean_diff_bpm:+.2f} bpm, CI [{r.ci95_low:.2f}, {r.ci95_high:.2f}], subjects {k}/3'
                        if r is not None else 'no data'))
    c1 = out[0]['result'] == 'SUPPORTED'
    others = [p for p in POS if p != 'centre' and (r := row(p, 'large')) is not None and not r.ci95_high < 0]
    out.append(dict(claim='C3', statement='The dynamic-vs-fixed effect of occlusion depends on location',
                    rule='C1 supported and >= 1 other position (large) with CI including 0 or favouring the fixed box',
                    result='SUPPORTED' if c1 and others else 'NOT supported',
                    detail='positions without a D5 advantage (large): ' + (', '.join(others) or 'none')))
    for p in POS:
        for s in SIZES:
            if p == 'centre':
                continue
            r = row(p, s)
            out.append(dict(claim='exploratory', statement=f'{p} {s}', rule='D5 - fixed, occluded windows (10 cells: not a claim alone)',
                            result=r.verdict if r is not None else 'no data',
                            detail=f'diff {r.mean_diff_bpm:+.2f} bpm, CI [{r.ci95_low:.2f}, {r.ci95_high:.2f}], subjects {subjects_won(p, s)}/3'
                            if r is not None else ''))
    return pd.DataFrame(out)


def coverage():
    """Hand coverage while present: face regions (generation) and B0 re-measured on the new run's person box."""
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
    from run_paper_evaluation import boxes_of
    rec = pd.read_csv(ROOT/'runs'/RUN/'outputs/recovery/recovery_results.csv', low_memory=False)
    rows = []
    for j in sorted(DATA.glob('subject*_occpos_*.json')):
        meta = json.loads(j.read_text())
        vid = 'NEW_' + j.stem
        boxes = boxes_of(rec[rec.video_id == vid])
        first = next((boxes[f] for f in sorted(boxes) if boxes[f] is not None), None)
        fr = pd.read_csv(DATA/f'{j.stem}_frames.csv')
        r = dict(video_id=vid, subject=meta['subject'], position=meta['position'], size=meta['size'],
                 **{f'cov_{k}': v for k, v in meta['coverage_while_present'].items()})
        if first is not None:
            x1, y1, x2, y2 = first
            dx, dy = (x2 - x1) / 4, (y2 - y1) / 4
            b0 = (x1 + dx, y1 + dy, x2 - dx, y2 - dy)
            hand = oc.Hand(meta['face_box'], meta['size'], meta['position'], 640, 480, meta['subject'])
            on = fr[fr.visibility >= .99].iloc[::15]
            cv = [oc.covered(hand.mask(x, y, a), b0) for x, y, a in zip(on.hand_x, on.hand_y, on.hand_angle_deg)]
            r['cov_fixed_box_run'] = round(float(np.mean(cv)), 3) if cv else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def d5_weight_on_hand():
    """Share of D5 fusion weight on ROIs >= 50 % covered by the hand, in fully occluded frames (sampled)."""
    import cv2
    t = pd.read_csv(ROOT/'module1_v2/runs'/RUN/'dynamic_roi_v2_D5/dynamic_roi_v2.csv',
                    usecols=['video_id', 'frame_index', 'roi_name', 'weight', 'pixel_polygon'], low_memory=False)
    t = t[t.weight > 0]
    rows = []
    for vid, g in t.groupby('video_id'):
        stem = vid.replace('NEW_', '')
        meta = json.loads((DATA/f'{stem}.json').read_text())
        fr = pd.read_csv(DATA/f'{stem}_frames.csv').set_index('frame')
        hand = oc.Hand(meta['face_box'], meta['size'], meta['position'], 640, 480, meta['subject'])
        frames = [f for f in sorted(g.frame_index.unique()) if f in fr.index and fr.visibility[f] >= .99][::10]
        tot = on = 0.
        for f in frames:
            r = fr.loc[f]
            a = hand.mask(r.hand_x, r.hand_y, r.hand_angle_deg)
            for _, s in g[g.frame_index == f].iterrows():
                poly = json.loads(s.pixel_polygon)
                if not poly:
                    continue
                m = np.zeros(a.shape, np.uint8)
                cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
                cov = float(a[m > 0].mean()) if m.any() else 0.
                tot += s.weight
                on += s.weight * (cov >= .5)
        rows.append(dict(video_id=vid, subject=meta['subject'], position=meta['position'], size=meta['size'],
                         frames_sampled=len(frames), d5_weight_on_covered_roi_pct=100 * on / tot if tot else np.nan))
    return pd.DataFrame(rows)


def fig_cells(met):
    m = met[met.windows_set == 'occluded']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(POS))
    for ax, s in zip(axes, SIZES):
        for j, (k, (name, colour, marker)) in enumerate(METHODS.items()):
            y = [m[(m.position == p) & (m['size'] == s) & (m.method_key == k)].mae.mean() for p in POS]
            ax.plot(x + (j - 1.5) * .08, y, marker=marker, color=colour, lw=0 if k == 'D8' else 1.5,
                    ms=6, label=name, alpha=.6 if k == 'D8' else 1)
        ax.set_xticks(x, [p.replace('_', ' ') for p in POS])
        ax.set_title(f'Hand size: {s}')
        ax.grid(axis='y', alpha=.3)
    axes[0].set_ylabel('MAE on occluded windows (bpm)')
    axes[0].legend(frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.2))
    fig.tight_layout()
    p = OUT/'fig_mae_by_position.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_forest(pair):
    g = pair[(pair.windows_set == 'occluded') & (pair.contrast == 'D5 - fixed')].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    y = np.arange(len(g))[::-1]
    for yi, (_, r) in zip(y, g.iterrows()):
        c = '#1baf7a' if r.ci95_high < 0 else '#2a78d6' if r.ci95_low > 0 else '#888'
        ax.plot([r.ci95_low, r.ci95_high], [yi, yi], color=c, lw=2)
        ax.plot(r.mean_diff_bpm, yi, 'o', color=c)
    ax.axvline(0, color='k', lw=.8)
    ax.set_yticks(y, [f"{r.position.replace('_', ' ')} · {r['size']}" + ('  (primary)' if r.position == 'centre' and r['size'] == 'large' else '')
                      for _, r in g.iterrows()])
    ax.set_xlabel('D5 − fixed box, MAE difference (bpm)  ← D5 better | fixed better →')
    ax.set_title('Occluded windows, pooled over subjects 1, 3, 4 (95 % block-bootstrap CI)', fontsize=10)
    fig.tight_layout()
    p = OUT/'fig_forest_d5_vs_fixed.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    OUT.mkdir(exist_ok=True)
    w = load()
    w.to_csv(OUT/'windows.csv', index=False)
    met, subj = metrics(w)
    pair = paired(w)
    cl = claims(pair, subj)
    cov = coverage()
    wt = d5_weight_on_hand()
    for name, t in (('metrics', met), ('per_subject', subj), ('paired', pair), ('claims', cl), ('coverage', cov),
                    ('d5_weight_on_hand', wt)):
        t.to_csv(OUT/f'{name}.csv', index=False)
    figs = [fig_forest(pair), fig_cells(met)]
    pd.set_option('display.width', 250)
    print(cl.to_string(index=False))
    tab = met[met.windows_set == 'occluded'].pivot_table(index=['position', 'size'], columns='method', values='mae').round(2)
    print(tab.to_string())
    covt = cov.groupby(['position', 'size'])[[c for c in cov.columns if c.startswith('cov_')]].mean().round(2)
    print(covt.to_string())
    print(wt.groupby(['position', 'size']).d5_weight_on_covered_roi_pct.mean().round(1).to_string())
    img = ''.join(f'<p><img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"></p>'
                  for p in figs)
    page = f"""<!doctype html><meta charset="utf-8"><title>Occlusion position test</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:3px 6px}}</style>
<h1>Occlusion-position test</h1><p>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Pre-registration: {html.escape(str(TEST/'PREREGISTRATION.md'))}.
Occluded windows = mean hand visibility &ge; {OCC_VIS}. Strict gate 2, band 45&ndash;180 bpm, reference = fingertip PPG.</p>
<h2>Pre-registered claims</h2>{cl.to_html(index=False)}
{img}
<h2>MAE on occluded windows (bpm)</h2>{tab.to_html()}
<h2>Hand coverage while present (share of region)</h2>{covt.to_html()}
<h2>D5 fusion weight on ROIs &ge; 50 % covered by the hand (%)</h2>{wt.to_html(index=False)}
<h2>All paired contrasts</h2>{pair.round(3).to_html(index=False)}"""
    (OUT/'occ_report.html').write_text(page)
    print('report:', OUT/'occ_report.html')


if __name__ == '__main__':
    main()
