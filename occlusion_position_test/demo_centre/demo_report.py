"""Report for the centre-occlusion demonstration video (designed case, not pre-registered).

    python occlusion_position_test/demo_centre/demo_report.py

Occluded windows = mean hand visibility >= 0.5 (same definition as the pre-registered test).
Writes demo_centre/report/: metrics.csv, paired.csv, d5_regions.csv, windows.csv, fig_timeline.png, demo_report.html.
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'scripts'))
import occluder as oc  # noqa: E402
ROOT = oc.ROOT
sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
from refmatch import nearest_reference  # noqa: E402

VID, STEM, RUN = 'NEW_subject1_democentre', 'subject1_democentre', 'ubfc_occdemo'
M2 = ROOT/'HR RR using module1 v2/runs'
OUT = HERE/'report'
METHODS = {'fixed': ('Fixed box', '#2a78d6', 's', '--'), 'v1': ('Module 1 v1', '#eb6834', '^', ':'),
           'D5': ('Module 1 v2 D5', '#1baf7a', 'o', '-'), 'D8': ('D8', '#7a4fd6', 'D', '-')}
BLOCK, N_BOOT = 8, 5000


def windows(run, vid):
    d = {k: M2/run/f'aug_eval_{k}'/'window_results_with_conditions.csv' for k in ('D5', 'D0', 'D8')}
    a = pd.read_csv(d['D5'])
    a = a[a.video_id == vid]
    w = a[['window', 'window_start_sec', 'window_end_sec', 'window_center_sec', 'hr_bpm_B0', 'hr_bpm_D5']].rename(
        columns={'hr_bpm_B0': 'hr_fixed', 'hr_bpm_D5': 'hr_D5'})
    for k, key in (('D0', 'v1'), ('D8', 'D8')):
        b = pd.read_csv(d[k])
        w = w.merge(b[b.video_id == vid][['window', 'hr_bpm_D5']].rename(columns={'hr_bpm_D5': f'hr_{key}'}), on='window', how='left')
    w['hr_ref'] = nearest_reference(pd.Series([vid] * len(w), index=w.index), w.window_center_sec, HERE/'reference_hr_ppg.csv')
    return w


def block_ci(d, rng):
    n = len(d)
    if n < 2 * BLOCK:
        return np.nan, np.nan
    k = int(np.ceil(n / BLOCK))
    m = [d[(rng.integers(0, n - BLOCK + 1, k)[:, None] + np.arange(BLOCK)).ravel()[:n]].mean() for _ in range(N_BOOT)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def d5_regions(fr, meta):
    t = pd.read_csv(ROOT/'module1_v2/runs'/RUN/'dynamic_roi_v2_D5/dynamic_roi_v2.csv',
                    usecols=['frame_index', 'timestamp_sec', 'roi_name', 'weight', 'pixel_polygon'], low_memory=False)
    t = t[t.weight > 0]
    oc.SIZES['demo'] = meta['palm_width_face']
    oc.POSITIONS['demo'] = tuple(meta['position_rule'])
    hand = oc.Hand(meta['face_box'], 'demo', 'demo', 640, 480, 1)
    fr = fr.set_index('frame')
    rows = []
    for f, g in t.groupby('frame_index'):
        if f not in fr.index:
            continue
        r = fr.loc[f]
        state = 'hand on' if r.visibility >= .99 else 'clean' if r.visibility == 0 else None
        if state is None:
            continue
        a = hand.mask(r.hand_x, r.hand_y, r.hand_angle_deg) if state == 'hand on' else None
        for _, s in g.iterrows():
            cov = 0.
            poly = json.loads(s.pixel_polygon)
            if a is not None and poly:
                m = np.zeros(a.shape, np.uint8)
                cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
                cov = float(a[m > 0].mean()) if m.any() else 0.
            rows.append(dict(state=state, frame=f, roi=s.roi_name, weight=s.weight, hand_cover=cov))
    d = pd.DataFrame(rows)
    share = d.groupby(['state', 'roi']).weight.sum() / d.groupby('state').weight.sum()
    on = d[d.state == 'hand on']
    covered_pct = 100 * on.weight[on.hand_cover >= .5].sum() / on.weight.sum() if len(on) else np.nan
    return (100 * share).round(1).rename('weight_share_pct').reset_index(), covered_pct


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((HERE/f'{STEM}.json').read_text())
    fr = pd.read_csv(HERE/f'{STEM}_frames.csv')
    w = windows(RUN, VID)
    w['hand_visibility'] = [fr.visibility[(fr.time_sec >= s) & (fr.time_sec < e)].mean() for s, e in zip(w.window_start_sec, w.window_end_sec)]
    w['set'] = np.where(w.hand_visibility >= .5, 'occluded', np.where(w.hand_visibility == 0, 'clean', 'partial'))
    w.to_csv(OUT/'windows.csv', index=False)
    orig = windows('ubfc_flat_38d8f553', 'NEW_subject1')
    rows = []
    for name, g in (('occluded', w[w.set == 'occluded']), ('all', w), ('original video (no hand)', orig)):
        for m, (lab, *_) in METHODS.items():
            e = (g[f'hr_{m}'] - g.hr_ref).abs()
            rows.append(dict(windows_set=name, method=lab, windows=len(g), no_hr=int(e.isna().sum()),
                             mae=round(e.mean(), 2), within5_pct=round(100 * (e <= 5).mean(), 1)))
    met = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    prs = []
    for name, g in (('occluded', w[w.set == 'occluded']), ('all', w)):
        for a, b in (('D5', 'fixed'), ('v1', 'fixed'), ('D8', 'fixed'), ('D5', 'v1')):
            h = g[g[f'hr_{a}'].notna() & g[f'hr_{b}'].notna()].sort_values('window')
            d = ((h[f'hr_{a}'] - h.hr_ref).abs() - (h[f'hr_{b}'] - h.hr_ref).abs()).to_numpy(float)
            lo, hi = block_ci(d, rng)
            prs.append(dict(windows_set=name, contrast=f'{a} - {b}', windows=len(d), mean_diff_bpm=round(d.mean(), 2) if len(d) else np.nan,
                            ci95_low=round(lo, 2), ci95_high=round(hi, 2),
                            verdict=f'{a} better' if hi < 0 else f'{b} better' if lo > 0 else 'no clear difference'))
    pair = pd.DataFrame(prs)
    regions, covered_pct = d5_regions(fr, meta)
    for n, t in (('metrics', met), ('paired', pair), ('d5_regions', regions)):
        t.to_csv(OUT/f'{n}.csv', index=False)
    # figure
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for s, e in meta['episodes_sec']:
        ax.axvspan(s, e, color='#f2c9a0', alpha=.35, lw=0)
    ax.plot(w.window_center_sec, w.hr_ref, color='k', lw=2.2, label='Reference HR (PPG)')
    for m, (lab, col, mk, ls) in METHODS.items():
        g = w[w.set == 'occluded']
        mae = (g[f'hr_{m}'] - g.hr_ref).abs().mean()
        ax.plot(w.window_center_sec, w[f'hr_{m}'], color=col, ls=ls, marker=mk, ms=4, lw=1.4,
                label=f'{lab} (MAE occluded {mae:.1f})')
    ax.set_xlabel('Time (s)  — shaded: hand over nose / centre / mid-cheeks')
    ax.set_ylabel('HR (bpm)')
    ax.set_title('UBFC subject 1, designed centre occlusion (demonstration, not pre-registered)', fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.18))
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT/'fig_timeline.png', dpi=150)
    plt.close(fig)
    pd.set_option('display.width', 200)
    print(met.to_string(index=False))
    print(pair.to_string(index=False))
    print('D5 weight on ROIs >= 50 % covered by the hand (hand on):', round(covered_pct, 1), '%')
    print(regions.to_string(index=False))
    img = base64.b64encode((OUT/'fig_timeline.png').read_bytes()).decode()
    prev = base64.b64encode((HERE/f'preview_{STEM}.png').read_bytes()).decode()
    (OUT/'demo_report.html').write_text(f"""<!doctype html><meta charset="utf-8"><title>Centre occlusion demo</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:3px 6px}}</style>
<h1>Centre-occlusion demonstration (designed case, not pre-registered)</h1>
<p>Hand coverage while present: {meta['coverage_while_present']}</p>
<p><img style="max-width:100%" src="data:image/png;base64,{prev}"></p><p><img style="max-width:100%" src="data:image/png;base64,{img}"></p>
<h2>MAE</h2>{met.to_html(index=False)}<h2>Paired</h2>{pair.to_html(index=False)}
<p>D5 weight on ROIs &ge; 50 % covered by the hand (hand on): {covered_pct:.1f} %</p><h2>D5 weight by region</h2>{regions.to_html(index=False)}""")
    print('report:', OUT/'demo_report.html')


if __name__ == '__main__':
    main()
