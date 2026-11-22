"""Paper-style figure for the centre-occlusion demonstration (same style as dataaugment/hr_eval fig_v2_vs_fixed_paper).

    python occlusion_position_test/demo_centre/make_paper_figure.py

(a) MAE vs PPG, (b) validated windows, for: original video / demo all windows / demo hand-present windows;
(c) timeline of the demo video (hand periods shaded); (d) frames: no hand vs hand, fixed box (blue) and the regions
Module 1 v2 D5 uses at that moment (green). Writes demo_centre/report/fig_occlusion_v2_vs_fixed_paper.png.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'dataaugment/hr_eval'), str(ROOT/'src'), str(ROOT/'scripts')]
from refmatch import nearest_reference  # noqa: E402
from run_paper_evaluation import boxes_of  # noqa: E402

M2 = ROOT/'HR RR using module1 v2/runs'
OUT = HERE/'report'
STEM = 'subject1_democentre'
OK = {'validated', 'validated_one_side'}
INK, INK2, GRID, SURFACE, MISS, HAND = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#c62828', '#f2c9a0'
METHODS = {'fixed': ('Fixed box', '#2a78d6', 's', '--'), 'D5': ('Module 1 v2 (dynamic ROI)', '#1baf7a', 'o', '-')}
YLIM = (45, 125)


def load(run, vid, ref):
    a = pd.read_csv(M2/run/'aug_eval_D5'/'window_results_with_conditions.csv')
    a = a[a.video_id == vid].sort_values('window_center_sec').reset_index(drop=True)
    d = a[['window', 'window_start_sec', 'window_end_sec', 'window_center_sec']].copy()
    d['hr_ref'] = np.asarray(nearest_reference(a.video_id, a.window_center_sec, ref))
    for key, col in (('fixed', 'B0'), ('D5', 'D5')):
        d[f'hr_{key}'], d[f'lab_{key}'] = a[f'hr_bpm_{col}'], a[f'label_{col}']
    return d


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.tick_params(labelsize=8.5, colors=INK2)


def block_ci(d, seed=42, block=8, n_boot=5000):
    rng = np.random.default_rng(seed)
    n, k = len(d), int(np.ceil(len(d) / block))
    m = [d[(rng.integers(0, n - block + 1, k)[:, None] + np.arange(block)).ravel()[:n]].mean() for _ in range(n_boot)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    meta = json.loads((HERE/f'{STEM}.json').read_text())
    fr = pd.read_csv(HERE/f'{STEM}_frames.csv')
    demo = load('ubfc_occdemo', 'NEW_subject1_democentre', HERE/'reference_hr_ppg.csv')
    demo['vis'] = [fr.visibility[(fr.time_sec >= s) & (fr.time_sec < e)].mean()
                   for s, e in zip(demo.window_start_sec, demo.window_end_sec)]
    orig = load('ubfc_flat_38d8f553', 'NEW_subject1', ROOT/'dataaugment/hr_eval/reference_hr_ppg.csv')
    present = demo[demo.vis >= .5]
    sets = {'original video\n(no hand)': orig, 'hand video\n(all windows)': demo, 'hand video\n(hand present)': present}
    h = present[present.hr_fixed.notna() & present.hr_D5.notna()]
    diff = ((h.hr_D5 - h.hr_ref).abs() - (h.hr_fixed - h.hr_ref).abs()).to_numpy(float)
    lo, hi = block_ci(diff)

    fig = plt.figure(figsize=(11, 11.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.05, 1.15], hspace=.55, wspace=.22, top=.88)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    x = np.arange(len(sets))
    for key, (name, col, mk, ls) in METHODS.items():
        mae = [(d[f'hr_{key}'] - d.hr_ref).abs().mean() for d in sets.values()]
        val = [100 * (d[f'lab_{key}'].isin(OK) & d[f'hr_{key}'].notna()).mean() for d in sets.values()]
        kw = dict(color=col, lw=2.2, marker=mk, ms=8, mec=SURFACE, mew=1.2, ls=ls, label=name)
        a.plot(x, mae, **kw)
        b.plot(x, val, **kw)
        for xi, v in zip(x, mae):
            a.annotate(f'{v:.1f}', (xi, v), xytext=(0, 9) if key == 'fixed' else ((0, -15) if v > 3.5 else (0, 9) if xi > 0 else (16, -4)), textcoords='offset points',
                       ha='center', fontsize=8.5, color=INK2)
    a.text(.40, .47, f'hand present: v2 − fixed box = {diff.mean():+.1f} bpm\n95 % CI [{lo:+.1f}, {hi:+.1f}] (block bootstrap)',
           transform=a.transAxes, fontsize=8, color=INK, va='top')
    for ax, yl, t in ((a, 'MAE vs PPG (bpm)', 'HR error (lower is better)'),
                      (b, 'validated windows (%)', 'HR the system reports (higher is better)')):
        style(ax)
        ax.set_xticks(x, list(sets), fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5, color=INK2)
        ax.set_title(t, loc='left', fontsize=10, color=INK)
        ax.set_xlim(-.4, len(sets) - .6)
    a.set_ylim(0, None)
    b.set_ylim(-3, 105)
    a.legend(fontsize=8.5, frameon=False, loc='upper left')

    # (c) timeline
    c = fig.add_subplot(gs[1, :])
    style(c)
    for s, e in meta['episodes_sec']:
        c.axvspan(s, e, color=HAND, alpha=.35, lw=0, zorder=0)
    t = demo.window_center_sec
    c.plot(t, demo.hr_ref, color=INK, lw=2.4, zorder=4)
    txt = []
    for key, (name, col, mk, ls) in METHODS.items():
        hr = demo[f'hr_{key}']
        valid = demo[f'lab_{key}'].isin(OK) & hr.notna()
        y = hr.clip(YLIM[0] + .5, YLIM[1] - .5)
        c.plot(t, y, color=col, lw=1.1, alpha=.5, ls=ls, zorder=2)
        c.scatter(t[valid], y[valid], marker=mk, s=36, color=col, edgecolor=SURFACE, lw=.8, zorder=5)
        c.scatter(t[~valid & hr.notna()], y[~valid & hr.notna()], marker=mk, s=32, facecolor='none', edgecolor=col, lw=1.3, zorder=5)
        miss = hr.isna()
        if miss.any():
            c.scatter(t[miss], np.full(miss.sum(), YLIM[0] + 1.5), marker='|', s=70, color=MISS, lw=1.6, zorder=6)
        txt.append(f'{"Fixed box" if key == "fixed" else "Module 1 v2"} {(present[f"hr_{key}"] - present.hr_ref).abs().mean():.1f}')
    c.set_ylim(*YLIM)
    c.set_ylabel('HR (bpm)', fontsize=8.5, color=INK2)
    c.set_xlabel('time (s), centre of each 8-s window   (shaded = hand over nose / face centre / mid-cheeks)', fontsize=8.5, color=INK2)
    c.set_title('Hand over the face centre   |   MAE on hand-present windows (bpm): ' + ',  '.join(txt), loc='left', fontsize=9.5, color=INK)

    # (d) frames
    cap = cv2.VideoCapture(str(HERE/f'{STEM}.mkv'))
    s0, e0 = meta['episodes_sec'][1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    roi = pd.read_csv(ROOT/'module1_v2/runs/ubfc_occdemo/dynamic_roi_v2_D5/dynamic_roi_v2.csv',
                      usecols=['frame_index', 'roi_name', 'weight', 'pixel_polygon'], low_memory=False)
    roi = roi[roi.weight > 0]
    rec = pd.read_csv(ROOT/'runs/ubfc_occdemo/outputs/recovery/recovery_results.csv', low_memory=False)
    boxes = boxes_of(rec)
    x1, y1, x2, y2 = next(boxes[f] for f in sorted(boxes) if boxes[f] is not None)
    dx, dy = (x2 - x1) / 4, (y2 - y1) / 4
    b0 = (int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy))
    frames = roi.frame_index.unique()
    tiles = []
    for want, label in ((int(5 * fps), 'no hand (t = 5 s)'), (int((s0 + e0) / 2 * fps), f'hand present (t = {(s0 + e0) / 2:.0f} s)')):
        f = int(frames[np.abs(frames - want).argmin()])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        over = img.copy()
        for _, r in roi[roi.frame_index == f].iterrows():
            p = np.round(np.asarray(json.loads(r.pixel_polygon))).astype(np.int32)
            cv2.fillPoly(over, [p], (27, 175, 122))
        img = cv2.addWeighted(over, .35, img, .65, 0)
        for _, r in roi[roi.frame_index == f].iterrows():
            p = np.round(np.asarray(json.loads(r.pixel_polygon))).astype(np.int32)
            cv2.polylines(img, [p], True, (27, 175, 122), 2)
        cv2.rectangle(img, b0[:2], b0[2:], (42, 120, 214), 3)
        tiles.append((img[60:440, 120:520], label, ', '.join(sorted(roi[roi.frame_index == f].roi_name.str.lower().str.replace('_', ' ')))))
    gap = np.full((tiles[0][0].shape[0], 24, 3), 252, np.uint8)
    d_ax = fig.add_subplot(gs[2, :])
    d_ax.imshow(np.hstack([tiles[0][0], gap, tiles[1][0]]))
    d_ax.set_xticks([])
    d_ax.set_yticks([])
    for s in d_ax.spines.values():
        s.set_visible(False)
    W0 = tiles[0][0].shape[1]
    for j, (img, label, names) in enumerate(tiles):
        x0 = j * (W0 + 24)
        d_ax.text(x0, -12, label, fontsize=9.5, color=INK)
        d_ax.text(x0, img.shape[0] + 8, f'Module 1 v2 regions:\n{names}', fontsize=8.5, color=INK2, va='top')
    d_ax.legend(handles=[Patch(fc='none', ec='#2a78d6', lw=2, label='fixed box'),
                         Patch(fc='#1baf7a', alpha=.5, ec='#1baf7a', label='Module 1 v2 regions')],
                loc='lower left', fontsize=8, framealpha=.85)
    for ax, letter in ((a, 'a'), (b, 'b'), (c, 'c'), (d_ax, 'd')):
        ax.text(-.06 if ax in (a, b) else -.03 if ax is c else .02, 1.12 if ax in (a, b, c) else 1.13, f'({letter})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=INK)
    h = [Line2D([], [], color=INK, lw=2.4, label='PPG reference'),
         Line2D([], [], color='#2a78d6', marker='s', ls='--', lw=1.1, ms=6, label='Fixed box'),
         Line2D([], [], color='#1baf7a', marker='o', ls='-', lw=1.1, ms=6, label='Module 1 v2 (dynamic ROI)'),
         Line2D([], [], color=INK2, marker='o', ls='', ms=6, label='filled = validated'),
         Line2D([], [], color=INK2, marker='o', ls='', ms=6, mfc='none', label='hollow = not validated'),
         Patch(fc=HAND, alpha=.6, label='hand present')]
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, fontsize=9, bbox_to_anchor=(.5, .955))
    fig.suptitle('Module 1 v2 dynamic ROI vs fixed box when a hand covers the face centre (UBFC subject1, designed case)',
                 x=.01, ha='left', y=.985, fontsize=12.5, color=INK)
    OUT.mkdir(exist_ok=True)
    p = OUT/'fig_occlusion_v2_vs_fixed_paper.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    print(p, f'diff {diff.mean():.2f} CI [{lo:.2f}, {hi:.2f}] n={len(diff)}')


if __name__ == '__main__':
    main()
