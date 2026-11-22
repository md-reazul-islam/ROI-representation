"""Fast, large motion: reference HR vs fixed box, Module 1 v1 and Module 1 v2 (D5), one panel per method.

Style of the Module 1 v2 HR figures: black line = PPG reference, dots = camera HR per 8 s window (filled = validated,
hollow = not validated), thin line = tracked (Kalman) HR. Windows where a method gives no HR at all are marked with
a red tick on the bottom axis. Values outside the plotted range are drawn on the border as a triangle.

Writes to report/:
  fig_fast_motion_<severity>.png   3 stacked panels (fixed box, v1, D5) for one severity
  fig_fast_motion_grid.png         all severities (rows) x methods (columns)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from refmatch import fix_reference  # noqa: E402
ROOT = HERE.parents[1]
M2RUNS = ROOT/'HR RR using module1 v2/runs'
import os  # noqa: E402
GATE2 = os.environ.get('HR_GATE2', 'strict')   # strict | fallback (see build_hr_report.py)
SUFFIX, DYN = ('_fb', 'D5_HR') if GATE2 == 'fallback' else ('', 'D5')
OUT = HERE/('report_fallback' if GATE2 == 'fallback' else 'report')
TAG = '  [fallback gate 2]' if GATE2 == 'fallback' else ''
SOURCES = {'none': ('ubfc_flat_38d8f553', 'NEW_subject1'),
           'small': ('ubfc_aug_fast', 'NEW_subject1_motion_fast_small'),
           'medium': ('ubfc_aug_fast', 'NEW_subject1_motion_fast_medium'),
           'large': ('ubfc_aug_fast', 'NEW_subject1_motion_fast_large'),
           'xlarge': ('ubfc_aug_fast_xl', 'NEW_subject1_motion_fast_xlarge')}
SEV_TITLE = {'none': 'Original (no motion added)', 'small': 'Small fast motion (up to 50 px)',
             'medium': 'Medium fast motion (up to 100 px)', 'large': 'Large fast motion (up to 160 px)',
             'xlarge': 'Extra-large fast motion (up to 230 px)'}
METHODS = {'fixed': ('Fixed box', '#2a78d6'), 'v1': ('Module 1 v1', '#eb6834'), 'D5': ('Module 1 v2 (D5)', '#1baf7a')}
OK = {'validated', 'validated_one_side'}
INK, INK2, GRID, SURFACE, MISS = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#c62828'
YLIM = (55, 125)


def load(sev):
    run, vid = SOURCES[sev]
    a = pd.read_csv(M2RUNS/run/f'aug_eval_D5{SUFFIX}'/'window_results_with_conditions.csv')
    b = pd.read_csv(M2RUNS/run/f'aug_eval_D0{SUFFIX}'/'window_results_with_conditions.csv')
    a, b = a[a.video_id == vid], b[b.video_id == vid]
    d = fix_reference(a[['video_id', 'window', 'window_center_sec', 'hr_ref']]).drop(columns='video_id')
    for key, src, col in (('fixed', a, 'B0'), ('D5', a, DYN), ('v1', b, DYN)):
        s = src.set_index('window')
        d[f'hr_{key}'] = d.window.map(s[f'hr_bpm_{col}'])
        d[f'kal_{key}'] = d.window.map(s[f'hr_kalman_bpm_{col}'])
        d[f'lab_{key}'] = d.window.map(s[f'label_{col}'])
    return d.sort_values('window_center_sec')


def panel(ax, d, key, title=True, xlabel=True):
    name, color = METHODS[key]
    t, ref, hr = d.window_center_sec, d.hr_ref, d[f'hr_{key}']
    valid = d[f'lab_{key}'].isin(OK) & hr.notna()
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.plot(t, ref, color=INK, lw=2.2, label='PPG reference', zorder=4)
    ax.plot(t, d[f'kal_{key}'].clip(*YLIM), color=color, lw=1, alpha=.55, label=f'{name}, tracked', zorder=2)
    y = hr.clip(YLIM[0] + .5, YLIM[1] - .5)
    out = (hr < YLIM[0]) | (hr > YLIM[1])
    ax.scatter(t[valid & ~out], y[valid & ~out], s=34, color=color, edgecolor=SURFACE, lw=.8, zorder=5,
               label=f'{name}, validated')
    ax.scatter(t[~valid & ~out], y[~valid & ~out], s=30, facecolor='none', edgecolor=color, lw=1.2, zorder=5,
               label=f'{name}, not validated')
    for lo_hi, mk in ((hr > YLIM[1], '^'), (hr < YLIM[0], 'v')):
        ax.scatter(t[lo_hi], y[lo_hi], marker=mk, s=34, facecolor='none', edgecolor=color, lw=1.2, zorder=5)
    miss = hr.isna()
    if miss.any():
        ax.scatter(t[miss], np.full(miss.sum(), YLIM[0] + 1.5), marker='|', s=70, color=MISS, lw=1.6, zorder=6,
                   label='no HR output')
    ax.set_ylim(*YLIM)
    err = (hr - ref).abs()
    mae = err.mean()
    cov = 100 * valid.mean()
    if title:
        txt = f'{name}:  MAE {mae:.1f} bpm,  validated {cov:.0f} % of windows'
        if miss.any():
            txt += f',  no HR in {miss.sum()} of {len(d)} windows'
        ax.set_title(txt, loc='left', fontsize=9.5, color=INK)
    if xlabel:
        ax.set_xlabel('time (s)', fontsize=8.5, color=INK2)
    ax.set_ylabel('HR (bpm)', fontsize=8.5, color=INK2)


def legend(fig, color_keys=('fixed', 'v1', 'D5'), y=1.0):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, lw=2.2, label='PPG reference')]
    for k in color_keys:
        h.append(Line2D([], [], color=METHODS[k][1], marker='o', ls='-', lw=1, ms=6, label=METHODS[k][0]))
    h += [Line2D([], [], color=INK2, marker='o', ls='', ms=6, label='filled = validated'),
          Line2D([], [], color=INK2, marker='o', ls='', ms=6, mfc='none', label='hollow = not validated'),
          Line2D([], [], color=INK2, lw=1, alpha=.6, label='thin line = tracked HR'),
          Line2D([], [], color=MISS, marker='|', ls='', ms=10, mew=1.6, label='no HR output')]
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, fontsize=8.5, bbox_to_anchor=(.5, y))


def main():
    OUT.mkdir(exist_ok=True)
    data = {s: load(s) for s in SOURCES}
    for sev, d in data.items():
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, facecolor=SURFACE)
        for i, (ax, key) in enumerate(zip(axes, METHODS)):
            panel(ax, d, key, xlabel=i == 2)
        fig.suptitle(f'NEW_subject1, {SEV_TITLE[sev]}: camera HR vs PPG reference' + TAG, x=.01, ha='left', y=1.02,
                     fontsize=12, color=INK)
        legend(fig, y=.995)
        fig.tight_layout(rect=(0, 0, 1, .965))
        fig.savefig(OUT/f'fig_fast_motion_{sev}.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
        plt.close(fig)
    fig, axes = plt.subplots(len(data), 3, figsize=(17, 2.6 * len(data) + .8), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for r, (sev, d) in enumerate(data.items()):
        for c, key in enumerate(METHODS):
            ax = axes[r, c]
            panel(ax, d, key, title=False, xlabel=r == len(data) - 1)
            hr, ref = d[f'hr_{key}'], d.hr_ref
            valid = d[f'lab_{key}'].isin(OK) & hr.notna()
            miss = int(hr.isna().sum())
            ax.set_title(f'{SEV_TITLE[sev].split(" (")[0]} | {METHODS[key][0]}\nMAE {(hr - ref).abs().mean():.1f} bpm, '
                         f'validated {100 * valid.mean():.0f} %' + (f', no HR {miss}/{len(d)}' if miss else ''),
                         loc='left', fontsize=9, color=INK)
            if c:
                ax.set_ylabel('')
    fig.suptitle('Fast, large motion (subject1): PPG reference vs fixed box, Module 1 v1 and Module 1 v2 (D5)' + TAG,
                 x=.01, ha='left', y=1.01, fontsize=13, color=INK)
    legend(fig, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT/'fig_fast_motion_grid.png', dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print('wrote', ', '.join(f'fig_fast_motion_{s}.png' for s in data), 'and fig_fast_motion_grid.png in', OUT)


if __name__ == '__main__':
    main()


# ------------------------------------------------------------------ claim figure: Module 1 v2 (D5) vs fixed box only
PAIR = ('fixed', 'D5')
SEV_SHORT = {'none': 'original', 'small': 'small\n(50 px)', 'medium': 'medium\n(100 px)', 'large': 'large\n(160 px)',
             'xlarge': 'extra-large\n(230 px)'}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.tick_params(labelsize=8.5, colors=INK2)


def pair_timeline(ax, d, sev, show_legend=False):
    style(ax)
    t = d.window_center_sec
    ax.plot(t, d.hr_ref, color=INK, lw=2.4, zorder=4, label='PPG reference')
    txt = []
    for key in PAIR:
        name, color = METHODS[key]
        hr = d[f'hr_{key}']
        valid = d[f'lab_{key}'].isin(OK) & hr.notna()
        y = hr.clip(YLIM[0] + .5, YLIM[1] - .5)
        ax.plot(t, y, color=color, lw=1.1, alpha=.5, zorder=2, ls='--' if key == 'fixed' else '-')
        mk = 's' if key == 'fixed' else 'o'
        ax.scatter(t[valid], y[valid], marker=mk, s=36, color=color, edgecolor=SURFACE, lw=.8, zorder=5)
        ax.scatter(t[~valid & hr.notna()], y[~valid & hr.notna()], marker=mk, s=32, facecolor='none',
                   edgecolor=color, lw=1.3, zorder=5)
        miss = hr.isna()
        if miss.any():
            ax.scatter(t[miss], np.full(miss.sum(), YLIM[0] + 1.5), marker='|', s=70, color=MISS, lw=1.6, zorder=6)
        txt.append(f'{"Fixed box" if key == "fixed" else "Module 1 v2"} {(hr - d.hr_ref).abs().mean():.1f}')
    ax.set_ylim(*YLIM)
    ax.set_ylabel('HR (bpm)', fontsize=8.5, color=INK2)
    ax.set_title(f'{SEV_TITLE[sev]}   |   MAE (bpm): ' + ',  '.join(txt), loc='left', fontsize=9.5, color=INK)


def pair_legend(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, lw=2.4, label='PPG reference'),
         Line2D([], [], color=METHODS['fixed'][1], marker='s', ls='--', lw=1.1, ms=6, label='Fixed box'),
         Line2D([], [], color=METHODS['D5'][1], marker='o', ls='-', lw=1.1, ms=6, label='Module 1 v2 (dynamic ROI)'),
         Line2D([], [], color=INK2, marker='o', ls='', ms=6, label='filled = validated'),
         Line2D([], [], color=INK2, marker='o', ls='', ms=6, mfc='none', label='hollow = not validated'),
         Line2D([], [], color=MISS, marker='|', ls='', ms=10, mew=1.6, label='no HR output (fixed box)')]
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, fontsize=9, bbox_to_anchor=(.5, y))


def summary_axes(ax_mae, ax_cov, data, pair_ci):
    sevs = list(data)
    x = np.arange(len(sevs))
    for key in PAIR:
        name, color = METHODS[key]
        mae = [(data[s][f'hr_{key}'] - data[s].hr_ref).abs().mean() for s in sevs]
        cov = [100 * (data[s][f'lab_{key}'].isin(OK) & data[s][f'hr_{key}'].notna()).mean() for s in sevs]
        lab = 'Fixed box' if key == 'fixed' else 'Module 1 v2 (dynamic ROI)'
        kw = dict(color=color, lw=2.2, marker='s' if key == 'fixed' else 'o', ms=8, mec=SURFACE, mew=1.2,
                  ls='--' if key == 'fixed' else '-', label=lab)
        ax_mae.plot(x, mae, **kw)
        ax_cov.plot(x, cov, **kw)
        for xi, v in zip(x, mae):
            ax_mae.annotate(f'{v:.1f}', (xi, v), xytext=(0, 9 if key == 'fixed' else -15), textcoords='offset points',
                            ha='center', fontsize=8.5, color=INK2)
    for xi, s in zip(x, sevs):
        miss = int(data[s].hr_fixed.isna().sum())
        if miss:
            ax_cov.annotate(f'fixed: no HR\nin {miss}/{len(data[s])}', (xi, 0), xytext=(0, 12),
                            textcoords='offset points', ha='center', fontsize=7.5, color=MISS)
        if s in pair_ci:
            d, lo, hi, sig = pair_ci[s]
            if sig:
                ax_mae.text(.04, .55, f'{SEV_SHORT[s].split(chr(10))[0]}: v2 − fixed box = {d:+.1f} bpm\n'
                            f'95 % CI [{lo:+.1f}, {hi:+.1f}] (block bootstrap)', transform=ax_mae.transAxes,
                            fontsize=8, color=INK, va='top')
    for ax, yl, t in ((ax_mae, 'MAE vs PPG (bpm)', 'HR error (lower is better)'),
                      (ax_cov, 'validated windows (%)', 'HR the system reports (higher is better)')):
        style(ax)
        ax.set_xticks(x, [SEV_SHORT[s] for s in sevs], fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5, color=INK2)
        ax.set_title(t, loc='left', fontsize=10, color=INK)
        ax.set_xlim(-.4, len(sevs) - .6)
    ax_mae.set_ylim(0, None)
    ax_cov.set_ylim(-3, 105)
    ax_mae.legend(fontsize=8.5, frameon=False, loc='upper left')


def claim_figures(data):
    p = pd.read_csv(OUT/'paired_differences.csv')
    p = p[(p.category == 'motion_fast') & (p.contrast == 'D5 - Fixed box')]
    pair_ci = {r.severity: (r.mean_abs_error_difference_bpm, r.ci95_low, r.ci95_high, r.ci95_high < 0)
               for r in p.itertuples()}
    # (1) timeline, all severities
    fig, axes = plt.subplots(len(data), 1, figsize=(11, 2.5 * len(data) + .9), sharex=True, facecolor=SURFACE)
    for ax, (sev, d) in zip(axes, data.items()):
        pair_timeline(ax, d, sev)
    axes[-1].set_xlabel('time (s), centre of each 8-s window', fontsize=8.5, color=INK2)
    fig.suptitle('Fast, large motion (UBFC subject1): Module 1 v2 dynamic ROI vs fixed box' + TAG, x=.01, ha='left',
                 y=1.01, fontsize=12.5, color=INK)
    pair_legend(fig, .995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(OUT/'fig_v2_vs_fixed_timeline.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    # (2) summary
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    summary_axes(a, b, data, pair_ci)
    fig.suptitle('Fast, large motion: Module 1 v2 dynamic ROI vs fixed box by severity' + TAG, x=.01, ha='left',
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT/'fig_v2_vs_fixed_summary.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    # (3) paper figure: (a) MAE, (b) validated, (c) large timeline, (d) extra-large timeline
    fig = plt.figure(figsize=(11, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=.62, wspace=.22, top=.87)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    summary_axes(a, b, data, pair_ci)
    c = fig.add_subplot(gs[1, :])
    e = fig.add_subplot(gs[2, :], sharex=c)
    pair_timeline(c, data['large'], 'large')
    pair_timeline(e, data['xlarge'], 'xlarge')
    e.set_xlabel('time (s), centre of each 8-s window', fontsize=8.5, color=INK2)
    for ax, letter in zip((a, b, c, e), 'abcd'):
        ax.text(-.06 if ax in (a, b) else -.03, 1.12, f'({letter})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', color=INK)
    pair_legend(fig, .955)
    fig.suptitle('Module 1 v2 dynamic ROI vs fixed box under fast, large motion (UBFC subject1)' + TAG, x=.01, ha='left',
                 y=.985, fontsize=12.5, color=INK)
    fig.savefig(OUT/'fig_v2_vs_fixed_paper.png', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print('wrote fig_v2_vs_fixed_timeline.png, fig_v2_vs_fixed_summary.png, fig_v2_vs_fixed_paper.png')


if __name__ == '__main__':
    claim_figures({s: load(s) for s in SOURCES})
