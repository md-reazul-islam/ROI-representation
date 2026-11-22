"""HR accuracy of fixed box vs Module 1 v1 vs Module 1 v2 (D5) on the DataAugment videos, against the PPG reference.

Sources (all made by run_hr_eval.sh with the same HR chain: POS, 8 s / 1 s windows, gate 1 + gate 2, band 45-180 bpm):
  HR RR using module1 v2/runs/<run>/aug_eval_D5/window_results_with_conditions.csv   -> Fixed box (B0), D5
  HR RR using module1 v2/runs/<run>/aug_eval_D0/window_results_with_conditions.csv   -> v1 (D0 = v1-equivalent)
  <run> = ubfc_aug (augmented videos) and ubfc_flat_38d8f553 (original videos = severity "none")

Outputs (dataaugment/hr_eval/report/):
  HR_evaluation_report.html  self-contained report (figures, tables, statements)
  metrics.csv                per category x severity x method: MAE, RMSE, within +-5 bpm, bias, r, coverage
  paired_differences.csv     D5 - fixed box and D5 - v1 in absolute error, moving-block bootstrap 95 % CI
  windows_merged.csv         every window: reference HR and the three methods' HR / label
  fig_timeline_<category>.png, fig_mae_by_severity.png, fig_coverage_by_severity.png
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
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

# Gate-2 variant (environment variable HR_GATE2): 'strict' (default, folders aug_eval_D5 / aug_eval_D0, method D5,
# report/) or 'fallback' (folders aug_eval_D5_fb / aug_eval_D0_fb, method D5_HR = HR-specific regions + fallback
# gate 2 as in the infant notebook, report_fallback/). The fixed box keeps its left/right gate 2 in both.
import os  # noqa: E402
GATE2 = os.environ.get('HR_GATE2', 'strict')
assert GATE2 in ('strict', 'fallback'), f'HR_GATE2 must be strict or fallback, not {GATE2}'
SUFFIX = '_fb' if GATE2 == 'fallback' else ''
DYN = 'D5_HR' if GATE2 == 'fallback' else 'D5'
REPORT_DIR = HERE/('report_fallback' if GATE2 == 'fallback' else 'report')
RUNS = ['ubfc_aug', 'ubfc_flat_38d8f553', 'ubfc_aug_fast', 'ubfc_aug_fast_xl']   # runs whose output is missing are skipped
ORIGINAL = {'NEW_subject1': 'motion', 'NEW_subject3': 'occlusion', 'NEW_subject4': 'lighting'}
CATS = ['motion', 'occlusion', 'lighting', 'motion_fast']
LABEL = {'motion': 'Motion (slow drift)', 'occlusion': 'Occlusion', 'lighting': 'Lighting',
         'motion_fast': 'Fast, large motion'}
SHARED_ORIGINAL = {'NEW_subject1': ['motion_fast']}   # the same original video is severity 'none' here too
SEVS = ['none', 'small', 'medium', 'large', 'xlarge']   # xlarge: fast motion only
SEV_LABEL = {'none': 'original', 'small': 'small', 'medium': 'medium', 'large': 'large', 'xlarge': 'extra-large'}
REPORTED = ('validated', 'validated_one_side')
TOL = 5.0
BLOCK = 8                       # windows overlap 7 s (8 s windows, 1 s step): bootstrap in blocks of 8 windows
N_BOOT = 5000

# Reference palette slots 1-3 (dataviz skill, validated adjacent set); identity also carried by marker + line style.
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
METHODS = {
    'fixed': dict(label='Fixed box', color='#2a78d6', marker='s', ls='--'),
    'v1':    dict(label='Module 1 v1', color='#eb6834', marker='^', ls=':'),
    'D5':    dict(label='Module 1 v2 (D5)', color='#1baf7a', marker='o', ls='-'),
}


def load(dirs_d5, dirs_d0):
    parts = []
    for d5, d0 in zip(dirs_d5, dirs_d0):
        a = pd.read_csv(d5/'window_results_with_conditions.csv')
        b = pd.read_csv(d0/'window_results_with_conditions.csv')
        a = a[['video_id', 'window', 'window_center_sec', 'hr_ref', 'hr_bpm_B0', 'label_B0', f'hr_bpm_{DYN}',
               f'label_{DYN}', 'snr_db_B0', f'snr_db_{DYN}']].rename(columns={
                   'hr_bpm_B0': 'hr_fixed', 'label_B0': 'label_fixed', f'hr_bpm_{DYN}': 'hr_D5', f'label_{DYN}': 'label_D5',
                   'snr_db_B0': 'snr_fixed', f'snr_db_{DYN}': 'snr_D5'})
        b = b[['video_id', 'window', f'hr_bpm_{DYN}', f'label_{DYN}', f'snr_db_{DYN}']].rename(
            columns={f'hr_bpm_{DYN}': 'hr_v1', f'label_{DYN}': 'label_v1', f'snr_db_{DYN}': 'snr_v1'})
        parts.append(a.merge(b, on=['video_id', 'window'], how='inner'))
    w = pd.concat(parts, ignore_index=True)
    w = fix_reference(w)            # nearest-centre PPG reference (see refmatch.py), as the UBFC check notebook

    def tag(v):
        if v in ORIGINAL:
            return ORIGINAL[v], 'none', v.replace('NEW_', '')
        s = v.replace('NEW_', '').split('_')
        return '_'.join(s[1:-1]), s[-1], s[0]
    w[['category', 'severity', 'subject']] = pd.DataFrame([tag(v) for v in w.video_id], index=w.index)
    extra = [w[w.video_id == v].assign(category=c) for v, cats in SHARED_ORIGINAL.items() for c in cats
             if (w.category == c).any()]
    return pd.concat([w] + extra, ignore_index=True)


def block_boot_ci(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 * BLOCK:
        return np.nan, np.nan
    starts = np.arange(n - BLOCK + 1)
    k = int(np.ceil(n / BLOCK))
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = (rng.choice(starts, k)[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
        means[i] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics(w):
    rows = []
    for (cat, sev), g in w.groupby(['category', 'severity']):
        g = g[g.hr_ref.notna()]
        for m in METHODS:
            hr, lab = g[f'hr_{m}'], g[f'label_{m}']
            ok = hr.notna()
            e = (hr - g.hr_ref)[ok]
            rep = ok & lab.isin(REPORTED)
            er = (hr - g.hr_ref)[rep]
            rows.append(dict(
                category=cat, severity=sev, method=METHODS[m]['label'], method_key=m, windows=len(g),
                mae_all=e.abs().mean(), rmse_all=np.sqrt((e ** 2).mean()), within5_all_pct=100 * (e.abs() <= TOL).mean(),
                bias_all=e.mean(), pearson_r_all=hr[ok].corr(g.hr_ref[ok]) if ok.sum() > 2 else np.nan,
                coverage_reported_pct=100 * rep.mean(), windows_reported=int(rep.sum()),
                mae_reported=er.abs().mean() if len(er) else np.nan,
                within5_reported_pct=100 * (er.abs() <= TOL).mean() if len(er) else np.nan,
                snr_median_db=g[f'snr_{m}'].median()))
    out = pd.DataFrame(rows)
    out['severity'] = pd.Categorical(out.severity, SEVS, ordered=True)
    out['category'] = pd.Categorical(out.category, CATS, ordered=True)
    out['method_key'] = pd.Categorical(out.method_key, list(METHODS), ordered=True)
    out = out.sort_values(['category', 'severity', 'method_key']).reset_index(drop=True)
    out['method_key'] = out.method_key.astype(str)
    return out


def paired(w):
    rng = np.random.default_rng(42)
    rows = []
    for (cat, sev), g in w.groupby(['category', 'severity']):
        g = g[g.hr_ref.notna() & g.hr_D5.notna()].sort_values('window')
        for other in ('fixed', 'v1'):
            h = g[g[f'hr_{other}'].notna()]
            d = ((h.hr_D5 - h.hr_ref).abs() - (h[f'hr_{other}'] - h.hr_ref).abs()).to_numpy()
            lo, hi = block_boot_ci(d, rng)
            rows.append(dict(category=cat, severity=sev, contrast=f'D5 - {METHODS[other]["label"]}', windows=len(d),
                             mean_abs_error_difference_bpm=d.mean() if len(d) else np.nan, ci95_low=lo, ci95_high=hi,
                             d5_better_windows_pct=100 * (d < 0).mean() if len(d) else np.nan,
                             verdict=('D5 better' if hi < 0 else f'{METHODS[other]["label"]} better' if lo > 0
                                      else 'no clear difference')))
    out = pd.DataFrame(rows)
    out['severity'] = pd.Categorical(out.severity, SEVS, ordered=True)
    out['category'] = pd.Categorical(out.category, CATS, ordered=True)
    return out.sort_values(['category', 'severity', 'contrast']).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(.6)
    ax.tick_params(colors=INK2, labelsize=8, width=.6)
    ax.grid(axis='y', color=GRID, lw=.6)
    ax.set_axisbelow(True)


def fig_timeline(w, met, cat, path):
    sub = w[w.category == cat]
    sevs = [s for s in SEVS if s in set(sub.severity)]
    fig, axes = plt.subplots(len(sevs), 1, figsize=(10, 2.35 * len(sevs) + .6), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    lo = np.nanpercentile(sub[['hr_ref', 'hr_fixed', 'hr_v1', 'hr_D5']].to_numpy(), 1) - 8
    hi = np.nanpercentile(sub[['hr_ref', 'hr_fixed', 'hr_v1', 'hr_D5']].to_numpy(), 99) + 8
    for ax, sev in zip(axes, sevs):
        style(ax)
        g = sub[sub.severity == sev].sort_values('window_center_sec')
        ax.plot(g.window_center_sec, g.hr_ref, color=INK, lw=2.2, label='Reference (PPG)', zorder=5)
        for m, s in METHODS.items():
            ax.plot(g.window_center_sec, g[f'hr_{m}'], color=s['color'], lw=1.6, ls=s['ls'], zorder=3)
            rep = g[f'label_{m}'].isin(REPORTED)
            ax.scatter(g.window_center_sec[rep], g[f'hr_{m}'][rep], s=26, marker=s['marker'], color=s['color'],
                       edgecolors=SURFACE, linewidths=.8, zorder=4)
            ax.scatter(g.window_center_sec[~rep], g[f'hr_{m}'][~rep], s=22, marker=s['marker'], facecolors=SURFACE,
                       edgecolors=s['color'], linewidths=1.0, zorder=4)
        r = met[(met.category == cat) & (met.severity == sev)].set_index('method_key')
        txt = '   '.join(f"{METHODS[m]['label']} {r.loc[m, 'mae_all']:.1f}" for m in METHODS)
        ax.set_title(f'{SEV_LABEL[sev].upper() if sev != "none" else "ORIGINAL (no augmentation)"}   |   MAE (bpm): {txt}',
                     loc='left', fontsize=9, color=INK)
        ax.set_ylabel('HR (bpm)', fontsize=8, color=INK2)
        ax.set_ylim(max(40, lo), min(185, hi))
    axes[-1].set_xlabel('time (s), centre of each 8-s window', fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=INK, lw=2.2, label='Reference (PPG)')] + \
              [plt.Line2D([], [], color=s['color'], lw=1.6, ls=s['ls'], marker=s['marker'], markersize=6,
                          label=s['label']) for s in METHODS.values()] + \
              [plt.Line2D([], [], color=INK2, lw=0, marker='o', markersize=6, label='filled = validated window'),
               plt.Line2D([], [], color=INK2, lw=0, marker='o', markersize=6, markerfacecolor=SURFACE,
                          label='hollow = not validated')]
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(.5, 1.0))
    subj = sub.subject.iloc[0]
    fig.suptitle(f'{LABEL[cat]} augmentation ({subj}): camera HR vs PPG reference', x=.01, y=1.035,
                 ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def fig_by_severity(met, column, ylabel, title, path, lower_better=True):
    cats = [c for c in CATS if (met.category == c).any()]
    fig, axes = plt.subplots(1, len(cats), figsize=(3.7 * len(cats), 3.4), sharey=True, facecolor=SURFACE)
    sevs = [v for v in SEVS if (met.severity == v).any()]
    x = np.arange(len(sevs))
    for ax, cat in zip(axes, cats):
        style(ax)
        r = met[met.category == cat]
        for m, s in METHODS.items():
            q = r[r.method_key == m].set_index('severity').reindex(sevs)
            ax.plot(x, q[column], color=s['color'], ls=s['ls'], lw=2, marker=s['marker'], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=s['label'])
            last = q[column].dropna()
            if len(last):
                ax.annotate(f'{last.iloc[-1]:.1f}', (x[sevs.index(last.index[-1])], last.iloc[-1]), xytext=(6, 0),
                            textcoords='offset points', fontsize=7.5, color=INK2, va='center')
        ax.set_xticks(x, [SEV_LABEL[v] for v in sevs], fontsize=8)
        ax.set_title(LABEL[cat], loc='left', fontsize=10, color=INK)
        ax.set_xlim(-.3, len(sevs) - .4)
    axes[0].set_ylabel(ylabel, fontsize=8.5, color=INK2)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(title + ('  (lower is better)' if lower_better else '  (higher is better)'), x=.01, ha='left',
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ report
def b64(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def table_html(df, fmt=None):
    fmt = fmt or {}
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    body = ''
    for _, r in df.iterrows():
        cells = ''
        for c in df.columns:
            v = r[c]
            s = fmt[c].format(v) if c in fmt and pd.notna(v) and not isinstance(v, str) else ('–' if pd.isna(v) else v)
            cells += f'<td>{html.escape(str(s))}</td>'
        body += f'<tr>{cells}</tr>'
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def statements(met, pair):
    out = []
    for cat in CATS:
        for sev in SEVS:
            r = met[(met.category == cat) & (met.severity == sev)].set_index('method_key')
            if r.empty:
                continue
            best = r.mae_all.idxmin()
            p = pair[(pair.category == cat) & (pair.severity == sev)].set_index('contrast')
            def ci(name):
                q = p.loc[name]
                return f"{q.mean_abs_error_difference_bpm:+.1f} bpm [95% CI {q.ci95_low:+.1f}, {q.ci95_high:+.1f}]"
            out.append(dict(category=cat, severity=sev, text=(
                f"<b>{LABEL[cat]}, {SEV_LABEL[sev]}:</b> MAE fixed box "
                f"{r.loc['fixed', 'mae_all']:.1f}, v1 {r.loc['v1', 'mae_all']:.1f}, D5 {r.loc['D5', 'mae_all']:.1f} bpm "
                f"(lowest: {METHODS[best]['label']}). D5 − fixed box {ci('D5 - Fixed box')}; "
                f"D5 − v1 {ci('D5 - Module 1 v1')}. Validated windows: fixed {r.loc['fixed', 'coverage_reported_pct']:.0f}%, "
                f"v1 {r.loc['v1', 'coverage_reported_pct']:.0f}%, D5 {r.loc['D5', 'coverage_reported_pct']:.0f}%.")))
    return out


def overall(met, pair):
    aug = pair[pair.severity != 'none']
    lines = []
    for name in ('D5 - Fixed box', 'D5 - Module 1 v1'):
        a = aug[aug.contrast == name]
        better, worse = (a.verdict == 'D5 better').sum(), a.verdict.str.endswith('better').sum() - (a.verdict == 'D5 better').sum()
        other = name.split(' - ')[1]
        lines.append(f'{name}: D5 clearly better (95 % CI below 0) in {better} of {len(a)} augmented conditions, '
                     f'{other} clearly better in {worse}, no clear difference in {len(a) - better - worse}.')
    m = met[met.severity != 'none'].groupby('method_key').mae_all.mean()
    lines.append(f'Mean MAE over the {len(aug) // 2} augmented conditions: ' +
                 ', '.join(f"{METHODS[k]['label']} {m[k]:.1f} bpm" for k in METHODS) + '.')
    return lines


CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--accent:#1baf7a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df}}
body{background:var(--surface);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--rule);padding-top:18px}
h3{font-size:15px;margin:20px 0 6px}.muted{color:var(--ink2)}img{max-width:100%;height:auto;display:block;margin:8px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin:8px 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
th{color:var(--ink2);font-weight:600}.box{border-left:3px solid var(--accent);padding:8px 14px;background:#f3f6f4;margin:12px 0}
li{margin:4px 0}code{font-size:13px}
"""


def write_html(target, w, met, pair, figs, meta):
    st = statements(met, pair)
    ov = overall(met, pair)
    show = met[['category', 'severity', 'method', 'windows', 'mae_all', 'rmse_all', 'within5_all_pct', 'bias_all',
                'pearson_r_all', 'coverage_reported_pct', 'mae_reported', 'within5_reported_pct']].copy()
    show.columns = ['Category', 'Severity', 'Method', 'Windows', 'MAE', 'RMSE', '±5 bpm %', 'Bias', 'r',
                    'Validated %', 'MAE (validated)', '±5 bpm % (validated)']
    fmt = {c: '{:.2f}' for c in ['MAE', 'RMSE', 'Bias', 'r', 'MAE (validated)']}
    fmt.update({c: '{:.0f}' for c in ['±5 bpm %', 'Validated %', '±5 bpm % (validated)']})
    pshow = pair[['category', 'severity', 'contrast', 'windows', 'mean_abs_error_difference_bpm', 'ci95_low',
                  'ci95_high', 'd5_better_windows_pct', 'verdict']].copy()
    pshow.columns = ['Category', 'Severity', 'Contrast', 'Windows', 'Δ |error| (bpm)', 'CI low', 'CI high',
                     'D5 better windows %', 'Verdict']
    pfmt = {c: '{:+.2f}' for c in ['Δ |error| (bpm)', 'CI low', 'CI high']}
    pfmt['D5 better windows %'] = '{:.0f}'
    GATE2_HTML = ('''<li><b>HR chain (identical for all):</b> POS pulse, 8 s windows every 1 s, HR band 45–180 bpm (adults), gate 1
(block-shuffle surrogate null) and gate 2 (left/right halves agree within 6 bpm) decide <i>validated</i>.</li>''' if GATE2 == 'strict' else
'''<li><b>HR chain — FALLBACK gate 2 variant:</b> POS pulse, 8 s windows every 1 s, HR band 45–180 bpm (adults), gate 1
(block-shuffle surrogate null). <b>v1 and v2</b> use the infant-notebook HR (method D5_HR: HR-specific regions chosen
from all Module 1 v2 candidates, and fallback gate 2: the first pair with both halves visible among left/right →
face/chest → top/bottom must agree within 6 bpm; if no pair is visible the window is <i>validated_one_side</i>, which
counts as reported). <b>The fixed box</b> has no fallback implementation and keeps the left/right gate 2, so in this
variant the methods differ in region choice and gate 2, not only in the ROI. Strict-gate results: <code>report/</code>.</li>''')
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Augmented UBFC HR Evaluation{' (fallback gate 2)' if GATE2 == 'fallback' else ''}</title><style>{CSS}</style></head><body>
<h1>HR accuracy under simulated motion, occlusion and lighting{' — fallback gate 2' if GATE2 == 'fallback' else ''}</h1>
<p class="muted">Fixed box vs Module 1 v1 vs Module 1 v2 (D5) on UBFC-rPPG videos augmented by DataAugment, against the
fingertip-PPG reference. Generated {meta['created']}.</p>
<div class="box"><b>Overall</b><ul>{''.join(f'<li>{l}</li>' for l in ov)}</ul>
<span class="muted">One subject per category: this is a controlled pilot, not a population result. Windows overlap
(8 s windows, 1 s step), so confidence intervals use a moving-block bootstrap (blocks of {BLOCK} windows).</span></div>
<h2>1. Set-up</h2><ul>
<li><b>Videos:</b> subject1 → motion (slow drift) and fast, large motion; subject3 → occlusion; subject4 → lighting;
severity <i>original</i> is the unaugmented video, then small / medium / large (same random trajectories, scaled).
Occlusions are simulated, not real swaddling.</li>
<li><b>Slow vs fast motion:</b> slow drift (below 0.08 Hz, up to 40 px) never moved the face out of the fixed box (at
least 57 % of the face stayed inside it in every frame). Fast motion = quick repositioning moves (0.25–0.6 s, random
times, mean every 3 s) of up to 50 / 100 / 160 / 230 px with rotation up to 4 / 8 / 12 / 15°; less than half of the
face is inside the fixed box in 0 / 7 / 46 / 78 % of frames, and at <i>extra-large</i> less than 10 % of it in 20 % of
frames (the face always stays at least 91 % inside the image).</li>
<li><b>Methods (only the ROI differs):</b> <i>Fixed box</i> = centre 50 % of the first-frame subject box, kept for the
whole video; <i>Module 1 v1</i> = configuration D0 (v1-equivalent dynamic ROI, no stabilizer); <i>Module 1 v2 (D5)</i> =
the headline configuration (stabilized selection, keypoint visibility). The stabilizer settings are the frozen final34
values; nothing was tuned on these videos.</li>
{GATE2_HTML}
<li><b>Reference:</b> FFT peak of the UBFC fingertip PPG waveform in the same 8 s windows. Augmentation keeps every frame
and timestamp, so the reference of the source subject applies unchanged.</li>
<li><b>Metrics:</b> MAE, RMSE, % of windows within ±{TOL:.0f} bpm, bias (camera − reference), Pearson r, on <i>all
windows</i> (robustness of the estimate itself) and on <i>validated windows</i> (what the system would report; read
with the validated %, because a method can look accurate by reporting less).</li></ul>
<h2>2. HR over time: reference vs the three methods</h2>
<p class="muted">Black line = PPG reference. Filled markers = validated windows, hollow = not validated. Panel titles give
the MAE over all windows.</p>"""]
    for cat in [c for c in CATS if c in figs]:
        parts.append(f'<h3>{LABEL[cat]}</h3><img alt="{cat} timeline" src="{b64(figs[cat])}">')
    if 'fast_grid' in figs:
        parts.append('<h3>Fast, large motion: one panel per method</h3><p class="muted">Same data as above, drawn like '
                     'the Module 1 v2 HR figures: black = PPG reference, filled dots = validated windows, hollow = not '
                     'validated, thin line = tracked (Kalman) HR, red ticks = windows where the method gave no HR at '
                     'all. Rows: severity; columns: fixed box, Module 1 v1, Module 1 v2 (D5).</p>'
                     f'<img alt="fast motion per method" src="{b64(figs["fast_grid"])}">'
                     '<p class="muted">Extra-large severity in detail:</p>'
                     f'<img alt="fast motion extra-large" src="{b64(figs["fast_xl"])}">')
    parts.append(f"""<h2>3. Error by severity</h2><img alt="MAE by severity" src="{b64(figs['mae'])}">
<img alt="MAE validated by severity" src="{b64(figs['mae_rep'])}">
<img alt="coverage by severity" src="{b64(figs['cov'])}">
<h2>4. Statements per condition</h2><ul>{''.join(f"<li>{s['text']}</li>" for s in st)}</ul>
<h2>5. Full metrics</h2>{table_html(show, fmt)}
<h2>6. Paired differences (D5 minus comparator, absolute error per window)</h2>
<p class="muted">Negative = D5 has the smaller error. Verdict uses the 95 % moving-block bootstrap CI.</p>{table_html(pshow, pfmt)}
<h2>7. Limitations</h2><ul>
<li>One adult subject per category and one video per severity: differences describe these videos only.</li>
<li>Augmentations are synthetic: global camera warp (no parallax or motion blur), flat pulse-free occluders,
gamma-2.2 lighting model. Real neonatal motion, swaddling and NICU lighting can behave differently.</li>
<li>The original videos were processed in an earlier run of the same pipeline (same code and settings).</li>
<li>All numbers are recomputed from the CSV files listed below by <code>build_hr_report.py</code>.</li></ul>
<p class="muted">Sources: {'; '.join(html.escape(str(s)) for s in meta['sources'])}</p></body></html>""")
    (target/'HR_evaluation_report.html').write_text(''.join(parts), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--d5', nargs='+', help='aug_eval_D5 folders (default: both runs)')
    ap.add_argument('--d0', nargs='+', help='aug_eval_D0 folders (same order)')
    ap.add_argument('--out', default=str(REPORT_DIR))
    a = ap.parse_args(argv)
    runs = [r for r in RUNS if (M2RUNS/r/f'aug_eval_D0{SUFFIX}'/'window_results_with_conditions.csv').exists()]
    d5 = [Path(p) for p in a.d5] if a.d5 else [M2RUNS/r/f'aug_eval_D5{SUFFIX}' for r in runs]
    d0 = [Path(p) for p in a.d0] if a.d0 else [M2RUNS/r/f'aug_eval_D0{SUFFIX}' for r in runs]
    target = Path(a.out)
    target.mkdir(parents=True, exist_ok=True)
    w = load(d5, d0)
    w.to_csv(target/'windows_merged.csv', index=False)
    met = metrics(w)
    met.to_csv(target/'metrics.csv', index=False)
    pair = paired(w)
    pair.to_csv(target/'paired_differences.csv', index=False)
    figs = {}
    for cat in CATS:
        if (w.category == cat).any():
            figs[cat] = target/f'fig_timeline_{cat}.png'
            fig_timeline(w, met, cat, figs[cat])
    if (w.category == 'motion_fast').any() and target == REPORT_DIR:
        import make_fast_motion_figures
        make_fast_motion_figures.main()
        figs['fast_grid'] = target/'fig_fast_motion_grid.png'
        figs['fast_xl'] = target/'fig_fast_motion_xlarge.png'
    figs['mae'] = target/'fig_mae_by_severity.png'
    fig_by_severity(met, 'mae_all', 'MAE, all windows (bpm)', 'HR error by augmentation severity', figs['mae'])
    figs['mae_rep'] = target/'fig_mae_validated_by_severity.png'
    fig_by_severity(met, 'mae_reported', 'MAE, validated windows (bpm)', 'HR error of the reported (validated) windows',
                    figs['mae_rep'])
    figs['cov'] = target/'fig_coverage_by_severity.png'
    fig_by_severity(met, 'coverage_reported_pct', 'validated windows (%)', 'Share of windows the system validates',
                    figs['cov'], lower_better=False)
    meta = dict(created=dt.datetime.now().strftime('%Y-%m-%d %H:%M'), sources=[str(p) for p in d5 + d0])
    if all(c in figs for c in CATS[:3]):
        write_html(target, w, met, pair, figs, meta)
    (target/'metadata.json').write_text(json.dumps(dict(meta, block=BLOCK, n_boot=N_BOOT, tolerance_bpm=TOL), indent=2))
    pd.set_option('display.width', 200)
    print(met[['category', 'severity', 'method', 'mae_all', 'within5_all_pct', 'coverage_reported_pct', 'mae_reported']]
          .round(2).to_string(index=False))
    print('\n'.join(overall(met, pair)) if (pair.severity != 'none').any() else '')
    print('wrote', target)


if __name__ == '__main__':
    main()
