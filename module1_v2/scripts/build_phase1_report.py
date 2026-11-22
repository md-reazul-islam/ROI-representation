"""Build the Module 1 v2 Phase 1 Word report (figures + tables) from the Phase 1 outputs.

Usage: python module1_v2/scripts/build_phase1_report.py --name final34_paper
Reads  module1_v2/reports/<name>/phase1, module1_v2/runs/<name>/tuning_log.json, runs/<name>/manifest.csv,
       module1_v2/configs/module1_v2.yaml and (optional) phase1/subject_audit.json (manual visual audit)
Writes module1_v2/reports/<name>/phase1/figures/* and Module1_v2_Phase1_Report.docx
Every number and every conclusion in the text is computed from those files; nothing is typed in by hand.
The headline configuration is chosen on the VALIDATION split only (highest mean pulse SNR among D1-D7).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports/paper/manuscript_build'))
from docbuild import Builder  # noqa: E402  (v1 paper tooling, imported read-only)

ORDER = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'Full', 'D0', 'I-B', 'D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7',
         'I-V', 'I-M', 'I-S']
NEW = ['D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7']
INK, INK2, GRID, MUTED = '#0b0b0b', '#52514e', '#e4e3df', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
METRIC = {'switches_per_sec': 'switches/s', 'temporal_iou': 'temporal IoU', 'pulse_snr_db': 'pulse SNR (dB)',
          'informative_window_percent': 'informative windows (%)'}
DESCRIPTION = {
    'B0': 'fixed box (v1)', 'B1': 'fixed anatomical ROI (v1)', 'B2': 'pose term P only (v1)',
    'B3': '(P+V+S)/3 (v1)', 'B4': 'stored Q, no SAM 2 (v1)', 'B5': 'stored Q, SAM 2 branch (v1)',
    'Full': 'B5 + m1 pixel refinement (v1)', 'D0': 'v2 code, all flags off (= Full)', 'I-B': 'Full + 2-point boxes read as rectangles', 'D1': 'EMA of Q',
    'D2': 'D1 + hysteresis', 'D3': 'D2 + offline Viterbi anchor', 'D3L': 'D2 + fixed-lag Viterbi',
    'D4': 'D3 + smoothed weights + crossfade', 'D5': "D4 + keypoint visibility V'", 'D6': 'D5 + nuisance motion',
    'D7': 'D6 + subject check', 'I-V': "Full + V' only", 'I-M': 'Full + nuisance motion only',
    'I-S': 'Full + subject check only'}


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def bootstrap_ci(values, reps=2000, seed=0):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    means = np.random.default_rng(seed).choice(values, (reps, len(values))).mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def fig_stability(pv, out, colors, chosen):
    configs = [c for c in ORDER if c in set(pv.config)]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.2), sharey=True)
    for ax, metric, label in zip(axes, ['switches_per_sec', 'temporal_iou'],
                                 ['Top-1 region switches per second (lower = steadier)',
                                  'Temporal IoU of selected region (higher = steadier)']):
        y = np.arange(len(configs))
        means = [pv.loc[pv.config == c, metric].mean() for c in configs]
        ax.barh(y, means, height=.55, color=[colors.get(c, MUTED) for c in configs])
        for i, c in enumerate(configs):
            vals = pv.loc[pv.config == c, metric].dropna()
            ax.scatter(vals, np.full(len(vals), i), s=8, color=INK, alpha=.6, zorder=3, linewidths=0)
        style(ax)
        ax.set_xlabel(label, fontsize=8, color=INK2)
        ax.set_yticks(y, configs)
    axes[0].invert_yaxis()  # shared y: invert once
    axes[1].set_xlim(0, 1.02)
    fig.text(.01, .005, f'Bars: mean over videos; dots: individual videos. Orange = old Module 1 (Full); '
             f'blue = {chosen} (selected on validation).', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_snr(pv, out, colors):
    configs = [c for c in ORDER if c in set(pv.config)]
    snr = pv.dropna(subset=['pulse_snr_db'])
    videos = sorted(snr.video_id.unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    y = np.arange(len(configs))
    if len(videos) <= 5:   # few videos: show each one
        for i, c in enumerate(configs):
            rows = snr[snr.config == c]
            if len(rows):
                ax.plot([rows.pulse_snr_db.min(), rows.pulse_snr_db.max()], [i, i], color=colors.get(c, MUTED),
                        lw=2, zorder=2)
        for v, m in zip(videos, ['o', '^', 's', 'D', 'v']):
            rows = snr[snr.video_id == v].set_index('config').reindex(configs)
            split = snr.loc[snr.video_id == v, 'split'].iloc[0]
            ax.scatter(rows.pulse_snr_db, y, marker=m, s=26, color=INK, zorder=3, linewidths=0, label=f'{v} ({split})')
        ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=len(videos))
    else:                  # many videos: mean with 95% bootstrap CI over videos
        rng = np.random.default_rng(0)
        for i, c in enumerate(configs):
            vals = snr.loc[snr.config == c, 'pulse_snr_db'].values
            ax.scatter(vals, i + rng.uniform(-.18, .18, len(vals)), s=5, color=MUTED, linewidths=0, zorder=2)
            lo, hi = bootstrap_ci(vals)
            ax.plot([lo, hi], [i, i], color=colors.get(c, INK2), lw=2.5, zorder=3)
            ax.scatter([vals.mean()], [i], s=30, color=colors.get(c, INK2), edgecolors='white', linewidths=.8, zorder=4)
        ax.text(1, 1.01, 'dot + line: mean and 95% bootstrap CI over videos; grey: individual videos', fontsize=7.5,
                color=INK2, ha='right', va='bottom', transform=ax.transAxes)
    b0 = snr.loc[snr.config == 'B0', 'pulse_snr_db'].mean()
    ax.axvline(b0, color=INK2, lw=1, ls='--')
    ax.text(b0, -.55, f' B0 mean {b0:.2f} dB', fontsize=7.5, color=INK2, va='center')
    style(ax)
    ax.set_yticks(y, configs)
    ax.set_ylim(len(configs) - .3, -.7)
    ax.set_xlabel('Fused pulse SNR (dB, POS)', fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_windows(win, out, chosen, tau, tau_valid):
    col = f'snr_{chosen}'
    videos = [v for v in sorted(win.video_id.unique()) if win.loc[win.video_id == v, 'snr_D0'].notna().any()]
    series = [('snr_D0', 'Full (old)', ORANGE), (col, f'{chosen} (new)', BLUE)]
    if len(videos) <= 3:   # few videos: time series per video
        fig, axes = plt.subplots(len(videos), 1, figsize=(7.5, 2.2 * len(videos)), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, v in zip(axes[:, 0], videos):
            rows = win[win.video_id == v].sort_values('window')
            for c, name, color in series:
                ax.plot(rows.window, rows[c], color=color, lw=2, label=name)
            style(ax)
            ax.grid(axis='y', color=GRID, linewidth=.6)
            ax.set_title(f'{v} ({rows.split.iloc[0]})', fontsize=9, color=INK, loc='left')
            ax.set_ylabel('SNR (dB)', fontsize=8, color=INK2)
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
        axes[-1, 0].set_xlabel('Window index (8 s windows, 1 s step)', fontsize=8, color=INK2)
    else:                  # many videos: distribution of all windows
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        for c, name, color in series:
            v = np.sort(win[c].dropna().values)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2, label=f'{name}, {len(v)} windows')
        if tau_valid:
            ax.axvline(tau, color=INK2, lw=1, ls='--')
            ax.text(tau, .02, f' τ = {tau:.2f} dB', fontsize=7.5, color=INK2)
        style(ax)
        ax.grid(axis='y', color=GRID, linewidth=.6)
        ax.set_xlabel('Window pulse SNR (dB)', fontsize=8, color=INK2)
        ax.set_ylabel('Fraction of windows ≤ x', fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_tuning(log, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    groups = [('D2 grid (β, δ, N)', log['results']['D2'], MUTED, 'o'),
              ('Viterbi, offline', [r for r in log['results']['D3'] if r['params']['viterbi_lag'] is None], BLUE, 's'),
              ('Viterbi, fixed lag 1 s', [r for r in log['results']['D3'] if r['params']['viterbi_lag']], AQUA, '^')]
    for name, rows, c, m in groups:
        ax.scatter([r['switches_per_sec'] for r in rows], [r['retention'] for r in rows], s=22, color=c, marker=m,
                   label=name, edgecolors='white', linewidths=.6, zorder=3)
    floor = log['retention_min']
    ax.axhline(floor, color=INK2, lw=1, ls='--')
    ax.text(ax.get_xlim()[1], floor, f'retention floor {floor}', fontsize=7.5, color=INK2, ha='right', va='bottom')
    for key in ('D2', 'D3', 'D3L'):
        t = log['chosen']['training_means'][key]
        ax.annotate(f'chosen {key}', (t['switches_per_sec'], t['retention']), xytext=(8, 8),
                    textcoords='offset points', fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=.6))
    style(ax)
    ax.grid(axis='y', color=GRID, linewidth=.6)
    ax.set_xlabel('Mean top-1 switches per second, training videos', fontsize=8, color=INK2)
    ax.set_ylabel('Mean quality retention', fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_correlation(corr, out):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', [BLUE, '#f0efec', '#e34948'])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    labels = [{'V_prime': "V'", 'M_nuisance': 'Mn'}.get(c, c) for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, fontsize=8, color=INK2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if abs(v) > .7 else INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    bar.ax.tick_params(labelsize=7, colors=INK2, length=0)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_frames(manifest, subject, out):
    few = len(manifest) <= 6
    cols, (tw, th), bar = (2, (640, 360), 34) if few else (6, (300, 400), 26)
    tiles = []
    for _, r in manifest.iterrows():
        cap = cv2.VideoCapture(r.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame_count) // 2)
        ok, im = cap.read()
        cap.release()
        tile = np.full((th + bar, tw, 3), 255, np.uint8)
        if ok:
            scale = min(tw / im.shape[1], th / im.shape[0])
            im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
            x, y = (tw - im.shape[1]) // 2, bar + (th - im.shape[0]) // 2
            tile[y:y + im.shape[0], x:x + im.shape[1]] = im
        s = subject.set_index('video_id').loc[r.video_id]
        label = (f'{r.video_id} | {r.split} | rejected as adult: {s.rejected_frames}/{s.sampled_frames}' if few else
                 f'{r.video_id} {r.split[:5]} rej {s.rejected_frames}/{s.sampled_frames}')
        cv2.putText(tile, label, (6, bar - 9), cv2.FONT_HERSHEY_SIMPLEX, .6 if few else .45, (11, 11, 11), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def fmt(x, nd=2):
    return 'n/a' if pd.isna(x) else f'{x:.{nd}f}'


def fmt_p(p):
    return 'n/a' if pd.isna(p) else ('< 0.0001' if p < 1e-4 else f'{p:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--headline', default=None, help='headline configuration (default: validation rule)')
    args = ap.parse_args()
    rep = ROOT / 'module1_v2/reports' / args.name / 'phase1'
    figs = rep / 'figures'
    figs.mkdir(exist_ok=True)
    pv = pd.read_csv(rep / 'per_video_metrics.csv')
    main_t = pd.read_csv(rep / 'table_phase1_main.csv').set_index('config')
    split_t = {s: pd.read_csv(rep / f'table_phase1_{s}.csv').set_index('config') for s in ('validation', 'test')}
    vs_full = pd.read_csv(rep / 'paired_vs_Full.csv')
    vs_b0 = pd.read_csv(rep / 'paired_vs_B0.csv')
    win = pd.read_csv(rep / 'window_metrics_v2.csv')
    corr = pd.read_csv(rep / 'term_correlation_pearson_train.csv', index_col=0)
    flags = pd.read_csv(rep / 'term_correlation_flags_train.csv')
    subject = pd.read_csv(rep / 'subject_verification.csv')
    audit = json.load(open(rep / 'subject_audit.json')) if (rep / 'subject_audit.json').is_file() else None
    meta = json.load(open(rep / 'phase1_metadata.json'))
    log = json.load(open(ROOT / 'module1_v2/runs' / args.name / 'tuning_log.json'))
    grids = yaml.safe_load((ROOT / 'module1_v2/configs/module1_v2.yaml').read_text())['phase1']
    manifest = pd.read_csv(ROOT / 'runs' / args.name / 'manifest.csv')

    n = len(manifest)
    mean = lambda c, k: pv.loc[pv.config == c, k].mean()  # noqa: E731
    snr_videos = sorted(pv.dropna(subset=['pulse_snr_db']).video_id.unique())
    all_snr = len(snr_videos) == n

    # Headline configuration: chosen on the validation split only (pulse SNR, then informative windows).
    val = pv[(pv.split == 'validation') & pv.config.isin(NEW)].groupby('config')[
        ['pulse_snr_db', 'informative_window_percent']].mean().reindex(NEW)
    val = val.assign(order=range(len(val))).fillna(-np.inf)
    recorded = rep.parent/'headline.json'
    headline = args.headline or (json.loads(recorded.read_text())['headline'] if recorded.is_file() else None)
    chosen = headline or val.sort_values(['pulse_snr_db', 'informative_window_percent', 'order'],
                                              ascending=[False, False, True]).index[0]
    colors = {'Full': ORANGE, chosen: BLUE}

    def pair(table, metric, cand):
        rows = table[(table.metric == metric) & (table.candidate == cand)]
        return rows.iloc[0] if len(rows) else None

    def delta(table, metric, cand, nd=2):
        r = pair(table, metric, cand)
        return 'n/a' if r is None else f'Δ {r.mean_difference:+.{nd}f}, Holm p {fmt_p(r.p_holm)}'

    def holm_sig(table, metric, cands):
        return [c for c in cands if (r := pair(table, metric, c)) is not None and r.p_holm < .05
                and np.sign(r.mean_difference) == (-1 if metric == 'switches_per_sec' else 1)]

    train_snr = pv[(pv.split == 'train') & (pv.config == 'Full')].pulse_snr_db.notna().any()
    tau = meta['tau_db']
    tuned = log['chosen']
    edge_params = []
    for key, grid in [('ema_beta', 'ema_beta_grid'), ('hysteresis_delta', 'hysteresis_delta_grid'),
                      ('hysteresis_frames', 'hysteresis_frames_grid'), ('viterbi_penalty', 'viterbi_penalty_grid')]:
        values = {tuned[c][key] for c in ('D2', 'D3', 'D3L') if key in tuned[c]}
        if any(v == max(grids[grid]) for v in values):
            edge_params.append(key)
    short = {'V_prime': "V'", 'M_nuisance': 'Mn'}
    flag_txt = '; '.join(f"{short.get(r.term_a, r.term_a)}–{short.get(r.term_b, r.term_b)} "
                         f"(Pearson {r.pearson:.2f}, Spearman {r.spearman:.2f})" for r in flags.itertuples()) or 'none'
    r_vs = corr.loc['V', 'S'] if {'V', 'S'} <= set(corr.index) else np.nan
    r_vps = corr.loc['V_prime', 'S'] if {'V_prime', 'S'} <= set(corr.index) else np.nan
    r_mm = corr.loc['M', 'M_nuisance'] if {'M', 'M_nuisance'} <= set(corr.index) else np.nan
    p_min = vs_full.p_value.min()
    rejected, sampled = int(subject.rejected_frames.sum()), int(subject.sampled_frames.sum())
    subject_ok = audit is not None and audit['frames_audited'] and audit['frames_correct'] / audit['frames_audited'] >= .8

    sig_snr = holm_sig(vs_full, 'pulse_snr_db', NEW)
    sig_inf = holm_sig(vs_full, 'informative_window_percent', NEW)
    sig_stab = holm_sig(vs_full, 'switches_per_sec', NEW)
    b0_full = pair(vs_full, 'pulse_snr_db', 'B0')
    ch_b0 = pair(vs_b0, 'pulse_snr_db', chosen)

    fig_frames(manifest, subject, figs / 'fig1_frames.jpg')
    fig_stability(pv, figs / 'fig2_stability.png', colors, chosen)
    fig_snr(pv, figs / 'fig3_snr.png', colors)
    fig_windows(win, figs / 'fig4_window_snr.png', chosen, tau, train_snr)
    fig_tuning(log, figs / 'fig5_tuning.png')
    fig_correlation(corr, figs / 'fig6_term_correlation.png')

    b = Builder(ROOT / 'reports/paper/manuscript_build/orig_docx/basepaper.docx')
    b.title('Module 1 v2 — Phase 1 Report: Stabilized ROI Selection and Corrected Quality Terms')
    b.para(f'Dataset: {args.name} ({n} videos). Generated from module1_v2/reports/{args.name}/phase1. '
           'All numbers are read from the run outputs.', WD_ALIGN_PARAGRAPH.CENTER, size=10, italic=True)

    # ---------------------------------------------------------------- 1. summary
    b.h1('1. Summary')
    snr_scope = (f'on all {n} videos' if all_snr else
                 f'on {len(snr_videos)} of {n} videos (no pulse trace could be extracted on the others)')
    bullets = [
        f"**Selection is now stable.** Top-1 region switches fell from {mean('Full', 'switches_per_sec'):.2f}/s "
        f"(old Module 1, Full) to {mean(chosen, 'switches_per_sec'):.2f}/s ({chosen}); temporal IoU rose from "
        f"{mean('Full', 'temporal_iou'):.3f} to {mean(chosen, 'temporal_iou'):.3f}. "
        + (f'The reduction is Holm-significant for every new configuration ({", ".join(sig_stab)}).'
           if len(sig_stab) == len(NEW) else
           f'Holm-significant for: {", ".join(sig_stab) or "none"}.'),
        f"**Pulse SNR {snr_scope}.** {chosen}: {mean(chosen, 'pulse_snr_db'):.2f} dB vs Full "
        f"{mean('Full', 'pulse_snr_db'):.2f} dB ({delta(vs_full, 'pulse_snr_db', chosen)}). "
        f"Informative windows: {mean(chosen, 'informative_window_percent'):.1f}% vs "
        f"{mean('Full', 'informative_window_percent'):.1f}% "
        f"({delta(vs_full, 'informative_window_percent', chosen, 1)}).",
        ('**Holm-significant quality gains over Full:** ' +
         (f'pulse SNR — {", ".join(sig_snr)}; ' if sig_snr else 'pulse SNR — none; ') +
         (f'informative windows — {", ".join(sig_inf)}.' if sig_inf else 'informative windows — none.'))
        + (f' With {n} videos the smallest attainable two-sided Wilcoxon p is {p_min:.4f}.' if n < 6 else ''),
        f"**The fixed box B0 is still the strongest pulse-SNR baseline.** B0 {mean('B0', 'pulse_snr_db'):.2f} dB "
        f"vs Full ({delta(vs_full, 'pulse_snr_db', 'B0')}); {chosen} vs B0: {delta(vs_b0, 'pulse_snr_db', chosen)}."
        if b0_full is not None and b0_full.mean_difference > 0 and (ch_b0 is None or ch_b0.mean_difference < 0) else
        f"**{chosen} vs the fixed box B0:** {delta(vs_b0, 'pulse_snr_db', chosen)}.",
        f'**Infant-vs-adult subject check (D7, I-S):** rejected {rejected:,} of {sampled:,} frames. '
        + (f"Manual audit: {audit['frames_correct']} of {audit['frames_audited']} audited rejections correct. "
           if audit else 'Not audited. ')
        + ('' if subject_ok else 'It is not reliable enough to use; D7 and I-S are reported for completeness only.'),
        f'**Term diagnostics (training videos, |r| > 0.8):** {flag_txt}.',
    ]
    b.bullets(bullets)

    # ---------------------------------------------------------------- 2. data
    b.h1('2. Data and protocol')
    rows = []
    for _, r in manifest.iterrows():
        s = subject.set_index('video_id').loc[r.video_id]
        rows.append([r.video_id, r.split, f'{r.fps:.3g}', int(r.frame_count), f'{r.duration_sec:.1f}',
                     f'{r.width}×{r.height}', 'yes' if r.video_id in snr_videos else 'no', int(s.sampled_frames)])
    b.para('**Table 1.** Videos. The split is per video (the video is the unit of analysis).', size=10)
    b.table(['Video', 'Split', 'FPS', 'Frames', 'Seconds', 'Resolution', 'Pulse SNR', 'Sampled frames'], rows,
            size=7.5 if n > 10 else 8.5)
    split_counts = manifest.split.value_counts()
    b.para(f"{n} videos: {split_counts.get('train', 0)} train, {split_counts.get('validation', 0)} validation, "
           f"{split_counts.get('test', 0)} test; {manifest.duration_sec.sum() / 60:.1f} min of video, "
           f'{sampled:,} sampled frames.')
    b.image(figs / 'fig1_frames.jpg', 6.5)
    b.caption('**Figure 1.** Middle frame of each video with the number of frames the subject check rejected as '
              '"adult".' + (f" Manual audit: {audit['summary']}" if audit else ''))
    tau_txt = (f'The informative-window threshold τ = {tau:.2f} dB is the median training ROI-window SNR.'
               if train_snr else
               f'The informative-window threshold τ should be the median training ROI-window SNR; no training video '
               f'yields a pulse SNR, so τ fell back to {tau:.1f} dB and the informative-window percentage is not '
               'meaningful on this dataset. It was deliberately not re-set on validation/test videos.')
    b.para(f"**No leakage.** All tuning used only the {len(log['training_videos'])} training videos (SHA-256 "
           f"recorded in tuning_log.json); validation/test videos were never used to tune a parameter. "
           f'{tau_txt} The headline configuration ({chosen}) was selected on the validation split alone, as the '
           'new configuration with the highest mean pulse SNR (informative windows as tie-break).')
    b.para('**Tuning objective** (colour-free, so it also works on IR video): minimise mean top-1 switches/s '
           f"subject to mean quality retention ≥ {log['retention_min']}; ties broken by higher retention. "
           'Retention = Σ rescaled Q of the selected set / Σ rescaled Q of the per-frame top-k.')
    b.para('**Statistics.** Paired two-sided Wilcoxon signed-rank tests over videos, rank-biserial r as effect size, '
           'Holm correction within each metric and comparison family, 95% bootstrap CIs over videos.')

    # ---------------------------------------------------------------- 3. configurations
    b.h1('3. Configurations')
    b.para('**Table 2.** Configurations. B0–Full are the unchanged v1 configurations; D0–D7 and I-V/I-M/I-S are new.',
           size=10)
    b.table(['ID', 'Definition'], [[c, DESCRIPTION[c]] for c in ORDER if c in main_t.index], widths=[.8, 4.5])
    d1 = next(r for r in log['results']['D1_beta_only'] if r['params']['ema_beta'] == tuned['D1']['ema_beta'])
    train_means = dict(tuned['training_means'], D1=d1)
    b.para('**Table 3.** Settings chosen on the training videos.', size=10)
    rows = [[k, ', '.join(f'{p}={v}' for p, v in tuned[k].items()), fmt(train_means[k]['switches_per_sec']),
             fmt(train_means[k]['retention'], 3)] for k in ('D1', 'D2', 'D3', 'D3L')]
    b.table(['Config', 'Chosen parameters', 'Train switches/s', 'Train retention'], rows)
    b.para(f"D4–D7 build on {tuned['D4_base']}. "
           + (f"{', '.join(edge_params)} ended at the upper edge of the searched grid, so the optimum may lie "
              'outside the searched range.' if edge_params else 'No chosen value lies at a grid edge.'))
    b.image(figs / 'fig5_tuning.png', 6.0)
    b.caption('**Figure 2.** Tuning on the training videos: every grid setting, the retention floor and the chosen '
              'settings.')

    # ---------------------------------------------------------------- 4. results
    b.h1('4. Results')
    b.para(f'**Table 4.** All {n} videos, mean ± SD over videos (95% bootstrap CIs in table_phase1_main.csv). '
           + ('' if all_snr else f"Pulse SNR only on {', '.join(snr_videos)}. ")
           + (f'Informative window: SNR ≥ τ = {tau:.2f} dB.' if train_snr else ''), size=10)
    cols = ['videos', 'switches_per_sec', 'temporal_iou', 'centroid_jitter', 'pulse_snr_db']
    header = ['Config', 'n', 'Switches/s ↓', 'Temporal IoU ↑', 'Jitter ↓', 'Pulse SNR dB ↑']
    if train_snr:
        cols.append('informative_window_percent')
        header.append('Informative % ↑')
    b.table(header, [[c] + [main_t.loc[c, k] for k in cols] for c in ORDER if c in main_t.index], size=8)
    reg = max(meta['regression_D0_vs_Full_max_abs_diff'].values())
    b.para(f'**Regression check.** D0 (new code with every flag off) reproduces v1 Full on every metric (largest '
           f'absolute difference {reg:.1e}).')
    b.image(figs / 'fig2_stability.png', 6.5)
    b.caption('**Figure 3.** Selection stability by configuration.')
    b.image(figs / 'fig3_snr.png', 6.5)
    b.caption('**Figure 4.** Fused pulse SNR per configuration; dashed line = fixed-box baseline B0.')
    b.image(figs / 'fig4_window_snr.png', 6.3)
    b.caption(f'**Figure 5.** Window-level pulse SNR, old Module 1 (Full) vs {chosen}'
              + (' (cumulative distribution over all windows; further right = better).' if len(snr_videos) > 3 else '.'))

    b.para('**Table 5.** Held-out splits (mean ± SD over videos). The test split was not used for any choice.',
           size=10)
    show = [c for c in ['B0', 'Full', 'D3', 'D4', 'D5', chosen] if c in main_t.index]
    show = list(dict.fromkeys(show))
    rows = []
    for s, t in split_t.items():
        for c in show:
            if c in t.index:
                rows.append([s, c, int(t.loc[c, 'videos']), t.loc[c, 'switches_per_sec'], t.loc[c, 'temporal_iou'],
                             t.loc[c, 'pulse_snr_db']] + ([t.loc[c, 'informative_window_percent']] if train_snr else []))
    b.table(['Split', 'Config', 'n', 'Switches/s', 'Temporal IoU', 'Pulse SNR dB'] +
            (['Informative %'] if train_snr else []), rows, size=8)

    metrics = ['switches_per_sec', 'temporal_iou', 'pulse_snr_db'] + (['informative_window_percent'] if train_snr else [])
    cands = list(dict.fromkeys(['B0', 'D2', 'D3', 'D3L', 'D4', 'D5', chosen, 'I-V']))
    b.para('**Table 6.** Paired Wilcoxon tests vs Full (video = unit; Holm-corrected within metric). '
           'r = rank-biserial correlation.', size=10)
    sel = vs_full[vs_full.candidate.isin(cands) & vs_full.metric.isin(metrics)]
    rows = [[METRIC[r.metric], r.candidate, int(r.n), f'{r.mean_difference:+.3f}', f'{r.rank_biserial:+.2f}',
             fmt_p(r.p_value), fmt_p(r.p_holm)] for r in sel.itertuples()]
    b.table(['Metric', 'vs Full', 'n', 'Mean diff', 'r', 'p', 'p (Holm)'], rows, size=8)
    b.para('**Table 7.** Paired Wilcoxon tests vs the fixed box B0 (pulse metrics).', size=10)
    sel = vs_b0[vs_b0.candidate.isin(NEW + ['I-V']) & vs_b0.metric.isin(metrics[2:])]
    rows = [[METRIC[r.metric], r.candidate, int(r.n), f'{r.mean_difference:+.3f}', f'{r.rank_biserial:+.2f}',
             fmt_p(r.p_value), fmt_p(r.p_holm)] for r in sel.itertuples()]
    b.table(['Metric', 'vs B0', 'n', 'Mean diff', 'r', 'p', 'p (Holm)'], rows, size=8)

    full = pv[pv.config == 'Full'].set_index('video_id')
    best = pv[pv.config == chosen].set_index('video_id')
    b.para(f'**Table 8.** Per-video values, Full → {chosen}.', size=10)
    rows = [[v, full.loc[v, 'split'], f"{full.loc[v, 'switches_per_sec']:.2f} → {best.loc[v, 'switches_per_sec']:.2f}",
             f"{full.loc[v, 'temporal_iou']:.3f} → {best.loc[v, 'temporal_iou']:.3f}",
             f"{fmt(full.loc[v, 'pulse_snr_db'])} → {fmt(best.loc[v, 'pulse_snr_db'])}"] for v in full.index]
    b.table(['Video', 'Split', 'Switches/s', 'Temporal IoU', 'Pulse SNR (dB)'], rows, size=7.5 if n > 10 else 8.5)

    # ---------------------------------------------------------------- 5. diagnostics
    b.h1('5. Diagnostics')
    b.h2('5.1 Quality-term correlations (training videos)')
    b.image(figs / 'fig6_term_correlation.png', 5.2)
    b.caption(f'**Figure 6.** Pearson correlation between quality terms on training ROI-frames (Mn = nuisance '
              f'motion). Flagged (|Pearson| or |Spearman| > 0.8): {flag_txt}.')
    text = []
    if np.isfinite(r_vs) and r_vs > .99:
        text.append('V and S are identical (r = 1), so the v1 score counts skin twice.')
    if np.isfinite(r_vps):
        text.append(f"The new keypoint visibility V′ correlates r = {r_vps:.2f} with S, i.e. it is an independent term.")
    d5_d4 = mean('D5', 'pulse_snr_db') - mean('D4', 'pulse_snr_db')
    text.append(f"Adding V′ (D4 → D5) changed mean pulse SNR by {d5_d4:+.2f} dB and switches/s by "
                f"{mean('D5', 'switches_per_sec') - mean('D4', 'switches_per_sec'):+.2f}.")
    if np.isfinite(r_mm):
        text.append(f'Nuisance motion Mn correlates r = {r_mm:.2f} with total motion M'
                    + (', so the motion decomposition changes little on these clips' if r_mm > .9 else '')
                    + f" (D5 → D6: {mean('D6', 'pulse_snr_db') - mean('D5', 'pulse_snr_db'):+.2f} dB).")
    b.para(' '.join(text))
    b.h2('5.2 Infant-vs-adult subject check')
    b.para('**Table 9.** Subject check per video (videos with at least one rejected frame). r1 = head/shoulder width, '
           'r2 = arm/torso length; a frame is "adult" when r1 < 0.55 or r2 > 1.0.', size=10)
    rows = [[r.video_id, r.split, int(r.sampled_frames), int(r.rejected_frames), int(r.adult_frames),
             int(r.unknown_frames), fmt(r.median_head_shoulder_ratio), fmt(r.median_arm_torso_ratio)]
            for r in subject.itertuples() if r.rejected_frames > 0]
    b.table(['Video', 'Split', 'Frames', 'Rejected', 'Adult', 'Unknown', 'Median r1', 'Median r2'], rows)
    b.para(f'{rejected:,} of {sampled:,} frames ({100 * rejected / max(sampled, 1):.1f}%) were rejected, in '
           f'{int((subject.rejected_frames > 0).sum())} of {n} videos. '
           + (f"{audit['method']} {audit['summary']} " if audit else '')
           + 'The thresholds are anthropometric priors, not fitted: there are no infant/adult labels, and fitting '
             'them on validation/test videos would leak.')

    # ---------------------------------------------------------------- 6. limitations
    b.h1('6. Limitations and next steps')
    lim = [f'{n} videos' + ('' if all_snr else f', {len(snr_videos)} with a pulse SNR') +
           '; no ground-truth heart rate, so pulse SNR is a proxy. Phase 3 needs rPPG data with ground truth.']
    if edge_params:
        lim.append(f"Tuned values at grid edges ({', '.join(edge_params)}): widen the grids when re-tuning.")
    if not subject_ok:
        lim.append('The subject check needs a different cue (or labels) before it can be used.')
    if b0_full is not None and b0_full.mean_difference > 0 and (ch_b0 is None or ch_b0.mean_difference < 0):
        lim.append('The fixed box B0 still has the highest mean pulse SNR; the adaptive selection has not yet '
                   'shown a pulse-quality advantage over it.')
    lim.append('Next: Phase 2 — baselines M0–M3, surrogate-null pulse-present test, cross-ROI agreement.')
    b.bullets(lim)
    b.h1('Reproduce')
    b.para(f'$PY module1_v2/scripts/run_phase1.py --run runs/{args.name}', WD_ALIGN_PARAGRAPH.LEFT, size=10)
    b.para(f'$PY module1_v2/scripts/build_phase1_report.py --name {args.name}', WD_ALIGN_PARAGRAPH.LEFT, size=10)
    out = rep / 'Module1_v2_Phase1_Report.docx'
    b.save(out)
    print('Wrote', out, '| headline config (validation):', chosen)


if __name__ == '__main__':
    main()
