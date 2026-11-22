"""Revision items 1, 2 and 5, read-only on the saved Phase 1 outputs. Writes ONLY to
module1_v2/results/revision_items1-4/.

1  training-split (21 videos) mean top-1 switches/s for D5 and D6, and whether they differ by > 0.01/s
2  D5 paired Wilcoxon + rank-biserial r + Holm, recomputed with the paper's families (each metric: all configurations
   vs Full; all configurations vs B0), checked against the saved paired_vs_*.csv
5  Figures 3 (fig2_stability) and 4 (fig3_snr) re-drawn with D5 in blue instead of D3

Usage: python module1_v2/scripts/revision_items_125.py --name final34_paper
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'scripts', ROOT/'src'):
    sys.path.insert(0, str(p))
import build_phase1_report as br                       # noqa: E402  (read-only reuse of the figure code)
from module2.ablation import holm, paired_test         # noqa: E402

OUT = V2/'results'/'revision_items1-4'
HIGHLIGHT = 'D5'


def fig_stability(pv, out, colors, chosen):
    """= build_phase1_report.fig_stability, only the footnote no longer says 'selected on validation'."""
    configs = [c for c in br.ORDER if c in set(pv.config)]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.2), sharey=True)
    for ax, metric, label in zip(axes, ['switches_per_sec', 'temporal_iou'],
                                 ['Top-1 region switches per second (lower = steadier)',
                                  'Temporal IoU of selected region (higher = steadier)']):
        y = np.arange(len(configs))
        means = [pv.loc[pv.config == c, metric].mean() for c in configs]
        ax.barh(y, means, height=.55, color=[colors.get(c, br.MUTED) for c in configs])
        for i, c in enumerate(configs):
            vals = pv.loc[pv.config == c, metric].dropna()
            ax.scatter(vals, np.full(len(vals), i), s=8, color=br.INK, alpha=.6, zorder=3, linewidths=0)
        br.style(ax)
        ax.set_xlabel(label, fontsize=8, color=br.INK2)
        ax.set_yticks(y, configs)
    axes[0].invert_yaxis()
    axes[1].set_xlim(0, 1.02)
    fig.text(.01, .005, f'Bars: mean over videos; dots: individual videos. Orange = old Module 1 (Full); '
             f'blue = {chosen}.', fontsize=7.5, color=br.INK2)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    name = ap.parse_args().name
    p1 = V2/'reports'/name/'phase1'
    OUT.mkdir(parents=True, exist_ok=True)
    pv = pd.read_csv(p1/'per_video_metrics.csv')
    train_ids = json.loads((V2/'runs'/name/'tuning_log.json').read_text())['training_videos']

    # ---- item 1
    tr = pv[(pv.split == 'train') & pv.config.isin(['D5', 'D6'])]
    wide = tr.pivot_table(index='video_id', columns='config', values='switches_per_sec')
    assert sorted(wide.index) == sorted(train_ids) and len(wide) == 21, 'training split mismatch'
    d = wide.D6 - wide.D5
    item1 = pd.DataFrame([dict(
        videos=len(wide), D5_mean_switches_per_sec=wide.D5.mean(), D5_sd=wide.D5.std(ddof=1),
        D6_mean_switches_per_sec=wide.D6.mean(), D6_sd=wide.D6.std(ddof=1),
        mean_difference_D6_minus_D5=d.mean(), abs_difference=abs(d.mean()),
        differs_by_more_than_0_01=bool(abs(d.mean()) > .01),
        videos_with_identical_rate=int((d.abs() < 1e-12).sum()), max_abs_per_video_difference=d.abs().max())])
    item1.to_csv(OUT/'item1_train_switches_D5_D6.csv', index=False)
    wide.assign(difference=d).to_csv(OUT/'item1_train_switches_D5_D6_per_video.csv')

    # ---- item 2 (families exactly as in run_v2_evaluation.paired: per metric, every other config vs reference)
    saved = {r: pd.read_csv(p1/f'paired_vs_{r}.csv') for r in ('Full', 'B0')}
    rows, fam_rows = [], []
    for ref, metrics in (('Full', ['pulse_snr_db', 'informative_window_percent']), ('B0', ['pulse_snr_db'])):
        for m in metrics:
            fam = saved[ref][saved[ref].metric == m].candidate.tolist()
            # exact ties: differences below 1e-9 (float residue, e.g. D0 = Full to 1.8e-15) count as zero
            w = pv.pivot_table(index='video_id', columns='config', values=m).round(9)
            block = [dict(candidate=c, **paired_test(w[ref], w[c])) for c in fam]
            for r, adj in zip(block, holm([b['p_value'] for b in block])):
                r['p_holm'] = adj
            r = next(b for b in block if b['candidate'] == HIGHLIGHT)
            s = saved[ref][(saved[ref].metric == m) & (saved[ref].candidate == HIGHLIGHT)].iloc[0]
            rows.append(dict(comparison=f'{HIGHLIGHT} vs {ref}', metric=m, family=f'all configurations vs {ref}',
                             family_size=len(fam), n_videos=int(s.n), mean_difference=s.mean_difference,
                             median_difference=s.median_difference, wilcoxon_p=s.p_value,
                             rank_biserial_r=s.rank_biserial, p_holm_as_in_paper=s.p_holm,
                             p_holm_exact_ties=r['p_holm'], wilcoxon_p_exact_ties=r['p_value'],
                             rank_biserial_exact_ties=r['rank_biserial'],
                             holm_significant_0_05=bool(max(s.p_holm, r['p_holm']) < .05)))
            fam_rows += [dict(reference=ref, metric=m, candidate=b['candidate'], p_exact_ties=b['p_value'],
                              p_holm_exact_ties=b['p_holm'],
                              p_as_in_paper=saved[ref][(saved[ref].metric == m) & (saved[ref].candidate == b['candidate'])].p_value.iloc[0],
                              p_holm_as_in_paper=saved[ref][(saved[ref].metric == m) & (saved[ref].candidate == b['candidate'])].p_holm.iloc[0])
                         for b in block]
    pd.DataFrame(rows).to_csv(OUT/'item2_D5_paired_tests.csv', index=False)
    pd.DataFrame(fam_rows).to_csv(OUT/'item2_holm_families.csv', index=False)

    # ---- item 5
    figs = OUT/'figures'
    figs.mkdir(exist_ok=True)
    colors = {'Full': br.ORANGE, HIGHLIGHT: br.BLUE}
    fig_stability(pv, figs/'fig3_stability_D5.png', colors, HIGHLIGHT)
    br.fig_snr(pv, figs/'fig4_snr_D5.png', colors)
    pd.DataFrame([dict(figure='Figure 3', file='figures/fig3_stability_D5.png', source='phase1/per_video_metrics.csv',
                       highlighted_blue=HIGHLIGHT, orange='Full'),
                  dict(figure='Figure 4', file='figures/fig4_snr_D5.png', source='phase1/per_video_metrics.csv',
                       highlighted_blue=HIGHLIGHT, orange='Full')]).to_csv(OUT/'item5_figures.csv', index=False)
    print(item1.T.to_string())
    print(pd.DataFrame(rows).T.to_string())


if __name__ == '__main__':
    main()
