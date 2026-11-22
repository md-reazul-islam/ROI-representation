"""Phase 1 figures (manuscript Figures 2-4) redrawn for the UPDATED manuscript only.

The Phase 1 report (module1_v2/scripts/build_phase1_report.py) drew them when D3 was the headline configuration, so
they highlight D3 and call the baseline "old Module 1". Here the same drawing functions are called, read-only, with the
headline configuration of the paper (D5) highlighted; text is renamed by the updated builder's matplotlib hook.
The tuning figure is redrawn with non-overlapping labels. The Phase 1 PNGs are not touched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'module1_v2/scripts'))
import build_phase1_report as P1  # noqa: E402

LABEL_OFFSET = {'D2': (8, 8), 'D3': (4, -18), 'D3L': (-6, 26)}   # points; D3 and D3L lie close together


def fig_tuning(log, out):
    plt, INK, INK2, GRID, MUTED = P1.plt, P1.INK, P1.INK2, P1.GRID, P1.MUTED
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    groups = [('D2 grid (β, δ, N)', log['results']['D2'], MUTED, 'o'),
              ('Viterbi, offline', [r for r in log['results']['D3'] if r['params']['viterbi_lag'] is None], P1.BLUE, 's'),
              ('Viterbi, fixed lag 1 s', [r for r in log['results']['D3'] if r['params']['viterbi_lag']], P1.AQUA, '^')]
    for name, rows, c, m in groups:
        ax.scatter([r['switches_per_sec'] for r in rows], [r['retention'] for r in rows], s=22, color=c, marker=m,
                   label=name, edgecolors='white', linewidths=.6, zorder=3)
    floor = log['retention_min']
    ax.axhline(floor, color=INK2, lw=1, ls='--')
    ax.text(ax.get_xlim()[1], floor, f'retention floor {floor}', fontsize=7.5, color=INK2, ha='right', va='bottom')
    for key in ('D2', 'D3', 'D3L'):
        t = log['chosen']['training_means'][key]
        ax.annotate(f'chosen {key}', (t['switches_per_sec'], t['retention']), xytext=LABEL_OFFSET[key],
                    textcoords='offset points', fontsize=7.5, color=INK, va='center',
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=.6))
    P1.style(ax)
    ax.grid(axis='y', color=GRID, linewidth=.6)
    ax.set_xlabel('Mean top-1 switches per second, training videos', fontsize=8, color=INK2)
    ax.set_ylabel('Mean quality retention', fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make(name, chosen, figs):
    """Redraw the three figures into `figs`; return {original file name: new path}."""
    import pandas as pd
    rep = ROOT/'module1_v2/reports'/name/'phase1'
    pv = pd.read_csv(rep/'per_video_metrics.csv')
    log = json.load(open(ROOT/'module1_v2/runs'/name/'tuning_log.json'))
    colors = {'Full': P1.ORANGE, chosen: P1.BLUE}
    out = {'fig2_stability.png': figs/'fig3_stability_updated.png', 'fig3_snr.png': figs/'fig4_snr_updated.png',
           'fig5_tuning.png': figs/'fig2_tuning_updated.png'}
    P1.fig_stability(pv, out['fig2_stability.png'], colors, chosen)
    P1.fig_snr(pv, out['fig3_snr.png'], colors)
    fig_tuning(log, out['fig5_tuning.png'])
    return out
