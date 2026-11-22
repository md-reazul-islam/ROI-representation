"""Write module1_v2/RESULTS.md (final deliverable) from the Phase 1 / 2 / 5 outputs of one run.

Usage: python module1_v2/scripts/write_results_md.py --name final34_paper --v1-report reports/paper_final34_paper
"""
import argparse
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2/'manuscript'))
sys.path.insert(0, str(V2.parent/'reports/paper/manuscript_build'))
import results_v2 as R  # noqa: E402


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--v1-report', default='reports/paper_final34_paper')
    ap.add_argument('--headline', default=None)
    args = ap.parse_args()
    D = R.load(args.name, V2.parent/args.v1_report, R.decided_headline(V2/'reports'/args.name, args.headline))
    c, n = D['chosen'], len(D['manifest'])
    m1, m2, f1, b01, f2 = D['main1'], D['main2'], D['full1'], D['b01'], D['full2']
    s = D['manifest'].split.value_counts()
    lines = [f'# Module 1 v2 — Results ({args.name})', '',
             f"{n} videos ({s.get('train', 0)} train / {s.get('validation', 0)} validation / {s.get('test', 0)} test, "
             f"video-level split). No physiological reference exists for these videos: every pulse number is a label-free "
             f"proxy. Headline configuration: **{c}** ({R.headline_sentence(D)}). All numbers: mean ± SD over videos.", '',
             '## Phase 1 — stabilization and corrected terms', '',
             md_table(['Config', 'Switches/s', 'Temporal IoU', 'Pulse SNR (dB)', 'Informative %'],
                      [[x, m1.loc[x, 'switches_per_sec'], m1.loc[x, 'temporal_iou'], m1.loc[x, 'pulse_snr_db'],
                        m1.loc[x, 'informative_window_percent']] for x in m1.index]), '',
             f"- D0 (all flags off) = v1 Full: max |diff| {max(D['meta1']['regression_D0_vs_Full_max_abs_diff'].values()):.1e}.",
             f"- {c} vs Full: switches {R.stat(f1, 'switches_per_sec', c, '/s')}; SNR {R.stat(f1, 'pulse_snr_db', c, ' dB')}; "
             f"informative {R.stat(f1, 'informative_window_percent', c, ' pp', 1)}.",
             f"- {c} vs B0: SNR {R.stat(b01, 'pulse_snr_db', c, ' dB')}; informative "
             f"{R.stat(b01, 'informative_window_percent', c, ' pp', 1)}.",
             f"- Term audit (train, |r| > 0.8): " + '; '.join(f'{r.term_a}-{r.term_b} r={r.pearson:.2f}'
                                                          for r in D['flags'].itertuples()) + '.',
             f"- Subject check: {int(D['subject'].rejected_frames.sum())} of {D['sampled_frames']} frames rejected"
             + (f"; audit {D['audit']['frames_correct']}/{D['audit']['frames_audited']} correct — not usable." if D['audit'] else '.'),
             '', '## Phase 2 — baselines and validity tests', '',
             md_table(['Config', 'SNR (dB)', 'Informative %', 'Pulse-present % (block-shuffle null)', 'Agreement %'],
                      [[x, m2.loc[x, 'pulse_snr_db'], m2.loc[x, 'informative_window_percent'],
                        m2.loc[x, 'present_shuffle_percent'], m2.loc[x, 'agreement_percent']] for x in m2.index]), '',
             f"- Chance level of the pulse-present test: {100 - D['p2']['null_percentile']:.0f}%. "
             f"{c} vs Full pulse-present: {R.stat(f2, 'present_shuffle_percent', c, ' pp', 1)}. The per-channel "
             'phase-randomization null is not used: on synthetic video it rejected a true pulse in every window '
             '(phase2/null_validation.json).',
             '- M3 is an oracle (same-window best region); its high SNR is selection bias, as its pulse-present rate shows.',
             '']
    if D['phase5']:
        st = D['stride']
        lines += ['## Phase 5 — efficiency', '',
                  md_table(['Config', 'Stride', 'Eff. FPS', 'Projected h', 'Speed-up', 'Switches/s', 'SNR (dB)', 'Informative %'],
                           [[r.config, int(r.stride), f'{r.effective_fps:.1f}', f'{r.projected_hours:.1f}',
                             f'{r.speedup:.2f}', f'{r.switches_per_sec:.2f}', f'{r.pulse_snr_db:.2f}',
                             f'{r.informative_window_percent:.1f}'] for r in st.itertuples()]), '',
                  f"- Optical flow per frame pair: Farnebäck {D['flow'].farneback_ms.mean():.0f} ms vs Lucas-Kanade "
                  f"{D['flow'].lucas_kanade_ms.mean():.0f} ms. Lighter pose model: not evaluated (no checkpoint).", '']
    lines += ['## Not done', '',
              '- Phases 3-4 (HR accuracy on rPPG data with reference, learned reliability): no reference data available.', '',
              '## Files', '',
              f'- Phase 1: `module1_v2/reports/{args.name}/phase1/` (+ `Module1_v2_Phase1_Report.docx`)',
              f'- Phase 2: `module1_v2/reports/{args.name}/phase2/`',
              f'- Phase 5: `module1_v2/reports/{args.name}/phase5/`',
              f'- Manuscript: `module1_v2/reports/{args.name}/Module1_v2_Manuscript.docx`',
              f'- Old Module 1: `{args.v1_report}/` (+ `Module1_Manuscript_completed.docx`)',
              '- How to run: `MODULE1_OLD_AND_NEW_V2.md`, `RUN_MODULE1_OLD_AND_NEW_V2.ipynb`']
    out = V2/'RESULTS.md'
    out.write_text('\n'.join(lines) + '\n')
    print('Wrote', out)


if __name__ == '__main__':
    main()
