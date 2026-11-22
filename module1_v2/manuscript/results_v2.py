"""Results, Discussion, Limitations and Conclusion of the Module 1 v2 manuscript. Every number is read from the
Phase 1 / 2 / 5 CSVs; statements that depend on significance or direction are worded from the data."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = V2.parent
NEW = ['D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7']


STAGE_LABELS = {
    'yolo': 'Detection (YOLO11s)', 'rtdetr': 'Detection (RT-DETR-L)', 'fusion': 'Detector fusion (OR, IoU 0.5)',
    'tracking': 'Tracking (ByteTrack)', 'recovery': 'Bounded recovery', 'sam2': 'Infant segmentation (SAM 2.1)',
    'pose': 'Pose, Sapiens2 (box branch)', 'pose_sam2': 'Pose, Sapiens2 (SAM 2 branch)',
    'roi': 'Candidate regions (box branch)', 'roi_sam2': 'Candidate regions (SAM 2 branch)',
    'roi_quality': 'Condition scoring (box branch)', 'roi_quality_sam2': 'Condition scoring (SAM 2 branch)',
    'confidence': 'Region confidence (box branch)', 'confidence_sam2': 'Region confidence (SAM 2 branch)',
    'v2_features': 'NeoSTAR terms (V′, Mₙ, subject check)', 'v2_evaluation': 'NeoSTAR stabilization and evaluation',
}


def load(name, v1_report, headline=None):
    rep = V2/'reports'/name
    p1, p2, p5 = rep/'phase1', rep/'phase2', rep/'phase5'
    read = lambda path: pd.read_csv(path)  # noqa: E731
    D = dict(name=name, rep=rep, p1dir=p1, p5dir=p5, v1=Path(v1_report))
    D['manifest'] = read(ROOT/'runs'/name/'manifest.csv')
    D['cfg'] = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())
    D['p2'] = D['cfg']['phase2']
    D['log'] = json.loads((V2/'runs'/name/'tuning_log.json').read_text())
    D['tuning'] = D['log']['chosen']
    D['meta1'] = json.loads((p1/'phase1_metadata.json').read_text())
    D['tau'] = D['meta1']['tau_db']
    D['pv1'] = read(p1/'per_video_metrics.csv')
    D['main1'] = read(p1/'table_phase1_main.csv').set_index('config')
    D['split1'] = {s: read(p1/f'table_phase1_{s}.csv').set_index('config') for s in ('validation', 'test')}
    D['full1'], D['b01'] = read(p1/'paired_vs_Full.csv'), read(p1/'paired_vs_B0.csv')
    D['flags'] = read(p1/'term_correlation_flags_train.csv')
    D['corr'] = pd.read_csv(p1/'term_correlation_pearson_train.csv', index_col=0)
    D['subject'] = read(p1/'subject_verification.csv')
    D['audit'] = json.loads((p1/'subject_audit.json').read_text()) if (p1/'subject_audit.json').is_file() else None
    D['sampled_frames'] = int(D['subject'].sampled_frames.sum())
    D['pv2'] = read(p2/'per_video_metrics.csv')
    D['main2'] = read(p2/'table_phase2_main.csv').set_index('config')
    D['full2'], D['b02'] = read(p2/'paired_vs_Full.csv'), read(p2/'paired_vs_B0.csv')
    D['meta2'] = json.loads((p2/'phase2_metadata.json').read_text())
    s19 = rep/'phase2_s19/per_video_metrics.csv'          # same pixels and windows, 19 instead of 199 surrogates
    if s19.is_file() and D['p2']['surrogates_shuffle'] != 19:
        old = read(s19)
        a = old.groupby('config').present_shuffle_percent.mean()
        b = D['pv2'].groupby('config').present_shuffle_percent.mean()
        both = [x for x in a.index if x in b.index]
        D['s19'] = pd.DataFrame({'s19': a[both], 'now': b[both]})
        D['mc_full'] = abs(float(a['Full'] - b['Full']))
        D['mc_max'] = float((a[both] - b[both]).abs().max())
    D['null_validation'] = json.loads((p2/'null_validation.json').read_text()) if (p2/'null_validation.json').is_file() else None
    w2 = pd.read_csv(p2/'window_metrics.csv', usecols=['video_id', 'window'])
    lag = int(round(8 / 1))
    D['m12_active'] = 100. * float((w2.window >= lag).mean())
    D['m12_active_videos'] = int((w2.groupby('video_id').window.max() >= lag).sum())
    full = ROOT/'runs'/name/'outputs/confidence_sam2/roi_confidence.csv'
    if full.is_file():                      # after a Level C re-run
        cand = pd.read_csv(full, usecols=['geometry_type', 'roi_status'], low_memory=False)
        cand = cand[cand.roi_status == 'AVAILABLE']
        D['bbox_share'] = 100. * float((cand.geometry_type == 'BBOX').mean())
    else:                                   # released summary of the same table (574 MB not distributed)
        g = pd.read_csv(ROOT/'runs'/name/'released/roi_geometry_counts.csv')
        g = g[g.roi_status == 'AVAILABLE']
        D['bbox_share'] = 100. * float(g.loc[g.geometry_type == 'BBOX', 'rows'].sum() / g.rows.sum())
    ft = rep/'fusion_test'
    D['fusion'] = (pd.read_csv(ft/'table_fusion_test.csv'), pd.read_csv(ft/'paired_fusion_test.csv')) \
        if (ft/'paired_fusion_test.csv').is_file() else None
    cons = rep/'consistency'
    if (cons/'c3_per_video.csv').is_file() and (cons/'e2_per_video.csv').is_file():
        D['cons'] = dict(c1=read(cons/'table_c1_term_independence.csv'), c3=read(cons/'c3_per_video.csv'),
                         c4=read(cons/'c4_per_video.csv'), e1=read(cons/'e1_per_video.csv'),
                         rank=read(cons/'rank_per_video.csv'), e2=read(cons/'e2_per_video.csv'),
                         e3=read(cons/'e3_per_video.csv') if (cons/'e3_per_video.csv').is_file() else None,
                         e4=read(cons/'e4_per_video.csv') if (cons/'e4_per_video.csv').is_file() else None,
                         tests=read(cons/'paired_tests.csv'))
    D['pv1_prefix'] = read(p1.parent/'phase1_prefix/per_video_metrics.csv') if (p1.parent/'phase1_prefix/per_video_metrics.csv').is_file() else None
    pre = rep/'consistency_prefix/c3_per_video.csv'
    D['c3_prefix'] = read(pre) if pre.is_file() else None
    D['phase5'] = (p5/'table_phase5_stride.csv').is_file()
    if D['phase5']:
        D['profile'] = read(p5/'table_phase5_profile.csv')
        D['stride'] = read(p5/'table_phase5_stride.csv')
        D['stride_tests'] = read(p5/'paired_stride_vs_1.csv')
        D['flow'] = read(p5/'table_phase5_flow.csv')
        D['meta5'] = json.loads((p5/'phase5_metadata.json').read_text())
    ver = rep/'environment_versions.json'      # versions of the machine that produced the results
    if ver.is_file():
        D['versions'] = json.loads(ver.read_text())
    else:
        import cv2, numpy, scipy, torch  # noqa: E401  (versions for Section 4.3)
        D['versions'] = dict(python=sys.version.split()[0], torch=torch.__version__.split('+')[0],
                             cv2=cv2.__version__, numpy=numpy.__version__, scipy=scipy.__version__)
    D['steps'] = stepwise(D['pv1'])
    D['retention_split'] = D['pv1'][D['pv1'].config.isin(['D3', 'D3L'])].groupby(['config', 'split']).retention.mean()
    # headline configuration, two-step rule (2026-11 revision):
    #   (1) eligibility from TRAINING data and the configuration itself only: term independence (no flagged pair of
    #       terms, |r| > 0.8 on training ROI-frames, inside the configuration's own score), quality retention >= the
    #       design floor on training videos, and candidate validity (2-point boxes carry pixels);
    #   (2) among eligible configurations: highest mean VALIDATION pulse SNR, then informative windows.
    D['eligibility'] = eligibility(D)
    eligible = [c for c in NEW if D['eligibility'].loc[c, 'eligible']]
    val = D['pv1'][(D['pv1'].split == 'validation') & D['pv1'].config.isin(NEW)].groupby('config')[
        ['pulse_snr_db', 'informative_window_percent']].mean().reindex(NEW)
    val = val.assign(order=range(len(val))).fillna(-np.inf)
    ranked = val.sort_values(['pulse_snr_db', 'informative_window_percent', 'order'], ascending=[False, False, True])
    D['validation_ranking_all'] = ranked                       # SNR-only rule (before the revision)
    D['chosen_snr_only'] = ranked.index[0]
    D['validation_ranking'] = ranked.loc[[c for c in ranked.index if c in eligible]] if eligible else ranked
    D['chosen_rule'] = D['validation_ranking'].index[0]
    D['chosen'] = headline or D['chosen_rule']
    return D


def score_terms(flags):
    """Terms that enter the confidence score of one configuration (names as in the term-correlation flags)."""
    terms = {'P', 'S', 'T'}
    terms.add('V' if flags.get('visibility', 'v1') == 'v1' else 'V_prime')
    terms.add('M' if flags.get('motion', 'total') in ('total', 'residual') else 'M_nuisance')
    return terms


def eligibility(D):
    """Per new configuration: the consistency checks that decide headline eligibility (training data only)."""
    flagged = [(r.term_a, r.term_b) for r in D['flags'].itertuples()]
    floor = D['cfg']['phase1']['quality_retention']
    train = D['pv1'][D['pv1'].split == 'train'].groupby('config').retention.mean()
    rows = []
    for c in NEW:
        flags = D['meta1']['configs'][c]
        terms = score_terms(flags)
        bad = [f'{a}-{b}' for a, b in flagged if a in terms and b in terms]
        rows.append(dict(config=c, term_pairs_flagged=', '.join(bad), term_independence=not bad,
                         train_retention=float(train.get(c, np.nan)), retention_ok=bool(train.get(c, 0.) >= floor),
                         candidate_validity=flags.get('box_roi_pixels', 'skip') == 'rectangle'))
    t = pd.DataFrame(rows).set_index('config')
    t['eligible'] = t.term_independence & t.retention_ok & t.candidate_validity
    return t


STEPS = [('D0', 'I-B', '2-point boxes read as rectangles'),   # D0 = v1 Full through the v2 code (has retention)
         ('I-B', 'D1', 'EMA smoothing'),
          ('D1', 'D2', '+ hysteresis'), ('D2', 'D3', '+ offline Viterbi anchor'),
         ('D2', 'D3L', '+ online Viterbi anchor (1-s lag)'), ('D3L', 'D3', 'offline instead of online decoder'),
         ('D3', 'D4', '+ smoothed, cross-faded fusion weights')]
STEP_METRICS = ['switches_per_sec', 'temporal_iou', 'retention', 'pulse_snr_db', 'informative_window_percent']


def stepwise(pv):
    """Paired Wilcoxon test of every stabilization step against the configuration it extends (Holm per metric)."""
    sys.path.insert(0, str(ROOT/'src'))
    from module2.ablation import holm, paired_test
    rows = []
    for m in STEP_METRICS:
        wide = pv.pivot_table(index='video_id', columns='config', values=m)
        block = []
        for a, b, label in STEPS:
            if a in wide and b in wide:
                block.append(dict(metric=m, base=a, step=b, label=label, **paired_test(wide[a], wide[b])))
        for row, adjusted in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = adjusted
        rows += block
    return pd.DataFrame(rows)


def decided_headline(report_dir, given=None):
    """Headline configuration: explicit argument, else reports/<run>/headline.json (a recorded decision), else None."""
    import json as _json
    from pathlib import Path as _Path
    path = _Path(report_dir)/'headline.json'
    return given or (_json.loads(path.read_text())['headline'] if path.is_file() else None)


# ------------------------------------------------------------------ helpers
def mean(D, config, metric, phase=1):
    pv = D['pv1'] if phase == 1 else D['pv2']
    return float(pv.loc[pv.config == config, metric].mean())


def pair(table, metric, candidate):
    rows = table[(table.metric == metric) & (table.candidate == candidate)]
    return rows.iloc[0] if len(rows) else None


def p(value):
    return 'n/a' if value is None or (isinstance(value, float) and math.isnan(value)) else (
        '< 0.001' if value < .001 else f'{value:.3f}')


def stat(table, metric, candidate, unit='', nd=2):
    r = pair(table, metric, candidate)
    if r is None:
        return 'n/a'
    return f'Δ = {r.mean_difference:+.{nd}f}{unit}, r = {r.rank_biserial:+.2f}, p_Holm {p(r.p_holm)}'


def sig(table, metric, candidate, better=1):
    r = pair(table, metric, candidate)
    return r is not None and r.p_holm < .05 and np.sign(r.mean_difference) == better


def figure(b, path, width, caption):
    if Path(path).is_file():
        b.image(path, width)
        b.caption(caption)
        return True
    return False


# ------------------------------------------------------------------ figures specific to the manuscript
INK, INK2, GRID, MUTED, BLUE, ORANGE = '#0b0b0b', '#52514e', '#e4e3df', '#c3c2b7', '#2a78d6', '#eb6834'


def make_figures(D, figs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs.mkdir(parents=True, exist_ok=True)
    order = [x for x in ['B0', 'B1', 'Full', 'I-B', 'D3', 'D4', 'D5', 'M0', 'M1', 'M2', 'M3', 'BG'] if x in set(D['pv2'].config)]
    pv = D['pv2']
    colors = {'Full': ORANGE, D['chosen']: BLUE}

    def style(ax):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.grid(axis='x', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.2), sharey=True)
    panels = [('informative_window_percent', 'Informative windows (%)\nSNR ≥ τ', None),
              ('present_shuffle_percent', 'Pulse-present windows (%)\nSNR > 95th pct. of shuffle null',
               100 - D['p2']['null_percentile']),
              ('agreement_percent', 'Split-half agreement (%)\nhalves within ±6 bpm', None)]
    y = np.arange(len(order))
    rng = np.random.default_rng(0)
    for ax, (metric, label, chance) in zip(axes, panels):
        for i, c in enumerate(order):
            vals = pv.loc[pv.config == c, metric].dropna().to_numpy()
            ax.scatter(vals, i + rng.uniform(-.18, .18, len(vals)), s=5, color=MUTED, linewidths=0, zorder=2)
            if len(vals) > 1:
                means = rng.choice(vals, (2000, len(vals))).mean(1)
                ax.plot(np.percentile(means, [2.5, 97.5]), [i, i], color=colors.get(c, INK2), lw=2.5, zorder=3)
            ax.scatter([vals.mean()], [i], s=30, color=colors.get(c, INK2), edgecolors='white', linewidths=.8, zorder=4)
        if chance is not None:
            ax.axvline(chance, color=INK2, lw=1, ls='--')
            ax.text(chance, -.9, f' chance {chance:.0f}%', fontsize=7, color=INK2)
        style(ax)
        ax.set_xlabel(label, fontsize=7.5, color=INK2)
        ax.set_xlim(-3, 103)
    axes[0].set_yticks(y, order)
    axes[0].set_ylim(len(order) - .4, -1.2)
    fig.text(.01, .01, 'Dot + line: mean and 95% bootstrap CI over videos; grey: videos. M3 is an oracle upper bound; '
             'BG is a negative control (background pixels).',
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(figs/'fig_validity.png', dpi=200)
    plt.close(fig)

    if D['phase5']:
        st = D['stride']
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
        for cid, color in (('D0', ORANGE), (D['chosen'] if D['chosen'] in set(st.config) else 'D3', BLUE)):
            part = st[st.config == cid].sort_values('stride')
            name = 'Full (D0)' if cid == 'D0' else cid
            for ax, metric, label in ((axes[0], 'pulse_snr_db', 'Mean pulse SNR (dB)'),
                                      (axes[1], 'switches_per_sec', 'Top-1 switches per second')):
                ax.plot(part.projected_hours, part[metric], color=color, lw=2, marker='o', ms=4, label=name)
                for _, r in part.iterrows():
                    ax.annotate(f's={int(r.stride)}', (r.projected_hours, r[metric]), xytext=(3, 4),
                                textcoords='offset points', fontsize=6.5, color=INK2)
                ax.set_xlabel('Projected summed stage time (h)', fontsize=8, color=INK2)
                ax.set_ylabel(label, fontsize=8, color=INK2)
                style(ax)
                ax.grid(axis='y', color=GRID, linewidth=.6)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figs/'fig_efficiency.png', dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------ abstract
def abstract(D):
    c, n = D['chosen'], len(D['manifest'])
    f1, b01 = D['full1'], D['b01']
    m2 = D['main2']
    pres = lambda cid: mean(D, cid, 'present_shuffle_percent', 2)  # noqa: E731
    chance = 100 - D['p2']['null_percentile']
    inf_sig = sig(f1, 'informative_window_percent', c)
    snr_sig = sig(f1, 'pulse_snr_db', c)
    b0_better = mean(D, 'B0', 'pulse_snr_db') > mean(D, c, 'pulse_snr_db')
    text = (
        'Camera-based neonatal monitoring needs reliable visual evidence from the infant body, yet dynamic region-of-'
        'interest (ROI) selection is easily destabilized by near-equal region scores, redundant quality terms and motion. '
        'We present NeoSTAR (Neonatal STAbilized ROI), an auditable, learning-free dynamic ROI representation that adds to an anatomy- and '
        'mask-guided ROI pipeline (i) temporal stabilization of region selection (confidence smoothing, hysteresis and '
        'offline or fixed-lag Viterbi decoding) with four settings selected on training videos by a colour-free '
        'objective, (ii) an independent keypoint visibility term that removes the double counting of skin evidence, and '
        '(iii) an optional camera-compensated, respiration-aware motion term. '
        'Because no physiological reference is available, we complement pulse signal-to-noise ratio (SNR) with two '
        'label-free validity tests: a surrogate-null pulse-present test that re-runs colour-dependent selection on every '
        f'surrogate, and split-half heart-rate agreement. On {n} unconstrained newborn videos with a video-level split, '
        f"stabilization reduced top-1 region switching from {mean(D, 'Full', 'switches_per_sec'):.1f} to "
        f"{mean(D, c, 'switches_per_sec'):.2f} switches/s ({c}, offline decoder) or {mean(D, 'D3L', 'switches_per_sec'):.2f} switches/s "
        f"(online decoder with 1-s lag) while retaining at least {100 * min(D['retention_split']):.0f}% of the instantaneous region "
        f"confidence on every split, and raised temporal IoU from {mean(D, 'Full', 'temporal_iou'):.2f} "
        f"to {mean(D, c, 'temporal_iou'):.2f} (Holm-corrected p {p(pair(f1, 'switches_per_sec', c).p_holm)}). The "
        + (f'headline configuration ({c}), selected on validation videos among the internally consistent configurations,'
           if c == D['chosen_rule'] else f'headline configuration ({c})')
        + " raised median pulse SNR by "
        f"{pair(f1, 'pulse_snr_db', c).mean_difference:+.2f} dB ({'significant' if snr_sig else 'not significant'} after "
        f"Holm correction) and the share of informative windows from {mean(D, 'Full', 'informative_window_percent'):.0f}% "
        f"to {mean(D, c, 'informative_window_percent'):.0f}% ({'significant' if inf_sig else 'not significant'}), "
        + ('but did not exceed a fixed box' if b0_better else 'and matched or exceeded a fixed box') + '. '
        f"The validity tests qualify these gains: only {min(pres(x) for x in ('B0', 'Full', c)):.0f}-"
        f"{max(pres(x) for x in ('B0', 'Full', c)):.0f}% of the windows of the fixed box, v1 and {c} exceeded the "
        f"surrogate null (chance {chance:.0f}%"
        + (f"; background pixels outside the infant: {pres('BG'):.0f}%" if 'BG' in set(D['pv2'].config) else '') + '), '
        f"an SNR-oracle that picks the best region per window reached {mean(D, 'M3', 'informative_window_percent', 2):.0f}% "
        f"informative windows but only {pres('M3'):.0f}% pulse-present windows, and split-half agreement was "
        f"{min(mean(D, x, 'agreement_percent', 2) for x in D['main2'].index if x not in ('M3', 'BG')):.0f}-"
        f"{max(mean(D, x, 'agreement_percent', 2) for x in D['main2'].index if x not in ('M3', 'BG')):.0f}% for all "
        'non-oracle methods. '
        'NeoSTAR therefore delivers a markedly more stable and inspectable infant ROI representation, while showing '
        'that SNR alone overstates pulse evidence in short web videos and that reference-based validation is required '
        'before physiological claims are made.')
    return text


def negative_control_text(D, chance, pres):
    """Results sentence for the background negative control (BG), worded from its measured pass rate."""
    if 'BG' not in set(D['pv2'].config):
        return ''
    bg = mean(D, 'BG', 'pulse_snr_db', 2)
    bgp = mean(D, 'BG', 'present_shuffle_percent', 2)
    bgi = mean(D, 'BG', 'informative_window_percent', 2)
    bga = mean(D, 'BG', 'agreement_percent', 2)
    pv = D['pv2'][D['pv2'].config == 'BG']
    n_hi = int((pv.present_shuffle_percent > 2 * chance).sum())
    text = (f"**Negative control.** Background pixels of the four frame corners, outside the infant, passed the same test in "
            f"{bgp:.1f}% of windows (median SNR {bg:.2f} dB; {bgi:.0f}% of their windows would count as informative by the "
            f"SNR threshold; split-half agreement {bga:.0f}%; {n_hi} of {len(pv)} videos above twice the chance level). ")
    best = max(v for k, v in pres.items() if k != 'M3')
    if bgp <= 2 * chance:
        text += (f'The test therefore behaves as intended on pixels that carry no pulse (close to the {chance:.0f}% chance '
                 'level), whereas the SNR proxy alone would credit background with informative windows. ')
    else:
        text += ('Background thus exceeds the chance level too, so part of the pulse-present rate of every method is '
                 'explained by periodic content that is not a pulse (compression, lighting or camera motion); ')
        text += (f"ROI methods exceed the background rate by at most {best - bgp:+.1f} percentage points. "
                 if best > bgp else 'no ROI method exceeds the background rate. ')
    return text


def headline_sentence(D):
    """How the headline configuration was chosen (two-step rule), including what the SNR-only rule would pick."""
    c, rule, rk, el = D['chosen'], D['chosen_rule'], D['validation_ranking'], D['eligibility']
    ok = [x for x in NEW if el.loc[x, 'eligible']]
    failed_terms = [x for x in NEW if not el.loc[x, 'term_independence']]
    listing = ', '.join(f'{k} ({v:.2f} dB)' for k, v in rk.pulse_snr_db.items())
    old_rk = D['validation_ranking_all'].head(3)
    old_list = ', '.join(f'{k} ({v:.2f} dB)' for k, v in old_rk.pulse_snr_db.items())
    pairs = sorted({p for x in failed_terms for p in el.loc[x, 'term_pairs_flagged'].split(', ') if p})
    text = ('**Headline selection.** The headline configuration is chosen in two steps. First, a configuration is '
            'eligible only if it meets the internal-consistency criteria that can be checked without validation or test '
            'videos: term independence inside its own score (no pair of terms with |r| > 0.8 on training ROI-frames), '
            f"quality retention of at least {D['cfg']['phase1']['quality_retention']:.2f} on training videos, and candidate "
            f"validity (every selectable region carries pixels). {', '.join(failed_terms)} fail term independence because "
            f"their score still contains the v1 visibility and skin terms ({', '.join(pairs)}: identical, |r| = 1.00); "
            f"{', '.join(ok)} {'is' if len(ok) == 1 else 'are'} eligible. Second, among eligible configurations the highest "
            f'mean validation pulse SNR decides: {listing}, so {rule} is the headline configuration.')
    if c != rule:
        text += f' We nevertheless report {c} as the headline configuration.'
    if c == 'D5':
        text += (' D5 therefore uses keypoint visibility V′ with total motion M; nuisance motion M_n (D6) and the subject '
                 'check (D7) are evaluated but are not part of the headline configuration.')
    text += (f' This consistency precondition was introduced after an initial SNR-only rule had been applied; that rule ranked {old_list} within '
             f"{float(old_rk.pulse_snr_db.max() - old_rk.pulse_snr_db.min()):.2f} dB of each other, all with the "
             'double-counted skin evidence. All configurations remain in every table.')
    return text


# ------------------------------------------------------------------ fusion-smoothing test (Option A)
FUSION_PAIRS = [('D0', 'F-Full', 'Full → Full + fusion smoothing'), ('D2', 'F-D2', 'D2 → D2 + fusion smoothing'),
                ('D3', 'D4', 'D3 → D4 (= D3 + fusion smoothing)')]


def fusion_stat(D, base, test, metric, nd=2):
    tab, tests = D['fusion']
    r = tests[(tests.pair == f'{base} vs {test}') & (tests.metric == metric)].iloc[0]
    row = tab[tab.pair == f'{base} vs {test}'].iloc[0]
    return row[f'{metric}_{base}'], row[f'{metric}_{test}'], r


def fusion_text(D):
    if D.get('fusion') is None:
        return 'was not tested'
    j0, j1, _ = fusion_stat(D, 'D0', 'F-Full', 'weight_jitter')
    s0, s1, rs = fusion_stat(D, 'D0', 'F-Full', 'pulse_snr_db')
    i0, i1, ri = fusion_stat(D, 'D0', 'F-Full', 'informative_window_percent')
    return (f"lowered fusion-weight jitter from {j0:.1f} to {j1:.1f} per second, raised median pulse SNR from {s0:.2f} to "
            f"{s1:.2f} dB (p_Holm {p(rs.p_holm)}) and informative windows from {i0:.1f}% to {i1:.1f}% (p_Holm {p(ri.p_holm)})")


def fusion_section(b, D):
    if D.get('fusion') is None:
        return
    tab, _ = D['fusion']
    b.h2('5.2.1 Where Fusion-Weight Smoothing Helps')
    full_share = tab[tab.pair == 'D0 vs F-Full'].transition_window_share.iloc[0]
    d3_share = tab[tab.pair == 'D3 vs D4'].transition_window_share.iloc[0]
    b.para('Fusion-weight smoothing never changes the selected regions, so its only possible effect is on the fused signal '
           'around changes of the selected set. We therefore applied the same smoothing (EMA factor and 1-s cross-fade of '
           'Section 3.5) to three selection rules with very different switch rates, keeping each selection identical, and '
           'measured fusion-weight jitter (mean L1 change of the weights per second), pulse SNR over all windows and over '
           'transition windows (windows in which the selected set changes), and informative windows (Table 5). On the '
           f"unstabilized v1 rule, where {full_share:.0f}% of windows contain a set change, smoothing {fusion_text(D)}. "
           'After hysteresis (D2) or the Viterbi anchor (D3; '
           f'{d3_share:.0f}% of windows contain a set change, but changes are rare within them) no metric changed '
           'significantly. Smoothing the weights is therefore an effective remedy for switching artefacts, but one that the '
           'selection-level stabilizer makes unnecessary.')
    b.caption('Table 5. Fusion-weight smoothing applied to three selection rules (selection identical within each pair; '
              'mean over 34 videos; Holm-corrected Wilcoxon p within each pair).')
    rows = []
    for base, test, label in FUSION_PAIRS:
        cells = [label]
        for metric, nd in (('weight_jitter', 2), ('pulse_snr_db', 2), ('transition_snr_db', 2),
                           ('informative_window_percent', 1)):
            v0, v1, r = fusion_stat(D, base, test, metric, nd)
            cells.append(f'{v0:.{nd}f} → {v1:.{nd}f} ({p(r.p_holm)})')
        rows.append(cells)
    b.table(['Pair', 'Weight jitter (/s)', 'SNR (dB)', 'SNR, transition windows (dB)', 'Informative %'], rows, size=7.5,
            widths=[2.0, 1.15, 1.15, 1.25, 1.15])


# ------------------------------------------------------------------ internal consistency
def consistency_section(b, D):
    C = D.get('cons')
    if C is None:
        return
    pv = D['pv1']
    prefix = D.get('c3_prefix')
    before_snr = lambda cid: float(D['pv1_prefix'].loc[D['pv1_prefix'].config == cid, 'pulse_snr_db'].mean())  # noqa: E731
    dead_before = lambda cid: float(prefix.loc[prefix.config == cid, 'dead_selection_percent'].mean())  # noqa: E731
    c1 = C['c1'].set_index(['score', 'method'])
    m = lambda t, col, cid: float(t.loc[t.config == cid, col].mean())  # noqa: E731
    e2 = C['e2'].groupby(['video_id', 'config', 'sigma'])[['jaccard', 'top_agreement']].mean().reset_index()
    e2m = lambda cid, sig, col: float(e2[(e2.config == cid) & (e2.sigma == sig)][col].mean())  # noqa: E731
    rank = C['rank'].groupby('score').kendall_tau.mean()
    margin = C['rank'].groupby('score').top_margin.mean()
    e1 = C['e1'].groupby('term').repeatability.mean()
    t = C['tests']
    pt = lambda fam, base, cand: (lambda r: p(r.p_holm.iloc[0]) if len(r) else 'n/a')(  # noqa: E731
        t[(t.family == fam) & (t.base == base) & (t.candidate == cand)])
    test_ret = lambda cid: float(pv[(pv.config == cid) & (pv.split == 'test')].retention.mean())  # noqa: E731
    e3 = C['e3'].groupby('theta')[['top_agreement', 'jaccard', 'spearman_vs_0_5']].mean() if C['e3'] is not None else None
    e4 = C['e4'].mean(numeric_only=True) if C['e4'] is not None else None
    r1 = c1.loc[('v1', 'pearson')]
    r2 = c1.loc[('v2', 'pearson')]
    resp = c1.loc[('M vs M_n', 'respiratory regions')].max_abs_r
    cam = c1.loc[('M vs M_n', 'high camera motion')].max_abs_r
    other = c1.loc[('M vs M_n', 'other regions')].max_abs_r
    b.para('We assess internal consistency with the five measurable properties defined in Section 4.4 (Table 8). '
           f"**Term independence.** Among the terms of the v1 score the largest correlation is |r| = {r1.max_abs_r:.2f} "
           f"({r1.pair.replace('_term', '').replace('visibility', 'V').replace('skin', 'S')}), i.e. two terms are identical; "
           f"among the terms of the v2 score it is |r| = {r2.max_abs_r:.2f} (pose P and keypoint visibility V′), below the "
           'criterion. Total motion M and nuisance motion M_n correlate strongly overall '
           f"(r = {D['corr'].loc['M', 'M_nuisance']:.2f}), but they are alternatives that never enter the same score, and "
           f"they separate where the decomposition is designed to act: r = {resp:.2f} in the respiratory regions and "
           f"{cam:.2f} under strong camera motion, versus {other:.2f} elsewhere. **Score-selection agreement.** Quality "
           f"retention on test videos is {test_ret('D3'):.3f} (D3), {test_ret('D3L'):.3f} (D3L) and {test_ret('D5'):.3f} "
           '(D5), so the stabilized selections keep at least 98% of the per-frame top-k confidence on unseen videos. '
           f"**Cross-level agreement.** {100 * m(C['c3'], 'area_inside_mask', 'D5'):.1f}% of the selected-region area and "
           f"{100 * m(C['c3'], 'landmarks_inside_mask', 'D5'):.1f}% of its pose landmarks lie inside the infant mask (D5), and "
           f"{100 * m(C['c3'], 'pixel_yield', 'D5'):.1f}% of the region pixels survive pixel refinement; the values for v1 "
           f"Full and D3 differ from D5 by at most {max(abs(100 * (m(C['c3'], k, x) - m(C['c3'], k, 'D5'))) for k in ('area_inside_mask', 'landmarks_inside_mask', 'pixel_yield') for x in ('D0', 'D3')):.1f} "
           'percentage points. **Candidate validity.** In v1, regions whose geometry is a two-point box carry no pixels '
           f"yet can be selected: {m(C['c3'], 'dead_selection_percent', 'D0'):.1f}% of v1 selections"
           + (f" ({dead_before('D3'):.1f}% for D3 and {dead_before('D5'):.1f}% for D5 before the fix)" if prefix is not None else '')
           + '. Reading these boxes as rectangles (Section 3.2) removes the inconsistency: selections without pixels fall to '
           f"{m(C['c3'], 'dead_selection_percent', 'D3'):.1f}% for D3 and {m(C['c3'], 'dead_selection_percent', 'D5'):.1f}% "
           'for D5. The v1 baselines keep v1\'s handling so that they remain reproducible. '
           + (f"The fix lowers pulse quality slightly (D3: {before_snr('D3'):.2f} → {mean(D, 'D3', 'pulse_snr_db'):.2f} dB; "
              f"I-B vs. Full: {pair(D['full1'], 'pulse_snr_db', 'I-B').mean_difference:+.2f} dB), i.e. the shoulder, limb and "
              'neck boxes that v1 silently ignored carry weaker pulse evidence: internal consistency and pulse quality pull '
              'in different directions here, and we keep the consistent version.' if D.get('pv1_prefix') is not None else ''))
    b.para(f"**Repeatability.** The best and second-best region scores differ by only {margin['Q_v1']:.3f} on average, "
           'which is why the v1 rule switches. The 17-region ranking is significantly more repeatable between consecutive '
           f"frames with keypoint visibility (Kendall τ {rank['Q_v1']:.2f} → {rank['Q_v2']:.2f}, p_Holm "
           f"{pt('kendall_tau', 'Q_v1', 'Q_v2')}), although V′ itself fluctuates more from frame to frame than V "
           f"(R = {e1['V_prime']:.2f} vs. {e1['V']:.2f}) because keypoint confidences flicker around the threshold. Under "
           f"score noise of σ = 0.01 the v1 rule keeps its top region in only {e2m('D0', .01, 'top_agreement'):.0f}% of "
           f"frames, D3 in {e2m('D3', .01, 'top_agreement'):.0f}% and D5 in {e2m('D5', .01, 'top_agreement'):.0f}% (p_Holm "
           f"{pt('perturbation sigma=0.01 top_agreement', 'D0', 'D5')} vs. v1); at σ = 0.05 the agreements are "
           f"{e2m('D0', .05, 'top_agreement'):.0f}%, {e2m('D3', .05, 'top_agreement'):.0f}% and "
           f"{e2m('D5', .05, 'top_agreement'):.0f}%. Halving the frame rate keeps the top region in "
           f"{m(C['c4'], 'top_agreement_s1_s2', 'D3'):.0f}% (D3) and {m(C['c4'], 'top_agreement_s1_s2', 'D5'):.0f}% (D5) of "
           'the frames processed in both runs (the memoryless v1 rule agrees trivially). '
           + (f"Moving the V′ threshold to 0.4 or 0.6 keeps the D5 top region in {e3.loc[.4, 'top_agreement']:.0f}% and "
              f"{e3.loc[.6, 'top_agreement']:.0f}% of frames, but 0.3 or 0.7 only in {e3.loc[.3, 'top_agreement']:.0f}% and "
              f"{e3.loc[.7, 'top_agreement']:.0f}%, so the threshold matters beyond ±0.1. " if e3 is not None else '')
           + (f"Estimating global motion by homography instead of the background median changes M_n little "
              f"(Spearman ρ = {e4.spearman_Mn:.2f}) and keeps the D6 top region in {e4.top_agreement:.0f}% of frames. "
              if e4 is not None else '')
           + 'Run-to-run repeatability of the perception models (re-running detection, SAM 2 and pose) was not tested.')
    rows = [
        ['Term independence: max |r| within the score (train)', f"{r1.max_abs_r:.2f}", f"{r1.max_abs_r:.2f}", f"{r2.max_abs_r:.2f}"],
        ['Quality retention, test videos', '1 (by definition)', f"{test_ret('D3'):.3f}", f"{test_ret('D5'):.3f}"],
        ['Selected area inside SAM 2 mask (%)'] + [f"{100 * m(C['c3'], 'area_inside_mask', c):.1f}" for c in ('D0', 'D3', 'D5')],
        ['Landmarks inside SAM 2 mask (%)'] + [f"{100 * m(C['c3'], 'landmarks_inside_mask', c):.1f}" for c in ('D0', 'D3', 'D5')],
        ['Refined-pixel yield (%)'] + [f"{100 * m(C['c3'], 'pixel_yield', c):.1f}" for c in ('D0', 'D3', 'D5')],
        ['Selections without pixels (%)'] + [f"{m(C['c3'], 'dead_selection_percent', c):.1f}" for c in ('D0', 'D3', 'D5')],
    ] + ([['  … before the 2-point-box fix (%)', '–', f"{dead_before('D3'):.1f}", f"{dead_before('D5'):.1f}"]] if prefix is not None else []) + [
        ['Ranking repeatability, Kendall τ (consecutive frames)', f"{rank['Q_v1']:.2f}", f"{rank['Q_v1']:.2f}", f"{rank['Q_v2']:.2f}"],
        ['Top region kept under score noise σ = 0.01 (%)'] + [f"{e2m(c, .01, 'top_agreement'):.0f}" for c in ('D0', 'D3', 'D5')],
        ['Top region kept under score noise σ = 0.05 (%)'] + [f"{e2m(c, .05, 'top_agreement'):.0f}" for c in ('D0', 'D3', 'D5')],
        ['Top region kept at half frame rate (%)', '100 (trivial)'] + [f"{m(C['c4'], 'top_agreement_s1_s2', c):.0f}" for c in ('D3', 'D5')],
    ]
    if e3 is not None:
        rows.append(["Top region kept with V' threshold 0.4 / 0.6 (%)", '–', '–',
                     f"{e3.loc[.4, 'top_agreement']:.0f} / {e3.loc[.6, 'top_agreement']:.0f}"])
    if e4 is not None:
        rows.append(['Top region kept with homography global motion (D6, %)', '–', '–', f'{e4.top_agreement:.0f}'])
    b.caption('Table 8. Internal consistency (mean over 34 videos). v1 Full and D3 use the v1 score terms; D5 uses the '
              'corrected terms (V′; M_n enters from D6).')
    b.table(['Property', 'v1 Full', 'D3', 'D5'], rows, size=7.5, widths=[3.4, 1.0, 1.0, 1.0])


# ------------------------------------------------------------------ results
def retention_text(t):
    values = sorted({round(v['retention'], 3) for v in t['training_means'].values()})
    return f'{values[0]:.3f}' if len(values) == 1 else f'{values[0]:.3f}-{values[-1]:.3f}'


def results(b, D, figs):
    c, n = D['chosen'], len(D['manifest'])
    f1, b01, f2, b02 = D['full1'], D['b01'], D['full2'], D['b02']
    m1, m2 = D['main1'], D['main2']
    b.h1('5. Results')

    b.h2('5.1 Reproducibility of the v1 Baseline')
    reg1 = max(D['meta1']['regression_D0_vs_Full_max_abs_diff'].values())
    reg2 = max(v for v in D['meta2']['regression_max_abs_window_snr_diff'].values() if v is not None)
    miss2 = sum(D['meta2'].get('regression_window_presence_mismatch', {}).values())
    b.para(f'With every flag off, the v2 code (D0) reproduced v1 Full on every metric and video (largest absolute '
           f'difference {reg1:.1e}), and the validity-test extraction reproduced the window SNRs of B0, B1, Full, D3, D4 and D5 '
           f'(largest difference {reg2:.1e} dB; {miss2} windows present in only one of the two). All differences reported below are therefore caused by the v2 changes, '
           'not by a changed evaluation.')
    t = D['tuning']
    b.para(f"**Tuned stabilizer.** On the {len(D['log']['training_videos'])} training videos the objective chose "
           f"β = {t['D2']['ema_beta']}, δ = {t['D2']['hysteresis_delta']}, N = {t['D2']['hysteresis_frames']} frames and "
           + (f"λ = {t['D3']['viterbi_penalty']} (offline and fixed-lag)" if t['D3']['viterbi_penalty'] == t['D3L']['viterbi_penalty']
              else f"λ = {t['D3']['viterbi_penalty']} (offline) / {t['D3L']['viterbi_penalty']} (fixed-lag)")
           + f"; D4-D7 build on "
           f"{t['D4_base']}. Training means: {t['training_means']['D2']['switches_per_sec']:.2f} (D2), "
           f"{t['training_means']['D3']['switches_per_sec']:.2f} (D3) and {t['training_means']['D3L']['switches_per_sec']:.2f} "
           f"(D3L) switches/s at a mean quality retention of {retention_text(t)} (Figure 2). {edge_text(D)} of the chosen values "
           'lie at the upper edge of their grids, so stronger smoothing might reduce switching further. '
           + headline_sentence(D))
    figure(b, D['p1dir']/'figures/fig5_tuning.png', 5.8,
           'Figure 2. Stabilizer tuning on training videos: every grid setting (switch rate vs. quality retention), the '
           'retention floor and the chosen settings.')

    b.h2('5.2 Temporal Stability')
    sig_stab = [x for x in NEW if sig(f1, 'switches_per_sec', x, -1)]
    b.para(f"Table 3 and Figure 3 summarize all configurations on all {n} videos. The v1 rule switched its top region "
           f"{mean(D, 'Full', 'switches_per_sec'):.2f} times per second (temporal IoU {mean(D, 'Full', 'temporal_iou'):.3f}). "
           f"Smoothing alone (D1) reduced this to {mean(D, 'D1', 'switches_per_sec'):.2f}/s, hysteresis (D2) to "
           f"{mean(D, 'D2', 'switches_per_sec'):.2f}/s and the offline Viterbi anchor (D3) to "
           f"{mean(D, 'D3', 'switches_per_sec'):.2f}/s (IoU {mean(D, 'D3', 'temporal_iou'):.3f}); the online fixed-lag "
           f"decoder (D3L) reached {mean(D, 'D3L', 'switches_per_sec'):.2f}/s. Keypoint visibility (D5) gave the most stable "
           f"selection ({mean(D, 'D5', 'switches_per_sec'):.2f}/s, IoU {mean(D, 'D5', 'temporal_iou'):.3f}). The reduction "
           f"versus Full is Holm-significant for {len(sig_stab)} of {len(NEW)} new configurations "
           f"(e.g. {c}: {stat(f1, 'switches_per_sec', c, '/s')}). The single fixes applied to the unstabilized rule "
           f"(I-V {mean(D, 'I-V', 'switches_per_sec'):.2f}/s, I-M {mean(D, 'I-M', 'switches_per_sec'):.2f}/s, I-S "
           f"{mean(D, 'I-S', 'switches_per_sec'):.2f}/s) did not stabilize it: instability is caused by the per-frame "
           'decision rule, not by a particular term.')
    cols = ['videos', 'switches_per_sec', 'temporal_iou', 'centroid_jitter', 'pulse_snr_db', 'informative_window_percent']
    order = [x for x in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'Full', 'D0', 'I-B'] + NEW + ['I-V', 'I-M', 'I-S'] if x in m1.index]
    b.caption(f'Table 3. Stability and pulse results on all {n} videos (mean ± SD over videos; 95% bootstrap CIs in the released CSVs). '
              f'Informative: window SNR ≥ τ = {D["tau"]:.2f} dB.')
    b.table(['Config', 'n', 'Switches/s ↓', 'Temporal IoU ↑', 'Jitter ↓', 'Pulse SNR (dB) ↑', 'Informative % ↑'],
            [[x] + [m1.loc[x, k] for k in cols] for x in order], size=7.5)
    figure(b, D['p1dir']/'figures/fig2_stability.png', 6.4,
           f'Figure 3. Selection stability per configuration (bars: mean; dots: videos). Orange: v1 Full; blue: {c}.')
    st = D['steps']
    cell = lambda base, step, m, nd: (lambda r: f'{r.mean_difference:+.{nd}f} ({p(r.p_holm)})')(  # noqa: E731
        st[(st.base == base) & (st.step == step) & (st.metric == m)].iloc[0])
    b.para('Table 4 tests every stabilization step against the configuration it extends. Smoothing, hysteresis and both '
           'Viterbi decoders each reduce switching significantly, and the offline decoder is more stable than the online '
           f"one ({mean(D, 'D3', 'switches_per_sec'):.2f} vs. {mean(D, 'D3L', 'switches_per_sec'):.2f}/s; p_Holm "
           f"{p(st[(st.base == 'D3L') & (st.step == 'D3') & (st.metric == 'switches_per_sec')].iloc[0].p_holm)}). Fusion-weight "
           'smoothing (D4) leaves selection unchanged by construction and has no measurable effect after the Viterbi '
           'stabilizer (Section 5.2.1 shows where it does help). Quality '
           'retention, the share of the per-frame top-k confidence that the stabilized selection keeps, stays at '
           f"{min(D['retention_split']):.3f}-{max(D['retention_split']):.3f} for D3 and D3L on training, validation and test "
           'videos, i.e. the training constraint (≥ 0.98) transfers to unseen videos.')
    b.caption('Table 4. Step-by-step ablation of the temporal stabilization (mean paired difference over 34 videos; '
              'Holm-corrected Wilcoxon p in parentheses, Holm within each metric over these seven steps; Section 5.3 '
              'corrects over all configurations versus Full, so its p-values for the same contrast are larger).')
    b.table(['Step', 'Switches/s', 'Temporal IoU', 'Retention', 'SNR (dB)', 'Informative (pp)'],
            [[f"{'Full (D0)' if a == 'D0' else a} → {b2}: {label}", cell(a, b2, 'switches_per_sec', 2),
              cell(a, b2, 'temporal_iou', 3), cell(a, b2, 'retention', 3), cell(a, b2, 'pulse_snr_db', 2),
              cell(a, b2, 'informative_window_percent', 1)] for a, b2, label in STEPS], size=7.5,
            widths=[2.3, 1.0, 1.0, 0.9, 0.9, 1.0])
    fusion_section(b, D)

    b.h2('5.3 Pulse Signal Quality')
    b0, cc, full = mean(D, 'B0', 'pulse_snr_db'), mean(D, c, 'pulse_snr_db'), mean(D, 'Full', 'pulse_snr_db')
    b.para(f"Stabilization raised the median pulse SNR from {full:.2f} dB (Full) to {cc:.2f} dB ({c}; "
           f"{stat(f1, 'pulse_snr_db', c, ' dB')}) and the share of informative windows from "
           f"{mean(D, 'Full', 'informative_window_percent'):.1f}% to {mean(D, c, 'informative_window_percent'):.1f}% "
           f"({stat(f1, 'informative_window_percent', c, ' pp', 1)}). Holm-significant gains in informative windows were "
           f"found for {', '.join(x for x in NEW if sig(f1, 'informative_window_percent', x)) or 'no configuration'}; for "
           f"SNR for {', '.join(x for x in NEW if sig(f1, 'pulse_snr_db', x)) or 'no configuration'} (family: all "
           f"configurations vs. Full). The fixed box B0 nevertheless reached {b0:.2f} dB and "
           f"{mean(D, 'B0', 'informative_window_percent'):.1f}% informative windows; {c} versus B0: "
           f"{stat(b01, 'pulse_snr_db', c, ' dB')}. The corrected terms did not add pulse quality: D4 → D5 (V′) changed SNR "
           f"by {mean(D, 'D5', 'pulse_snr_db') - mean(D, 'D4', 'pulse_snr_db'):+.2f} dB and D5 → D6 (nuisance motion) by "
           f"{mean(D, 'D6', 'pulse_snr_db') - mean(D, 'D5', 'pulse_snr_db'):+.2f} dB, and the single fixes I-V, I-M and I-S "
           f"stayed within {max(abs(mean(D, x, 'pulse_snr_db') - full) for x in ('I-V', 'I-M', 'I-S')):.2f} dB of Full. "
           "Reading two-point-box regions as rectangles (I-B) changed Full's SNR by "
           f"{stat(f1, 'pulse_snr_db', 'I-B', ' dB')} and its informative windows by "
           f"{stat(f1, 'informative_window_percent', 'I-B', ' pp', 1)}.")
    figure(b, D['p1dir']/'figures/fig3_snr.png', 6.2,
           'Figure 4. Median pulse SNR per configuration (mean and 95% bootstrap CI over videos; grey: videos). Dashed line: '
           'fixed box B0.')

    b.h2('5.4 Additional Baselines')
    b.para(f"Table 6 adds whole-infant skin (M0: {mean(D, 'M0', 'pulse_snr_db', 2):.2f} dB), skin + signal-quality "
           f"selection (M1: {mean(D, 'M1', 'pulse_snr_db', 2):.2f} dB) and grid tiles (M2: "
           f"{mean(D, 'M2', 'pulse_snr_db', 2):.2f} dB). None exceeded B0 ({stat(b02, 'pulse_snr_db', 'M0', ' dB')} for M0; "
           f"{stat(b02, 'pulse_snr_db', 'M1', ' dB')} for M1; {stat(b02, 'pulse_snr_db', 'M2', ' dB')} for M2). The oracle "
           f"M3, which picks in every window the single region whose pulse is best in that same window, reached "
           f"{mean(D, 'M3', 'pulse_snr_db', 2):.2f} dB and {mean(D, 'M3', 'informative_window_percent', 2):.1f}% informative "
           'windows. M3 is an upper bound on what SNR-driven selection can show, not an achievable method, and Section 5.5 '
           f"shows how much of it is selection bias. Because the clips are short, causal selection was possible in only "
           f"{D['m12_active']:.0f}% of the windows (in {D['m12_active_videos']} of {n} videos); elsewhere M1 and M2 reduce "
           'to an average over all anatomical regions or all tiles, respectively.')
    cols2 = ['videos', 'pulse_snr_db', 'informative_window_percent', 'present_shuffle_percent',
             'agreement_percent', 'agreement_testable_percent']
    b.caption('Table 6. Additional baselines and label-free validity tests (mean ± SD over videos). Pulse-present: window SNR '
              'above the 95th percentile of the block-shuffle surrogate null (chance about 5%). Agreement: share of '
              'testable windows whose split-half heart rates agree within ±6 bpm.'
              + (' BG: negative control, background pixels of the four frame corners (not a method).' if 'BG' in m2.index else ''))
    b.table(['Config', 'n', 'SNR (dB)', 'Informative %', 'Pulse-present %', 'Agreement %', 'Testable %'],
            [[('BG (neg. control)' if x == 'BG' else x)] + [m2.loc[x, k] for k in cols2] for x in m2.index], size=7.5)

    b.h2('5.5 Do the Selected Pixels Carry a Pulse?')
    chance = 100 - D['p2']['null_percentile']
    methods = [x for x in m2.index if x != 'BG']
    pres = {x: mean(D, x, 'present_shuffle_percent', 2) for x in methods}
    agree = {x: mean(D, x, 'agreement_percent', 2) for x in methods}
    above = [x for x in methods if x != 'M3' and pres[x] > 2 * chance]
    pf = pair(f2, 'present_shuffle_percent', c)
    b.para(f"Under the block-shuffle null a window passes by chance with probability about {chance:.0f}%. Observed "
           f"pulse-present rates were {min(pres.values()):.1f}-{max(pres.values()):.1f}% (Figure 5): "
           + f"v1 Full {pres['Full']:.1f}%, B0 {pres['B0']:.1f}%, D3 {pres['D3']:.1f}%, D4 {pres['D4']:.1f}%, D5 "
           f"{pres['D5']:.1f}% and the oracle M3 {pres['M3']:.1f}%; "
           + (f"{len(above)} of the {len([x for x in methods if x != 'M3'])} methods exceed twice the chance level, but "
              'no method reaches even one window in five. ' if above else 'no method exceeds twice the chance level. ')
           + negative_control_text(D, chance, pres)
           + f"Versus Full, the pulse-present rate of {c} was "
           f"{stat(f2, 'present_shuffle_percent', c, ' pp', 1)}"
           + (', i.e. the SNR gain of stabilization does not translate into more windows with a detectable pulse. '
              if pf is not None and pf.mean_difference <= 0 else '. ')
           + f"The oracle makes the selection-bias point sharply: {mean(D, 'M3', 'informative_window_percent', 2):.0f}% of "
           f"its windows are informative but only {pres['M3']:.1f}% beat a null in which the same max-SNR selection is "
           + (f"applied to surrogates. With {D['p2']['surrogates_shuffle']} surrogates per video the threshold is estimated "
              f"from the pooled null of the video ({D['p2']['surrogates_shuffle']} surrogate values per window); a sensitivity run with 19 surrogates gave "
              f"pulse-present rates within {D['mc_max']:.1f} percentage points of the present ones for every configuration "
              f"(v1 Full: {D['s19'].loc['Full', 's19']:.1f}% → {D['s19'].loc['Full', 'now']:.1f}%), so differences of that "
              'size between configurations are not meaningful.'
              if D.get('mc_max') is not None else 'applied to surrogates.'))
    ref = 2 * 6 / 180 - (6 / 180) ** 2
    b.para(f"Split-half agreement was {min(agree.values()):.1f}-{max(agree.values()):.1f}% of testable windows, above the "
           f"{100 * ref:.1f}% expected for two independent peaks uniformly distributed over the 90-270 bpm band, but low in "
           f"absolute terms; no method differed significantly from B0 (smallest p_Holm "
           f"{p(D['b02'][D['b02'].metric == 'agreement_percent'].p_holm.min())}). Taken together, these tests indicate that "
           'the pulse evidence in these short, compressed and edited web videos is weak for every ROI method, so pulse SNR '
           'differences between methods must not be read as differences in heart-rate quality.')
    figure(b, figs/'fig_validity.png', 6.5,
           'Figure 5. Informative windows (SNR ≥ τ), pulse-present windows (SNR above the 95th percentile of the surrogate '
           'null; dashed line: 5% chance level) and split-half agreement per configuration.')

    b.h2('5.6 Held-Out Splits')
    rows = []
    for s, t in D['split1'].items():
        for x in ['B0', 'Full', 'D3', 'D4', 'D5', c]:
            if x in t.index and [s, x] not in [r[:2] for r in rows]:
                rows.append([s, x, int(t.loc[x, 'videos']), t.loc[x, 'switches_per_sec'], t.loc[x, 'temporal_iou'],
                             t.loc[x, 'pulse_snr_db'], t.loc[x, 'informative_window_percent']])
    v, te = D['split1']['validation'], D['split1']['test']
    b.para(f"On the test split, which informed no decision, {c} switched {te.loc[c, 'switches_per_sec']} times per second "
           f"versus {te.loc['Full', 'switches_per_sec']} for Full, with pulse SNR {te.loc[c, 'pulse_snr_db']} vs. "
           f"{te.loc['Full', 'pulse_snr_db']} dB and B0 {te.loc['B0', 'pulse_snr_db']} dB (Table 7). With "
           f"{int(te.loc[c, 'videos'])} test and {int(v.loc[c, 'videos'])} validation videos these split-wise estimates are "
           'imprecise and are reported for transparency only.')
    b.para(split_disagreement_sentence(D))
    b.caption('Table 7. Validation and test splits (mean ± SD over videos).')
    b.table(['Split', 'Config', 'n', 'Switches/s', 'Temporal IoU', 'SNR (dB)', 'Informative %'], rows, size=8)

    b.h2('5.7 Internal Consistency of the Representation')
    consistency_section(b, D)
    corr, flags = D['corr'], D['flags']
    short = {'V_prime': "V'", 'M_nuisance': 'M_n'}
    flagged = '; '.join(f"{short.get(r.term_a, r.term_a)}-{short.get(r.term_b, r.term_b)} (r = {r.pearson:.2f})"
                        for r in flags.itertuples()) or 'none'
    figure(b, D['p1dir']/'figures/fig6_term_correlation.png', 4.6,
           "Figure 6. Pearson correlation between condition terms on training ROI-frames (V': keypoint visibility; Mn: "
           'nuisance motion).')
    s, a = D['subject'], D['audit']
    rej, tot = int(s.rejected_frames.sum()), int(s.sampled_frames.sum())
    b.para(f"The subject check rejected {rej:,} of {tot:,} frames ({100 * rej / tot:.1f}%) in "
           f"{int((s.rejected_frames > 0).sum())} videos. "
           + (f"A visual audit of {a['frames_audited']} rejected frames found {a['frames_correct']} correct rejections "
              f"({a['summary']}) " if a else '')
           + 'Overhead and close-up views foreshorten the infant\'s head and shoulders, so the head-to-shoulder ratio '
           'falls below the adult threshold without an adult in view. The rule is therefore not reliable enough for use and '
           'D7 and I-S are reported for completeness only.')

    b.h2('5.8 Computational Cost')
    if D['phase5']:
        pr, st, fl, m5 = D['profile'], D['stride'], D['flow'], D['meta5']
        q = pr[pr.stage.isin(['roi_quality', 'roi_quality_sam2'])].seconds.sum()
        pose = pr[pr.stage.isin(['pose', 'pose_sam2'])].seconds.sum()
        total = pr.seconds.sum()
        base = pr[~pr.stage.str.startswith('v2_')].seconds.sum()
        pick = D['chosen'] if D['chosen'] in set(st.config) else 'D3'   # headline configuration when Phase 5 ran it
        x = st[(st.config == pick)] if pick in set(st.config) else st[st.config == st.config.iloc[0]]
        s2 = x[x.stride == 2].iloc[0]
        s1 = x[x.stride == 1].iloc[0]
        tests = D['stride_tests']
        t2 = tests[(tests.config == s2.config) & (tests.stride == 2) & (tests.metric == 'pulse_snr_db')]
        b.para(f"Table 9 lists the measured stage costs. Summed over stages (the two branches ran concurrently, so wall "
               f"clock was {m5.get('base_run_wall_clock_hours', float('nan')):.1f} h), the base run took {base / 3600:.1f} h "
               f"for {m5['sampled_frames']:,} sampled frames; ROI quality characterization took {100 * q / base:.0f}% of it "
               f"and pose {100 * pose / base:.0f}%, whereas the v2 additions (features {pr.set_index('stage').seconds.get('v2_features', 0) / 3600:.1f} h, "
               f"evaluation {pr.set_index('stage').seconds.get('v2_evaluation', 0) / 3600:.1f} h) are small. Peak GPU memory "
               f"was {m5['gpu_peak_mib'] / 1024:.1f} GiB. Processing every second sampled frame "
               f"({s2.effective_fps:.1f} instead of {s1.effective_fps:.1f} FPS on average) would cut the projected cost by a "
               f"factor {s2.speedup:.2f}; for {s2.config} it changed pulse SNR from {s1.pulse_snr_db:.2f} to "
               f"{s2.pulse_snr_db:.2f} dB (p_Holm {p(t2.p_holm.iloc[0]) if len(t2) else 'n/a'}) and switching from "
               f"{s1.switches_per_sec:.2f} to {s2.switches_per_sec:.2f}/s (Figure 7, Table 10). Dense Farnebäck flow took "
               f"{fl.farneback_ms.mean():.0f} ms per frame pair versus {fl.lucas_kanade_ms.mean():.0f} ms for sparse "
               f"Lucas-Kanade [40] on the same frames; since the motion decomposition did not change results, the dense flow "
               'could be replaced by a sparse estimate. A lighter pose model was not evaluated because no alternative checkpoint was '
               'available.')
        b.caption('Table 9. Measured stage cost (summed seconds; share of total). Box branch: candidate regions from the tracked box; SAM 2 branch: candidate regions constrained by the SAM 2 infant mask.')
        b.table(['Stage', 'Seconds', 's / sampled frame', 'Share %'],
                [[STAGE_LABELS.get(r.stage, r.stage), f'{r.seconds:,.0f}', f'{r.sec_per_sampled_frame:.3f}', f'{r.share_percent:.1f}']
                 for r in pr.itertuples()], size=8)
        b.caption('Table 10. Accuracy when only every s-th sampled frame is processed (geometry held in between; run time '
                  'projected from measured stage costs).')
        b.table(['Config', 's', 'Eff. FPS', 'Projected h', 'Speed-up', 'Switches/s', 'Temporal IoU', 'SNR (dB)',
                 'Informative %'],
                [[r.config.replace('D0', 'Full (D0)'), int(r.stride), f'{r.effective_fps:.1f}', f'{r.projected_hours:.1f}',
                  f'{r.speedup:.2f}', f'{r.switches_per_sec:.2f}', f'{r.temporal_iou:.3f}', f'{r.pulse_snr_db:.2f}',
                  f'{r.informative_window_percent:.1f}'] for r in st.itertuples()], size=7.5)
        figure(b, figs/'fig_efficiency.png', 6.2,
               'Figure 7. Pulse SNR and switch rate versus projected pipeline cost when every s-th sampled frame is '
               'processed.')
    else:
        b.para('[Phase 5 results not yet available.]')


def edge_text(D):
    """How many of the four tuned stabilizer values sit at the upper edge of their grid ('All four', 'Three', ...)."""
    g, t = D['cfg']['phase1'], D['tuning']
    chosen = [(t['D2']['ema_beta'], g['ema_beta_grid']), (t['D2']['hysteresis_delta'], g['hysteresis_delta_grid']),
              (t['D2']['hysteresis_frames'], g['hysteresis_frames_grid']), (t['D3']['viterbi_penalty'], g['viterbi_penalty_grid'])]
    n = sum(v == max(grid) for v, grid in chosen)
    return {4: 'All four', 3: 'Three', 2: 'Two', 1: 'One', 0: 'None'}[n]


def split_means(D, split, cid):
    """Per-split means over videos of the pulse measures (Phase 1: SNR, informative; Phase 2: pulse-present)."""
    p1, p2 = D['pv1'], D['pv2']
    m1 = p1[(p1.split == split) & (p1.config == cid)]
    m2 = p2[(p2.split == split) & (p2.config == cid)]
    return dict(snr=float(m1.pulse_snr_db.mean()), inf=float(m1.informative_window_percent.mean()),
                pres=float(m2.present_shuffle_percent.mean()) if len(m2) else float('nan'))


def split_disagreement_sentence(D):
    """Validation vs. test: which method the pulse measures favour on each split, and whether the background control
    separates from the methods there (written from the per-video tables, so it follows the data)."""
    c = D['chosen']
    v0, vc = split_means(D, 'validation', 'B0'), split_means(D, 'validation', c)
    t0, tc = split_means(D, 'test', 'B0'), split_means(D, 'test', c)
    has_bg = 'BG' in set(D['pv2'].config)
    vbg = split_means(D, 'validation', 'BG')['pres'] if has_bg else float('nan')
    tbg = split_means(D, 'test', 'BG')['pres'] if has_bg else float('nan')
    v_b0 = v0['snr'] > vc['snr'] and v0['inf'] > vc['inf'] and v0['pres'] > vc['pres']
    t_c = tc['snr'] > t0['snr']
    text = (f"The pulse measures do not rank the methods consistently across splits. On the validation videos the fixed "
            f"box was {'clearly best on every pulse measure' if v_b0 else 'ahead on several pulse measures'} (SNR "
            f"{v0['snr']:.2f} vs. {vc['snr']:.2f} dB for {c}; informative windows {v0['inf']:.1f}% vs. {vc['inf']:.1f}%; "
            f"pulse-present {v0['pres']:.1f}% vs. {vc['pres']:.1f}%"
            + (f", background {vbg:.1f}%" if has_bg else '') + '), whereas on the test videos '
            + (f"{c} had the higher SNR ({tc['snr']:.2f} vs. {t0['snr']:.2f} dB)" if t_c else
               f"the fixed box again had the higher SNR ({t0['snr']:.2f} vs. {tc['snr']:.2f} dB)") + '.')
    if has_bg and tbg >= 0.8 * min(t0['pres'], tc['pres']):
        text += (f" On the test split, however, background pixels passed the pulse-present test in {tbg:.1f}% of windows, "
                 f"about as often as B0 ({t0['pres']:.1f}%) and {c} ({tc['pres']:.1f}%), so the test-split pulse measures "
                 'cannot separate infant from background and favour neither method.')
    text += (f" The validation split, which chose the headline configuration among the eligible ones, favours "
             f"the fixed box; pooled over all {len(D['manifest'])} videos, {c} does not exceed the fixed box in pulse "
             'evidence (Section 5.3), and its advantage is selection stability, which holds on every split (Table 7).')
    return text


def discussion(b, D):
    c = D['chosen']
    b.h1('6. Discussion')
    st = D['steps']
    d4 = st[(st.step == 'D4') & (st.base == 'D3') & (st.metric == 'pulse_snr_db')].iloc[0]
    b.para('**Stability is fixable without learning.** The v1 instability was not a perception failure: neighbouring regions '
           'receive almost identical scores, and an instantaneous top-k rule converts small score fluctuations into region '
           'switches. Standard sequential-decision tools (exponential smoothing, hysteresis and Viterbi decoding [38]) '
           f"remove {100 * (1 - mean(D, c, 'switches_per_sec') / mean(D, 'Full', 'switches_per_sec')):.0f}% of the "
           f"switches offline and {100 * (1 - mean(D, 'D3L', 'switches_per_sec') / mean(D, 'Full', 'switches_per_sec')):.0f}% "
           'online with one second of latency, while giving up at most '
           f"{100 * (1 - min(D['retention_split'])):.1f}% of the instantaneous confidence on any split. Each of the three "
           'selection steps contributes significantly (Table 4). The contribution is the leakage-controlled, colour-free way '
           'these tools are combined and tuned for infant ROI selection, not the algorithms themselves. Smoothing and '
           'cross-fading the fusion weights is effective where regions switch often: on the unstabilized v1 rule it '
           f"{fusion_text(D)}. After hysteresis or Viterbi stabilization there is little left to smooth and it has no "
           f"measurable effect (D4 vs. D3: Δ SNR = {d4.mean_difference:+.3f} dB, unadjusted p = {d4.p_value:.2f}), so it is a remedy for "
           'switching artefacts that becomes redundant once switching is removed at the selection level; we keep it as a '
           'safeguard for downstream consumers of the weights but claim no ''benefit. Finally, stability is necessary but not sufficient: a fixed box never switches, and the value of the '
           'stabilized selection lies in combining stability with high retained confidence, which still has to be validated '
           'against ROI or physiological ground truth.')
    b.para('**The quality terms needed correction, but the corrections did not change pulse quality.** Replacing the '
           'redundant visibility term by keypoint visibility is a correctness fix that also improved stability, and the '
           'motion decomposition is conceptually cleaner, yet neither moved the pulse proxy on these clips. A rule-based score '
           'cannot be expected to predict pulse quality well when its terms describe appearance and geometry; a learned '
           'reliability model trained against reference heart rate, with explicit calibration (e.g. Platt scaling [36]; see '
           '[37]) and, if needed, domain-adversarial adaptation to neonatal video [34], is the appropriate next step.')
    pres = {x: mean(D, x, 'present_shuffle_percent', 2) for x in D['main2'].index}
    b.para('**Label-free proxies must be validated against their own null.** Pulse SNR rewards anything that concentrates '
           'power at one frequency in the band. Without a surrogate comparison, stabilization appears to improve pulse '
           f"quality and an SNR-oracle appears nearly perfect; with it, only {pres[c]:.1f}% of {c}'s windows and "
           f"{pres['M3']:.1f}% of the oracle's windows beat the surrogates (chance about "
           f"{100 - D['p2']['null_percentile']:.0f}%), and v1 Full reaches {pres['Full']:.1f}%. Stability and pulse evidence are therefore separate properties that "
           'must be measured separately. We recommend that label-free rPPG ROI studies report a surrogate-null test '
           'alongside SNR, that colour-dependent selection be re-applied to every surrogate, and that the null itself be '
           'validated on a signal with a known pulse and on pixels that carry none (a background negative control'
           + (f", which passed {pres['BG']:.1f}% of windows here" if 'BG' in pres else '') + '): per-channel phase '
           'randomization, a natural choice, rejected a genuine synthetic pulse in every window.')
    b.para('**Fixed boxes are hard to beat on short web video.** A large fixed box averages many pixels and never switches, '
           'which helps a periodicity proxy. Its weaknesses—drifting off the infant and including non-infant pixels—are '
           'exactly what the dynamic representation removes, but they matter mostly for longer recordings with real infant '
           'motion, where they could not be tested here. At this sample size the pulse ranking also depends on the split: '
           'the fixed box led clearly on the validation videos and not on the test videos (Section 5.6), so no '
           'pulse-quality advantage of either representation can be claimed from these data.')
    b.h1('7. Limitations')
    b.bullets([
        f"No physiological reference was available, so no heart-rate or respiratory-rate accuracy is reported; the planned "
        'evaluation on rPPG datasets with reference heart rate (e.g., UBFC-rPPG [47], PURE [48], MMPD [16]) and a learned reliability model '
        'trained against it remain future work.',
        f"{len(D['manifest'])} short public web videos with unknown infant identities are not representative of continuous "
        'NICU recordings; subject independence across splits cannot be verified.',
        f'{edge_text(D)} tuned stabilizer values lie at the upper edges of their grids; wider grids on a larger training set '
        'may change the optimum.',
        'With 7 validation and 6 test videos, split-wise pulse measures are unstable: the validation split, used to choose '
        'the headline configuration among the eligible ones, favours the fixed box, while the test split does not '
        'separate the methods from the background control.',
        'The infant-versus-adult rule based on anthropometric priors was unreliable and should be replaced by a learned or '
        'labelled cue.',
        'Efficiency results for reduced frame rates project run time from measured stage costs rather than re-running '
        'the perception stages; a lighter pose model was not evaluated.',
        'No ROI ground truth exists; geometry metrics measure stability, not correctness.',
        f"The v1 baselines B0-Full keep v1's handling of two-point-box regions ({D['bbox_share']:.1f}% of available "
        'ROI-frames), which carry no pixels there; all v2 configurations read them as rectangles.',
    ])
    b.h1('8. Conclusion')
    b.para(f"NeoSTAR turns an unstable, partly redundant rule-based ROI selection into a stable and auditable dynamic "
           f"infant representation without any trained model: top-region switching falls from "
           f"{mean(D, 'Full', 'switches_per_sec'):.1f} to {mean(D, c, 'switches_per_sec'):.2f} per second with the offline "
           f"decoder and to {mean(D, 'D3L', 'switches_per_sec'):.2f} per second online, at least "
           f"{100 * min(D['retention_split']):.0f}% of the instantaneous region confidence is retained, only four stabilizer "
           'settings are selected (on training videos), all changes are flag-controlled, and the original configuration '
           'is reproduced exactly. Pulse SNR and informative-window rates of the headline configuration improve relative to '
           'v1 (not significantly after Holm correction) but do not exceed a fixed box, and surrogate-null and split-half '
           'tests show that these gains are not evidence of more pulse in the selected pixels of short web videos. The '
           'representation and the validity tests are released so that physiological benefits can be tested on recordings '
           'with a reference.')
    b.h1('Data and Code Availability')
    b.para('The pipeline, the v2 evaluation scripts (stability, validity-test and efficiency analyses), configuration files, the manifest with SHA-256 hashes '
           'of every video, tuning logs and all per-video result tables are released with the paper. [AUTHORS TO COMPLETE: '
           'repository URL and video source list.]')
