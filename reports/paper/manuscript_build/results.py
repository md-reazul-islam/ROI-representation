"""Abstract, Results, Discussion, Limitations and Conclusion, filled from reports/paper (every number read from the CSVs)."""
import json

from results_setup import load, setup, references, words, REFERENCES  # noqa: F401

IDS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'Full']


def by_id(table):
    """Table rows (one per configuration, in CONFIGS order) keyed by short ID."""
    return {cid: row for cid, (_, row) in zip(IDS, table.iterrows())}


def figure(b, path, width):
    """Insert a figure if it was drawn (e.g. Figure 4 needs held-out ROI-windows)."""
    if not path.is_file():
        return False
    b.image(path, width)
    return True


def strata_n(t5):
    counts = t5[[c for c in t5.columns if c.endswith(' videos')]].to_numpy().ravel()
    counts = counts[counts == counts]
    return f'{int(counts.min())}-{int(counts.max())}'


def mean(text):
    return float(str(text).split('±')[0])


def test(R, reference, metric, baseline):
    name = 'table3_paired_tests_vs_full' if reference == 'Full' else 'table3_paired_tests_vs_B4'
    t = R[name]
    return t[(t.metric == metric) & (t.baseline == baseline)].iloc[0]


def p(value):
    return '< 0.001' if value < .001 else f'{value:.3f}'


def pulse_sentence(R, m):
    """Abstract sentence on pixel refinement and the fixed box, worded by what the statistics actually show."""
    t = test(R, 'Full', 'informative_window_percent', 'B5')
    b5, full = mean(m['B5'].informative_window_percent), mean(m['Full'].informative_window_percent)
    verb = 'increased' if full > b5 else 'did not increase'
    text = (f'Pixel refinement {verb} the share of pulse-informative windows relative to region-level SAM 2 selection '
            f'({b5:.1f}% to {full:.1f}%; p_Holm = {t.p_holm:.3f})')
    best_fixed = max(mean(m[c].informative_window_percent) for c in ('B0', 'B1'))
    best_dynamic = max(mean(m[c].informative_window_percent) for c in IDS[2:])
    if best_dynamic <= best_fixed:
        text += ', and no dynamic configuration exceeded a fixed ROI on the pulse proxy'
    return text + '.'


def abstract(R):
    m = by_id(R['table3_main_all_videos'])
    q = R['table_q_validity'].set_index(['configuration', 'split'])
    b5 = q.loc[(R['table_q_validity'].configuration.iloc[-1], 'all')]
    n = R['meta']['videos']
    return (
        'Camera-based neonatal monitoring depends on extracting reliable visual evidence from the infant body, yet fixed '
        'regions of interest (ROIs) mix skin with blankets, devices, caregiver hands and background, and drift under motion, '
        'occlusion and illumination change. We present Module 1, a dynamic ROI representation framework that treats ROI '
        'construction as a standalone, auditable vision problem. Complementary detectors, multi-object tracking and bounded '
        'recovery localize the infant; SAM 2 propagates an infant mask; a 308-keypoint pose model defines 17 anatomical '
        'candidate regions; and an interpretable condition vector (anatomy, skin, temporal overlap, motion, blur, '
        'illumination, swaddling and mask support) drives a transparent region-level confidence, followed by pixel-level '
        'refinement to region ∩ mask ∩ adaptive-skin pixels. Because ROI ground truth is rarely available, we also propose '
        'a leakage-controlled, label-free evaluation protocol that uses a pulse signal-quality proxy computed from colour '
        'evidence the scorer never sees, together with calibration analysis, condition-stratified robustness, failure-'
        f'episode mining and computational profiling. On {n} unconstrained, in-the-wild neonatal videos with a predeclared '
        'video-level split, anatomy- and mask-guided selection raised the fraction of used pixels inside the infant mask '
        f'from {mean(m["B0"].mask_support):.2f} for a fixed box to {mean(m["B4"].mask_support):.2f} (region-level), '
        f'{mean(m["B5"].mask_support):.2f} (with SAM 2) and {mean(m["Full"].mask_support):.2f} (pixel-refined) '
        f'(Holm-corrected p {p(test(R, "Full", "mask_support", "B0").p_holm)}). {pulse_sentence(R, m)} '
        'The analysis also exposes two limitations of the '
        f'training-free rule: condition-aware scoring increased top-1 region switching from {mean(m["B2"].switches_per_sec):.1f} '
        f'to {mean(m["B4"].switches_per_sec):.1f} switches/s, and the confidence was not predictive of pulse quality '
        f'(Spearman ρ = {b5.spearman_q_snr:.2f}, AUROC = {b5.auroc:.2f}) and was overconfident (ECE = {b5.ece_raw:.2f}). '
        'Module 1 thus provides a reproducible, inspectable infant-evidence representation and a benchmark protocol that '
        'quantifies where dynamic ROI selection helps and where learned, temporally regularized reliability is required.')


def results(b, R):
    m = by_id(R['table3_main_all_videos'])
    meta = R['meta']
    b.h1('5. Results')

    # ---------------- 5.1 main ablation ----------------
    b.h2('5.1 Progressive Component Ablation')
    sup_b0, sup_full = test(R, 'Full', 'mask_support', 'B0'), test(R, 'Full', 'mask_support', 'B5')
    sup_b4b0 = test(R, 'B4', 'mask_support', 'B0')
    b.para(f'Table 3 and Figure 3 summarize the ablation over all {R["n_videos"]} videos. **Infant evidence purity improves '
           'monotonically with each localization component.** With a fixed box, only '
           f'{mean(m["B0"].mask_support):.3f} of the used pixels lie inside the reliable SAM 2 infant mask; anatomy-guided '
           f'dynamic selection raises this to {mean(m["B2"].mask_support):.3f}-{mean(m["B4"].mask_support):.3f} '
           f'(B4 vs. B0: Δ = {sup_b4b0.mean_difference:+.3f}, r = {sup_b4b0.rank_biserial:.2f}, p_Holm {p(sup_b4b0.p_holm)}), '
           f'SAM 2 guidance raises it further to {mean(m["B5"].mask_support):.3f}, and pixel refinement uses mask pixels '
           f'exclusively (1.000 by construction; Full vs. B5 Δ = {sup_full.mean_difference:+.3f}, p_Holm '
           f'{p(sup_full.p_holm)}). Coverage, the share of sampled frames with a valid representation, is lower for '
           f'dynamic configurations ({mean(m["B4"].coverage_percent):.1f}% without and '
           f'{mean(m["Full"].coverage_percent):.1f}% with SAM 2) than for the fixed baselines (100% by construction): '
           'dynamic configurations deliberately return no ROI when the infant is not localized or no candidate region is '
           'valid, whereas a fixed ROI keeps reporting pixels that may no longer contain the infant.')
    sw = test(R, 'B4', 'switches_per_sec', 'B2')
    b.para('**Temporal stability degrades as condition terms are added.** Region selection driven by anatomy alone is '
           f'stable ({mean(m["B2"].switches_per_sec):.2f} top-1 switches/s, temporal IoU {mean(m["B2"].temporal_iou):.3f}). '
           f'Adding skin and visibility raises switching to {mean(m["B3"].switches_per_sec):.2f}/s, and adding the temporal '
           f'and motion terms to {mean(m["B4"].switches_per_sec):.2f}/s (B4 vs. B2: Δ = {sw.mean_difference:+.2f}/s, '
           f'r = {sw.rank_biserial:.2f}, p_Holm {p(sw.p_holm)}), with temporal IoU falling to '
           f'{mean(m["B4"].temporal_iou):.3f}. Selection Jaccard drops from {mean(m["B2"].selection_jaccard):.3f} to '
           f'{mean(m["B4"].selection_jaccard):.3f}. The equal-weight rule produces many near-ties between neighbouring '
           'regions (e.g., face, head and cheek), and the softmax top-k without hysteresis flips between them from frame '
           'to frame; SAM 2 guidance and pixel refinement do not change the selected geometry (B5 and Full share geometry by '
           'design). This instability is a property of the training-free rule rather than of the candidate geometry.')
    inf_full = test(R, 'Full', 'informative_window_percent', 'B5')
    inf_b0 = test(R, 'Full', 'informative_window_percent', 'B0')
    significant = []
    for name in ('table3_paired_tests_vs_full', 'table3_paired_tests_vs_B4'):
        t = R[name]
        for _, row in t[(t.metric == 'pulse_snr_db') & (t.p_holm < .05)].iterrows():
            significant.append(f'{row.candidate} − {row.baseline} = {row.mean_difference:+.2f} dB (n = {int(row.n)}, '
                               f'p_Holm {p(row.p_holm)})')
    snr_text = ('After Holm correction, no paired SNR contrast was significant. ' if not significant else
                'After Holm correction, the significant SNR contrasts were ' + '; '.join(significant) + '. ')
    sig = inf_full.p_holm < .05
    best = max(IDS, key=lambda c: mean(m[c].informative_window_percent))
    b.para('**Pulse-signal validity.** Median window SNR was low for all configurations (between '
           f'{min(mean(m[c].pulse_snr_db) for c in IDS):.2f} and {max(mean(m[c].pulse_snr_db) for c in IDS):.2f} dB), as '
           'expected for short, compressed and frequently edited web video. ' + snr_text +
           f'Pixel refinement changed the share of pulse-informative windows from '
           f'{mean(m["B5"].informative_window_percent):.1f}% (B5) to {mean(m["Full"].informative_window_percent):.1f}% '
           f'(Full; r = {inf_full.rank_biserial:.2f}, nominal p = {inf_full.p_value:.3f}, p_Holm = {inf_full.p_holm:.3f}), '
           + ('a difference that remains significant after multiplicity correction. ' if sig else
              'a difference that does not survive multiplicity correction. ') +
           f'The highest informative-window share was reached by {best} '
           f'({mean(m[best].informative_window_percent):.1f}%; Full − B0 = {inf_b0.mean_difference:+.1f} pp, '
           f'p_Holm = {inf_b0.p_holm:.3f}).' +
           (' We attribute the advantage of the fixed box to two effects: a large box averages over many pixels, which '
            'suppresses compression noise, and periodic content in the box need not originate from infant skin '
            '(Section 6). The pulse proxy therefore does not show that dynamic selection yields stronger pulse signals in '
            'this corpus; it shows that dynamic selection yields purer infant evidence at similar pulse quality.'
            if best in ('B0', 'B1') else ''))

    b.caption(f'Table 3. Main ablation over all {R["n_videos"]} videos (mean ± SD over videos; 95% bootstrap CIs in the released tables). '
              '↑ higher is better, ↓ lower is better. Mask support for Full is 1 by construction.')
    rows = []
    for c in IDS:
        r = m[c]
        rows.append([c, r.coverage_percent, r.temporal_iou, r.centroid_jitter, r.switches_per_sec, r.mask_support,
                     r.pulse_snr_db, r.informative_window_percent])
    b.table(['ID', 'Coverage % ↑', 'Temporal IoU ↑', 'Centroid jitter ↓', 'Switches/s ↓', 'Mask support ↑',
             'Pulse SNR (dB) ↑', 'Informative windows % ↑'], rows, size=8)

    b.caption('Table 3b. Paired Wilcoxon signed-rank comparisons (video-level). Δ = candidate − baseline; r = matched-pairs '
              'rank-biserial correlation; p Holm-corrected within metric.')
    pairs = [('Full', 'B5', 'pixel refinement'), ('Full', 'B0', 'Full vs. fixed box'), ('B4', 'B2', 'condition terms'),
             ('B4', 'B3', 'temporal/motion terms'), ('B4', 'B1', 'dynamic vs. fixed anatomy')]
    metrics = [('mask_support', 'Mask support', 3), ('switches_per_sec', 'Switches/s', 2), ('pulse_snr_db', 'SNR (dB)', 2),
               ('informative_window_percent', 'Informative %', 1)]
    rows = []
    for candidate, baseline, label in pairs:
        for key, name, digits in metrics:
            t = test(R, candidate, key, baseline)
            rows.append([f'{candidate} vs. {baseline} ({label})', name, int(t.n), f'{t.mean_difference:+.{digits}f}',
                         f'{t.rank_biserial:.2f}', p(t.p_holm)])
    b.table(['Comparison', 'Metric', 'n', 'Δ mean', 'r', 'p (Holm)'], rows, size=8)

    if figure(b, R['report']/'figures'/'fig3_ablation_metrics.png', 6.5):
        b.caption(f'Figure 3. Ablation metrics per configuration (mean over {R["n_videos"]} videos, error bars = 95% bootstrap CI). Baselines '
                  'in grey ramp; the Full model in blue.')

    # ---------------- 5.2 qualitative ----------------
    b.h2('5.2 Qualitative Analysis')
    b.para('Figure 2 traces the representation through every stage for four automatically selected, uncurated frames: a '
           'typical held-out frame, the highest-motion frame, the frame with the strongest torso occlusion, and the darkest '
           'frame. The pixel-weight map makes explicit which pixels feed the fused trace: within the same box, devices, '
           'tubing, blankets and background receive zero weight, and weight concentrates on exposed skin inside the infant '
           'mask. Because the frames are chosen by diagnostics rather than by hand, the figure also shows failure modes that '
           'the label-free metrics cannot detect on their own: when an adult caregiver dominates the frame, the person-class '
           'detector and the adult-trained pose model can lock onto the adult rather than the infant, and SAM 2 then '
           'faithfully propagates the wrong subject. This is the principal open problem for in-the-wild deployment and '
           'motivates an infant-specific detector (Section 7).')
    if figure(b, R['report']/'figures'/'fig2_qualitative.png', 6.5):
        b.caption('Figure 2. Qualitative stage-wise outputs. Columns: frame, fused box, SAM 2 mask, 308-keypoint pose, 17 '
                  'candidate ROIs, pixel-weight map ω_t(p), final selected ROIs with fusion weights. Frames are selected '
                  'automatically and may include failure cases.')

    # ---------------- 5.3 motion ----------------
    t4 = by_id(R['table4_motion_stratified'])
    b.h2('5.3 Motion-Stratified Robustness')
    b.para('Table 4 and Figure 5 stratify 8 s windows by optical-flow motion inside the infant box, with tertile '
           f'thresholds fixed on training videos ({meta["thresholds"]["motion_tertiles"][0]:.3f} and '
           f'{meta["thresholds"]["motion_tertiles"][1]:.3f} normalized flow). In low-motion windows the Full model has the '
           f'highest informative share ({mean(t4["Full"]["low informative %"]):.1f}% vs. '
           f'{mean(t4["B0"]["low informative %"]):.1f}% for B0 and {mean(t4["B4"]["low informative %"]):.1f}% for B4), '
           f'but its share falls to {mean(t4["Full"]["medium informative %"]):.1f}% and '
           f'{mean(t4["Full"]["high informative %"]):.1f}% in medium- and high-motion windows, while the fixed box stays '
           f'near {mean(t4["B0"]["high informative %"]):.0f}%. Motion therefore remains the dominant failure condition '
           'for the pixel-refined representation: under strong motion the selected regions switch, refined pixel sets '
           'shrink, and the hold of 0.25 s between sampled frames is insufficient. Few videos contain windows in both the '
           'lowest and highest tertile, so within-video drops cannot be estimated reliably and are not reported.')
    b.caption('Table 4. Pulse-informative windows (%) and median SNR by motion tertile (mean ± SD over videos; n = videos '
              'with windows in the stratum).')
    rows = [[c, t4[c]['low informative %'], t4[c]['medium informative %'], t4[c]['high informative %'],
             t4[c]['low SNR dB'], t4[c]['high SNR dB'],
             f"{int(t4[c]['low videos'])}/{int(t4[c]['medium videos'])}/{int(t4[c]['high videos'])}"] for c in IDS]
    b.table(['ID', 'Low motion %', 'Medium %', 'High %', 'Low SNR (dB)', 'High SNR (dB)', 'n (L/M/H)'], rows, size=8)
    if figure(b, R['report']/'figures'/'fig5_motion_strata.png', 4.2):
        b.caption('Figure 5. Pulse-informative windows by motion tertile for the fixed box (B0), anatomy-only selection (B2), '
                  'the condition-aware rule without SAM 2 (B4) and the Full model.')

    # ---------------- 5.4 conditions ----------------
    t5 = R['table5_condition_stratified']
    b.h2('5.4 Condition-Stratified Robustness')
    strata = [('Occlusion proxy', ['low', 'partial', 'strong']), ('Illumination', ['low light', 'mid', 'bright']),
              ('Swaddling proxy', ['low', 'mid', 'swaddled']), ('Illumination transition', ['stable', 'transition'])]
    show = ['B0', 'B2', 'B4', 'B5', 'Full']
    rows = []
    for condition, levels in strata:
        part = by_id(t5[t5.condition == condition].reset_index(drop=True))
        for level in levels:
            rows.append([condition if level == levels[0] else '', level] +
                        [part[c][f'{level} informative %'] for c in show])
    occ = by_id(t5[t5.condition == 'Occlusion proxy'].reset_index(drop=True))
    light = by_id(t5[t5.condition == 'Illumination'].reset_index(drop=True))
    strong = lambda c: mean(occ[c]['strong informative %'])
    b.para('Table 5 stratifies windows by the occlusion proxy (1 − skin likelihood), mean luminance in the infant box, the '
           'swaddling proxy (1 − mean limb keypoint confidence), and abrupt luminance transitions (frame-to-frame change '
           'above the training 90th percentile). Two patterns stand out. First, under strong occlusion, adding skin, '
           f'temporal and motion terms lowers the informative share from {strong("B2"):.1f}% (B2) to {strong("B4"):.1f}% '
           f'(B4), and SAM 2 guidance and pixel refinement recover it ({strong("B5"):.1f}% and {strong("Full"):.1f}%), to '
           f'the level of the fixed box ({strong("B0"):.1f}%); restricting colour evidence to mask and skin pixels is thus '
           'what keeps a partially covered infant usable. Second, low light is difficult for every SAM 2 configuration '
           f'(Full {mean(light["Full"]["low light informative %"]):.1f}% vs. {mean(light["Full"]["mid informative %"]):.1f}% '
           'at mid luminance), because adaptive skin masks become sparse in '
           'dark frames. Strata contain '
           f'{strata_n(t5)} videos each, so these comparisons are descriptive.')
    b.caption('Table 5. Pulse-informative windows (%) by condition stratum (mean ± SD over videos; thresholds from training '
              'videos).')
    b.table(['Condition', 'Stratum'] + show, rows, size=8)

    # ---------------- 5.5 Q validity ----------------
    q = R['table_q_validity']
    b.h2('5.5 Validity and Calibration of the Region Confidence')
    names = dict(zip(R['table3_main_all_videos'].configuration, IDS))
    q_all = {names[r.configuration]: r for _, r in q[q.split == 'all'].iterrows()}
    q_test = {names[r.configuration]: r for _, r in q[q.split == 'test'].iterrows()}
    def assoc(c):
        r = q_all[c]
        kind = 'uninformative' if r.auroc < .55 and abs(r.spearman_q_snr) < .1 else 'weakly informative' if r.auroc < .65 else 'informative'
        return f'{c} ρ = {r.spearman_q_snr:.3f} (p = {r.spearman_p:.3f}), AUROC {r.auroc:.3f}: {kind}'
    b5 = q_all['B5']
    base_brier = b5.positive_rate*(1 - b5.positive_rate)
    collapsed = b5.brier_platt >= base_brier - .005
    train_rho = q[q.split == 'train'].iloc[-1].spearman_q_snr
    if 'B5' in q_test:
        t = q_test['B5']
        flip = (t.spearman_q_snr > 0) != (train_rho > 0)
        test_text = (f'On the held-out test split, B5 reaches AUROC {t.auroc:.2f} (ρ = {t.spearman_q_snr:.2f}) on '
                     f'{int(t.roi_windows)} ROI-windows from {R["n_split"].get("test", 0)} videos' +
                     (f', but the sign disagrees with the training split (ρ = {train_rho:.2f}), so we do not interpret it. '
                      if flip else f' (training split ρ = {train_rho:.2f}). '))
    else:
        test_text = ''
    b.para(f'We paired {int(q_all["B4"].roi_windows):,} ROI-windows (B2-B4; {int(q_all["B5"].roi_windows):,} for the SAM 2 '
           'branch) with the pulse SNR of the same ROI. Association with pulse quality: ' +
           '; '.join(assoc(c) for c in ('B2', 'B3', 'B4', 'B5')) + '. '
           f'The raw rescaled confidence has ECE {q_all["B4"].ece_raw:.2f} for B4 and {b5.ece_raw:.2f} for B5 (Figure 4). '
           f'Platt scaling fitted on training videos reduces the ECE of B5 to {b5.ece_platt:.3f} on all videos' +
           (f', but its Brier score ({b5.brier_platt:.3f}) equals that of predicting the base rate ({base_brier:.3f}), '
            'i.e. calibration is achieved by collapsing the confidence to a constant, not by gaining discrimination. '
            if collapsed else f' and its Brier score to {b5.brier_platt:.3f} (base rate {base_brier:.3f}). ') +
           test_text +
           ('We conclude that the equal-weight condition score is a useful auditable descriptor but not a validated '
            'reliability estimate for pulse extraction; Section 6 discusses learning it.' if b5.auroc < .65 else
            'The condition score therefore carries usable information about pulse quality, although calibration still '
            'requires a fitted map; Section 6 discusses learning it.'))
    b.caption('Table 6. Confidence validity against the pulse proxy (ROI-window level). ECE/Brier after Platt scaling use a '
              'calibrator fitted on training videos only.')
    rows = []
    for c in ('B2', 'B3', 'B4', 'B5'):
        for split, table in (('all', q_all), ('test', q_test)):
            if c not in table:
                continue
            r = table[c]
            rows.append([c, split, int(r.roi_windows), f'{r.spearman_q_snr:.3f}', f'{r.auroc:.3f}', f'{r.auprc:.3f}',
                         f'{r.positive_rate:.2f}', f'{r.ece_raw:.3f}', f'{r.ece_platt:.3f}', f'{r.brier_raw:.3f}',
                         f'{r.brier_platt:.3f}'])
    b.table(['ID', 'Split', 'ROI-windows', 'Spearman ρ', 'AUROC', 'AUPRC', 'Pos. rate', 'ECE raw', 'ECE Platt',
             'Brier raw', 'Brier Platt'], rows, size=7.5)
    if figure(b, R['report']/'figures'/'fig4_q_reliability.png', 3.4):
        b.caption('Figure 4. Reliability diagram of the B5 region confidence on held-out (validation + test) ROI-windows, raw '
                  'and after Platt scaling fitted on training videos.')
    w = json.loads((R['report']/'table6_weight_regimes.json').read_text())['weakly_supervised']['fitted_weights']
    b.para('As a weight-sensitivity check, the weakly supervised regime of Module 1 (projected-gradient fitting of the five '
           'weights to a pseudo-reliability target on training videos only) moves weight towards anatomy '
           f'(w_P = {w["pose"]:.2f}, w_C = {w["temporal"]:.2f}, w_V = {w["visibility"]:.2f}, w_S = {w["skin"]:.2f}, '
           f'w_M = {w["motion"]:.2f}). Because its target is itself built from the same condition terms, it cannot add the '
           'missing physiological signal to the score and is reported only as a configuration option.')

    # ---------------- 5.6 generalization ----------------
    g = R['table6_split_generalization']
    b.h2('5.6 Cross-Video Generalization')
    b.para('No component is fitted to the evaluation videos, so the split tests whether conclusions transfer to unseen recordings '
           'rather than whether a model overfits. Table 7 reports the training, validation and test splits separately. The '
           'geometric pattern (mask support and the stability cost of condition terms) is preserved on held-out videos. The '
           f'pulse proxy is not: all dynamic configurations reach almost no informative windows on the {words(R["n_split"].get("test", 0))} test videos, '
           'which coincide with the hardest perception conditions in the corpus (Section 5.7: fused detection rate '
           f'{R["table_perception_diagnostics"].set_index(["component", "metric"]).loc[("Detection: YOLO+RT-DETR:or", "detection_rate"), "test"]} '
           'on test vs. '
           f'{R["table_perception_diagnostics"].set_index(["component", "metric"]).loc[("Detection: YOLO+RT-DETR:or", "detection_rate"), "train"]} '
           'on training videos). ' + (f'With {words(min(R["n_split"].get("validation", 0), R["n_split"].get("test", 0)))} '
           'videos per held-out split these results characterize the variability of in-the-wild data rather than '
           'establish generalization.' if min(R['n_split'].get('validation', 0), R['n_split'].get('test', 0)) < 10 else ''))
    b.caption('Table 7. Results by predeclared split (mean ± SD over videos).')
    rows = []
    gid = {**names}
    for _, r in g.iterrows():
        cid = gid[r.configuration]
        if cid in ('B0', 'B2', 'B4', 'B5', 'Full'):
            rows.append([r.split, cid, int(r.videos), r.temporal_iou, r.centroid_jitter, r.pulse_snr_db,
                         r.informative_window_percent])
    b.table(['Split', 'ID', 'Videos', 'Temporal IoU', 'Centroid jitter', 'Pulse SNR (dB)', 'Informative %'], rows, size=8)

    # ---------------- 5.7 perception ----------------
    d = R['table_perception_diagnostics'].set_index(['component', 'metric'])
    b.h2('5.7 Perception Diagnostics')
    get = lambda c, k, s='all': d.loc[(c, k), s]
    b.para('Table 8 reports the upstream components that bound Module 1. OR fusion of YOLO11s and RT-DETR-L raises the '
           f'per-frame detection rate to {get("Detection: YOLO+RT-DETR:or", "detection_rate")} from '
           f'{get("Detection: YOLO", "detection_rate")} (YOLO) and {get("Detection: RT-DETR", "detection_rate")} (RT-DETR) '
           f'and shortens the longest miss; AND fusion is the most conservative ({get("Detection: YOLO+RT-DETR:and", "detection_rate")}). '
           f'ByteTrack covers {get("Tracking: ByteTrack", "track_coverage")} of frames. Bounded recovery bridged only '
           f'{get("Bounded recovery", "recovered_frames")} frames per video: nearly all misses in this corpus '
           'exceed the 10-frame cap and are counted as failures rather than extrapolated. '
           f'SAM 2 produces a reliable mask in {get("SAM 2 segmentation", "mask_observation_rate")} of frames with mean '
           f'predicted quality {get("SAM 2 segmentation", "mean_mask_quality")}. Pose is usable (status VALID or FLAGGED) in '
           f'{get("Pose (no SAM 2)", "usable_pose_rate")} of frames, but no frame passes the strict VALID criterion of the '
           'adult-calibrated pose quality check, a direct measurement of the adult-to-infant domain gap. SAM 2-guided pose '
           f'raises mean keypoint confidence on test videos from {get("Pose (no SAM 2)", "mean_keypoint_confidence", "test")} '
           f'to {get("Pose (SAM 2-guided)", "mean_keypoint_confidence", "test")}.')
    keep = [('Detection: YOLO', 'detection_rate'), ('Detection: RT-DETR', 'detection_rate'),
            ('Detection: YOLO+RT-DETR:and', 'detection_rate'), ('Detection: YOLO+RT-DETR:or', 'detection_rate'),
            ('Detection: YOLO+RT-DETR:or', 'longest_consecutive_miss'), ('Tracking: ByteTrack', 'track_coverage'),
            ('Tracking: ByteTrack', 'observed_primary_id_changes'), ('Bounded recovery', 'detector_miss_rate'),
            ('Bounded recovery', 'recovered_frames'), ('SAM 2 segmentation', 'mask_observation_rate'),
            ('SAM 2 segmentation', 'mean_mask_quality'), ('SAM 2 segmentation', 'mean_temporal_consistency'),
            ('Pose (no SAM 2)', 'usable_pose_rate'), ('Pose (no SAM 2)', 'mean_keypoint_confidence'),
            ('Pose (SAM 2-guided)', 'usable_pose_rate'), ('Pose (SAM 2-guided)', 'mean_keypoint_confidence')]
    b.caption('Table 8. Perception diagnostics (mean ± SD over videos; label-free).')
    b.table(['Component', 'Metric', 'All', 'Train', 'Validation', 'Test'],
            [[c.replace('Detection: ', '').replace('YOLO+RT-DETR:', 'Fusion: ').replace(': and', ' (AND)').replace(': or', ' (OR)'), k.replace('_', ' ')] +
             [d.loc[(c, k), s] for s in ('all', 'train', 'validation', 'test')] for c, k in keep], size=7.5)

    # ---------------- 5.8 failure ----------------
    f = R['table8_failure_cases']
    measured = f[f['Videos (n)'].notna()]
    b.h2('5.8 Failure-Case Analysis')
    ext = measured.iloc[0]
    b.para('Table 9 mines failure episodes from diagnostics with thresholds fixed on training videos and reports how often '
           'the representation becomes conservative (no ROI, or best confidence below the training 25th percentile, '
           f'{meta["thresholds"]["q_low"]:.2f}) and whether it recovers within 1 s. Extended detector loss is the most severe '
           f'condition ({int(ext["Episodes (n)"])} episodes in {int(float(ext["Videos (n)"]))} videos, recovery within 1 s in '
           f'{ext["Recovered within 1 s (%)"]}%); in these segments the representation correctly abstains rather than '
           'reporting background pixels. High nuisance motion produces the most episodes, and the representation is '
           'conservative in nearly all of them. Four clinically defined failure types (near-complete swaddling, caregiver '
           'occlusion with infant motion, reflective incubator artifacts and unusual postures) require expert labels and '
           'are left to the released annotation kit.')
    b.caption('Table 9. Failure episodes (thresholds from training videos).')
    b.table(list(f.columns), [[('' if str(v) == 'nan' else (int(float(v)) if isinstance(v, float) and float(v).is_integer() else v))
                               for v in row] for row in f.values.tolist()], size=7.5)

    # ---------------- 5.9 compute ----------------
    c = R['table_compute']
    total = c.iloc[-1]
    params = meta['compute']['parameters']
    b.h2('5.9 Computational Cost')
    b.para(f'The full pipeline processed {meta["compute"]["native_frames"]:,} native frames ({meta["compute"]["sampled_frames"]:,} '
           f'sampled) in {total.wall_time_sec/3600:.1f} GPU-hours for both branches, i.e. {total.native_fps_equivalent} '
           'native frames/s including model loading, with a peak GPU memory of '
           f'{meta["compute"]["peak_gpu_memory_mib"]/1024:.1f} GiB. The cost is dominated by 308-keypoint pose at 1024×768 '
           f'(~{c.set_index("phase").loc["pose", "native_fps_equivalent"]:.1f} native frames/s) and per-ROI feature extraction '
           f'with dense optical flow (~{c.set_index("phase").loc["roi_quality", "native_fps_equivalent"]:.1f} frames/s), whereas '
           f'detection, tracking, SAM 2 ({c.set_index("phase").loc["sam2", "native_fps_equivalent"]:.1f} frames/s) and the '
           'confidence rule are not limiting. The models total '
           f'{sum(params.values())/1e6:.0f} M parameters, of which {params["Sapiens2 0.4B pose"]/1e6:.0f} M are in the pose '
           'backbone. Module 1 is therefore an offline representation at present; a smaller pose model, sparse '
           'flow and sampling below 20 FPS are the obvious routes to real time.')
    b.caption('Table 10. Per-phase wall time and throughput (one A100-80GB; both branches).')
    b.table(['Phase', 'Wall time (s)', 'Sampled FPS', 'Native FPS equiv.'],
            [[r.phase, f'{r.wall_time_sec:,.1f}', r.sampled_fps, r.native_fps_equivalent] for _, r in c.iterrows()], size=8)


def discussion(b, R):
    m = by_id(R['table3_main_all_videos'])
    b.h1('6. Discussion')
    b.para('**What dynamic selection buys.** The clearest effect in this study is purity of evidence. A fixed box admits '
           f'about {100*(1-mean(m["B0"].mask_support)):.0f}% non-infant pixels on average, and anatomy-guided selection '
           f'reduces this to {100*(1-mean(m["B4"].mask_support)):.0f}%, SAM 2 to {100*(1-mean(m["B5"].mask_support)):.0f}% '
           'and pixel refinement to zero, while preserving explicit anatomical identity per region. This matters most for '
           'downstream users that need to know where a signal comes from (e.g., separating thoraco-abdominal motion for RR '
           'from facial colour change for HR), which a fixed box cannot provide.')
    best = max(IDS, key=lambda c: mean(m[c].informative_window_percent))
    if best in ('B0', 'B1'):
        b.para('**Why purity did not translate into a stronger pulse proxy.** The fixed box reached the highest '
               'informative-window share. Two mechanisms explain this without contradicting the purity result. Averaging over '
               'many pixels suppresses compression and sensor noise, which dominates in short, re-encoded web video; and the '
               'POS projection extracts any in-band periodic chrominance, including from caregiver skin, monitor flicker or '
               'camera motion, so a high SNR is not proof of infant pulse. Conversely, the rapid region switching of the '
               'condition-aware rule injects step changes into the fused trace that spread energy across the spectrum. The '
               'pixel-level stage partly compensates (B5 → Full), indicating that the pixel set, not the region identity, is '
               'what the pulse proxy rewards.')
    b.para('**What the confidence analysis implies.** A training-free, equal-weight score built from appearance descriptors '
           'is auditable but not a reliability estimate: it does not rank ROI-windows by pulse quality and is overconfident. '
           'Two concrete changes follow from the measurements and do not alter the architecture: (i) temporal hysteresis or '
           'smoothing of the fusion weights to remove the near-tie switching, and (ii) learning the weights, or a small '
           'reliability head over the same condition vector, against an external signal (expert ROI labels or reference '
           'HR) rather than a pseudo-target derived from the same terms. Domain-adversarial training of such a head [34] '
           'is a natural extension once labels from more than one site exist.')

    b.h1('7. Limitations')
    b.bullets([
        f'**Data.** The dataset consists of {R["n_videos"]} short, publicly available NICU-themed web videos with captions, scene cuts, '
        'split screens and segments without an infant. They stress-test robustness but are not continuous bedside '
        'recordings, contain no synchronized physiological reference, and '
        f'{R["n_split"].get("validation", 0)} validation and {R["n_split"].get("test", 0)} test videos are too few to '
        'establish generalization. Evaluation on clinical NICU recordings with ECG/impedance references is required.',
        '**No ground truth.** All results are label-free diagnostics or signal-quality proxies. Mask support depends on '
        'SAM 2 being correct, and the pulse SNR measures in-band periodicity rather than HR accuracy. Expert ROI validity, '
        'precision/recall and Dice are not reported; the annotation kit provides the protocol.',
        '**Subject localization.** Detection uses a generic person class and pose an adult-trained model; in frames '
        'dominated by caregivers the pipeline can track an adult (Figure 2). An infant-specific detector or identity '
        'verification is needed.',
        '**Scoring rule.** The confidence is training-free with equal weights; visibility is computed as one minus the '
        'occlusion proxy, which itself is one minus skin likelihood, so skin evidence is effectively double-weighted. The '
        'rule is unstable over time and not calibrated to pulse quality (Sections 5.1 and 5.5).',
        '**Motion.** Motion is a total-flow penalty without separating camera, caregiver and infant motion, and high-motion '
        'windows remain the dominant failure condition.',
        '**Compute.** Throughput of about 0.4 native frames/s for both branches precludes real-time use without model '
        'reduction.',
    ])

    b.h1('8. Conclusion')
    b.para('We presented Module 1, a dynamic ROI representation for neonatal video that combines complementary '
           'localization, SAM 2 masks, 308-keypoint anatomy and an interpretable condition vector to select regions and '
           'pixels per frame, together with a leakage-controlled, label-free evaluation protocol. '
           f'On {R["n_videos"]} in-the-wild neonatal '
           'videos, the representation keeps colour evidence on the infant (mask support '
           f'{mean(m["B0"].mask_support):.2f} → 1.00)' + (' and pixel refinement increases the share of pulse-informative windows, '
           if mean(m['Full'].informative_window_percent) > mean(m['B5'].informative_window_percent) else ', ') +
           'while the evaluation shows that the training-free confidence rule is temporally unstable and not predictive of '
           'pulse quality. These findings define the next steps (temporal regularization of the fusion weights, a learned '
           'reliability estimate, infant-specific localization, and clinical validation with physiological references) and '
           'provide the benchmark against which they can be measured. Code, configuration, per-video tables and the '
           'annotation kit are released.')
