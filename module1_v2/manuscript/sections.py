"""Front matter, Introduction, Related Work, Method and Experimental Setup of the Module 1 v2 manuscript.

Text that depends on the data (dataset size, tuned values, thresholds) is filled from the loaded results `D`.
"""
from results_setup import REFERENCES as V1_REFERENCES   # v1 reference list [1]-[37], reused unchanged

NEW_REFERENCES = [
    '[38] G. D. Forney, "The Viterbi algorithm," Proceedings of the IEEE, vol. 61, no. 3, pp. 268-278, 1973, doi: 10.1109/PROC.1973.9030.',
    '[39] J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, and J. D. Farmer, "Testing for nonlinearity in time series: the method of surrogate data," Physica D, vol. 58, no. 1-4, pp. 77-94, 1992, doi: 10.1016/0167-2789(92)90102-S.',
    '[40] B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision," in IJCAI, 1981, pp. 674-679.',
    '[41] F. Wilcoxon, "Individual Comparisons by Ranking Methods," Biometrics Bulletin, vol. 1, no. 6, pp. 80-83, 1945, doi: 10.2307/3001968.',
    '[42] S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65-70, 1979.',
    '[43] D. S. Kerby, "The Simple Difference Formula: An Approach to Teaching Nonparametric Correlation," Comprehensive Psychology, vol. 3, 2014, doi: 10.2466/11.IT.3.1.',
    '[44] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.',
    '[45] G. Jocher and J. Qiu, "Ultralytics YOLO11," version 11.0.0, software, 2024. [Online]. Available: https://github.com/ultralytics/ultralytics',
    '[46] R. Khirodkar, H. Wen, J. Martinez, Y. Dong, Z. Su, and S. Saito, "Sapiens2," in ICLR, 2026, arXiv:2604.21681.',
    '[47] S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography," Pattern Recognition Letters, vol. 124, pp. 82-90, 2019, doi: 10.1016/j.patrec.2017.10.017.',
    '[48] R. Stricker, S. Müller, and H.-M. Gross, "Non-contact video-based pulse rate measurement on a mobile service robot," in IEEE RO-MAN, 2014, pp. 1056-1062, doi: 10.1109/ROMAN.2014.6926392.',
]
REFERENCES = V1_REFERENCES + NEW_REFERENCES


def introduction(b, D):
    b.h1('1. Introduction')
    b.para('Camera-based sensing offers a non-contact pathway for neonatal monitoring: an ordinary RGB camera records both the '
           'subtle colour changes caused by the blood volume pulse and the visible motion of breathing, without attaching '
           'anything to fragile skin [1]-[3], [19]-[23]. In neonatal video, however, the region from which such signals are '
           'read is itself the hard part. Infants are supine, prone or side-lying, swaddled, handled by caregivers and '
           'surrounded by equipment, so a fixed facial or thoraco-abdominal crop mixes skin with blankets, gloves, tubes and '
           'background, and drifts as the infant moves.')
    b.para('In our baseline pipeline (hereafter v1) we framed ROI construction as a standalone, auditable vision problem: '
           'complementary detectors, tracking and bounded recovery localize the infant, SAM 2 [31] propagates an infant mask, '
           'a 308-keypoint pose model [32] defines 17 anatomical candidate regions, an interpretable condition vector scores '
           'each region, and selected regions are refined to region ∩ mask ∩ skin pixels. A label-free evaluation of that '
           'design exposed three concrete defects. (i) The rule-based per-frame top-k selection was temporally unstable, '
           'switching the top-ranked region roughly ten times per second because neighbouring regions have nearly equal '
           'scores. (ii) Two of its terms were redundant: the visibility term was defined as one minus an occlusion proxy that '
           'is itself derived from skin likelihood, so skin evidence was counted twice. (iii) Its motion penalty used total '
           'in-region optical flow, which conflates camera motion, gross body motion and the periodic chest motion that '
           'carries respiratory information.')
    b.para('This paper presents NeoSTAR (Neonatal STAbilized ROI), which addresses these defects with design changes that keep every decision '
           'inspectable, and evaluates them with a stricter protocol. All new behaviour sits behind configuration flags; with '
           'every flag off the new code reproduces the v1 results exactly, which we verify on every run. Parameters are tuned '
           'on training videos only, and no ground-truth physiological reference is used anywhere. Because the evaluation '
           'videos have no heart-rate reference, we pay particular attention to the validity of the label-free proxy itself: '
           'besides pulse signal-to-noise ratio we test whether a pulse is present at all against surrogate nulls built from '
           'the same pixels, and whether two disjoint halves of the selected skin agree on the pulse frequency.')
    b.para('The main contributions are as follows:')
    b.bullets([
        'A learning-free temporal stabilization layer for dynamic ROI selection, combining exponential smoothing of region '
        'confidence, hysteresis and Viterbi decoding of the anchor region [38] (offline, and online with a 1-s lag), whose '
        'four parameters are selected on training videos by an explicit, colour-free objective. It reduces top-region '
        f"switching from {D['sw']['Full']:.1f} to {D['sw']['D3']:.2f} switches/s offline with the stabilizer alone (D3; "
        f"{D['sw'][D['chosen']]:.2f} switches/s in the headline configuration {D['chosen']}) and to "
        f"{D['sw']['D3L']:.2f} switches/s online, while retaining at least {100 * D['ret_min']:.0f}% of the instantaneous region confidence on held-out "
        'videos.',
        'Corrected condition terms and an operational definition of internal consistency: an independent keypoint-based '
        'visibility term that removes the double counting of skin, a motion decomposition into global (camera) and residual '
        'motion with a respiratory-periodicity exemption, and measurable consistency criteria (term independence, '
        'score-selection agreement, cross-level agreement, candidate validity, and repeatability under frame-to-frame '
        'change, score perturbation, frame-rate reduction and alternative term definitions).',
        'An expanded, leakage-controlled baseline set (whole-infant skin, skin + signal-quality selection, grid tiles and an '
        'oracle upper bound) evaluated on identical pixels and pulse extraction.',
        'Label-free validity tests beyond SNR: a surrogate-null pulse-present test [39] that re-runs colour-dependent '
        'selection on every surrogate, and a split-half heart-rate agreement test.',
        'An efficiency study of the full pipeline, including measured stage costs and the accuracy of reduced frame rates.',
    ])


def related_work(b):
    b.h1('2. Related Work')
    b.h2('2.1 Static and Dynamic ROI Selection')
    b.para('Colour-based camera physiology selects skin pixels and extracts temporal colour variation [1]-[3]. ROI choice '
           'materially changes signal quality: Kim et al. compared anatomically defined facial regions [12], and Kiddle et al. '
           'selected high-quality facial tiles dynamically [11]. In neonates, Villarroel et al. combined skin-region analysis '
           'with signal-quality assessment [19], and Khanam et al. used neural ROI selection [20]. Grid tiling and skin + '
           'signal-quality selection are therefore natural baselines for an anatomy-guided method, and we include both '
           '(M1, M2) together with whole-infant skin (M0).')
    b.h2('2.2 Learned Spatial Attention and Motion Robustness')
    b.para('Deep rPPG models learn spatial attention and motion robustness end to end [4]-[10], [13]-[18]. Their attention is '
           'optimized for a downstream physiological estimate and requires physiological labels; it does not produce an '
           'explicit, temporally tracked, infant-specific ROI representation. NeoSTAR is complementary: it produces such a '
           'representation without labels, and a learned reliability model is a natural extension once labelled data exist.')
    b.h2('2.3 Localization, Segmentation and Infant Anatomy')
    b.para('The front-end uses established models as building blocks: YOLO-family and RT-DETR detectors [27], [45], [28], ByteTrack [29] '
           '(BoT-SORT [30] is a comparable tracker), the promptable segmenter SAM [33] and its video extension SAM 2 [31], and a '
           'Sapiens pose models [32], [46]. Infant pose literature documents the adult-to-infant domain gap of such '
           'models [24]-[26], which is one reason we measure, rather than assume, the reliability of anatomy-derived cues.')
    b.h2('2.4 Temporal Decision Smoothing and Surrogate Testing')
    b.para('Selecting one of several noisy candidates over time is a classic sequential decision problem; hysteresis and '
           'Viterbi decoding [38] trade instantaneous score against the cost of switching. Surrogate-data testing [39] '
           'compares a statistic with its distribution under a null that preserves selected properties of the data. We use '
           'block shuffling in time, which keeps the colour distribution of the pixels but destroys any rhythm. A common '
           'alternative, independent per-channel phase randomization, keeps each channel\'s power spectrum; we show that it '
           'therefore also keeps a genuine pulse and is unsuitable as a pulse-present null.')
    b.h2('2.5 Research Gap')
    b.para('Prior neonatal studies establish clinical feasibility [19]-[23] but do not provide a temporally stable, auditable '
           'infant ROI representation together with label-free tests of whether the selected pixels carry a pulse at all. '
           'NeoSTAR targets exactly this gap.')


def method(b, D):
    t = D['tuning']
    p1 = D['cfg']['phase1']
    b.h1('3. Method')
    b.h2('3.1 Overview and Problem Formulation')
    b.para('Let V = {I_t} be an RGB video sampled at up to 20 FPS with native timestamps. For each sampled frame NeoSTAR '
           'produces a set S_t of k = 3 selected anatomical regions out of K = 17 candidates, fusion weights α_{t,r}, a '
           'refined pixel set R̃_{t,r} for every selected region, and the condition vector z_{t,r} of every candidate:')
    b.eq('Z_t = { R̃_{t,r}, α_{t,r}, Q_{t,r}, z_{t,r} }_{r ∈ S_t}.')
    b.para('The pipeline has eight stages (Figure 1). Stages 1-4 are unchanged from v1: (1) frame validation and '
           'sampling at up to 20 FPS with native timestamps; (2) dual infant detection (YOLO11s [45], RT-DETR-L [28]) with OR '
           'fusion, ByteTrack tracking and bounded recovery; (3) SAM 2.1 infant segmentation with mask reliability gating; '
           'and (4) mask-constrained Sapiens2-0.4B pose [46] with anatomical templates for 17 candidate regions. NeoSTAR '
           'changes stages 5-7: (5) condition-aware characterization with an independent visibility term, a '
           'camera-compensated motion term and an optional subject check (Sections 3.3, 3.4, 3.6); (6) region confidence '
           'with temporal stabilization of the selection (Section 3.5); and (7) rule-based pixel refinement with smoothed, '
           'cross-faded fusion weights (Sections 3.2, 3.5). Stage 8 is the output representation Z_t. No stage contains a '
           'trained model and no physiological signal is used as input; the only data-dependent choices are the four '
           'stabilizer settings of stage 6, selected on training videos (Section 3.5).')
    src = D.get('fig1_source', {})
    b.image(D['fig1'], 6.5)
    b.caption('Figure 1. NeoSTAR architecture, illustrated with real data from one evaluation video '
              f"({src.get('video', '')}, frame {src.get('frame', '')}): sampled frames (1); fused, tracked infant box (2); "
              'SAM 2 infant mask (3); mask-constrained pose skeleton and the anatomical candidate regions (4); examples of '
              'conditions the characterization must handle, from other videos (5); top-region switches over time for the '
              'v1 per-frame rule and the v2 stabilized selection, with the mean switch rates over all videos (6); refined '
              'pixels (green: ROI ∩ SAM 2 mask ∩ adaptive skin) of the regions selected by v2 (7); and the selected regions '
              'with their fusion weights (8). Stages marked v2 were changed in this work. No stage contains a trained model; only '
              'four stabilizer settings (stage 6) are selected on training videos.')
    b.h2('3.2 Condition Vector (Stage 5) and Pixel Refinement (Stage 7) in v1')
    b.para('For every available candidate r the v1 condition vector contains the anatomical keypoint confidence P, the mean '
           'YCrCb skin likelihood S, an occlusion proxy O = 1 − S, the polygon IoU with the previous sampled frame C, the '
           'Farnebäck optical-flow magnitude M [35] bounded by 1 − exp(−m/20 px), blur, exposure and saturation scores, a '
           'swaddling proxy and SAM 2 mask support. The v1 region confidence is the equal-weight rule')
    b.eq('Q_{t,r} = 0.2 (P + V + S + C) − 0.2 M,   V = 1 − O,')
    b.para('followed by per-frame top-k selection and a softmax over the selected set. Selected regions are refined to '
           'shrink(R) ∩ M_t ∩ K_{t,r} ∩ {15 < max_c I < 250}, where M_t is the reliable SAM 2 mask and K_{t,r} an adaptive '
           'YCrCb skin set; native frames between sampled frames reuse the latest geometry for at most 0.25 s. Because '
           'O = 1 − S, the term V equals S exactly, so skin evidence carries twice the weight of the other terms.')
    b.para(f"**Candidate validity (v2).** Some candidate regions, mostly shoulders, limbs and the neck ({D['bbox_share']:.1f}% "
           'of available ROI-frames), are stored as two-point bounding boxes. v1 extracts no pixels from them although they '
           'can be selected, so a selected region may contribute nothing to the fused signal. v2 reads such a box as its '
           'four-corner rectangle before pixel refinement, so every selectable region carries pixels. The fix changes '
           'pixels only, not geometry or selection; it is part of D1-D7, and I-B applies it alone to Full.')
    b.h2('3.3 Independent Visibility Term (Stage 5)')
    b.para('v2 replaces V by a keypoint visibility V′_{t,r}: the fraction of the keypoints that define region r\'s anatomical '
           f'template whose pose confidence is at least {p1["keypoint_confidence"]}. V′ measures whether the body part is seen, '
           'independently of its colour, so the region confidence becomes')
    b.eq('Q′_{t,r} = 0.2 (P + V′ + S + C) − 0.2 M_n,')
    b.para('with M_n the nuisance motion of Section 3.4 (or total motion M when that flag is off). Term redundancy is audited on '
           'training ROI-frames with Pearson and Spearman correlation; pairs with |r| > 0.8 are flagged.')
    b.h2('3.4 Motion Decomposition (Stage 5)')
    b.para('Dense Farnebäck flow is computed between consecutive sampled frames at native resolution. Global motion is the '
           f'median flow over background pixels, i.e. pixels outside the SAM 2 mask dilated by {p1["background_dilation_px"]} '
           'px and outside the infant box enlarged by 10% (a homography fit on the same pixels is available as an option). '
           'The residual flow (total minus global) inside each region gives a residual magnitude, bounded with the same '
           f'{p1["motion_scale_pixels"]:.0f} px scale as v1, and a mean residual velocity vector. For the thoraco-abdominal '
           'regions, the share ρ of the residual velocity power in the respiratory band '
           f'{p1["respiratory_band_hz"][0]}-{p1["respiratory_band_hz"][1]} Hz along the principal motion axis is computed over a '
           f'trailing (causal) {p1["periodicity_window_sec"]:.0f} s window. The penalized nuisance motion is')
    b.eq('M_n = m_res · (1 − ρ_{t,r}) for respiratory regions,   M_n = m_res otherwise,')
    b.para('so periodic chest motion is not penalized, while camera motion no longer penalizes every region equally. The raw '
           'motion descriptors are kept in the representation for downstream respiration analysis.')
    b.h2('3.5 Temporal Stabilization of Selection and Fusion (Stages 6-7)')
    b.para('**Smoothing.** Region confidence is smoothed per region with an exponential moving average '
           'Q̄_{t,r} = β Q̄_{t−1,r} + (1 − β) Q_{t,r}; a region keeps its smoothed value while it is unavailable (it cannot '
           'be selected then), and all memory is reset when the infant is lost. **Hysteresis.** An unselected region replaces '
           'the weakest selected region only after its smoothed score has exceeded the incumbent\'s by more than δ on N '
           'consecutive sampled frames; a vacancy (a selected region becomes unavailable) is filled immediately. '
           '**Viterbi anchor.** The top-ranked (anchor) region is decoded as the path maximizing Σ_t Q̄_{t,a_t} − λ · '
           '#switches [38]; the remaining k − 1 regions are filled around the anchor by hysteresis. The offline decoder uses '
           'the whole video; the online variant decides frame t from frames up to t + L only, with a fixed lag of '
           f'{p1["viterbi_lag_sec"]:.0f} s. **Fusion.** Target weights are the v1 softmax over the selected set; v2 smooths '
           f'them with an EMA (factor {p1["alpha_ema"]}) and limits every weight change to 1/(τ_x · fps) per frame with '
           f'τ_x = {p1["crossfade_sec"]:.0f} s, so a region entering or leaving the set is cross-faded rather than switched.')
    b.para('**Tuning protocol.** β, δ, N and λ are chosen on training videos only, by a colour-free objective that does not '
           'look at any pulse signal: minimize the mean top-1 switch rate subject to a mean quality retention of at least '
           f'{p1["quality_retention"]}, where retention is Σ rescaled Q of the selected set divided by Σ rescaled Q of the '
           'per-frame top-k (how much instantaneous confidence stabilization gives up). Ties are broken by higher retention. '
           f'The grids are β ∈ {p1["ema_beta_grid"]}, δ ∈ {p1["hysteresis_delta_grid"]}, N ∈ {p1["hysteresis_frames_grid"]} '
           f'and λ ∈ {p1["viterbi_penalty_grid"]}. The fusion constants (EMA factor, cross-fade time) are fixed design '
           'constants and were not tuned.')
    b.h2('3.6 Infant-versus-Adult Subject Check (Stage 5)')
    b.para('Caregivers are frequently in view, and a generic person detector and pose model may lock onto them. v2 includes an '
           'optional abstention rule based on two anthropometric ratios from the pose keypoints: head width relative to '
           f'shoulder width r1 (newborns ≈ 0.7, adults ≈ 0.4) and arm length relative to torso length r2. A frame is labelled '
           f'adult when r1 < {p1["head_shoulder_ratio_min"]} or r2 > {p1["arm_torso_ratio_max"]}; a track is rejected while '
           f'adult frames form the majority of the known decisions in a trailing {p1["subject_window_sec"]:.0f} s window, and '
           'rejected frames contribute no region. The thresholds are priors, not fitted: no infant/adult labels exist, and '
           'fitting them on validation or test videos would leak.')


def setup(b, D):
    m, n = D['manifest'], len(D['manifest'])
    split = m.split.value_counts()
    small, large = m.loc[(m.width * m.height).idxmin()], m.loc[(m.width * m.height).idxmax()]
    b.h1('4. Experimental Setup')
    b.h2('4.1 Dataset')
    b.para(f'NeoSTAR is evaluated on {n} colour newborn videos ({int(m.frame_count.sum()):,} native frames, '
           f'{m.duration_sec.sum() / 60:.1f} min; {m.duration_sec.min():.1f}-{m.duration_sec.max():.1f} s per clip; '
           f'{m.fps.min():.0f}-{m.fps.max():.0f} FPS; {int(small.width)}×{int(small.height)} to '
           f'{int(large.width)}×{int(large.height)} px), {D["sampled_frames"]:,} sampled frames in total. The clips are '
           'publicly available short-form web videos of newborns in hospital settings, mostly recorded on phones in portrait '
           'orientation; they contain caregiver hands and gloves, clinicians, warmers and monitoring equipment, swaddling, '
           'burned-in captions and handheld camera motion. They are therefore an unconstrained stress test rather than '
           'continuous bedside recordings. No frame was removed or curated. Every file is identified by its SHA-256 hash, and '
           'no physiological reference signal is available for any video.')
    b.para(f'The split is predeclared and video-level ({split.get("train", 0)} training, {split.get("validation", 0)} '
           f'validation, {split.get("test", 0)} test videos; seeded shuffle, seed 42). Training videos are used to tune the '
           'stabilizer and to fix every evaluation threshold; the validation split is used only to choose the headline '
           'configuration among those that meet the internal-consistency criteria (Section 4.4, checked on training data); '
           'the test split is used for no decision. Infant identities are unknown, so subject independence '
           'cannot be verified. **Ethics statement:** [AUTHORS TO COMPLETE: data source and licence/terms of use of the web '
           'videos, institutional review board determination, and any de-identification applied before publication of '
           'figures.]')
    b.caption('Table 1. Dataset composition by split.')
    rows = []
    for s in ('train', 'validation', 'test'):
        part = m[m.split == s]
        rows.append([s, len(part), f'{int(part.frame_count.sum()):,}', f'{part.duration_sec.sum():.1f}',
                     f'{part.fps.median():.1f}'])
    b.table(['Split', 'Videos', 'Native frames', 'Duration (s)', 'Median FPS'], rows)

    b.h2('4.2 Configurations')
    b.para('All configurations share the sampled frames, perception outputs, k = 3, pixel refinement (where applicable), '
           'pulse extraction and analysis windows, so differences are attributable to the ROI mechanism. B0-Full are the v1 '
           'configurations, unchanged. D0 runs the v2 code with every flag off and must reproduce Full exactly. D1-D7 add the '
           'v2 changes cumulatively, and I-V, I-M and I-S apply a single fix to Full to isolate its effect. M0-M2 are '
           'additional baselines and M3 is an oracle upper bound that is not a method: it chooses, in every window, the region '
           'whose pulse is best in that same window. M1 and M2 select from the previous non-overlapping 8 s window only, so '
           'they are causal; where no such window exists yet (the first 8 s of a clip) they average all of their units. M3 '
           'is not causal.')
    b.caption('Table 2. Configurations.')
    b.table(['ID', 'Definition'], [
        ['B0', 'Fixed box: central half of the first-frame infant box'],
        ['B1', 'Fixed anatomical ROI: first-frame region with the highest anatomical confidence'],
        ['B2 / B3', 'Per-frame top-k by anatomy P / by mean(P, V, S)'],
        ['B4', 'v1 rule Q, no SAM 2'], ['B5', 'v1 rule Q on the SAM 2 branch'],
        ['Full', 'B5 + pixel refinement (v1 baseline)'], ['D0', 'v2 code, all flags off (= Full)'],
        ['I-B', 'Full + two-point-box regions read as rectangles for pixels (part of D1-D7)'],
        ['D1', 'Full + EMA of Q'], ['D2', 'D1 + hysteresis'], ['D3', 'D2 + offline Viterbi anchor'],
        ['D3L', 'D2 + fixed-lag (online) Viterbi anchor'], ['D4', 'D3 + smoothed, cross-faded fusion weights'],
        ['D5', "D4 + keypoint visibility V'"], ['D6', 'D5 + nuisance motion M_n'], ['D7', 'D6 + subject check'],
        ['I-V / I-M / I-S', "Full + V' / + M_n / + subject check only"],
        ['F-Full / F-D2', 'Full / D2 + smoothed, cross-faded fusion weights (fusion-smoothing test, Section 5.2.1)'],
        ['M0', 'SAM 2 mask ∩ adaptive skin over the whole infant box; no anatomy, no scoring'],
        ['M1', 'Anatomical regions (mask ∩ skin), top-3 by signal quality sigmoid(SNR) × (1 − motion) of the previous window'],
        ['M2', '4 × 4 grid tiles of the infant box (mask ∩ skin), top-3 by SNR of the previous window'],
        ['M3', 'Oracle: single anatomical region with the best SNR in the same window (upper bound)']], widths=[1.1, 5.2])

    b.h2('4.3 Implementation Details')
    b.para('Perception settings are those of v1: Ultralytics YOLO11s [45] and RT-DETR-L (person class, 640 px, confidence 0.25), '
           'OR fusion at IoU 0.5, ByteTrack, SAM 2.1 Hiera-Tiny (96-frame chunks), Sapiens2-0.4B at 1024×768. No component '
           'is trained or fine-tuned; v2 adds only the four stabilizer parameters tuned on training videos. The software stack '
           f'is Python {D["versions"]["python"]}, PyTorch {D["versions"]["torch"]}, OpenCV {D["versions"]["cv2"]}, NumPy '
           f'{D["versions"]["numpy"]} and SciPy {D["versions"]["scipy"]} on one NVIDIA A100-SXM4-80GB GPU with an AMD EPYC '
           '7742 CPU (2 cores allocated); the seed is 42.')

    b.h2('4.4 Evaluation Metrics')
    b.para('**Temporal stability** is measured on the selected regions: top-1 switches per second, temporal IoU of the union '
           'of selected regions between consecutive frames, centroid jitter normalized by the infant-box diagonal, and Jaccard '
           'overlap of the selected region names. **Pulse proxy.** The fused colour trace is converted to a pulse with POS [3] '
           '(1.6 s window), detrended (smoothness priors, λ = 100) and band-passed to 90-270 bpm [19]; in 8 s windows with a '
           '1 s step, the de Haan-Jeanne SNR [2] at the in-band spectral peak is computed. We report the per-video median '
           f'window SNR and the share of informative windows, SNR ≥ τ = {D["tau"]:.2f} dB, the median ROI-window SNR of '
           'training videos. SNR measures periodicity in the pulse band, not heart-rate accuracy.')
    nv = D.get('null_validation') or {}
    tp, sp = nv.get('true_pulse_120bpm', {}), nv.get('static_patch', {})
    b.para(f'**Pulse-present test.** For every video and configuration, {D["p2"]["surrogates_shuffle"]} surrogate colour '
           f'traces are generated from the same pixels by permuting {D["p2"]["shuffle_block_sec"]} s blocks in time [39], '
           'passed through the identical pulse extraction and fusion, and windowed. Colour-dependent selection (M1-M3) is '
           're-run on each surrogate, so selection cannot inflate the observed statistic relative to its null. A window is '
           f'pulse-present when its SNR exceeds the {D["p2"]["null_percentile"]:.0f}th percentile of the pooled surrogate '
           'window SNRs, so a window passes by chance with probability about 5%. **Null validation.** On a synthetic video '
           'with a known 120 bpm skin pulse, the shuffle null passed '
           f"{tp.get('shuffle_null_pass_percent', float('nan')):.0f}% of the windows and on a static skin patch "
           f"{sp.get('shuffle_null_pass_percent', float('nan')):.0f}%. Independent per-channel phase randomization, which "
           'keeps every channel\'s spectrum and therefore the pulse line itself, passed '
           f"{tp.get('phase_null_pass_percent', float('nan')):.0f}% of the true-pulse windows and "
           f"{sp.get('phase_null_pass_percent', float('nan')):.0f}% of the static ones; it is computed in our pipeline but "
           'not used, because it rejects genuine pulses. **Negative control.** The same test is applied to pixels that '
           'cannot carry a pulse: four patches at the frame corners (each 20% of the frame width and height), restricted '
           'to background pixels (outside the SAM 2 mask dilated by 15 px, or outside the infant box enlarged by 10% '
           'when no mask is available; the same background as the camera-motion estimate), not skin-filtered, averaged '
           'with equal weights (left and right corners form the two halves for split-half agreement), and passed through '
           'the identical pulse extraction, windows and surrogates. A valid test should pass close to 5% of these windows. **Split-half '
           'agreement.** Each selected region is split at its own vertical midline into two disjoint halves; the two '
           'half-fusions, with the configuration\'s own weights, must give in-band spectral peaks within '
           f'±{D["p2"]["agreement_bpm"]:.0f} bpm. We report the agreeing share of testable windows and the testable share.')
    b.para('**Internal consistency** is defined by measurable properties of the representation, each computed per video: '
           '(i) term independence: no pair of terms that enter the same score correlates with |r| > 0.8 on training '
           'ROI-frames; (ii) score-selection agreement: the selected regions keep at least 98% of the per-frame top-k '
           'confidence (quality retention) on every split; (iii) cross-level agreement: the fusion-weighted share of '
           'selected-region area and of its pose landmarks inside the SAM 2 mask, and the refined-pixel yield (pixels kept by '
           'ROI ∩ mask ∩ adaptive skin over all pixels of the shrunk polygon, every 10th sampled frame); refined pixels lie in '
           'the selected anatomy by construction; (iv) candidate validity: the share of selections that carry no pixels '
           '(two-point boxes); and (v) repeatability: test-retest agreement of every term and of the 17-region ranking '
           '(Kendall τ) between consecutive sampled frames, top-region and set agreement when Gaussian noise (σ = 0.01-0.05) '
           'is added to every score (3 seeds), when only every second sampled frame is processed, when the V′ keypoint '
           'threshold is varied from 0.3 to 0.7, and when global motion is estimated by a RANSAC homography instead of the '
           'background median. Repeatability R = 1 − E[(x_t − x_{t+1})²] / (2 Var x) within each region.')
    b.para('**Efficiency.** Per-stage wall time, peak GPU memory, and accuracy versus projected cost when only every s-th '
           'sampled frame (s = 1-4) carries detection, pose, ROI and quality, with geometry held in between and the '
           'stabilizer\'s time constants preserved (β → β^s, N → N/s).')
    b.h2('4.5 Statistical Analysis')
    b.para('The video is the statistical unit. We report mean ± standard deviation over videos and 95% bootstrap confidence '
           'intervals [44]. Paired comparisons use the two-sided Wilcoxon signed-rank test [41] with the matched-pairs '
           'rank-biserial correlation r as effect size [43]; p-values are Holm-corrected [42] within each metric and family of '
           'comparisons (versus Full, versus B0). All thresholds and tuned parameters are fixed before held-out videos are '
           'evaluated.')
