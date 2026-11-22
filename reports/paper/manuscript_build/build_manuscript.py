"""Completed Module 1 manuscript (.docx). Front matter + method here; results from reports/paper via results.py."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from docbuild import Builder
import results

# Usage: python build_manuscript.py [REPORT_DIR] [OUTPUT_DOCX]   (defaults: this folder's parent report)
REPORT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else REPORT/'Module1_Manuscript_completed.docx'

b = Builder(HERE/'orig_docx/basepaper.docx')
R = results.load(REPORT)

b.title('Motion-Aware, Pixel-Refined Dynamic ROI Representation for Robust Neonatal Video Analysis')
b.para('**Research question:** Does a condition-aware, anatomy-guided and pixel-refined dynamic ROI representation provide '
       'more stable and more physiologically informative infant visual evidence than fixed or region-level ROI selection '
       'under unconstrained neonatal video conditions?')
b.para('**Abstract:** ' + results.abstract(R))
b.para('**Keywords:** neonatal video, NICU, dynamic ROI, pixel-level refinement, motion-aware representation, '
       'infant segmentation, pose estimation, label-free evaluation, non-contact monitoring')

b.h1('1. Introduction')
b.para('Camera-based sensing provides a promising non-contact pathway for neonatal monitoring because a conventional RGB '
       'camera can capture both subtle skin appearance changes and visible body motion without attaching additional hardware '
       'to fragile skin. Early remote photoplethysmography (rPPG) work established that physiological information can be '
       'recovered from ordinary video under ambient light [1]-[3], and subsequent deep models have learned increasingly '
       'powerful spatiotemporal representations [4]-[10]. In the NICU, however, the central visual problem is substantially '
       'harder than in adult frontal-face benchmarks. Neonates may be supine, prone, side-lying, partially covered, swaddled, '
       'touched by caregivers, surrounded by equipment, or observed through changing illumination and viewpoint. Clinical '
       'feasibility studies confirm the potential of camera-based neonatal HR/RR monitoring, but also expose the importance '
       'of robust localization, signal-quality assessment, and condition-aware region selection [19]-[23].')
b.para('A major weakness of conventional pipelines is the assumption that one fixed facial or thoraco-abdominal crop remains '
       'informative over time. ROI choice is known to affect rPPG quality: anatomical regions contribute differently [12], and '
       'dynamic tiling can improve signal quality by rejecting low-quality facial areas [11]. Neural attention mechanisms can '
       'also learn spatial weighting [4], [6]-[9], yet these mechanisms are optimized inside a downstream physiological '
       'estimator and are not designed to produce an interpretable, temporally tracked, infant-specific ROI representation. '
       'For neonates, region-level selection is especially limiting because a nominal ROI can simultaneously contain exposed '
       'skin, blanket, shadow, background, medical devices, caregiver hands, or motion-corrupted pixels.')
b.para('This work therefore treats ROI construction as a standalone computer-vision problem whose output can be audited. '
       'The objective is not only to localize or segment the infant, but to determine which anatomical regions, and which '
       'pixels inside them, should contribute evidence at each time point, while keeping the selection temporally tracked and '
       'every decision explainable by measured conditions. The design uses complementary detection, tracking and bounded '
       'recovery for stable infant localization [27]-[30], SAM 2 to turn recovered boxes into temporally propagated infant '
       'masks [31], and a 308-keypoint human-centric pose backbone from the Sapiens family [32]. Infant-specific pose '
       'literature motivates explicit neonatal anatomy and an awareness of the adult-to-infant domain gap [24]-[26].')
b.para('The paper deliberately stops at the output of Module 1: a multi-channel dynamic ROI representation. Heart-rate (HR) '
       'and respiratory-rate (RR) estimation against clinical references is reserved for a separate study. Because ROI '
       'ground truth is rarely available, we additionally contribute a label-free evaluation protocol in which a pulse '
       'signal-quality proxy, computed from colour time series that the ROI scorer never sees, serves as independent '
       'evidence about whether selected pixels carry physiological information.')
b.para('The main contributions are as follows:')
b.bullets([
    'We formulate neonatal ROI construction as a temporally tracked, condition-aware selection problem at two levels: '
    'region-level confidence over 17 anatomical candidates and pixel-level refinement inside each selected region.',
    'We introduce an interpretable condition vector (anatomical confidence, skin likelihood, temporal overlap, optical-flow '
    'motion, blur, illumination, saturation, swaddling proxy and SAM 2 mask support) and a transparent, training-free '
    'confidence-and-fusion rule whose every term can be audited per frame.',
    'We show how a promptable video segmenter (SAM 2) can be used as a pixel-level constraint on pose, ROI support and '
    'colour evidence rather than as a detector, and we isolate its effect with paired with/without-SAM 2 branches.',
    'We define a reproducible, leakage-controlled, label-free evaluation protocol (temporal stability, mask support, '
    'pulse-SNR-based validity, calibration, motion- and condition-stratified robustness, failure-episode mining and '
    'computational cost) with video-level statistics, together with an expert-annotation kit for label-based follow-up.',
])

b.h1('2. Related Work')
b.h2('2.1 Static and Dynamic ROI Selection')
b.para('Classical camera physiology typically begins by selecting skin or facial pixels and then extracting temporal colour '
       'variation. Verkruysse et al. demonstrated ambient-light plethysmographic imaging [1], de Haan and Jeanne introduced '
       'the motion-robust chrominance formulation [2], and Wang et al. analyzed the algorithmic principles underlying '
       'colour-based rPPG and the plane-orthogonal-to-skin (POS) approach [3]. These methods made dependence on skin '
       'appearance and illumination explicit, but they generally assume a sufficiently stable region. Kim et al. '
       'systematically evaluated anatomically defined facial regions and showed that ROI choice materially changes rPPG '
       'quality [12]. Kiddle et al. later proposed a dynamic tiling-and-aggregation strategy that selects higher-quality '
       'facial tiles instead of using the full midface [11]. These findings directly motivate adaptive ROI construction, but '
       'adult-face tiling does not address neonatal pose, swaddling, body-region transitions, or infant-level segmentation.')
b.h2('2.2 Spatial Attention and Motion-Robust Representation Learning')
b.para('DeepPhys combined appearance-guided attention with a learned motion representation [4], while PhysNet modeled '
       'physiological signals with spatiotemporal convolution [5]. MTTS-CAN used multi-task temporal-shift attention for '
       'contactless pulse and respiration estimation [6]. PhysFormer and PhysFormer++ introduced temporal-difference '
       'transformers [7], [8], and EfficientPhys simplified preprocessing [9]. More recent work emphasizes robustness: '
       'Motion Matters augments rPPG data through neural motion transfer [13], masked-attention regularization targets ROI '
       'localization and motion artifacts [14], RhythmMamba targets efficient long-context modeling [18], and PhysDiff '
       'disentangles physiological dynamics from interference [17]. These approaches demonstrate that spatial weighting and '
       'motion robustness matter, but their attention is learned for downstream prediction and requires physiological labels; '
       'it is not an explicit, inspectable neonatal ROI representation.')
b.h2('2.3 Temporal Localization, Segmentation, and Infant Anatomy')
b.para('The proposed front-end uses established perception modules as building blocks rather than claiming them as '
       'independent contributions. YOLO established a unified real-time detection paradigm [27], and RT-DETR provides an '
       'end-to-end transformer detector designed for real-time operation [28]. ByteTrack and BoT-SORT provide complementary '
       'strategies for temporal association under missed or low-confidence detections [29], [30]. SAM introduced promptable '
       'segmentation [33], and SAM 2 extended the paradigm to video with streaming memory for temporal mask propagation [31]. '
       'Sapiens provides high-resolution human-centric foundation models for pose and related tasks [32]. These generic models '
       'face an adult-to-infant distribution shift: the babyPose dataset exposed the need for preterm-infant pose benchmarks '
       '[24], AggPose developed an infant pose transformer [25], and adult-to-infant adaptation work reports degraded transfer '
       '[26]. This literature motivates a pipeline that uses generic perception backbones as spatial priors while scoring '
       'neonatal ROI reliability separately and auditably.')
b.h2('2.4 Domain Shift and Generalization')
b.para('Domain shift is a persistent challenge in camera physiology. DANN showed that a gradient-reversal domain classifier '
       'can encourage domain-invariant features [34]; Dual-Bridging addresses cross-domain discrepancy in rPPG [15]; and MMPD '
       'exposes models to diverse motion, lighting and appearance [16]. Learned adaptation, however, presupposes training data '
       'from the source domain and, typically, labels. Module 1 takes the complementary route of a training-free scoring rule '
       'on top of frozen foundation models, so its behaviour on a new recording does not depend on having seen that '
       'recording\'s appearance; we test this with a strict video-level split and report split-wise results (Section 5.5).')
b.h2('2.5 Neonatal Monitoring Context and Research Gap')
b.para('Clinical and engineering studies have shown that non-contact neonatal physiological monitoring is feasible using RGB '
       'or RGB-D cameras [19]-[23]. Villarroel et al. combined skin-region analysis and signal-quality assessment in preterm '
       'infants [19]; Khanam et al. used neural ROI selection for HR/RR estimation [20]; Zeng et al. studied camera-based '
       'cardiorespiratory monitoring across preterm infants [21]; Ahani et al. estimated infant respiratory rate from video '
       '[22]; and Ruhrberg Estevez et al. reported continuous RGB-D neonatal monitoring [23]. These studies establish clinical '
       'motivation but do not resolve the upstream representation problem addressed here: how to construct a temporally '
       'tracked, infant-specific ROI whose selection is explicitly conditioned on anatomy, skin evidence, segmentation support, '
       'motion and temporal stability, refined at pixel level, and evaluable without ROI ground truth.')
b.caption('Table 1. Positioning of representative prior work relative to the proposed Module 1 objective.')
b.table(['Representative work', 'Primary focus', 'Dynamic ROI', 'Pixel-level', 'Motion-aware', 'Interpretable score', 'Neonatal'],
        [['Kim et al. [12]', 'Anatomical facial ROI study', 'No', 'No', 'Limited', 'Yes', 'No'],
         ['Kiddle et al. [11]', 'Dynamic facial tiling', 'Yes', 'Tile-level', 'Indirect', 'Yes', 'No'],
         ['DeepPhys [4]', 'Physiology + attention', 'Learned', 'Attention', 'Yes', 'No', 'No'],
         ['MAR-rPPG [14]', 'Motion-robust rPPG', 'Attention', 'Masked attention', 'Yes', 'No', 'No'],
         ['Dual-Bridging [15]', 'Domain-adaptive rPPG', 'No', 'No', 'Indirect', 'No', 'No'],
         ['Villarroel et al. [19]', 'Neonatal vital signs', 'Quality-aware', 'Skin-region', 'Yes', 'Partly', 'Yes'],
         ['Khanam et al. [20]', 'Neonatal HR/RR', 'Automatic ROI', 'Region-level', 'Noise reduction', 'No', 'Yes'],
         ['Proposed Module 1', 'Reusable visual ROI representation', 'Yes (17 regions)', 'Yes (ROI ∩ mask ∩ skin)', 'Yes',
          'Yes (per-term)', 'Yes']])

b.h1('3. Proposed Method')
b.h2('3.1 Problem Formulation')
b.para('Let V = {I_t}_{t=1}^T denote an RGB video. Module 1 maps each sampled frame to a representation Z_t that emphasizes '
       'reliable infant evidence and suppresses unreliable content. Candidate anatomical ROIs are indexed by r ∈ {1,...,K} '
       '(K = 17) and pixels by p. The module produces a region-level confidence Q_{t,r}, fusion weights α_{t,r}, and a pixel '
       'support set R̃_{t,r} for each selected region:')
b.eq('Z_t = { R̃_{t,r}, α_{t,r}, Q_{t,r}, z_{t,r}, pose_t, M_t }_{r ∈ S_t},')
b.para('where S_t is the selected set, z_{t,r} the condition vector, pose_t the 308 keypoints and M_t the infant mask. The '
       'output is not a single rectangular crop; it is a weighted, multi-region representation with the metadata needed by '
       'downstream HR and RR models. Gross motion is treated as a nuisance for region confidence, whereas the motion '
       'descriptors themselves are retained in the metadata channel for downstream respiration analysis.')
b.h2('3.2 Overall Framework')
b.para('The implemented framework has eight stages (Figure 1): (i) manifest-based frame validation and deterministic '
       'sampling at 20 FPS with native timestamps; (ii) dual infant detection; (iii) detection fusion, tracking and bounded '
       'temporal recovery; (iv) SAM 2 temporal infant segmentation; (v) mask-constrained pose estimation and anatomical '
       'mapping; (vi) condition-aware ROI characterization; (vii) region-level confidence and dynamic selection; and (viii) '
       'pixel-level refinement and confidence-weighted fusion. Stages (ii)-(v) use frozen, publicly available models; the '
       'methodological contribution lies in how stages (vi)-(viii) combine their cues into an auditable representation.')
b.figure1_original()
b.caption('Figure 1. Module 1 pipeline. Standard perception models produce localization, mask and anatomy cues; Module 1 '
          'characterizes each anatomical candidate, scores and selects regions with a transparent condition-aware rule, '
          'refines selected regions at pixel level with the SAM 2 mask and skin evidence, and outputs a dynamic ROI '
          'representation with its confidence and motion metadata.')
b.h2('3.3 Infant Localization, Segmentation and Anatomical Candidates')
b.para('Each sampled frame is processed by YOLO11s and RT-DETR-L [27], [28] (person class, confidence ≥ 0.25, 640 px). '
       'Their highest-confidence boxes are matched by IoU (threshold 0.5) and fused with an OR rule, so that disagreement '
       'cases remain available for temporal arbitration rather than being discarded by a hard AND rule. The fused box is '
       'associated over time with ByteTrack [29] (high/low thresholds 0.25/0.10, buffer 30 frames). A bounded recovery state '
       'machine bridges detector gaps of at most 10 sampled frames using tracker prediction when scale change ≤ 0.3, centre '
       'displacement ≤ 0.5 box diagonals and trajectory error ≤ 0.25, with confidence decaying by 0.9 per frame and a '
       'minimum of 0.35, so that errors cannot propagate indefinitely.')
b.para('The recovered box prompts SAM 2.1 (Hiera-T) [31], which propagates an infant mask M_t through chunks of 96 frames '
       'and is re-prompted when the box becomes unreliable. Each mask receives explicit states (initialized, propagated, '
       'reinitialized, low-confidence, failed) and a quality score combining area plausibility, box-mask agreement, temporal '
       'mask IoU and boundary stability; only reliable masks are used downstream. SAM 2 is used as a pixel constraint, never '
       'as a detector. Sapiens2-0.4B [32] then estimates 308 keypoints P_t = {(x_k, y_k, c_k)} at 1024×768 on the box crop in '
       'which pixels outside a reliable mask are replaced by the neutral ImageNet mean, suppressing caregiver hands and '
       'background. Anatomical templates over confident keypoints (c ≥ 0.5, at least two support points) generate 17 '
       'candidates: head, face, forehead, both cheeks, upper face, neck, both shoulders, upper and lower chest, abdomen, '
       'thoraco-abdominal region, both arms and both legs.')
b.h2('3.4 Condition-Aware ROI Characterization')
b.para('For every available candidate r at time t, Module 1 measures a condition vector')
b.eq('z_{t,r} = [P, S, O, C, M, B, L, Sat, W, G]_{t,r},')
b.para('where P is the anatomical (keypoint) confidence of the region; S the mean YCrCb skin likelihood of its pixels, '
       'S = mean_p sqrt(exp(-(Cr-150)²/2·35²)·exp(-(Cb-110)²/2·25²)); O = 1 − S an occlusion proxy; C the polygon IoU with '
       'the same region in the previous sampled frame (temporal overlap); M the mean Farnebäck optical-flow magnitude inside '
       'the region [35], mapped to [0,1] by 1 − exp(−m/20 px); B a Laplacian-variance sharpness score; L an exposure score '
       '1 − 2|Y − 0.5|; Sat a saturation score; W a swaddling proxy, one minus the mean confidence of limb keypoints; and G the '
       'fraction of region pixels inside the reliable SAM 2 mask. Landmark velocity, centroid displacement and area change '
       'are also stored. All features are explicit so that every selection decision can be audited.')
b.h2('3.5 Region-Level Confidence and Dynamic Selection')
b.para('Region confidence is a transparent linear rule over normalized terms,')
b.eq('Q_{t,r} = w_P P + w_V (1 − O) + w_S S + w_C C − w_M M + λ_G G,')
b.para('followed by a temperature softmax over available candidates of the same frame and top-k selection (k = 3):')
b.eq('α_{t,r} = exp(Q_{t,r}/T) / Σ_j exp(Q_{t,j}/T),   S_t = top-k_r Q_{t,r}.')
b.para('In the default regime the weights are equal (w = 0.2, T = 1) and λ_G = 0, so that the with/without-SAM 2 comparison '
       'isolates the effect of the mask on pose, ROI geometry and pixels rather than on the score. Because the occlusion proxy '
       'is defined from skin likelihood, the visibility and skin terms coincide in the current implementation, i.e. skin '
       'evidence carries twice the weight of the other terms. Two optional regimes fit non-negative, sum-to-one weights by '
       'projected gradient descent on a transparent pseudo-reliability target: one on the training split and one on the '
       'validation split; neither uses physiological labels. A first frame without a predecessor receives neutral C = 1 and '
       'M = 0, and the imputation is recorded.')
b.h2('3.6 Pixel-Level Refinement and Fusion')
b.para('Region confidence cannot distinguish a stable cheek from a shadowed or occluded part of the same cheek, nor exclude '
       'a caregiver\'s glove that overlaps the chest polygon. Each selected region is therefore refined at pixel level:')
b.eq('R̃_{t,r} = shrink(R_{t,r}) ∩ M_t ∩ K_{t,r} ∩ {p : 15 < max_c I_t(p) < 250},')
b.para('where shrink scales the polygon by 0.85 about its centroid to reduce boundary jitter, M_t is the reliable SAM 2 mask '
       'and K_{t,r} is an adaptive skin set: pixels in a broad YCrCb skin range whose chromaticity lies within three robust '
       'deviations of the region\'s own median, which rejects tape, tubes and blanket edges while adapting to neonatal, '
       'jaundiced or phototherapy-lit skin. A region with fewer than 20 valid pixels or less than 10% valid area contributes '
       'nothing at that frame. The pixel weight of the fused representation is')
b.eq('ω_t(p) = Σ_{r ∈ S_t} α̃_{t,r} · 1[p ∈ R̃_{t,r}] / |R̃_{t,r}|,   α̃ = α renormalized over S_t,')
b.para('so every frame yields an explicit, inspectable pixel-weight map (Figure 2, column 6). The fused colour trace used by '
       'downstream rPPG is the ω-weighted mean, and native frames between sampled frames reuse the latest sampled geometry '
       'for at most 0.25 s. The pixel-level weights in this work are rule-based rather than learned; Section 6 discusses '
       'learning them.')
b.h2('3.7 Motion Awareness')
b.para('Motion has a dual role in neonatal video: abrupt body, camera or caregiver motion corrupts a region, but periodic '
       'thoraco-abdominal motion carries respiratory information. Module 1 therefore penalizes motion only in the region '
       'confidence (−w_M M), so that strongly moving regions lose rank, while the raw flow magnitude, landmark velocity and '
       'centroid displacement are preserved unchanged in the representation for the downstream RR model. Temporal overlap C '
       'rewards regions whose geometry is stable between frames. The penalty uses total in-region flow; separating global '
       'camera motion from local motion is left for future work.')
b.h2('3.8 Generalization by Design')
b.para('All perception models are frozen and the default scoring rule has no trainable parameters, so Module 1 cannot overfit '
       'the recording-specific appearance of the evaluation videos. Instead of a domain-adversarial objective, which would require '
       'source-domain training and would make the representation harder to audit, we evaluate cross-recording generalization '
       'directly: every threshold of the evaluation is fixed on training videos, and validation and test videos are reported '
       'separately. A learned, domain-adversarial reliability head [34] is a natural extension once expert labels exist.')

b.h1('4. Experimental Setup')
results.setup(b, R)
results.results(b, R)
results.discussion(b, R)
results.references(b)
b.save(OUT)
print(OUT)
