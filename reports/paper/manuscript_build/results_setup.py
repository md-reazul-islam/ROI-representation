"""Experimental setup section and references (data-independent apart from dataset/compute tables)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]   # Module1 folder


def load(report):
    report = Path(report)
    read = lambda name: pd.read_csv(report/name)
    R = dict(report=report, meta=json.loads((report/'evaluation_metadata.json').read_text()))
    for name in ('table3_main_all_videos', 'table3_main_test_split', 'table3_main_heldout_val_test', 'table3_paired_tests_vs_full',
                 'table3_paired_tests_vs_B4', 'table4_motion_stratified', 'table5_condition_stratified', 'table_q_validity',
                 'table6_split_generalization', 'table8_failure_cases', 'table_perception_diagnostics', 'table_compute',
                 'per_video_metrics', 'window_metrics'):
        R[name] = read(f'{name}.csv')
    R['dataset'] = pd.read_csv(report/'table_dataset.csv')
    run = Path(R['meta']['run'])
    run = run if run.is_absolute() else ROOT/run
    R['manifest'] = pd.read_csv(run/'manifest.csv')
    R['n_videos'] = len(R['manifest'])
    R['n_split'] = R['manifest'].split.value_counts().to_dict()
    return R


def words(n):
    """Number as a word for small counts (prose), digits otherwise."""
    return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][n] if n <= 10 else str(n)


def setup(b, R):
    d = R['dataset'].set_index('split')
    total_frames, total_sec = int(d.frames.sum()), float(d.duration_sec.sum())
    b.h2('4.1 Dataset, Data Integrity and Ethics')
    m = R['manifest']
    pixels = m.width*m.height
    small, large = m.loc[pixels.idxmin()], m.loc[pixels.idxmax()]
    b.para(f'Module 1 is evaluated on {R["n_videos"]} neonatal videos ({total_frames:,} native frames, {total_sec/60:.1f} min; '
           f'{m.duration_sec.min():.1f}-{m.duration_sec.max():.1f} s per clip; {m.fps.min():.1f}-{m.fps.max():.1f} FPS; '
           f'{int(small.width)}×{int(small.height)} to {int(large.width)}×{int(large.height)} px). '
           'The clips are publicly available, short-form NICU-themed web videos '
           'rather than continuous bedside recordings, and therefore constitute an unconstrained, in-the-wild stress test: they '
           'contain incubators, ventilators and monitors, caregiver hands and gloves, phototherapy and low light, swaddling and '
           'costumes, burned-in captions, abrupt scene cuts, split screens, and segments without any infant (e.g., adults or '
           'empty rooms). These properties are hostile to any fixed-ROI assumption and are retained deliberately; no frame was '
           'removed or curated. Each file is validated for readability, native frame rate, frame count, resolution, duration '
           'and deterministic timestamps and is identified by its SHA-256 hash. Frames are sampled at 20 FPS with native '
           'timestamps preserved.')
    n = R['n_split']
    b.para(f'The split is predeclared and video-level: {n.get("train", 0)} training, {n.get("validation", 0)} validation '
           f'and {n.get("test", 0)} test videos; every frame inherits the '
           'split of its video. No component of Module 1 is trained on any of these videos; the split is used only to fix '
           'evaluation thresholds and calibration maps on training videos before validation and test videos are inspected. '
           'Infant identities are not available, so subject-independent generalization cannot be verified and no identity is '
           'inferred from appearance. **Ethics statement:** [AUTHORS TO COMPLETE: data source and licence/terms of use of the '
           'web videos, institutional review board determination, and any de-identification applied before publication of '
           'figures.]')
    b.caption('Table 2a. Dataset composition by split.')
    b.table(['Split', 'Videos', 'Native frames', 'Duration (s)', 'Median FPS'],
            [[s, int(d.loc[s, 'videos']), f'{int(d.loc[s, "frames"]):,}', f'{d.loc[s, "duration_sec"]:.1f}',
              f'{d.loc[s, "median_fps"]:.1f}'] for s in ('train', 'validation', 'test') if s in d.index])

    b.h2('4.2 Configurations and Ablation Protocol')
    b.para('All configurations use the same sampled frames, the same Phases 2-6 outputs (detection, fusion, tracking, recovery), '
           'the same selection size k = 3 and the same fusion softmax, so differences are attributable to the ROI mechanism '
           'alone. B0 holds the central half of the first-frame infant box fixed; B1 holds the first-frame region with the '
           'highest anatomical confidence fixed; B2 selects regions per frame by anatomical confidence only; B3 adds skin and '
           'visibility evidence; B4 is the full condition-aware rule including temporal overlap and the motion penalty, '
           'without SAM 2 (control S0); B5 runs the same rule on the SAM 2 branch, i.e. mask-constrained pose, ROI geometry and '
           'mask support; and the Full model additionally restricts colour evidence to ROI ∩ mask ∩ skin pixels (control S1). '
           'B5 and Full share geometry and differ only in the pixels used, which isolates the pixel-level refinement.')
    b.caption('Table 2. Progressive component ablation and SAM 2 control conditions.')
    b.table(['ID', 'Configuration', 'Per-frame selection', 'Condition terms', 'Temporal / motion', 'SAM 2 guidance', 'Pixel refinement'],
            [['B0', 'Fixed box ROI', 'No', 'No', 'No', 'No', 'No'],
             ['B1', 'Fixed anatomical ROI', 'No', 'Anatomy (frame 1)', 'No', 'No', 'No'],
             ['B2', 'Region-level dynamic', 'Yes', 'Anatomy', 'No', 'No', 'No'],
             ['B3', '+ skin / visibility', 'Yes', 'Anatomy, skin', 'No', 'No', 'No'],
             ['B4 (S0)', '+ temporal / motion', 'Yes', 'All', 'Yes', 'No', 'No'],
             ['B5', '+ SAM 2 guidance', 'Yes', 'All', 'Yes', 'Yes', 'No'],
             ['Full (S1)', '+ pixel refinement', 'Yes', 'All', 'Yes', 'Yes', 'Yes']])

    b.h2('4.3 Implementation Details')
    b.para('Detection uses Ultralytics 8.4.183 with YOLO11s and RT-DETR-L checkpoints (person class, 640 px input, confidence '
           '0.25, NMS IoU 0.45); fusion uses the OR strategy at IoU 0.5; tracking uses ByteTrack with the settings of '
           'Section 3.3. Segmentation uses the official SAM 2 implementation with the SAM 2.1 Hiera-Tiny checkpoint '
           '(mask threshold 0, 96-frame chunks, re-prompt after 10 frames without a reliable box, at least 3 frames between '
           're-prompts). Pose uses the Sapiens2-0.4B 308-keypoint checkpoint at 1024×768 in float32. No component is trained '
           'or fine-tuned; the confidence rule uses equal weights (0.2), T = 1, k = 3 and λ_G = 0. The software stack is Python '
           '3.12.15, PyTorch 2.11.0 (CUDA 12.8), OpenCV 4.14.0, NumPy 2.5.4 and SciPy 1.18.1, run on one NVIDIA A100-SXM4-80GB '
           'GPU with an AMD EPYC 7742 CPU; the random seed is 42. Every run stores its configuration, model checkpoint paths, '
           'per-phase metadata and the SHA-256 of every input video and intermediate table.')

    b.h2('4.4 Evaluation Metrics')
    t = R['meta']['thresholds']
    b.para('No ROI, mask or box ground truth exists for these videos, so we separate label-free diagnostics from validity '
           'evidence and never reuse a model output as ground truth for the same metric. **Temporal stability** is measured '
           'on the union box of the selected ROIs: consecutive-frame IoU, centroid jitter (centroid displacement normalized by '
           'the infant-box diagonal), Jaccard overlap of the selected region names, and top-1 switches per second. **Mask '
           'support** is the fusion-weighted fraction of used ROI pixels inside the reliable SAM 2 mask; it is 1 for the Full '
           'model by construction and is reported to quantify how much non-infant content the other configurations admit.')
    b.para('**Pulse-based validity (label-free proxy).** For every configuration the fused colour trace is converted to a pulse '
           'signal with POS [3] (1.6 s projection window), detrended (smoothness priors, λ = 100) and band-passed to the '
           'neonatal HR band of 90-270 bpm [19]. In 8 s windows with 1 s step, the de Haan-Jeanne SNR [2] at the in-band '
           'spectral peak measures how much of the trace is concentrated at a single pulse frequency and its harmonic. We '
           'report the per-video median SNR and the percentage of pulse-informative windows, i.e. SNR ≥ τ with '
           f'τ = {t["informative_snr_db"]:.2f} dB, the median ROI-window SNR of training videos. The ROI scorer never sees '
           'colour time series, so agreement between Q and SNR is independent evidence about Q. Because no reference HR is '
           'used, SNR measures periodicity in the pulse band, not HR accuracy; it is a proxy and is labeled as such.')
    b.para('**Confidence validity and calibration.** For every ROI and window we pair the mean rescaled confidence Q ∈ [0,1] '
           'with that ROI\'s own pulse SNR and report Spearman correlation, AUROC and AUPRC for pulse-informative windows, and '
           'expected calibration error (ECE, 10 bins) and Brier score both for raw Q and after Platt scaling [36] fitted on '
           'training videos only [37]. **Robustness** is stratified by optical-flow motion, low skin likelihood (an occlusion '
           'proxy), the swaddling proxy, luminance inside the infant box, and abrupt luminance transitions; tertile thresholds '
           'are fixed on training videos. **Failure episodes** are mined from diagnostics (extended detector loss, SAM 2 '
           'failures, severe underexposure, extreme motion or skin loss) and summarized by how often the representation '
           'becomes conservative (no ROI or best Q below the training 25th percentile) and whether it recovers within 1 s. '
           '**Perception diagnostics** (detection, tracking, recovery, segmentation, pose) and **computational cost** '
           '(per-phase throughput, peak GPU memory, parameter counts) complete the evaluation. Precision, recall, AP, Dice and '
           'expert-valid ROI rates require annotations; an annotation kit and the corresponding metrics are released, and '
           'these columns are not filled without labels.')
    b.h2('4.5 Statistical Analysis and Reproducibility')
    b.para('The video is the statistical unit throughout, so temporal correlation between frames cannot inflate the sample '
           'size. We report mean ± standard deviation over videos with 95% percentile bootstrap confidence intervals (5,000 '
           'resamples). Paired comparisons use the two-sided Wilcoxon signed-rank test with the matched-pairs rank-biserial '
           'correlation r as effect size, and p-values are Holm-corrected within each metric. All thresholds are fixed on '
           'training videos before held-out videos are evaluated. The pipeline driver, evaluation and figure scripts, '
           'configuration, manifest with hashes and all per-video result tables are released with the paper.')


REFERENCES = [
    '[1] W. Verkruysse, L. O. Svaasand, and J. S. Nelson, "Remote plethysmographic imaging using ambient light," Optics Express, vol. 16, no. 26, pp. 21434-21445, 2008, doi: 10.1364/OE.16.021434.',
    '[2] G. de Haan and V. Jeanne, "Robust pulse rate from chrominance-based rPPG," IEEE Transactions on Biomedical Engineering, vol. 60, no. 10, pp. 2878-2886, 2013, doi: 10.1109/TBME.2013.2266196.',
    '[3] W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic Principles of Remote PPG," IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479-1491, 2017, doi: 10.1109/TBME.2016.2609282.',
    '[4] W. Chen and D. McDuff, "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks," in ECCV, 2018, pp. 349-365, doi: 10.1007/978-3-030-01216-8_22.',
    '[5] Z. Yu, X. Li, and G. Zhao, "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks," in BMVC, 2019, Art. 277.',
    '[6] X. Liu, J. Fromm, S. Patel, and D. McDuff, "Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement," in NeurIPS, vol. 33, 2020.',
    '[7] Z. Yu, Y. Shen, J. Shi, H. Zhao, P. H. S. Torr, and G. Zhao, "PhysFormer: Facial Video-Based Physiological Measurement With Temporal Difference Transformer," in CVPR, 2022, pp. 4186-4196.',
    '[8] Z. Yu, Y. Shen, J. Shi, H. Zhao, Y. Cui, J. Zhang, P. Torr, and G. Zhao, "PhysFormer++: Facial Video-Based Physiological Measurement with SlowFast Temporal Difference Transformer," International Journal of Computer Vision, vol. 131, pp. 1307-1330, 2023, doi: 10.1007/s11263-023-01758-1.',
    '[9] X. Liu, B. Hill, Z. Jiang, S. Patel, and D. McDuff, "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement," in WACV, 2023, pp. 5008-5017.',
    '[10] X. Liu et al., "rPPG-Toolbox: Deep Remote PPG Toolbox," in NeurIPS, vol. 36, pp. 68485-68510, 2023.',
    '[11] A. Kiddle, H. Barham, S. Wegerif, and C. Petronzio, "Dynamic Region of Interest Selection in Remote Photoplethysmography: Proof-of-Concept Study," JMIR Formative Research, vol. 7, e44575, 2023, doi: 10.2196/44575.',
    '[12] D.-Y. Kim, K. Lee, and C.-B. Sohn, "Assessment of ROI Selection for Facial Video-Based rPPG," Sensors, vol. 21, no. 23, 7923, 2021, doi: 10.3390/s21237923.',
    '[13] A. Paruchuri, X. Liu, Y. Pan, S. Patel, D. McDuff, and S. Sengupta, "Motion Matters: Neural Motion Transfer for Better Camera Physiological Measurement," in WACV, 2024, pp. 5933-5942.',
    '[14] P. Zhao, Q. Sun, X. Tian, Y. Yang, S. Tao, J. Cheng, and J. Chen, "Toward Motion Robustness: A Masked Attention Regularization Framework in Remote Photoplethysmography," in CVPR Workshops, 2024, pp. 7829-7838.',
    '[15] J. Du, S.-Q. Liu, B. Zhang, and P. C. Yuen, "Dual-Bridging With Adversarial Noise Generation for Domain Adaptive rPPG Estimation," in CVPR, 2023, pp. 10355-10364.',
    '[16] J. Tang, K. Chen, Y. Wang, Y. Shi, S. Patel, D. McDuff, and X. Liu, "MMPD: Multi-Domain Mobile Video Physiology Dataset," in IEEE EMBC, 2023, pp. 1-5, doi: 10.1109/EMBC40787.2023.10340857.',
    '[17] W. Qian, G. Su, D. Guo, J. Zhou, X. Li, B. Hu, S. Tang, and M. Wang, "PhysDiff: Physiology-based Dynamicity Disentangled Diffusion Model for Remote Physiological Measurement," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, pp. 6568-6576, 2025, doi: 10.1609/aaai.v39i6.32704.',
    '[18] B. Zou, Z. Guo, X. Hu, and H. Ma, "RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 10, pp. 11077-11085, 2025.',
    '[19] M. Villarroel et al., "Non-contact physiological monitoring of preterm infants in the Neonatal Intensive Care Unit," npj Digital Medicine, vol. 2, 128, 2019, doi: 10.1038/s41746-019-0199-5.',
    '[20] F.-T.-Z. Khanam, A. G. Perera, A. Al-Naji, K. Gibson, and J. Chahl, "Non-Contact Automatic Vital Signs Monitoring of Infants in a Neonatal Intensive Care Unit Based on Neural Networks," Journal of Imaging, vol. 7, no. 8, 122, 2021, doi: 10.3390/jimaging7080122.',
    '[21] Y. Zeng, D. Yu, X. Song, Q. Wang, L. Pan, H. Lu, and W. Wang, "Camera-Based Cardiorespiratory Monitoring of Preterm Infants in NICU," IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-13, 2024, doi: 10.1109/TIM.2024.3395314.',
    '[22] S. Ahani, N. Niknafs, P. M. Lavoie, L. Holsti, and G. A. Dumont, "Video-Based Respiratory Rate Estimation for Infants in the NICU," IEEE Journal of Translational Engineering in Health and Medicine, vol. 12, pp. 684-696, 2024, doi: 10.1109/JTEHM.2024.3488523.',
    '[23] S. Ruhrberg Estevez, A. Grafton, L. Thomson, J. Warnecke, K. Beardsall, and J. Lasenby, "Continuous non-contact vital sign monitoring of neonates in intensive care units using RGB-D cameras," Scientific Reports, vol. 15, 16863, 2025, doi: 10.1038/s41598-025-00539-9.',
    '[24] L. Migliorelli, S. Moccia, R. Pietrini, V. P. Carnielli, and E. Frontoni, "The babyPose dataset," Data in Brief, vol. 33, 106329, 2020, doi: 10.1016/j.dib.2020.106329.',
    '[25] X. Cao, X. Li, L. Ma, Y. Huang, X. Feng, Z. Chen, H. Zeng, and J. Cao, "AggPose: Deep Aggregation Vision Transformer for Infant Pose Estimation," in IJCAI, 2022, pp. 5045-5051, doi: 10.24963/ijcai.2022/700.',
    '[26] S. Bose, H. Dela Cruz, A. Dutta, E. Kokkoni, K. Karydis, and A. K. Roy-Chowdhury, "Leveraging Synthetic Adult Datasets for Unsupervised Infant Pose Estimation," in CVPR Workshops, 2025, doi: 10.1109/CVPRW67362.2025.00553.',
    '[27] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in CVPR, 2016, pp. 779-788, doi: 10.1109/CVPR.2016.91.',
    '[28] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs Beat YOLOs on Real-time Object Detection," in CVPR, 2024, pp. 16965-16974, doi: 10.1109/CVPR52733.2024.01605.',
    '[29] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," in ECCV, 2022, pp. 1-21, doi: 10.1007/978-3-031-20047-2_1.',
    '[30] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," arXiv:2206.14651, 2022.',
    '[31] N. Ravi et al., "SAM 2: Segment Anything in Images and Videos," in ICLR, 2025.',
    '[32] R. Khirodkar, T. Bagautdinov, J. Martinez, Z. Su, A. T. James, P. Selednik, S. Anderson, and S. Saito, "Sapiens: Foundation for Human Vision Models," in ECCV, 2024, pp. 206-228, doi: 10.1007/978-3-031-73235-5_12.',
    '[33] A. Kirillov et al., "Segment Anything," in ICCV, 2023, pp. 4015-4026.',
    '[34] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. March, and V. Lempitsky, "Domain-Adversarial Training of Neural Networks," Journal of Machine Learning Research, vol. 17, no. 59, pp. 1-35, 2016.',
    '[35] G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," in Scandinavian Conference on Image Analysis (SCIA), LNCS 2749, 2003, pp. 363-370, doi: 10.1007/3-540-45103-X_50.',
    '[36] J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," in Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61-74.',
    '[37] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in ICML, 2017, pp. 1321-1330.',
]


def references(b):
    b.h1('References')
    for ref in REFERENCES:
        b.para(ref, size=10)
