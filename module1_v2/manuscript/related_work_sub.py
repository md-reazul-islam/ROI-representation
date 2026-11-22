"""Expanded Related Work (Section 2) with a literature comparison table, for the SUBMISSION build.

Citations use the reference numbers of references.json ([1]-[59]); build_submission.py renumbers them in order of first
appearance. Every statement about a cited study was checked against its abstract or full text; the comparison table
states only what each paper reports about its ROI (no performance numbers are compared, because the studies use
different data, references and metrics).
"""

DEEP_RELATED = (
    'Deep rPPG models learn spatial attention and motion robustness end to end [4]-[10], [13]-[18]. They do not remove '
    'the ROI problem; they move it upstream: each model expects a face crop from a separate detector or tracker, and the '
    'crop defines what the network can see. Their spatial attention is implicit, optimized for a downstream physiological '
    'estimate, not constrained to be stable over time and not open to inspection; it requires physiological labels for '
    'training; the models cover the face only, whereas respiration is read from the chest and abdomen; and they are '
    'trained almost exclusively on adult faces, with a documented adult-to-infant domain gap [24]-[26]. NeoSTAR is '
    'therefore complementary: it provides the explicit, temporally tracked, infant-specific input representation such '
    'models consume, without labels, and a learned extractor or reliability model can be placed on top of it once '
    'labelled neonatal data exist.')

# (study, subjects/data, ROI definition, temporal handling, labels needed for the ROI, how the ROI itself is evaluated)
COMPARISON = [
    ('Poh et al. [50]', 'Adults, webcam', 'Face-detector box', 'Detected in every frame', 'None (pretrained detector)',
     'Through HR error only'),
    ('Li et al. [51]', 'Adults (MAHNOB-HCI)', 'Face region from facial landmarks', 'Landmarks tracked (KLT)',
     'None (pretrained landmarks)', 'Through HR error only'),
    ('Kumar et al. [52]', 'Adults', 'Facial regions, quality-weighted average', 'Regions tracked; weights from the signal',
     'None', 'Through SNR and HR error'),
    ('Tulyakov et al. [54]', 'Adults (MAHNOB-HCI, MMSE-HR)', 'Facial regions selected by matrix completion',
     'Selection adapted over time', 'None', 'Through HR error only'),
    ('Bobbia et al. [47]', 'Adults (UBFC-rPPG)', 'Superpixels merged by pulsatility, no face detection',
     'Temporal superpixels', 'None', 'Through HR error only'),
    ('Kim et al. [12]', 'Adults (UBFC-rPPG, LGI-PPGI)', '39 anatomical facial regions compared', 'Fixed per region',
     'None', 'Per-region signal similarity and HR error'),
    ('Kiddle et al. [11]', 'Adults', '62 midface tiles weighted by spectral SNR', 'Per recording', 'None',
     'Blinded grading of signal quality'),
    ('Deep rPPG, e.g. [4], [7]', 'Adults', 'Face crop with implicit learned attention', 'Learned, per frame or clip',
     'Physiological signals', 'Through HR error only'),
    ('Chaichulee et al. [56]; Villarroel et al. [19]', 'Preterm infants, NICU', 'Patient detection and skin segmentation '
     '(multi-task CNN)', 'Per frame, with signal-quality gating', 'Annotated NICU frames',
     'Detection and segmentation accuracy; HR and RR error'),
    ('Hoog Antink et al. [58]', 'Neonates, two NICUs (RGB and NIR)', 'Body-part segmentation (deep encoder-decoder)',
     'Segmentation and tracking', 'Manually segmented neonatal frames, adult data',
     'Segmentation accuracy against manual masks'),
    ('Khanam et al. [20]', 'NICU infants', 'ROI selected by a convolutional network', 'Not the focus',
     'Trained detector', 'Through HR and RR error'),
    ('Huang et al. [59]', 'Newborns (NBHR dataset)', 'Facial video into a spatio-temporal network', 'Learned',
     'Physiological signals', 'Through HR error only'),
    ('NeoSTAR (this work)', 'Public newborn videos; UBFC-rPPG with controlled perturbations',
     '17 pose-derived anatomical regions ∩ SAM 2 mask ∩ skin, with a condition vector per region',
     'Confidence smoothing, hysteresis and Viterbi decoding (offline or 1-s lag)',
     'None (four stabilizer settings tuned on training videos)',
     'Stability, internal consistency, surrogate-null and split-half tests; HR against PPG'),
]


def related_work(b):
    b.h1('2. Related Work')
    b.h2('2.1 Region-of-Interest Selection in Camera Physiology')
    b.para('Colour-based camera physiology reads the blood-volume pulse from small colour changes of skin pixels and '
           'separates it from motion and illumination by projections of the RGB channels [1]-[3]. Which pixels enter this '
           'average has been a design question from the start. Verkruysse et al. used manually placed regions [1]; Poh et '
           'al. automated the choice with a face detector and blind source separation [50]; Li et al. defined the face '
           'region from landmarks, tracked it with the Kanade-Lucas-Tomasi tracker [40] and corrected illumination with '
           'the background [51]; and Wang et al. treated every skin pixel as an independent sensor, with motion-compensated '
           'pixel-to-pixel pulse extraction and spatial pruning, to gain motion robustness [53]. A second line of work selects or weights regions by the quality of their '
           'signal: DistancePPG weights tracked facial regions by an estimate of their blood perfusion and illumination '
           '[52], Tulyakov et al. select reliable facial regions over time by matrix completion [54], Kiddle et al. weight '
           'midface tiles by spectral signal-to-noise ratio [11], and Bobbia et al. merge temporal superpixels by their '
           'pulsatility without any face detection [47]. Kim et al. compared 39 anatomically defined facial regions and '
           'showed that the choice of region changes accuracy [12].')
    b.para('Two observations from this literature motivate our design. First, the ROI is almost always evaluated through '
           'the error of the final heart-rate estimate; its temporal stability and the consistency of its own scores are '
           'rarely measured as properties in their own right. Second, selecting regions by a colour-derived quality score '
           'favours whatever concentrates power at one frequency, which is a pulse but can also be noise; a label-free '
           'study therefore needs a null against which such a selection is tested (Section 2.5). Grid tiling and skin + '
           'signal-quality selection are natural baselines for an anatomy-guided method, and we include both (M1, M2) '
           'together with whole-infant skin (M0).')
    b.h2('2.2 Learned Spatial Attention and Motion Robustness')
    b.para(DEEP_RELATED)
    b.h2('2.3 Camera-Based Monitoring of Neonates')
    b.para('Neonatal studies established that camera-based monitoring is feasible in the clinic. Aarts et al. recorded 19 '
           'infants in two neonatal intensive care units (NICUs) under ambient light and measured the pulse signal in all '
           'of them, with low light and infant motion as the main causes of failure [55]. '
           'Chaichulee et al. trained a multi-task convolutional network for patient detection and skin segmentation on '
           'annotated NICU frames [56] and extended it to cardio-respiratory signal extraction [57]; with it, Villarroel '
           'et al. monitored 30 preterm infants over several days and estimated vital signs only in time periods judged '
           'reliable by signal-quality assessment [19]. Hoog Antink et al. segmented the body parts of neonates in RGB and '
           'near-infrared video with a deep network pretrained on adult data and fine-tuned on manually segmented '
           'neonatal frames [58]. Khanam et al. selected the ROI of seven NICU infants with a convolutional network [20]; '
           'Huang et al. released a large newborn facial-video database with reference vital signs and a spatio-temporal '
           'network for heart rate [59]; Zeng et al. measured heart rate, respiration, their variability and actigraphy '
           'in 50 preterm infants [21]; Ahani et al. estimated respiratory rate of swaddled and covered infants from '
           'manually and automatically selected regions [22]; and Ruhrberg Estévez et al. combined colour, infrared and '
           'depth to monitor neonates in intensive care [23].')
    b.para('In these systems the ROI is a means to a vital-sign estimate: it is drawn by hand or produced by a network '
           'trained on annotated NICU frames, and it is judged by the accuracy of the vital sign or of the segmentation '
           'mask. None of them provides an explicit, anatomically labelled infant ROI whose temporal stability, internal '
           'consistency and pulse evidence are measured without labels, which is what a representation intended for '
           'several downstream signals and for clinical audit needs.')
    b.h2('2.4 Localization, Segmentation and Infant Anatomy')
    b.para('The front-end uses established models as building blocks: YOLO-family and RT-DETR detectors [27], [45], [28], '
           'ByteTrack [29] (BoT-SORT [30] is a comparable tracker), the promptable segmenter SAM [33] and its video '
           'extension SAM 2 [31], and Sapiens pose models [32], [46]. Infant pose literature documents the adult-to-infant '
           'domain gap of such models [24]-[26], which is one reason we measure, rather than assume, the reliability of '
           'anatomy-derived cues.')
    b.h2('2.5 Temporal Decision Smoothing and Surrogate Testing')
    b.para('Selecting one of several noisy candidates over time is a classic sequential decision problem; hysteresis and '
           'Viterbi decoding [38] trade instantaneous score against the cost of switching. Surrogate-data testing [39] '
           'compares a statistic with its distribution under a null that preserves selected properties of the data. We use '
           'block shuffling in time, which keeps the colour distribution of the pixels but destroys any rhythm. A common '
           'alternative, independent per-channel phase randomization, keeps each channel\'s power spectrum; we show that it '
           'therefore also keeps a genuine pulse and is unsuitable as a pulse-present null.')
    b.h2('2.6 Summary and Research Gap')
    b.para('Table @1 compares representative ROI strategies by what they use as the ROI, how they handle it over time, '
           'which labels they need and how the ROI itself is evaluated. Performance figures are deliberately not compared: '
           'the studies use different subjects, references and metrics. Prior neonatal studies establish clinical '
           'feasibility [19]-[23], [55]-[59] but do not provide a temporally stable, auditable infant ROI representation '
           'together with label-free tests of whether the selected pixels carry a pulse at all. Nor do they test, against '
           'a physiological reference, under which conditions a dynamic ROI is worth its complexity compared with a fixed '
           'crop. NeoSTAR targets exactly this gap.')
    b.caption('Table @1. Comparison of ROI strategies for camera-based physiological monitoring (as reported by each study).')
    b.table(['Study', 'Subjects / data', 'ROI definition', 'Temporal handling', 'Labels needed for the ROI',
             'How the ROI is evaluated'],
            [list(r) for r in COMPARISON], size=7, widths=[1.15, 1.1, 1.45, 1.2, 1.05, 1.15])
