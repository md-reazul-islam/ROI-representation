# D9: swaddling handling for Module 1 v2. Pre-registration

Fixed on 2026-11-16, after development on UBFC subject 1 swaddling videos and **before** any D9 result on the test
videos. Nothing here may be changed after test results are seen. Any bug fix is recorded under *Deviations*.

## Development (disclosed)

**Development data:** `module1_v2_d8/data_swaddle/subject1_swaddle_{small,medium,large}`. These were already seen
in the D8 evaluation.

**Diagnosis of the D5 failure** at swaddle-large (cloth up to the mouth): MAE 15.9 bpm, no HR in 37 of 46 windows,
while the fixed box gave 1.8 bpm.
1. **Subject mask.** The person detector sees only the head, and SAM 2 then segments the hair instead of the face. The
   HR extraction keeps only pixels inside the SAM 2 mask, so the face ROIs lose almost all their pixels. SAM 2 covers
   0–9 % of UPPER_FACE and FACE, against 100 % on the original video. The fixed box ignores the mask.
2. **Landmarks.** The face keypoints fail: mean keypoint visibility of the face group is 0.42, and below 0.5 in 63 %
   of frames, from the first frame. The landmark ROIs then jump 11–30 px per frame and shrink by 2–7×.

**Literature basis:**
* NICU camera systems decide visible skin from **pixels (skin segmentation)**, not body pose (Villarroel et al.,
  npj Digit Med 2019; RGB-D NICU, arXiv 2412.06012, which drops non-skin pixels with a skin mask).
* Infant-face tracking falls back on skin colour when the body is covered by a blanket (PMC8642856).

**Variants tried on the development videos only (2):**

| Variant | swaddle-large MAE | swaddle-medium MAE |
|---|---|---|
| D5 | 15.9 (37 no-HR) | 2.44 |
| D9A: mask repair | 9.6 (0 no-HR) | 1.96 |
| D9B: mask repair + geometry stabilisation | 3.25 (0 no-HR) | 1.96 |

An earlier attempt with geometry stabilisation alone gave 16.5. D9B is chosen as D9. **These development numbers are
not evidence** and are not reported as results.

## D9 rules (module1_v2_d9/scripts/export_d9.py, `--stabilise --name D9`)

**R1: mask repair.**
* For each sampled frame, compute the coverage: the share of face-skin ROI pixels (the union of FOREHEAD, UPPER_FACE,
  LEFT_CHEEK and RIGHT_CHEEK polygons) inside the SAM 2 mask.
* If coverage < 0.5, add the union to the mask.
* The extraction's own adaptive YCrCb skin filter still selects the pixels inside each ROI.

**R2: geometry stabilisation.**
* A frame counts as unreliable when the mean keypoint visibility of the face group (FOREHEAD, UPPER_FACE, LEFT_CHEEK,
  RIGHT_CHEEK, FACE) is < 0.5.
* In those frames, the face-group ROIs and HEAD get a temporal majority-vote shape: a pixel is inside if it was inside
  in ≥ 50 % of the frames within ±1 s.
* The shape is computed at half resolution and the largest outer contour is kept.

**Unchanged from D5:** selection, ranks, fusion weights and everything else.

**Implementation:**
* The repaired masks go to a mirror run `runs/<run>_d9`. Every item there is a symlink to the original run, except
  `outputs/sam2`.
* The unchanged HR comparison tool runs on the mirror run.
* The fixed box ignores the mask, so it is identical in both runs. This was checked on the development videos.

## Test data (new, never used for development)

**Swaddling videos of UBFC subjects 3 and 4.** They use the same generator rules as the subject 1 videos
(module1_v2_d8 pre-registration):
* Cloth top edge at chin + 0.6 / 0 / −0.25 face heights for small / medium / large.
* Edge waves 6 px over 120 px.
* Drift ±0.05 face heights, low-pass filtered at 0.05 Hz.
* The same pale-blue cloth.
* Seed 20261116.
* Lossless, with checks.

Script: `scripts/make_swaddle_test_videos.py`.

**Evaluation settings:**
* Reference: fingertip PPG, one value per 8 s window, matched by the nearest window centre within 1 s.
* Strict gate 2, HR band 45–180 bpm.
* Paired moving-block bootstrap: block 8 windows, 5000 resamples, seed 42, resampled within video and pooled over the
  two subjects.

## Claim rules (applied automatically by scripts/d9_report.py)

* **S1 (primary): "D9 recovers HR under heavy swaddling where D5 fails."**
  On the test videos at large: D9 − D5 MAE CI upper bound < 0, pooled over subjects 3 and 4.
* **S2: "D9 is comparable to the fixed box under swaddling."**
  At every test level, D9 − fixed box CI upper bound ≤ +2 bpm (non-inferiority margin 2 bpm).
* **S3: "D9 beats the fixed box under swaddling."**
  D9 − fixed box CI upper bound < 0 at some test level. It is reported but not expected.
* **No harm (required before D9 replaces D5 for any claim):**
  * On every existing condition — original UBFC, slow motion, fast motion incl. extra-large, occlusion, lighting and
    the centre-occlusion demo — D9 − D5 CI lower bound ≤ 0 at every level, i.e. never clearly worse than D5.
  * On final34 infant data (label-free), validated % and gate-1-above-BG fall by ≤ 5 percentage points vs D5.
* **Also reported, not used for test claims:**
  * The development videos (subject 1).
  * The rate of R1 and R2 triggers per condition.

**Scope:** simulated cloth on 3 adult UBFC subjects, 2 of them test subjects. This is not real swaddled infants.

## Deviations

None so far.
