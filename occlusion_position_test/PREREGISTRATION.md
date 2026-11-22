# Occlusion-position test: pre-registration

Fixed on 2026-11-16, **before** any method (fixed box, v1, D5, D8) was run on these videos. Nothing here may be
changed after results are seen. Any bug fix is recorded under *Deviations* with its date and reason.

## Background (disclosed)

* On the existing UBFC occlusion videos (subject 3, a moving hand plus a blanket), D5 was worse than the fixed box.
* D8 (module1_v2_d8, pre-registered) did not fix this.
* The working idea is that dynamic ROI should help **only when the occluder covers the region the fixed box measures**.
  The earlier occlusion mostly covered the side and lower face, where the fixed box keeps clean skin.
* This test is designed *after* those failures. To keep it from being fitted to D5's strengths, the design commits to
  the following before any method runs:
  * **5 positions**, chosen as a standard face grid, not from D5 behaviour.
  * **2 sizes**.
  * **All 3 subjects**.
  * **All results reported**, including the positions where the fixed box wins.
* One geometric fact found while designing (not a result): on UBFC, the fixed box (B0) is the centre 50 % of Module 1's
  first-frame **person box**, which covers the upper body. B0 therefore spans most of the face below the upper forehead,
  plus the neck and some background. It is **not** a small face-centre box. Each video's JSON records how much of the
  estimated B0 the hand covers, and the report re-measures this on the new run.

## Videos (scripts/make_occlusion_videos.py, scripts/occluder.py)

* **Sources:** UBFC-rPPG subjects 1, 3 and 4. The original files are unchanged (SHA-256 checked before and after).
* **Occluder:** DataAugment's simulated hand (same renderer, skin colour and soft shadow). No blanket.
* **Hand position:** the palm centre is placed relative to the Haar face box (median of the first frames).
  * dx is in face widths and dy in face heights, measured from the face-box centre.
  * Fingers point away from the face centre.

  | Position | dx | dy | Fingers |
  |---|---|---|---|
  | centre (nose, mid-cheeks) | 0 | +0.05 | down |
  | forehead | 0 | −0.30 | up |
  | left_cheek (image left) | −0.28 | +0.12 | left |
  | right_cheek (image right) | +0.28 | +0.12 | right |
  | lower_face (mouth, chin) | 0 | +0.40 | down |

* **Sizes:** palm width = 0.30 (small) or 0.45 (large) face widths.
* **Timing:**
  * No hand for the first 10 s.
  * After that, aperiodic episodes: hand present for U(10, 14) s, then absent for U(4, 8) s, with 0.3 s fades.
  * Slow jitter: ±0.05 face sizes in position and ±8° in rotation, low-pass filtered at 0.08 Hz.
* **Common random numbers:** every position and size of one subject uses the same schedule and jitter, so videos of a
  subject differ only in where the hand is and how big it is. Seed 20261116.
* **Checks:**
  * lossless FFV1 (bit-exact)
  * frame count equal to the source
  * original file unchanged
  * no narrowband component in the HR band (0.7–4 Hz) in any control signal

  RR-band flags are recorded only, because RR is not evaluated.
* **Total:** 3 subjects × 5 positions × 2 sizes = 30 videos.

## Methods compared (all frozen, nothing tuned)

* **Fixed box (B0):** as in the comparison tool.
* **Module 1 v1:** D0.
* **Module 1 v2 D5:** the method under test.
* **BG:** background control.
* **D8:** reported only as a secondary result, frozen as pre-registered in module1_v2_d8. No claim is made for D8.

Settings:
* Comparison tool `HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py`, unchanged.
* Strict gate 2, HR band 45–180 bpm, frozen final34 stabilizer.
* Module 1 base run with the SAM2 branch, as for every other UBFC run.

## Evaluation (scripts/occ_report.py)

**Reference HR:**
* Fingertip PPG, one value per 8-s window, from `dataaugment/hr_eval/reference_hr_ppg.csv` (copied per new video ID).
* Matched to each analysis window by the nearest window centre within 1 s.

**Window sets:**
* **Occluded windows (primary):** mean hand visibility over the window ≥ 0.5.
* **All windows:** reported as well.

**Error:** |HR − reference|. A window with no HR is counted as missing. The number of no-HR windows is reported per method.

**Paired contrast D5 − fixed box (primary):**
* Computed on windows where both methods give HR, pooled over the 3 subjects.
* 95 % CI from a moving-block bootstrap: blocks of 8 windows, resampled within each video, 5000 resamples, seed 42.

The same contrast is reported for v1 − fixed, D8 − fixed and D5 − v1.

**Descriptive outputs:**
* Coverage by the hand of B0 and of the face regions.
* MAE per subject.
* Share of D5's fusion weight on ROIs that the hand covers by ≥ 50 % in occluded frames. This checks whether D5
  actually moves away from the hand.

## Claim rules (applied automatically by occ_report.py)

* **C1 (primary): "Dynamic ROI (D5) recovers HR when the occluder covers the centre of the face."**
  * At centre + large: the D5 − fixed box CI upper bound is < 0 on occluded windows,
  * **and** D5's occluded-window MAE is lower than the fixed box's in at least 2 of the 3 subjects.
* **C2 (secondary):** the same rule at centre + small.
* **C3: "The effect of occlusion on dynamic vs fixed ROI depends on location."**
  * C1 is supported,
  * **and** at least one other position (large size) has a D5 − fixed CI that includes 0 or favours the fixed box.
* **Other positions:** the 8 other position × size cells are reported with their verdicts but are **exploratory**.
  * There are 10 cells in total, so a CI excluding 0 at one non-primary cell is not a claim on its own.
  * Any paper sentence about them must say they are exploratory.
* **No general claim:** none of these rules supports "D5 handles occlusion". At most it supports "D5 helps when the
  occluder covers the fixed region". A lost cell is reported as lost.
* **Scope:** only simulated hand occlusion, on 3 adult UBFC subjects, at one camera geometry.

## Deviations / additions

* **2026-11-16: demonstration video added at the user's request** (`demo_centre/`). It is **not** part of the
  pre-registered test, and its result cannot support C1–C3.
  * Content: UBFC subject 1, one large hand (palm width 0.6 face widths) reaching in sideways from the image right.
  * Purpose: cover the nose, face centre and mid-cheeks (coverage 0.95 / 0.83–0.90) and leave the forehead free (0.00).
  * The geometry was chosen from coverage alone, before any method ran on it.
  * The video is designed to show the situation where dynamic ROI *should* help, so it may be reported only as an
    illustrative case, labelled as designed.
  * Even this hand covers only 34 % of the fixed box, because on UBFC the fixed box also spans the neck and collar.
  * Schedule, jitter, shadow and checks are the same as the pre-registered subject-1 videos.
  * The 30 pre-registered videos and their rules are unchanged.
