# Module 1 v2 configuration D8 — pre-registration

**Fixed on 2026-11-15, before any D8 result was computed.** Nothing in this file may be changed after the first D8
HR result exists. If a bug is found, it is fixed and recorded below in "Deviations" with a date and reason; rules and
thresholds are not adjusted to results.

## Context (disclosed)
D5 (Module 1 v2 headline) and a fixed box were evaluated on augmented UBFC-rPPG videos (`dataaugment/`): D5 beat the
fixed box under fast, large motion, equalled it under lighting changes and was worse under simulated occlusion (hand +
blanket). D8 is motivated by that occlusion failure and is evaluated on the same augmented videos (no other subjects with
reference HR are available). This must be stated in any report. The thresholds below are a-priori choices, not tuned.

## Definition
D8 = D5 (same scores, same frozen final34 stabilizer, same top-3 / weights / cross-fade) plus three rules, applied per
video on the sampled frames (time order). D8 only removes regions or changes the fusion set; when no rule fires anywhere
in a video, D8's output equals D5's.

Inputs per sampled frame t and region r: `visible_skin_proportion` s[t,r] (Module 1), availability, polygon, and a new
colour feature over ALL pixels of the polygon (no skin filter): chromaticity rn = R/(R+G+B), gn = G/(R+G+B) and
luminance L = mean(R+G+B)/3. Polygons smaller than 50 px have no colour feature (rules R2 not applied).

Baseline of a region at frame t: its values over the previous 10 s (causal, frame t excluded) on frames where the region
was available and **not flagged** by R1 or R2 (a frozen baseline, so a long occlusion does not become the new normal).
At least 2 s of clean baseline frames are required; otherwise no decision (region not flagged).

* **R1 Coverage (cloth / blanket / swaddling).** Flag when s[t,r] < 0.5 × median baseline s. Once flagged, the region
  stays flagged until s[t,r] >= 0.75 × the baseline median frozen at the flagging time.
* **R2 Appearance change (skin-coloured occluder, e.g. a hand).** Deviation dev[t,r] = max(dc / sc, dl / sl), where
  dc = Euclidean distance of (rn, gn) to the baseline median (rn, gn), dl = |L / L_baseline − 1|, sc and sl = 1.4826 ×
  MAD of the region's own baseline dc and dl (floors: sc >= 0.005, sl >= 0.02). Global compensation: excess[t,r] =
  dev[t,r] − median over the available regions of dev[t,·] (a lighting change moves all regions; an occluder moves
  some). Flag when excess > 3 for at least 0.5 s of consecutive sampled frames; unflag when excess < 1.5.
* Flagged regions are **excluded**: their score is set to unavailable before the D5 smoothing and selection.
* **R3 Redundancy.** HR group = FACE, FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, UPPER_FACE, UPPER_CHEST (the HR candidate list
  of `HR RR using module1 v2/configs/hr_rr_m1v2.yaml`). A frame is *affected* when any HR-group region was flagged in
  the last 2 s (including t) or fewer than 3 clean available HR-group regions exist at t. In affected frames the fusion
  set is **all clean available HR-group regions**, weighted by visible skin area (s × polygon area), instead of the
  top-3. Outside affected frames: the D5 selection on the rule-filtered scores.

## Evaluation (run once)
Same HR chain as the strict comparison (`compare_roi_fixed_vs_dynamic.py`, region_set module1, strict gate 2, band
45–180 bpm), reference = fingertip-PPG HR per 8-s window, matched at the nearest window centre within 1 s
(`dataaugment/hr_eval/refmatch.py`). Videos: the 13 augmented UBFC videos + the 3 original UBFC videos, plus a new
swaddling augmentation (below). Infant check: final34 (no reference): D8 vs D5 label-free metrics (validated %, gate-1
above background, SNR), infant band.

**Swaddling augmentation (new test videos, defined now):** UBFC subject1; a cloth layer (DataAugment cloth texture)
rising from the bottom of the frame to a top edge at: small = 0.6 face heights below the chin (chest and shoulders
covered), medium = at the chin (neck covered), large = 0.25 face heights above the chin (chin and mouth covered);
gentle edge waves and an aperiodic vertical drift of ±0.05 face heights (low-pass 0.05 Hz). Face box = Haar detection
(median over the video), as DataAugment.

## Claim rules (fixed now)
Per condition and severity, paired per window, absolute error, moving-block bootstrap 95 % CI (blocks of 8 windows,
5,000 resamples, seed 42):
* "D8 handles occlusion" only if D8 − fixed box has CI upper bound < 0 at occlusion medium or large.
* "D8 handles swaddling" only if D8 − fixed box has CI upper bound < 0 at swaddling medium or large.
* "No harm" in a condition if D8 − D5 has CI lower bound <= 0 (i.e. D8 not clearly worse than D5) and, on final34,
  D8's validated % and gate-1-above-background are not lower than D5's by more than 5 percentage points.
* "Best" for a condition only if D8 − fixed box CI < 0 and D8 − D5 CI upper bound <= 0.
Results are reported as they come out, including failures.

## Deviations
(none)
