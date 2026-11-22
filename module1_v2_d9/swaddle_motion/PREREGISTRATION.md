# Swaddling + fast motion test: pre-registration

Fixed on 2026-11-16, **before** any method ran on these videos. Nothing here may be changed after results are seen.
Any bug fix is recorded under *Deviations*.

## Motivation (disclosed)

* **Swaddling alone:** on UBFC subject 1, the fixed box was as good as or better than dynamic ROI. The cloth stays
  below the face, the subject is still, and the fixed box keeps face skin.
* **Fast, large motion alone:** the fixed box failed (15.1 bpm at extra-large) while D5 stayed at about 2 bpm.
* **Swaddled infants move inside the wrap.** This test combines the two conditions.
* **Chosen to be realistic:** the design reuses the existing swaddling rules (D8 / D9) and the existing fast-motion
  rules (DataAugment `motion_fast`, severity large) unchanged. Nothing was chosen from method results on these videos.

## Videos (make_videos.py)

**Sources:** UBFC subjects 1, 3 and 4. Originals are unchanged (SHA-256 checked).

**Swaddling** (same rule as module1_v2_d8 / module1_v2_d9):
* Cloth top edge at chin + 0 face heights (**medium**: neck covered) or chin − 0.25 face heights (**large**: chin and
  mouth covered).
* Edge waves 6 px over 120 px.
* Drift ±0.05 face heights, low-pass filtered at 0.05 Hz.
* Pale-blue cloth.

**Fast motion:** DataAugment `motion_fast`, severity **large**, applied to the whole swaddled frame, so subject and
cloth move together. Settings:
* translation 160 px, rotation 12°, zoom 10 %, drift 15 px, tremor 1.5 px RMS;
* Poisson repositioning moves every 3 s on average, each lasting 0.25–0.6 s with a minimum-jerk profile;
* the first 2 s are still.

**Seeds:**
* cloth and swaddle drift: [20261117, subject];
* motion: [20261117, subject, 1];
* both levels of one subject share the same motion (common random numbers).

**Total:** 3 subjects × 2 swaddle levels = **6 videos**, lossless FFV1 with checks.

## Methods (frozen, nothing tuned)

* **Fixed box** (B0).
* **Module 1 v1** (D0).
* **Module 1 v2 D5**: the current system.
* **Module 1 v2 D9**: D5 plus the swaddling rules, pre-registered in ../PREREGISTRATION.md, evaluated on the mirror run.
* **BG**: background control.

Settings: unchanged comparison tool, strict gate 2, HR band 45–180 bpm.

## Evaluation (swmo_report.py)

* **Reference:** fingertip PPG, one value per 8 s window, matched by the nearest window centre within 1 s.
* **Error:** |HR − reference| on all windows. Windows with no HR are counted and reported per method.
* **Paired difference:** computed on windows where both methods give HR.
* **95 % CI:** moving-block bootstrap with blocks of 8 windows, resampled within video and pooled over the 3 subjects,
  5000 resamples, seed 42.

## Claim rules (applied automatically)

* **M1 (primary): "Under swaddling (to the neck) with fast movement, dynamic ROI (D5) gives more accurate HR than a
  fixed box."**
  * At medium: D5 − fixed box CI upper bound < 0, pooled over the 3 subjects,
  * **and** D5's MAE is lower than the fixed box's in at least 2 of the 3 subjects.
* **M2:** the same rule at large (cloth to the mouth).
* **M3 / M4:** the same rules for **D9** at medium / large.
  * A D9 claim may be used only if D9 also passes its own no-harm rule in the D9 report.
* **Also reported, not claims:**
  * D9 − D5 and v1 − fixed;
  * no-HR counts;
  * context cells from earlier runs: swaddle only and motion only.
* **Scope:** simulated cloth and simulated whole-frame motion on 3 adult UBFC subjects. This is not real infants.

## Deviations

None so far.
