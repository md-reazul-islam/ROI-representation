# HR RR using module1 v2: HR-specific region set

**Purpose:** HR and RR from the Module 1 v2 output. HR chooses **its own regions** instead of using the Module 1 v2
top 3. Those are ranked for ROI quality and usually land on the torso, which suits RR but often not HR.

- **Notebook:** `HR RR using module1 v2.ipynb` (repo root). Run top to bottom; edit only cell 1.
- **Code:** folder `HR RR using module1 v2/`, a copy of `module2_v2/`. Its package is renamed `hr_rr_m1v2`, so the
  two can never mix.
- **Unchanged:** Module 1 v2, `module2_v2/`, `RUN_HR_RR_V2_FULL.ipynb` and `Updated HR RR(v3).ipynb`.

## 1. Settings (cell 1)

| Setting | Meaning | Usual value |
|---|---|---|
| `VIDEO_FOLDER` | Folder inside `Module1` with the video(s) | `'HR RR V2'` |
| `HR_REGION_SET` | `'hr_specific'` = own HR regions (new); `'module1'` = Module 1 v2 top 3 (identical to Module 2 v2) | `'hr_specific'` |
| `DETECT_CAREGIVER` | `True` = detect caregiver contact and exclude the overlapping windows (cell 6b; see the last section) | `True` |
| `EXISTING_RUN`, `MODULE1_CONFIG`, `TUNING_LOG`, `REFERENCE_HR`, `REFERENCE_RR`, `WORKERS` | as in `RUN_HR_RR_V2_FULL.ipynb` | – |

The default in the code's config (`configs/hr_rr_m1v2.yaml`, `hr.region_set`) stays `module1`. The notebook passes
`--hr-region-set`. Outputs go to `HR RR using module1 v2/runs/<RUN>/D5_hr_specific/` (or `D5/` for `module1`), so the two
modes never overwrite each other.

## 2. Steps

Same as `RUN_HR_RR_V2_FULL.ipynb`:
1. Module 1 base run (skipped if done)
2. v2 features (skipped if done)
3. v2 representation
4. HR + RR, with the plots
5. **Monitor video**: red = the HR regions actually used in each window, cyan = RR regions
6. ROI comparison v1 vs v2
7. Zip (`<RUN>_HR_RR_using_module1_v2.zip`)

About 5 min for a 40 s video once Module 1 is done.

## 3. How the HR-specific region set works

| Item | Rule |
|---|---|
| Source | **All** Module 1 v2 candidates in `dynamic_roi_v2.csv` (every region, every frame, with its pixel polygon), not only the `selected` rows |
| Candidates | Face, forehead, left cheek, right cheek, upper face, upper chest. Limbs and head/hair are excluded |
| Eligible | At least 150 skin pixels, a skin fraction of at least 0.30 of the polygon (covered regions drop out), and available in at least 90% of the window |
| Prior (no pulse information) | Module 1 v2 quality q × log(skin pixels) |
| Pulse evidence | Each region's SNR minus **its own** block-shuffle null threshold, averaged over **past windows only** (windows that end before the current one starts, within 20 s) |
| Score | prior × sigmoid(evidence / 2 dB). With history, a region joins only if it beats its own null (when any region does) and scores at least 25% of the best. Without history (first seconds), the prior alone |
| Overlap | Two regions overlapping by more than 30% (e.g. face + cheek) are never both selected |
| Stability | A new set must score 10% higher for 3 windows before it replaces the current set. A member that loses its evidence leaves at once |
| Fusion | Score-weighted mean of the selected regions' pulses. Split-half agreement uses their left and right halves |
| **Validity** | The **whole select-then-measure procedure** is repeated on block-shuffled surrogates of all candidates. A window is *present* only if its fused SNR beats that null, and *validated* if the two halves also agree. So selecting regions on noise cannot validate itself |

RR is exactly as in Module 2 v2: chest and abdomen candidates, camera-compensated motion.

## 4. New columns in `hr_windows.csv` (hr_specific)

| Column | Meaning |
|---|---|
| `hr_regions` | Regions used in the window and their weights, e.g. `{"FACE": 0.6, "UPPER_CHEST": 0.4}` |
| `selection_source` | `prior` (warm-up, no history yet) or `history` (ranked on past pulse evidence) |
| `region_switch` | True when the set changed from the previous window |
| `region_snr_db`, `region_null_db` | Every candidate's SNR in this window and its own null threshold |

## 5. Tests (`HR RR using module1 v2/tests`, 8 pass)

**Copied from Module 2 v2 (5):** HR and RR recovered and validated on true signals; noise and static patches not validated.

**New (3)**, on a synthetic video where the face pulses (130 bpm) and the larger "chest", which the stand-in Module 1 marks
as selected, does not:

| Test | Result |
|---|---|
| `module1` mode (chest only) | not validated, correctly |
| `hr_specific` mode | moves to the face once it has pulse history, never pairs overlapping regions, HR 130 bpm validated |
| `hr_specific` with no pulse anywhere | not validated (selection on noise does not validate itself) |

## 6. Result on the real video `NEW_1104` (`HR RR V2/1104.mp4`, true HR 157)

| | `module1` (= Module 2 v2) | `hr_specific` |
|---|---|---|
| HR regions | Thoraco-abdominal, upper chest, abdomen | Upper chest + both cheeks at first, then face + upper chest |
| Validated windows | 3 of 30 (10%), at about 135 bpm | 0 of 30 |
| Windows within ±10 of 157 | 0 | 0 |
| RR (validated) | about 25 breaths/min | same |

**How to read this:**
- **No region carries a pulse.** Every candidate's SNR is −1 to −5 dB, below its own null (about −1 dB). The HR-specific mode therefore reports no HR, which is correct.
- **The 156.5 median is not a result.** `hr_bpm_median_all` is 156.5 for the HR-specific mode, but the per-window values jump between about 180 and about 120–135, and the median just falls between them. Ignore it.
- **The `module1` validated windows are not trustworthy either.** Its 3 validated windows (about 135) are 22 bpm below 157.
- **What would show a gain:** a recording where the pulse is visible (original phone file, steady light, camera closer)
  with a monitor HR over the same minute.

## 7. Shell commands (same as the notebook)

```bash
cd .
PY=python
$PY "HR RR using module1 v2/scripts/run_vitals_v2.py" --run "runs/HR RR V2_e234ae6f" --module1-config D5 --hr-region-set hr_specific
$PY "HR RR using module1 v2/scripts/render_monitor_v2.py" --run "runs/HR RR V2_e234ae6f" --module1-config D5 \
    --results "HR RR using module1 v2/runs/HR RR V2_e234ae6f/D5_hr_specific"
$PY -m pytest -q "HR RR using module1 v2/tests"        # 8 tests, about 1 min
```

## Change after NEW_1104 (hand-held phone video), 2026-11-05: camera-confound check for RR
Module 1 v2 estimates camera motion as the median background flow. On a video with black bars (NEW_1104: portrait clip
inside 1920×1080) the bars dominate the background, so `global_dx`/`global_dy` are 0 and the residual motion still
contains the camera sway (21–27 /min). Module 1 v2 is not changed; instead `rr.confound_check` (config `rr.confound`)
marks a window `not_assessable` when its rate is also the rate of the camera motion or of a non-breathing ROI (HEAD,
FACE) with comparable amplitude. NEW_1104: 9 "validated" windows before (all at the camera rate), 1 after.
Columns added to `rr_windows.csv`: `camera_confounded`, `confound_source`, `confound_rate_brpm`.
Best fix for such videos: crop the black bars before Module 1 (then Module 1 v2 can measure the camera motion).
Refinement: the head check is used only when the camera estimate is degenerate (NEW_1104) with a strict ratio (0.5), or
when the measured camera moves at the window's rate; otherwise a swaddled infant's head (moves with breathing) would veto
real breathing. Known limit: this RR branch uses Module 1 v2's per-ROI motion, so it cannot bridge torso ROI gaps
(NEW_1105: 0 % validated); RR+ (cell 7d) bridges them and is the RR to report for such videos.

## Caregiver contact → *caregiver intervention* windows (added 2026-11-08)

**Why:** when a caregiver handles the infant, the HR / RR signals measure the handling, not the infant (NEW_10: RR
jumped from 28 to 39 breaths/min when the hands arrived). Windows during handling are now labelled
`caregiver_intervention` and **excluded from the reported rate and the accuracy evaluation**, as is standard in NICU
camera studies. The HR / RR estimates themselves are unchanged; only labels and summaries are added.

**Setting (cell 1):** `DETECT_CAREGIVER = True` (cell 6b). Settings: `HR RR using module1 v2/configs/caregiver.yaml`,
fixed a priori (geometry, not tuned on any evaluation video).

**How contact is detected** (`scripts/detect_caregiver_contact.py`, whole video at 10 frames/s, GPU, ~1 min per minute
of video):

| Cue | Rule | Used for contact |
|---|---|---|
| **hand** | Standard COCO pose model (Ultralytics YOLO11s-pose, `models/yolo11s-pose.pt`): every person that is not the infant gives its wrists; hand point = wrist + 0.35 × (wrist − elbow). Contact when a hand point is within 2 % of the frame diagonal (~44 px at 1080p) of the SAM 2 infant mask | yes |
| **SAM 2 mask growth** | The infant mask area jumps above 1.40 × the median of the clean masks of the previous 10 s: SAM 2 absorbed the caregiver's hands / arms into the "infant" mask (NEW_10 at 19 s: 0.24 → 0.44 of the frame). The last clean mask is then the infant reference | yes |
| mask shrink | Area below 0.60 × baseline | reported only (SAM 2 also shrinks when it corrects an over-segmentation: NEW_1104, no caregiver) |
| body | Adult silhouette (YOLO11s-seg, `models/yolo11s-seg.pt`) touching the infant mask | reported only (in 2-D an adult standing *behind* the infant looks the same: NEW_10, 0–18 s) |

Which detection is the infant: highest IoU (≥ 0.30) with the infant box; a pose whose confident shoulders / hips lie
≥ 50 % inside the infant mask; a silhouette ≥ 60 % inside the infant mask. An adult needs ≥ 25 % of its box outside the
(5 % enlarged) infant box. Adult hand points within 1 % of the diagonal of an infant wrist are dropped.
Time line: majority vote over 5 frames (0.5 s), episodes merged across gaps ≤ 1 s, episodes < 0.3 s dropped, every
episode **padded by 1 s** on both sides. A window overlapping a padded episode at all is `caregiver_intervention`.

**Outputs**

| File | Content |
|---|---|
| `HR RR using module1 v2/runs/<RUN>/caregiver/caregiver_frames.csv` | per analysed frame: adults, hand points, distance to the infant mask, mask ratio, contact and its source |
| `.../caregiver/caregiver_segments.csv` | contact episodes: `start_sec`/`end_sec` and the excluded span `pad_start_sec`/`pad_end_sec` |
| `.../caregiver/caregiver_summary.csv` | per video: % adult in view, % contact, episodes, excluded seconds |
| `.../caregiver/preview/<video>_caregiver.mp4` | **check the detection here**: white = infant reference, orange = adults, red dot = hand in contact, yellow = hand not in contact, purple bar = contact |
| `<OUT_V2>/caregiver/{hr,rr,rr_plus}_windows_caregiver.csv` | window tables + `caregiver_overlap_sec`, `caregiver_overlap_fraction`, `adult_in_view_fraction`, `caregiver_intervention`, `label_with_caregiver` |
| `<OUT_V2>/caregiver/caregiver_window_summary.csv` | per signal and video: validated % of all windows and of the windows **without** intervention; median validated rate with / without |
| `<OUT_V2>/caregiver/evaluation_*_caregiver.csv` | with references: MAE / RMSE / within tolerance for all and validated windows, with and without intervention windows (report the `*_excl_intervention` rows) |

Plots (cell 7): purple dots and shading. Monitor video (cell 7b): purple banner "CAREGIVER CONTACT" and
`EXCLUDED: caregiver intervention` instead of the value.

**Check on the videos available (no ground-truth contact labels; judged from the preview videos):**

| Video | Detected | Correct? |
|---|---|---|
| NEW_10 (`10.mp4`, 34 s) | adult in view 98 %; contact **18.95 s – end** (hand + mask); excluded 17.95–34.1 s | yes: gloved hands from ~19 s; 0–18 s the adult only stands behind (no contact) |
| NEW_1105 (`1105.mp4`, 63 s) | no adult, no contact | yes |
| NEW_1104 (`1104.mp4`, 38 s) | no adult, no contact | yes (after making the shrink cue report-only) |
| NEW_1107_2 (`1107 (2).mp4`, 17 s) | no contact | **missed**: fingertips touch the shoulder at ~15 s, but no person is visible |

**Limitations (state them in the paper):** contact is judged in the 2-D image (a hand hovering over the infant counts);
a caregiver not visible as a person (only fingertips) is missed unless SAM 2 absorbs it; while SAM 2 is contaminated
the last clean mask is used, which lags when the camera moves; there are no ground-truth contact labels yet: annotate
contact start / end on a few videos to report sensitivity / specificity.

**Methods text (draft):** *Caregiver contact was detected automatically at 10 frames/s. A COCO-pretrained pose model
(YOLO11s-pose) located adult wrists; a hand within 2 % of the frame diagonal of the SAM 2 infant mask, or a sudden
growth of the infant mask (> 1.4 × its median over the previous 10 s, indicating that the caregiver's hands had been
absorbed into the segmentation), was scored as contact. Contact was smoothed over 0.5 s, merged across gaps ≤ 1 s and
padded by 1 s; HR and RR windows overlapping a contact episode were labelled caregiver intervention and excluded from
the accuracy analysis. All thresholds were fixed before analysis.*

**Shell commands**
```
python "HR RR using module1 v2/scripts/detect_caregiver_contact.py" --run runs/<RUN> [--video-root <folder if the video moved>]
python "HR RR using module1 v2/scripts/apply_caregiver_labels.py" --run runs/<RUN> --results "HR RR using module1 v2/runs/<RUN>/D5_hr_specific" \
       [--rr-plus rr_plus/runs/<RUN>/D5] [--reference-hr ref_hr.csv] [--reference-rr ref_rr.csv]
python "HR RR using module1 v2/scripts/render_monitor_v2.py" --run runs/<RUN> --module1-config D5 \
       --results "HR RR using module1 v2/runs/<RUN>/D5_hr_specific" --caregiver "HR RR using module1 v2/runs/<RUN>/D5_hr_specific/caregiver"
```
Tests: `HR RR using module1 v2/tests/test_caregiver.py` (6 tests; all 14 tests of the folder pass).
