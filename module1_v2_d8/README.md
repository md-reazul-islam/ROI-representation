# module1_v2_d8 — Module 1 v2 configuration D8 (D5 + occlusion / coverage handling)

Separate add-on. **No existing file is changed**: Module 1, Module 1 v2 (D5 and all other configurations), the HR code,
DataAugment, the notebooks and the manuscripts are only read or run.

* `PREREGISTRATION.md`: rules, thresholds, evaluation and claim rules, fixed on 2026-11-15 before any D8 result.
* `d8/rules.py`: R1 coverage (visible skin drop), R2 appearance change relative to the other regions (skin-coloured
  occluder), R3 redundancy (area-weighted fusion of all clean HR-group regions while occlusion is present).
* `scripts/colour_features.py`: per-region colour of all polygon pixels (R2 input).
* `scripts/export_d8.py`: D8 table in the D5 export format (+ `d8_flags.csv`); `--no-rules` reproduces D5 exactly
  (checked: all 54,434 rows and columns of the UBFC D5 export identical).
* `scripts/make_swaddle_videos.py`: the pre-registered swaddling test videos (`data_swaddle/`).
* `scripts/d8_report.py`: metrics, paired bootstrap tests and the pre-registered claim checks -> `report/D8_report.html`.
* `run_d8_eval.sh`: the whole evaluation, run once (resumable), log `run_d8_eval.log`.
* `tests/test_rules.py`: synthetic checks of the rules (`pytest module1_v2_d8/tests`).

New outputs outside this folder (additions only): `runs/ubfc_swaddle/`, `module1_v2/runs/ubfc_swaddle/`, symlinks
`module1_v2/runs/<run>/dynamic_roi_v2_D8 -> module1_v2_d8/runs/<run>/dynamic_roi_v2_D8` (the unchanged comparison tool
reads D8 there), and `HR RR using module1 v2/runs/<run>/aug_eval_D8`, `.../ubfc_swaddle/aug_eval_D5|D0`,
`.../final34_paper/d8check_D5|D8`. Deleting this folder plus those additions removes D8 completely.
