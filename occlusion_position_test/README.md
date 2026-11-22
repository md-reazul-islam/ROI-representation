# occlusion_position_test: does dynamic ROI help when the occluder covers the fixed region?

Separate test. **No existing file is changed**: Module 1, Module 1 v2 (D5), D8, the HR code, DataAugment, the notebooks
and the manuscripts are only read or run.

* `PREREGISTRATION.md`: the design, evaluation and claim rules. Fixed before any method ran on these videos.
* `scripts/occluder.py`: the hand positions, sizes, timing and jitter, shared by the generator and the report.
* `scripts/make_occlusion_videos.py`: writes the 30 test videos to `data/`. That is UBFC subjects 1, 3 and 4 × 5 positions
  × 2 sizes, lossless FFV1. Each video gets a `_frames.csv` (hand placement and coverage per frame) and a `.json`
  (checks). There is also a `preview_subject<k>.png` (orange rectangle = estimated fixed box) and `videos_summary.csv`.
* `run_occ_eval.sh`: the whole evaluation, run once and resumable. Log in `run_occ_eval.log`. Steps:
  1. reference
  2. Module 1 base run
  3. v2 features
  4. D5 / D0
  5. HR
  6. D8 (secondary)
  7. report
* `scripts/occ_report.py`: metrics, paired bootstrap tests and the pre-registered claims. Writes `report/occ_report.html`.

New outputs outside this folder (additions only):
* `runs/ubfc_occpos/`
* `module1_v2/runs/ubfc_occpos/`
* `module1_v2_d8/runs/ubfc_occpos/`, plus its symlink
* `HR RR using module1 v2/runs/ubfc_occpos/`

Run: `cd Module1; setsid nohup occlusion_position_test/run_occ_eval.sh > occlusion_position_test/run_occ_eval.log 2>&1 &`
