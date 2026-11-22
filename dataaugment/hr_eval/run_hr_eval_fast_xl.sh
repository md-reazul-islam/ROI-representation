#!/bin/bash
# HR evaluation of the EXTRA-LARGE fast motion video (category motion_fast, subject1): fixed box (B0) vs Module 1 v1 (D0)
# vs Module 1 v2 (D5), same HR chain as run_hr_eval.sh. Resumable: each step is skipped when its output exists.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json        # frozen stabilizer (nothing tuned on these videos)
REF=dataaugment/hr_eval/reference_hr_ppg.csv
R=ubfc_aug_fast_xl
log(){ echo "[$(date '+%F %T')] $*"; }

while pgrep -f run_hr_eval_fast.sh >/dev/null; do sleep 60; done   # one GPU job at a time
log "1/6 Module 1 base run (SAM 2 branch) on the extra-large fast-motion video"
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || \
  $PY scripts/run_new_video.py dataaugment/hr_eval/videos_fast_xl --branch sam2 --name $R
log "2/6 Module 1 v2 features"
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || \
  $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
log "3/6 dynamic ROI export: D5 and D0"
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done
log "4/6 reference HR"
$PY dataaugment/hr_eval/make_reference.py
log "5/6 HR on the same chain (adult band 45-180 bpm)"
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
log "6/6 report"
$PY dataaugment/hr_eval/build_hr_report.py
log "done"
