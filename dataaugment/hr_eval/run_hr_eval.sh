#!/bin/bash
# HR evaluation of the DataAugment videos: fixed box (B0) vs Module 1 v1 (D0) vs Module 1 v2 (D5), same HR chain.
# Every step is skipped when its output exists, so the script can be re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json        # frozen stabilizer (nothing tuned on these videos)
REF=dataaugment/hr_eval/reference_hr_ppg.csv
AUG=ubfc_aug; ORIG=ubfc_flat_38d8f553                       # ORIG = the unaugmented UBFC run (severity "none")
log(){ echo "[$(date '+%F %T')] $*"; }

log "1/6 Module 1 base run (SAM 2 branch) on the 9 augmented videos"
[ -f runs/$AUG/outputs/confidence_sam2/roi_confidence.csv ] || \
  $PY scripts/run_new_video.py dataaugment/hr_eval/videos --branch sam2 --name $AUG
log "2/6 Module 1 v2 features"
[ -f module1_v2/runs/$AUG/features/roi_features_median.csv ] || \
  $PY module1_v2/scripts/run_v2_features.py --run runs/$AUG --workers 2
log "3/6 dynamic ROI export: D5 (v2 headline) and D0 (v1-equivalent)"
for R in $AUG $ORIG; do for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done; done
log "4/6 reference HR (PPG waveform, 8 s windows) for original and augmented video IDs"
$PY dataaugment/hr_eval/make_reference.py
log "5/6 HR on the same chain (adult band 45-180 bpm)"
for R in $AUG $ORIG; do
  [ -f "HR RR using module1 v2/runs/$R/aug_eval_D5/window_results.csv" ] || \
    $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
  [ -f "HR RR using module1 v2/runs/$R/aug_eval_D0/window_results.csv" ] || \
    $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
done
log "6/6 report"
$PY dataaugment/hr_eval/build_hr_report.py
log "done"
