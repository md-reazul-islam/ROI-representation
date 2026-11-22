#!/bin/bash
# D8 evaluation, run once (PREREGISTRATION.md). Resumable: every step is skipped when its output exists.
# Existing results are not touched: new outputs only (module1_v2_d8/, runs/ubfc_swaddle, module1_v2/runs/ubfc_swaddle,
# module1_v2/runs/<run>/dynamic_roi_v2_D8 symlinks, "HR RR using module1 v2/runs/<run>/aug_eval_D8 | d8check_*").
set -euo pipefail
cd "$(dirname "$0")/.."                          # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json
REF=module1_v2_d8/reference_hr_ppg.csv
D8=module1_v2_d8
log(){ echo "[$(date '+%F %T')] $*"; }

log "0 reference HR incl. swaddling video IDs (copy of the UBFC PPG reference; existing file untouched)"
$PY - <<'EOF'
import pandas as pd
ref = pd.read_csv('dataaugment/hr_eval/reference_hr_ppg.csv')
s1 = ref[ref.video_id == 'NEW_subject1']
extra = [s1.assign(video_id=f'NEW_subject1_swaddle_{s}') for s in ('small', 'medium', 'large')]
pd.concat([ref] + extra, ignore_index=True).to_csv('module1_v2_d8/reference_hr_ppg.csv', index=False)
EOF

log "1 D8 on the existing UBFC runs (colour features -> D8 export -> HR)"
for R in ubfc_flat_38d8f553 ubfc_aug ubfc_aug_fast ubfc_aug_fast_xl; do
  [ -f $D8/runs/$R/colour_features.csv ] || $PY $D8/scripts/colour_features.py --run runs/$R
  [ -f $D8/runs/$R/dynamic_roi_v2_D8/dynamic_roi_v2.csv ] || $PY $D8/scripts/export_d8.py --run runs/$R --tuning-log $TUNE
  [ -f "HR RR using module1 v2/runs/$R/aug_eval_D8/window_results.csv" ] || \
    $PY "$CMP" --run runs/$R --module1-config D8 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D8
done

log "2 swaddling videos: Module 1 base run, v2 features, D5 / D0 / D8, HR"
R=ubfc_swaddle
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || \
  $PY scripts/run_new_video.py $D8/data_swaddle --branch sam2 --name $R
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done
[ -f $D8/runs/$R/colour_features.csv ] || $PY $D8/scripts/colour_features.py --run runs/$R
[ -f $D8/runs/$R/dynamic_roi_v2_D8/dynamic_roi_v2.csv ] || $PY $D8/scripts/export_d8.py --run runs/$R --tuning-log $TUNE
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D8/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D8 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D8

log "3 infant data final34 (no reference): D5 vs D8, label-free, infant band"
R=final34_paper
[ -f $D8/runs/$R/colour_features.csv ] || $PY $D8/scripts/colour_features.py --run runs/$R
[ -f $D8/runs/$R/dynamic_roi_v2_D8/dynamic_roi_v2.csv ] || $PY $D8/scripts/export_d8.py --run runs/$R --tuning-log $TUNE
[ -f "HR RR using module1 v2/runs/$R/d8check_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods D5,BG --out-name d8check_D5
[ -f "HR RR using module1 v2/runs/$R/d8check_D8/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D8 --methods D5,BG --out-name d8check_D8

log "4 report"
$PY $D8/scripts/d8_report.py
log "done"
