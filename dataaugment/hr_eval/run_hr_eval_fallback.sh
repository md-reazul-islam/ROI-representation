#!/bin/bash
# HR evaluation with FALLBACK gate 2 (existing implementation = method D5_HR of the comparison script: Module 1 v2
# dynamic ROI + HR-specific regions + gate 2 fallback left/right -> face/chest -> top/bottom, no visible pair ->
# 'validated_one_side'; configs/roi_comparison.yaml d5_hr_gate2_mode: fallback). Same as the infant notebook's HR.
# The fixed box (B0) has no fallback implementation and keeps its left/right gate 2.
# Needs the Module 1 base runs + D5/D0 exports of run_hr_eval*.sh (reused, not recomputed). Output folders end in _fb,
# so the strict results (aug_eval_D5 / aug_eval_D0) are not touched. Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
REF=dataaugment/hr_eval/reference_hr_ppg.csv
log(){ echo "[$(date '+%F %T')] $*"; }
for R in ubfc_flat_38d8f553 ubfc_aug ubfc_aug_fast ubfc_aug_fast_xl; do
  for C in D5 D0; do
    [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || { echo "missing export $R $C: run run_hr_eval*.sh first"; exit 1; }
  done
  log "fallback HR: $R, Module 1 v2 (D5) + fixed box"
  [ -f "HR RR using module1 v2/runs/$R/aug_eval_D5_fb/window_results.csv" ] || \
    $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5_HR,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5_fb
  log "fallback HR: $R, Module 1 v1 (D0)"
  [ -f "HR RR using module1 v2/runs/$R/aug_eval_D0_fb/window_results.csv" ] || \
    $PY "$CMP" --run runs/$R --module1-config D0 --methods D5_HR,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0_fb
done
log "report (fallback)"
HR_GATE2=fallback $PY dataaugment/hr_eval/build_hr_report.py
HR_GATE2=fallback $PY dataaugment/hr_eval/make_fast_motion_figures.py
log "done"
