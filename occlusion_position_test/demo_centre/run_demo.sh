#!/bin/bash
# Demonstration video only (not pre-registered). Resumable. New outputs: runs/ubfc_occdemo, module1_v2/runs/ubfc_occdemo,
# module1_v2_d8/runs/ubfc_occdemo, "HR RR using module1 v2/runs/ubfc_occdemo/aug_eval_D5|D0|D8".
set -euo pipefail
cd "$(dirname "$0")/../.."                       # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json
D=occlusion_position_test/demo_centre
REF=$D/reference_hr_ppg.csv
R=ubfc_occdemo
log(){ echo "[$(date '+%F %T')] $*"; }

log "0 reference"
$PY - <<'EOF'
import pandas as pd
ref = pd.read_csv('dataaugment/hr_eval/reference_hr_ppg.csv')
s1 = ref[ref.video_id == 'NEW_subject1'].assign(video_id='NEW_subject1_democentre')
pd.concat([ref, s1], ignore_index=True).to_csv('occlusion_position_test/demo_centre/reference_hr_ppg.csv', index=False)
EOF
mkdir -p $D/video && ln -sf ../subject1_democentre.mkv $D/video/subject1_democentre.mkv
log "1 Module 1 base run"
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || $PY scripts/run_new_video.py $D/video --branch sam2 --name $R
log "2 v2 features, D5 / D0"
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done
log "3 HR"
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
log "4 D8 (frozen)"
[ -f module1_v2_d8/runs/$R/colour_features.csv ] || $PY module1_v2_d8/scripts/colour_features.py --run runs/$R
[ -f module1_v2_d8/runs/$R/dynamic_roi_v2_D8/dynamic_roi_v2.csv ] || $PY module1_v2_d8/scripts/export_d8.py --run runs/$R --tuning-log $TUNE
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D8/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D8 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D8
log "5 report"
$PY $D/demo_report.py
log "done"
