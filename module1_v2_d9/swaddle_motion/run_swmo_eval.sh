#!/bin/bash
# Swaddling + fast motion test, run once (PREREGISTRATION.md in this folder). Resumable. New outputs only.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json
T=module1_v2_d9/swaddle_motion
REF=$T/reference_hr_ppg.csv
HRR="HR RR using module1 v2/runs"
R=ubfc_swaddle_motion
log(){ echo "[$(date '+%F %T')] $*"; }

log "0 videos and reference"
[ -f $T/data/subject4_swmo_large.json ] || $PY $T/make_videos.py
$PY - <<'EOF'
import pandas as pd
from pathlib import Path
ref = pd.read_csv('dataaugment/hr_eval/reference_hr_ppg.csv')
parts = [ref] + [ref[ref.video_id == 'NEW_' + p.stem.split('_')[0]].assign(video_id='NEW_' + p.stem)
                 for p in Path('module1_v2_d9/swaddle_motion/data').glob('*.mkv')]
pd.concat(parts, ignore_index=True).to_csv('module1_v2_d9/swaddle_motion/reference_hr_ppg.csv', index=False)
EOF

log "1 Module 1 base run, v2 features, D5 / D0, D9"
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || $PY scripts/run_new_video.py $T/data --branch sam2 --name $R
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done
[ -f module1_v2_d9/runs/$R/D9/dynamic_roi_v2.csv ] || $PY module1_v2_d9/scripts/export_d9.py --run runs/$R --stabilise --name D9

log "2 HR: fixed + D5, v1, D9 (mirror run)"
[ -f "$HRR/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "$HRR/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
[ -f "$HRR/${R}_d9/aug_eval_D9/window_results.csv" ] || \
  $PY "$CMP" --run runs/${R}_d9 --module1-config D9 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D9

log "3 report"
$PY $T/swmo_report.py
log "done"
