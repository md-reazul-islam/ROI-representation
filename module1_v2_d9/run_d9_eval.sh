#!/bin/bash
# D9 evaluation, run once (PREREGISTRATION.md). Resumable. New outputs only: module1_v2_d9/, runs/ubfc_swaddle_test,
# runs/<run>_d9 mirrors, module1_v2/runs/ubfc_swaddle_test + <run>_d9, "HR RR using module1 v2/runs/<...>/aug_eval_D9 | d9check".
set -euo pipefail
cd "$(dirname "$0")/.."                          # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json
D=module1_v2_d9
REF=$D/reference_hr_ppg.csv
HRR="HR RR using module1 v2/runs"
log(){ echo "[$(date '+%F %T')] $*"; }
d9(){ [ -f $D/runs/$1/D9/dynamic_roi_v2.csv ] || $PY $D/scripts/export_d9.py --run runs/$1 --stabilise --name D9; }

log "0 test videos (subjects 3, 4) and reference"
[ -f $D/data_swaddle_test/subject4_swaddle_large.json ] || $PY $D/scripts/make_swaddle_test_videos.py
$PY - <<'EOF'
import pandas as pd
from pathlib import Path
ref = pd.read_csv('dataaugment/hr_eval/reference_hr_ppg.csv')
parts = [ref]
ids = [p.stem for p in Path('module1_v2_d9/data_swaddle_test').glob('*.mkv')] + \
      [p.stem for p in Path('module1_v2_d8/data_swaddle').glob('*.mkv')] + ['subject1_democentre']
for s in ids:
    parts.append(ref[ref.video_id == 'NEW_' + s.split('_')[0]].assign(video_id='NEW_' + s))
pd.concat(parts, ignore_index=True).to_csv('module1_v2_d9/reference_hr_ppg.csv', index=False)
EOF

log "1 test run: Module 1 base run, v2 features, D5 / D0 exports, D9"
R=ubfc_swaddle_test
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || $PY scripts/run_new_video.py $D/data_swaddle_test --branch sam2 --name $R
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done
d9 $R

log "2 test run HR: fixed + D5 (original run), v1, D9 (mirror run)"
[ -f "$HRR/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "$HRR/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0
[ -f "$HRR/${R}_d9/aug_eval_D9/window_results.csv" ] || \
  $PY "$CMP" --run runs/${R}_d9 --module1-config D9 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D9

log "3 no-harm: D9 on every existing UBFC condition (+ development swaddle run, reported separately)"
for R in ubfc_flat_38d8f553 ubfc_aug ubfc_aug_fast ubfc_aug_fast_xl ubfc_occdemo ubfc_swaddle; do
  d9 $R
  [ -f "$HRR/${R}_d9/aug_eval_D9/window_results.csv" ] || \
    $PY "$CMP" --run runs/${R}_d9 --module1-config D9 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D9
done

log "4 infant data final34 (label-free, infant band)"
d9 final34_paper
[ -f "$HRR/final34_paper_d9/d9check_D9/window_results.csv" ] || \
  $PY "$CMP" --run runs/final34_paper_d9 --module1-config D9 --methods D5,BG --out-name d9check_D9

log "5 report"
$PY $D/scripts/d9_report.py
log "done"
