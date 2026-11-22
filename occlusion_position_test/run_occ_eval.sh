#!/bin/bash
# Occlusion-position test, run once (PREREGISTRATION.md). Resumable: every step is skipped when its output exists.
# New outputs only: occlusion_position_test/, runs/ubfc_occpos, module1_v2/runs/ubfc_occpos,
# module1_v2_d8/runs/ubfc_occpos (D8, secondary), "HR RR using module1 v2/runs/ubfc_occpos/aug_eval_D5|D0|D8".
set -euo pipefail
cd "$(dirname "$0")/.."                          # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
TUNE=module1_v2/runs/final34_paper/tuning_log.json
T=occlusion_position_test
REF=$T/reference_hr_ppg.csv
R=ubfc_occpos
log(){ echo "[$(date '+%F %T')] $*"; }

log "0 reference HR per new video ID (copy of the UBFC PPG reference; existing file untouched)"
$PY - <<'EOF'
import pandas as pd
from pathlib import Path
ref = pd.read_csv('dataaugment/hr_eval/reference_hr_ppg.csv')
parts = [ref]
for p in sorted(Path('occlusion_position_test/data').glob('subject*_occpos_*.mkv')):
    subj = p.stem.split('_occpos_')[0]
    parts.append(ref[ref.video_id == f'NEW_{subj}'].assign(video_id=f'NEW_{p.stem}'))
pd.concat(parts, ignore_index=True).to_csv('occlusion_position_test/reference_hr_ppg.csv', index=False)
print(len(parts) - 1, 'videos')
EOF

log "1 Module 1 base run (SAM2) on the 30 videos"
[ -f runs/$R/outputs/confidence_sam2/roi_confidence.csv ] || $PY scripts/run_new_video.py $T/data --branch sam2 --name $R

log "2 Module 1 v2 features and D5 / D0 exports"
[ -f module1_v2/runs/$R/features/roi_features_median.csv ] || $PY module1_v2/scripts/run_v2_features.py --run runs/$R --workers 2
for C in D5 D0; do
  [ -f module1_v2/runs/$R/dynamic_roi_v2_$C/dynamic_roi_v2.csv ] || \
    $PY module1_v2/scripts/export_dynamic_roi_v2.py --run runs/$R --config $C --tuning-log $TUNE
done

log "3 HR: fixed box + D5 + BG, then v1"
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D5/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D5
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D0/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D0 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D0

log "4 D8 (secondary, frozen)"
[ -f module1_v2_d8/runs/$R/colour_features.csv ] || $PY module1_v2_d8/scripts/colour_features.py --run runs/$R
[ -f module1_v2_d8/runs/$R/dynamic_roi_v2_D8/dynamic_roi_v2.csv ] || \
  $PY module1_v2_d8/scripts/export_d8.py --run runs/$R --tuning-log $TUNE
[ -f "HR RR using module1 v2/runs/$R/aug_eval_D8/window_results.csv" ] || \
  $PY "$CMP" --run runs/$R --module1-config D8 --methods D5,BG --hr-band 45,180 --reference-hr $REF --out-name aug_eval_D8

log "5 report"
$PY $T/scripts/occ_report.py
log "done"
