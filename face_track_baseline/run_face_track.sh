#!/bin/bash
# Tracked face box (FACE_TRACK) baseline: UBFC robustness runs + final34 (label-free). New outputs only (out-name facetrack).
# FACE_TRACK = bounding rectangle of Module 1's FACE region per sampled frame, same pixel rules and HR chain as B0 / D5.
set -uo pipefail
cd "$(dirname "$0")/.."                          # Module1 root
PY=${PY:-python}
CMP="HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py"
T=face_track_baseline
HRR="HR RR using module1 v2/runs"
log(){ echo "[$(date '+%F %T')] $*"; }

log "0 reference (union of the existing reference files)"
$PY - <<'PYEOF'
import pandas as pd
parts = [pd.read_csv(p) for p in ('module1_v2_d9/reference_hr_ppg.csv', 'module1_v2_d9/swaddle_motion/reference_hr_ppg.csv')]
pd.concat(parts, ignore_index=True).drop_duplicates(['video_id', 'timestamp_sec']).to_csv('face_track_baseline/reference_hr_ppg.csv', index=False)
PYEOF

log "1 UBFC runs: B0 (consistency check) + FACE_TRACK, adult band"
for R in ubfc_flat_38d8f553 ubfc_aug ubfc_aug_fast ubfc_aug_fast_xl ubfc_swaddle_test ubfc_swaddle_motion ubfc_occdemo; do
  if [ -f "$HRR/$R/facetrack/window_results.csv" ]; then log "  $R done, skipped"; continue; fi
  log "  $R"
  $PY "$CMP" --run runs/$R --module1-config D5 --methods B0,FACE_TRACK --hr-band 45,180 \
      --reference-hr $T/reference_hr_ppg.csv --out-name facetrack || log "  $R FAILED"
done

log "2 final34 (label-free, infant band): B0, FACE_TRACK, D5, BG"
[ -f "$HRR/final34_paper/facetrack/window_results.csv" ] || \
  $PY "$CMP" --run runs/final34_paper --module1-config D5 --methods B0,FACE_TRACK,D5,BG --out-name facetrack || log "  final34 FAILED"

log "3 report"
$PY $T/scripts/face_track_report.py || log "  report FAILED"
log "done"
