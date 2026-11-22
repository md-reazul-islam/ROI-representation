#!/bin/bash
# Whole HR evaluation of the DataAugment videos, one run after the other (one GPU job at a time):
#   1) slow motion / occlusion / lighting (9 videos)  2) fast motion small/medium/large (3)  3) fast motion extra-large (1)
#   4) only with argument "fallback": the same videos with fallback gate 2 (run_hr_eval_fallback.sh; reuses 1-3)
# Every step inside is skipped when its output exists, so re-running resumes. Used by "HR_all Handle.ipynb".
#   bash run_all.sh            strict gate 2 (default)
#   bash run_all.sh fallback   strict + fallback gate 2
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-strict}
log(){ echo "[$(date '+%F %T')] $*"; }
log "== part 1/3: run_hr_eval.sh";         bash ./run_hr_eval.sh
log "== part 2/3: run_hr_eval_fast.sh";    bash ./run_hr_eval_fast.sh
log "== part 3/3: run_hr_eval_fast_xl.sh"; bash ./run_hr_eval_fast_xl.sh
if [ "$MODE" = fallback ]; then
  log "== part 4: run_hr_eval_fallback.sh (fallback gate 2)"; bash ./run_hr_eval_fallback.sh
fi
log "== ALL DONE ($MODE)"
