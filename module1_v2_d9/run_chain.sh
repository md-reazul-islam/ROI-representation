#!/bin/bash
# 1) swaddling + fast motion test, 2) resume D9 test, 3) refresh the swaddle+motion report (adds D9 'none' cells)
set -uo pipefail
cd "$(dirname "$0")"
./swaddle_motion/run_swmo_eval.sh > swaddle_motion/run_swmo_eval.log 2>&1; echo "[$(date '+%F %T')] swmo exit $?"
./run_d9_eval.sh >> run_d9_eval.log 2>&1; echo "[$(date '+%F %T')] d9 exit $?"
${PY:-python} swaddle_motion/swmo_report.py > swaddle_motion/report_refresh.log 2>&1; echo "[$(date '+%F %T')] refresh exit $?"
echo "[$(date '+%F %T')] chain done"
