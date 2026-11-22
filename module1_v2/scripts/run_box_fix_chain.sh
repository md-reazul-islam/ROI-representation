#!/bin/bash
# Re-run every pixel-dependent v2 evaluation after the 2-point-box fix (selection, tuning and features unchanged).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
RUN=runs/final34_paper; V1=reports/paper_final34_paper; LOG=module1_v2/runs/final34_boxfix
step() { echo "== $1 $(date)"; shift; "$PY" "$@" > "$LOG.$STEP.log" 2>&1; echo "   exit $? $(date)"; tail -3 "$LOG.$STEP.log"; }
STEP=phase1;      step "Phase 1 evaluation" module1_v2/scripts/run_v2_evaluation.py --run $RUN --v1-report $V1 --workers 2
STEP=phase2;      step "Phase 2"            module1_v2/scripts/run_phase2.py --run $RUN --v1-report $V1 --workers 2
STEP=fusion;      step "Fusion test"        module1_v2/scripts/run_fusion_test.py --run $RUN --v1-report $V1 --workers 2
STEP=phase5;      step "Phase 5"            module1_v2/scripts/run_phase5.py --run $RUN --v1-report $V1 --workers 2
STEP=consistency; step "Consistency checks" module1_v2/scripts/run_consistency_checks.py --run $RUN --workers 2
echo "CHAIN DONE $(date)"
