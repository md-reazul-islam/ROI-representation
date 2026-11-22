#!/bin/bash
# Module 2 (HR + RR, unchanged code) on the same test videos with three Module 1 inputs: v1, v2 D3, v2 D5.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
RUN=runs/final34_paper; OUT=module1_v2/runs/final34_paper/module2_check
VIDEOS="VID_002 VID_005 VID_008 VID_034"
mkdir -p "$OUT"
declare -A SRC=([v1]=$RUN/outputs/confidence_sam2 [v2_D3]=module1_v2/runs/final34_paper/module2_input_D3 [v2_D5]=module1_v2/runs/final34_paper/module2_input_D5)
for name in v1 v2_D3 v2_D5; do
  for branch in rr hr; do
    script=scripts/run_rr.py; [ $branch = hr ] && script=scripts/run_hr_rppg.py
    dest=$OUT/${branch}_$name; rm -rf "$dest"
    extra=""; [ $branch = hr ] && [ -f $OUT/rr_$name/rr_windows.csv ] && extra="--rr-windows $OUT/rr_$name/rr_windows.csv"
    echo "== $branch $name $(date)"
    $PY $script --manifest $PWD/$RUN/manifest.csv --confidence-dir $PWD/${SRC[$name]} \
        --recovery $PWD/$RUN/outputs/recovery/recovery_results.csv --videos $VIDEOS --output-dir $PWD/$dest --no-plots $extra \
        > $dest.log 2>&1; echo "   exit $?"; tail -2 $dest.log
  done
done
echo "MODULE2 CHECK DONE $(date)"
