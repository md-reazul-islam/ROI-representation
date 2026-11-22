"""Phase 1 end to end for a finished v1 run: v1 evaluation (only if no report is given) -> v2 features ->
training-only tuning -> Phase 1 evaluation. Every step is a separate script and can also be run alone.

Usage (from the Module1 folder):
    python module1_v2/scripts/run_phase1.py --run runs/dataset5_paper
    python module1_v2/scripts/run_phase1.py --run runs/pilot24_paper --v1-report reports/paper
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent


def step(*args):
    print('>>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 v2 Phase 1 pipeline.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--v1-report', help='existing v1 evaluation of this run (default: computed into module1_v2/reports)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    name = Path(args.run).name
    report = args.v1_report or f'module1_v2/reports/{name}/v1_eval'
    if not (ROOT/report/'per_video_metrics.csv').is_file():
        step('scripts/run_paper_evaluation.py', '--run', args.run, '--output-dir', report,
             '--gpu-log', f'{args.run}/gpu_monitor.csv', '--workers', args.workers)
    step('module1_v2/scripts/run_v2_features.py', '--run', args.run, '--workers', args.workers)
    step('module1_v2/scripts/tune_phase1.py', '--run', args.run)
    step('module1_v2/scripts/run_v2_evaluation.py', '--run', args.run, '--v1-report', report, '--workers', args.workers)
    print('Phase 1 done ->', V2/'reports'/name/'phase1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
