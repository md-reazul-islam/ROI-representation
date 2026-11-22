"""Evaluate Module 2 camera RR against a reference RR (impedance pneumography / bedside monitor / manual count).

Reference CSV columns: video_id, timestamp_sec, rr_brpm (any sampling rate, time in the video's clock).
Each analysis window is compared with the mean reference RR inside that window. Metrics are reported
per method x strategy x estimate (window RR, Kalman-smoothed RR), at window and video level, for all
windows and for reliable windows only, together with coverage (fraction of windows kept).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module2.evaluation import agreement


def pair_windows(windows, reference):
    by_video = {video: group for video, group in reference.groupby('video_id')}
    values = []
    for row in windows.itertuples():
        group = by_video.get(row.video_id)
        inside = (group[(group.timestamp_sec >= row.window_start_sec) & (group.timestamp_sec <= row.window_end_sec)]
                  if group is not None else None)
        values.append(inside.rr_brpm.mean() if inside is not None and len(inside) else np.nan)
    return windows.assign(reference_rr_brpm=values)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate camera RR against a reference RR.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--rr-dir', default='outputs/rr')
    parser.add_argument('--reference', required=True, help='CSV with video_id, timestamp_sec, rr_brpm')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--output-dir', help='Default: <rr-dir>/evaluation')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rr_dir = project_path(cfg, args.rr_dir)
    output = project_path(cfg, args.output_dir) if args.output_dir else rr_dir/'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(project_path(cfg, args.reference))
    missing = {'video_id', 'timestamp_sec', 'rr_brpm'} - set(reference.columns)
    if missing:
        raise ValueError(f'Reference CSV is missing columns: {sorted(missing)}')
    windows = pd.read_csv(rr_dir/'rr_windows.csv')
    if args.split != 'all':
        windows = windows[windows.split == args.split]
    pairs = pair_windows(windows, reference)
    pairs = pairs[pairs.reference_rr_brpm.notna()]
    if pairs.empty:
        raise ValueError('No analysis window overlaps the reference RR (check video_id and timestamps).')
    pairs.to_csv(output/'rr_window_pairs.csv', index=False)

    rows = []
    for (method, strategy), group in pairs.groupby(['method', 'strategy']):
        for estimate in ('rr_brpm', 'rr_kalman_brpm'):
            for subset, part in (('all_windows', group), ('reliable_windows', group[group.reliable])):
                coverage = len(part) / len(group) if len(group) else 0.
                rows.append(dict(method=method, strategy=strategy, estimate=estimate, level='window', subset=subset,
                                 coverage=coverage, videos=part.video_id.nunique(),
                                 **agreement(part[estimate], part.reference_rr_brpm)))
                video = part.groupby('video_id')[[estimate, 'reference_rr_brpm']].median()
                rows.append(dict(method=method, strategy=strategy, estimate=estimate, level='video', subset=subset,
                                 coverage=coverage, videos=len(video), **agreement(video[estimate], video.reference_rr_brpm)))
    # agreement() names its columns in bpm; for RR the unit is breaths/min (brpm).
    metrics = pd.DataFrame(rows).rename(columns=lambda c: c.replace('bpm', 'brpm'))
    metrics.to_csv(output/'rr_metrics.csv', index=False)
    (output/'evaluation_metadata.json').write_text(json.dumps(dict(
        reference=str(project_path(cfg, args.reference)), split=args.split, windows_paired=len(pairs),
        pairing='mean reference RR inside each analysis window'), indent=2), encoding='utf-8')
    view = metrics[metrics.level == 'window'].reindex(columns=['method', 'strategy', 'estimate', 'subset', 'coverage', 'n',
                                                               'mae_brpm', 'rmse_brpm', 'pearson_r'])
    print(view.to_string(index=False))
    print(f'Done -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
