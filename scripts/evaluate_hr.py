"""Evaluate Module 2 camera HR against a reference HR (ECG / pulse oximeter / bedside monitor).

Reference CSV columns: video_id, timestamp_sec, hr_bpm (any sampling rate, time in the video's
clock). Each analysis window is compared with the mean reference HR inside that window. Metrics are
reported per method x strategy x estimate (AR, FFT, Kalman-smoothed), at window level and video level, for all windows and for reliable
(SNR-passing) windows only, together with coverage (fraction of windows kept).
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
    by_video = {video: group.sort_values('timestamp_sec') for video, group in reference.groupby('video_id')}
    values = []
    for row in windows.itertuples():
        group = by_video.get(row.video_id)
        if group is None:
            values.append(np.nan)
            continue
        inside = group[(group.timestamp_sec >= row.window_start_sec) & (group.timestamp_sec <= row.window_end_sec)]
        values.append(inside.hr_bpm.mean() if len(inside) else np.nan)
    return windows.assign(reference_hr_bpm=values)


def bland_altman_plot(pairs, target, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mean = (pairs.hr_bpm + pairs.reference_hr_bpm) / 2
    difference = pairs.hr_bpm - pairs.reference_hr_bpm
    bias, sd = difference.mean(), difference.std(ddof=1)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(mean, difference, s=10, alpha=.6, color='#d6589f')
    for value, style in ((bias, '-'), (bias - 1.96*sd, '--'), (bias + 1.96*sd, '--')):
        axis.axhline(value, color='k', ls=style, lw=1)
    axis.set_xlabel('mean of camera and reference HR (bpm)')
    axis.set_ylabel('camera - reference (bpm)')
    axis.set_title(f'{title}\nbias {bias:.1f} bpm, LoA [{bias-1.96*sd:.1f}, {bias+1.96*sd:.1f}]')
    axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate camera HR against a reference HR.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--hr-dir', default='outputs/hr_rppg')
    parser.add_argument('--reference', required=True, help='CSV with video_id, timestamp_sec, hr_bpm')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--output-dir', help='Default: <hr-dir>/evaluation')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    hr_dir = project_path(cfg, args.hr_dir)
    output = project_path(cfg, args.output_dir) if args.output_dir else hr_dir/'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(project_path(cfg, args.reference))
    missing = {'video_id', 'timestamp_sec', 'hr_bpm'} - set(reference.columns)
    if missing:
        raise ValueError(f'Reference CSV is missing columns: {sorted(missing)}')
    windows = pd.read_csv(hr_dir/'hr_windows.csv')
    if args.split != 'all':
        windows = windows[windows.split == args.split]
    pairs = pair_windows(windows, reference)
    pairs = pairs[pairs.reference_hr_bpm.notna()]
    if pairs.empty:
        raise ValueError('No analysis window overlaps the reference HR (check video_id and timestamps).')
    pairs.to_csv(output/'hr_window_pairs.csv', index=False)

    rows = []
    estimates = [column for column in ('hr_bpm', 'hr_fft_bpm', 'hr_ar_bpm', 'hr_kalman_bpm') if column in pairs.columns]
    for (method, strategy), group in pairs.groupby(['method', 'strategy']):
        for estimate in estimates:
            for subset, part in (('all_windows', group), ('reliable_windows', group[group.reliable])):
                coverage = len(part) / len(group) if len(group) else 0.
                rows.append(dict(method=method, strategy=strategy, estimate=estimate, level='window', subset=subset,
                                 coverage=coverage, videos=part.video_id.nunique(),
                                 **agreement(part[estimate], part.reference_hr_bpm)))
                video = part.groupby('video_id')[[estimate, 'reference_hr_bpm']].median()
                rows.append(dict(method=method, strategy=strategy, estimate=estimate, level='video', subset=subset,
                                 coverage=coverage, videos=len(video), **agreement(video[estimate], video.reference_hr_bpm)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output/'hr_metrics.csv', index=False)
    primary = pairs[(pairs.method == 'POS') & (pairs.strategy == 'dynamic_fusion') & pairs.hr_bpm.notna()]
    if len(primary) >= 3:
        bland_altman_plot(primary, output/'bland_altman_POS_dynamic_fusion.png', 'POS + confidence-weighted dynamic ROI fusion')
    (output/'evaluation_metadata.json').write_text(json.dumps(dict(
        reference=str(project_path(cfg, args.reference)), split=args.split, windows_paired=len(pairs),
        pairing='mean reference HR inside each analysis window'), indent=2), encoding='utf-8')
    view = metrics[(metrics.level == 'window')].reindex(columns=['method', 'strategy', 'estimate', 'subset', 'coverage', 'n', 'mae_bpm',
                                                                 'rmse_bpm', 'pearson_r', 'bland_altman_bias_bpm'])
    print(view.to_string(index=False))
    print(f'Done -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
