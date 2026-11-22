"""Render monitor-style HR videos from Module 2 output.

Left: the original video (every native frame) with the heart-rate ROIs; the ROI with the highest
fusion weight is red. Right: a dark monitor panel with the current heart rate in bpm, the scrolling
camera pulse waveform, the signal quality (SNR) and the per-ROI fusion weights. The displayed HR is
causal: at time t it is the SQI-weighted Kalman estimate of the latest analysis window that ended at
or before t, as a monitor would show; the raw window HR, SNR and SQI are listed beneath it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest
from module2.extraction import hold_indices, roi_polygon
from module2.neural import NEURAL_METHODS
from module2.rppg import METHODS, segments


FONT = cv2.FONT_HERSHEY_SIMPLEX
BEST_COLOR = (0, 0, 255)
ROI_COLOR = (80, 220, 80)
PULSE_COLOR = (200, 150, 245)
PANEL = (24, 20, 18)
GOOD, LOW, MUTED = (110, 220, 110), (0, 190, 255), (150, 150, 150)
STRATEGIES = ('dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box')


def text(canvas, value, origin, scale, color, thickness=2, outline=True):
    x, y = int(origin[0]), int(origin[1])
    if outline:
        cv2.putText(canvas, value, (x, y), FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, value, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def current_window(windows, t):
    done = windows[windows.window_end_sec <= t + 1e-9]
    return None if done.empty else done.iloc[-1]


def draw_video(image, rois, weights, scale, u):
    frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else image.copy()
    best = max(weights, key=weights.get) if weights and max(weights.values()) > 0 else None
    for name, polygon in rois.items():
        weight = weights.get(name, 0.)
        if polygon is None or weight <= 0 or name == best:
            continue
        points = np.rint(polygon * scale).astype(np.int32)
        cv2.polylines(frame, [points], True, ROI_COLOR, max(1, int(2*u)), cv2.LINE_AA)
    if best is not None and rois.get(best) is not None:
        points = np.rint(rois[best] * scale).astype(np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], BEST_COLOR)
        frame = cv2.addWeighted(overlay, .3, frame, .7, 0)
        cv2.polylines(frame, [points], True, BEST_COLOR, max(2, int(3*u)), cv2.LINE_AA)
        label = f'HR ROI: {best} w={weights[best]:.2f}'
        (width, height), _ = cv2.getTextSize(label, FONT, .6*u, max(1, int(1.5*u)))
        x, y = points[:, 0].min(), points[:, 1].min() - int(8*u)
        x = int(min(max(x, 3), frame.shape[1] - width - 6))
        y = int(max(y, height + 6))
        cv2.rectangle(frame, (x-4, y-height-6), (x+width+4, y+6), BEST_COLOR, -1)
        text(frame, label, (x, y), .6*u, (255, 255, 255), max(1, int(1.5*u)), outline=False)
    return frame, best


def draw_panel(height, width, u, info, window, history, weights, best, args):
    panel = np.full((height, width, 3), PANEL, dtype=np.uint8)
    margin = int(24*u)
    y = int(40*u)
    cv2.circle(panel, (margin + int(7*u), y - int(7*u)), int(7*u), GOOD, -1, cv2.LINE_AA)
    text(panel, f'Camera-derived signals (rPPG, {args.method})', (margin + int(22*u), y), .65*u, (220, 220, 220), max(1, int(1.5*u)))
    text(panel, f"{info['video_id']}   t = {info['t']:.1f} s", (margin, y + int(30*u)), .5*u, MUTED, max(1, int(u)))

    # Heart-rate number.
    y = int(115*u)
    text(panel, 'Heart rate', (margin, y), .65*u, (220, 220, 220), max(1, int(1.5*u)))
    if window is None:
        value, color, status = '--', MUTED, f'acquiring signal (first {args.window_sec:g} s)...'
    elif pd.isna(window.hr_kalman_bpm):
        value, color, status = '--', LOW, 'no usable skin signal yet'
    else:
        raw = '--' if pd.isna(window.hr_bpm) else f'{window.hr_bpm:.0f}'
        snr = 'NA' if pd.isna(window.snr_db) else f'{window.snr_db:.1f} dB'
        quality = 'GOOD' if window.reliable else 'LOW'
        value = f'{window.hr_kalman_bpm:.0f}'
        color = PULSE_COLOR if window.reliable else LOW
        status = f'signal quality {quality}: SQI {window.sqi:.2f}, SNR {snr}, window HR {raw}'
    number_scale = 2.6*u
    (number_width, number_height), _ = cv2.getTextSize(value, FONT, number_scale, max(2, int(5*u)))
    text(panel, value, (margin, y + int(15*u) + number_height), number_scale, color, max(2, int(5*u)), outline=False)
    text(panel, 'bpm', (margin, y + int(45*u) + number_height), .65*u, (220, 220, 220), max(1, int(1.5*u)))

    # Scrolling pulse waveform.
    x0 = margin + max(number_width, int(150*u)) + int(30*u)
    x1, top, bottom = width - margin, y - int(15*u), y + int(45*u) + number_height
    cv2.rectangle(panel, (x0, top), (x1, bottom), (55, 50, 48), 1)
    times, values = history
    if len(times) > 1:
        span = args.waveform_sec
        xs = x0 + (times - (info['t'] - span)) / span * (x1 - x0)
        mid, amplitude = (top + bottom) / 2, (bottom - top) / 2 * .85
        ys = mid - np.clip(values, -3, 3) / 3 * amplitude
        for start, stop in segments(np.isfinite(ys), 2):  # gaps (no usable ROI) are left blank
            points = np.column_stack([xs[start:stop], ys[start:stop]]).astype(np.int32)
            cv2.polylines(panel, [points], False, PULSE_COLOR, max(1, int(2*u)), cv2.LINE_AA)
    text(panel, f'pulse waveform (last {args.waveform_sec:g} s)', (x0, bottom + int(22*u)), .45*u, MUTED, max(1, int(u)))

    y = bottom + int(65*u)
    text(panel, status, (margin, y), .6*u, GOOD if window is not None and window.reliable else LOW, max(1, int(1.5*u)))
    text(panel, f'{args.window_sec:g} s window, {args.method}, {args.strategy}, SQI-weighted Kalman', (margin, y + int(28*u)),
         .45*u, MUTED, max(1, int(u)))

    # ROI fusion weights.
    y += int(70*u)
    text(panel, 'Dynamic ROI weights (Module 1 confidence x SNR)', (margin, y), .55*u, (220, 220, 220), max(1, int(1.5*u)))
    bar_x, bar_w = margin + int(230*u), width - 2*margin - int(300*u)
    ranked = sorted(info['pool'], key=lambda name: -weights.get(name, 0.))[:6]
    for name in ranked:
        y += int(30*u)
        if y > height - int(60*u):
            break
        weight = weights.get(name, 0.)
        color = BEST_COLOR if name == best else (ROI_COLOR if weight > 0 else MUTED)
        text(panel, name, (margin, y), .5*u, color, max(1, int(1.5*u)))
        cv2.rectangle(panel, (bar_x, y - int(14*u)), (bar_x + bar_w, y), (55, 50, 48), -1)
        cv2.rectangle(panel, (bar_x, y - int(14*u)), (bar_x + int(bar_w*min(max(weight, 0.), 1.)), y), color, -1)
        text(panel, f'{weight:.2f}', (bar_x + bar_w + int(10*u), y), .5*u, color, max(1, int(1.5*u)))
    text(panel, 'RED = highest-weight ROI    research prototype, not for clinical use',
         (margin, height - int(20*u)), .42*u, MUTED, max(1, int(u)))
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render monitor-style heart-rate videos from Module 2 output.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--hr-dir', default='outputs/hr_rppg')
    parser.add_argument('--confidence-dir', help='Module 1 Phase 10 output used by run_hr_rppg.py (default: from its metadata)')
    parser.add_argument('--method', choices=METHODS + NEURAL_METHODS, default='POS')
    parser.add_argument('--strategy', choices=STRATEGIES, default='dynamic_fusion')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--scale', type=float, default=2.)
    parser.add_argument('--waveform-sec', type=float, default=6.)
    parser.add_argument('--output-dir', help='Default: <hr-dir>/monitor_videos')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg['_project_root'])
    hr_dir = project_path(cfg, args.hr_dir)
    metadata = json.loads((hr_dir/'experiment_metadata.json').read_text(encoding='utf-8'))
    pool = list(metadata['extraction_config']['roi_pool'])
    max_hold = float(metadata['extraction_config']['max_hold_sec'])
    args.window_sec = float(metadata['hr_config']['estimation_window_sec'])
    confidence_csv = (project_path(cfg, args.confidence_dir)/'roi_confidence.csv' if args.confidence_dir
                      else Path(metadata['inputs']['confidence']))
    output = project_path(cfg, args.output_dir) if args.output_dir else hr_dir/'monitor_videos'
    output.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest.load(project_path(cfg, args.manifest))
    records = manifest.df
    if args.videos:
        records = records[records.video_id.isin(args.videos)]
    elif args.split != 'all':
        records = records[records.split == args.split]
    windows = pd.read_csv(hr_dir/'hr_windows.csv')
    windows = windows[(windows.method == args.method) & (windows.strategy == args.strategy)]
    signals = pd.read_csv(hr_dir/'pulse_signals.csv', low_memory=False)
    pulse_column = f'pulse_{args.method}_{args.strategy}'
    confidence = pd.read_csv(confidence_csv, usecols=lambda c: c in ('video_id', 'frame_index', 'roi_name', 'roi_status',
                                                                     'polygon', 'bbox'), low_memory=False)
    confidence = confidence[confidence.roi_name.isin(pool)]

    written = 0
    for _, record in records.iterrows():
        video = str(record.video_id)
        frames = signals[signals.video_id == video].sort_values('frame_index')
        if frames.empty:
            print(f'{video}: no Module 2 output, skipped')
            continue
        video_windows = windows[windows.video_id == video].sort_values('window_end_sec')
        rows = confidence[confidence.video_id == video]
        geometry = {(row['roi_name'], int(row['frame_index'])): roi_polygon(row) for row in rows.to_dict('records')}
        fps = float(video_windows.fps.iloc[0]) if not video_windows.empty else float(record.get('fps', 30.))
        hold = hold_indices(rows.frame_index.unique(), len(frames), fps, max_hold)
        times, pulse = frames.timestamp_sec.to_numpy(), frames[pulse_column].to_numpy()
        weight_columns = {name: frames[f'weight_{name}'].to_numpy() for name in pool if f'weight_{name}' in frames}
        capture = cv2.VideoCapture(str(manifest.resolve_video_path(root, record)))
        writer, target = None, output/f'{video}_hr_monitor_{args.method}_{args.strategy}.mp4'
        try:
            for index in range(len(frames)):
                ok, image = capture.read()
                if not ok:
                    break
                u = image.shape[0] * args.scale / 720 * 1.25
                source = int(hold[index])
                rois = {name: geometry.get((name, source)) if source >= 0 else None for name in pool}
                weights = {name: float(column[index]) for name, column in weight_columns.items()}
                left, best = draw_video(image, rois, weights, args.scale, u)
                t = times[index]
                recent = (times > t - args.waveform_sec) & (times <= t)
                info = dict(video_id=video, t=t, pool=pool)
                panel = draw_panel(left.shape[0], max(int(left.shape[0] * .9), int(620*u)), u, info,
                                   current_window(video_windows, t), (times[recent], pulse[recent]), weights, best, args)
                canvas = np.hstack([left, panel])
                if writer is None:
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot open a video writer for {target}')
                writer.write(canvas)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        written += 1
        print(f'{video}: -> {target}', flush=True)
    print(f'Done: {written} video(s) in {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
