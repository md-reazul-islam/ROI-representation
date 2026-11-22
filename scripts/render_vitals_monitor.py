"""Render joint HR + RR monitor videos from the Module 2 HR and RR outputs.

Left: the original video (every native frame) with the dynamic ROIs: the highest-weight HR ROI in red,
the highest-weight RR ROI in cyan, other weighted ROIs thin. Right: a dark monitor panel with the
current heart rate (bpm) and respiration rate (breaths/min), their scrolling camera waveforms (pulse
and breathing), signal quality and the top ROI weights of both. Both numbers are causal: at time t
they are the SQI-weighted Kalman estimates of the latest window that ended at or before t. A number
is coloured when its window is reliable, amber when the signal quality is low, '--' with no estimate.
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
from module2.respiration import RR_METHODS
from module2.neural import NEURAL_METHODS
from module2.rppg import METHODS, segments


FONT = cv2.FONT_HERSHEY_SIMPLEX
HR_COLOR = (0, 0, 255)            # red: HR ROI
RR_COLOR = (255, 220, 0)          # cyan: RR ROI
OTHER_COLOR = (80, 220, 80)
PULSE_COLOR = (200, 150, 245)
BREATH_COLOR = (255, 220, 90)
PANEL = (24, 20, 18)
GOOD, LOW, MUTED, WHITE = (110, 220, 110), (0, 190, 255), (150, 150, 150), (225, 225, 225)
STRATEGIES = ('dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box')
# HR-only multi-site strategies (run_hr_rppg.py): they reuse the dynamic-fusion ROI weights and pulse
# waveform, and RR has no counterpart, so the waveforms and RR come from dynamic_fusion.
MULTISITE_STRATEGIES = ('consensus_fusion', 'cardiorespiratory_fusion')


def text(canvas, value, origin, scale, color, thickness=2, outline=False):
    x, y = int(origin[0]), int(origin[1])
    if outline:
        cv2.putText(canvas, value, (x, y), FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, value, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def current_window(windows, t):
    done = windows[windows.window_end_sec <= t + 1e-9]
    return None if done.empty else done.iloc[-1]


def best_of(weights):
    return max(weights, key=weights.get) if weights and max(weights.values()) > 0 else None


def label_box(frame, points, label, color, u, below=False):
    (width, height), _ = cv2.getTextSize(label, FONT, .55*u, max(1, int(1.5*u)))
    x = int(min(max(points[:, 0].min(), 3), frame.shape[1] - width - 6))
    y = int(points[:, 1].max() + height + int(10*u)) if below else int(points[:, 1].min() - int(8*u))
    y = int(min(max(y, height + 6), frame.shape[0] - 8))
    cv2.rectangle(frame, (x-4, y-height-6), (x+width+4, y+6), color, -1)
    text(frame, label, (x, y), .55*u, (0, 0, 0) if color == RR_COLOR else (255, 255, 255), max(1, int(1.5*u)), outline=False)


def draw_video(image, rois, hr_weights, rr_weights, scale, u):
    frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else image.copy()
    hr_best, rr_best = best_of(hr_weights), best_of(rr_weights)
    for name, polygon in rois.items():
        if polygon is None or name in (hr_best, rr_best):
            continue
        if hr_weights.get(name, 0.) > 0 or rr_weights.get(name, 0.) > 0:
            cv2.polylines(frame, [np.rint(polygon * scale).astype(np.int32)], True, OTHER_COLOR, max(1, int(1.5*u)), cv2.LINE_AA)
    for best, weights, color, kind, below in ((rr_best, rr_weights, RR_COLOR, 'RR', True),
                                              (hr_best, hr_weights, HR_COLOR, 'HR', False)):
        if best is None or rois.get(best) is None:
            continue
        points = np.rint(rois[best] * scale).astype(np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        frame = cv2.addWeighted(overlay, .25, frame, .75, 0)
        cv2.polylines(frame, [points], True, color, max(2, int(3*u)), cv2.LINE_AA)
        label_box(frame, points, f'{kind} ROI: {best} w={weights[best]:.2f}', color, u, below=below)
    return frame, hr_best, rr_best


def waveform(panel, box, times, values, t, span, color, u):
    x0, top, x1, bottom = box
    cv2.rectangle(panel, (x0, top), (x1, bottom), (55, 50, 48), 1)
    if len(times) < 2:
        return
    xs = x0 + (times - (t - span)) / span * (x1 - x0)
    mid, amplitude = (top + bottom) / 2, (bottom - top) / 2 * .85
    ys = mid - np.clip(values, -3, 3) / 3 * amplitude
    for start, stop in segments(np.isfinite(ys), 2):
        points = np.column_stack([xs[start:stop], ys[start:stop]]).astype(np.int32)
        cv2.polylines(panel, [points], False, color, max(1, int(2*u)), cv2.LINE_AA)


def vital_block(panel, y, u, margin, width, title, unit, window, value_column, rate_column, color, warmup_sec,
                history, t, span, wave_label, wave_color):
    """One vital sign: label, big number, unit, waveform and the quality line. Returns the next y."""
    text(panel, title, (margin, y), .62*u, WHITE, max(1, int(1.5*u)))
    if window is None:
        value, value_color, status, status_color = '--', MUTED, f'acquiring signal (first {warmup_sec:g} s)...', MUTED
    elif pd.isna(window[value_column]):
        value, value_color, status, status_color = '--', LOW, 'no usable signal', LOW
    else:
        raw = '--' if pd.isna(window[rate_column]) else f'{window[rate_column]:.0f}'
        snr = 'NA' if pd.isna(window.snr_db) else f'{window.snr_db:.1f} dB'
        quality = 'GOOD' if window.reliable else 'LOW'
        value = f'{window[value_column]:.0f}'
        value_color = color if window.reliable else LOW
        status = f'quality {quality}: SQI {window.sqi:.2f}, SNR {snr}, window {raw}'
        status_color = GOOD if window.reliable else LOW
    number_scale = 1.9*u
    thickness = max(2, int(4*u))
    (number_width, number_height), _ = cv2.getTextSize(value, FONT, number_scale, thickness)
    text(panel, value, (margin, y + int(12*u) + number_height), number_scale, value_color, thickness, outline=False)
    text(panel, unit, (margin, y + int(34*u) + number_height), .55*u, WHITE, max(1, int(1.3*u)))
    x0 = margin + max(number_width, int(int(150*u))) + int(25*u)
    box = (x0, y - int(16*u), width - margin, y + int(36*u) + number_height)
    waveform(panel, box, *history, t, span, wave_color, u)
    text(panel, wave_label, (x0, box[3] + int(17*u)), .42*u, MUTED, max(1, int(u)))
    y = box[3] + int(40*u)
    text(panel, status, (margin, y), .5*u, status_color, max(1, int(1.3*u)))
    return y


def weight_column(panel, x, y, u, column_width, title, weights, best, color):
    text(panel, title, (x, y), .47*u, WHITE, max(1, int(1.3*u)))
    ranked = [name for name in sorted(weights, key=lambda name: -weights[name]) if weights[name] > 0][:3]
    name_width = int(column_width * .58)
    bar_width = int(column_width * .22)
    for name in ranked:
        y += int(24*u)
        weight = weights[name]
        shade = color if name == best else OTHER_COLOR
        text(panel, name, (x, y), .42*u, shade, max(1, int(1.2*u)))
        cv2.rectangle(panel, (x + name_width, y - int(11*u)), (x + name_width + bar_width, y), (55, 50, 48), -1)
        cv2.rectangle(panel, (x + name_width, y - int(11*u)), (x + name_width + int(bar_width*min(weight, 1.)), y), shade, -1)
        text(panel, f'{weight:.2f}', (x + name_width + bar_width + int(6*u), y), .42*u, shade, max(1, int(1.2*u)))


def draw_panel(height, width, u, info, hr_window, rr_window, hr_history, rr_history, hr_weights, rr_weights,
               hr_best, rr_best, args):
    panel = np.full((height, width, 3), PANEL, dtype=np.uint8)
    margin = int(22*u)
    y = int(36*u)
    cv2.circle(panel, (margin + int(7*u), y - int(7*u)), int(7*u), GOOD, -1, cv2.LINE_AA)
    text(panel, 'Camera-derived vital signs (HR + RR)', (margin + int(22*u), y), .62*u, WHITE, max(1, int(1.5*u)))
    text(panel, f"{info['video_id']}   t = {info['t']:.1f} s", (margin, y + int(26*u)), .47*u, MUTED, max(1, int(u)))
    y = int(92*u)
    y = vital_block(panel, y, u, margin, width, 'Heart rate', 'bpm', hr_window, 'hr_kalman_bpm', 'hr_bpm', PULSE_COLOR,
                    args.hr_window_sec, hr_history, info['t'], args.hr_waveform_sec,
                    f'pulse waveform ({args.hr_method}, last {args.hr_waveform_sec:g} s)', PULSE_COLOR)
    y += int(40*u)
    y = vital_block(panel, y, u, margin, width, 'Respiration rate', 'breaths/min', rr_window, 'rr_kalman_brpm', 'rr_brpm',
                    BREATH_COLOR, args.rr_min_window_sec, rr_history, info['t'], args.rr_waveform_sec,
                    f'breathing waveform ({args.rr_method}, last {args.rr_waveform_sec:g} s)', BREATH_COLOR)
    y += int(32*u)
    column_width = (width - 3*margin) // 2
    if y + int(72*u) < height - int(28*u):
        weight_column(panel, margin, y, u, column_width, 'HR ROI weights (conf x SNR)', hr_weights, hr_best, HR_COLOR)
        weight_column(panel, 2*margin + column_width, y, u, column_width, 'RR ROI weights (conf x SNR)', rr_weights,
                      rr_best, RR_COLOR)
    text(panel, 'RED = HR ROI   CYAN = RR ROI   research prototype, not for clinical use',
         (margin, height - int(16*u)), .4*u, MUTED, max(1, int(u)))
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render joint HR + RR monitor videos from Module 2 output.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--hr-dir', default='outputs/hr_rppg')
    parser.add_argument('--rr-dir', default='outputs/rr')
    parser.add_argument('--confidence-dir', help='Module 1 Phase 10 output (default: from the HR metadata)')
    parser.add_argument('--hr-method', choices=METHODS + NEURAL_METHODS, default='POS')
    parser.add_argument('--rr-method', choices=RR_METHODS, default='FLOW')
    parser.add_argument('--strategy', choices=STRATEGIES + MULTISITE_STRATEGIES, default='dynamic_fusion')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--scale', type=float, default=2.)
    parser.add_argument('--hr-waveform-sec', type=float, default=6.)
    parser.add_argument('--rr-waveform-sec', type=float, default=15.)
    parser.add_argument('--output-dir', help='Default: <rr-dir>/vitals_monitor_videos')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg['_project_root'])
    hr_dir, rr_dir = project_path(cfg, args.hr_dir), project_path(cfg, args.rr_dir)
    hr_meta = json.loads((hr_dir/'experiment_metadata.json').read_text(encoding='utf-8'))
    rr_meta = json.loads((rr_dir/'experiment_metadata.json').read_text(encoding='utf-8'))
    hr_pool, rr_pool = list(hr_meta['extraction_config']['roi_pool']), list(rr_meta['extraction_config']['roi_pool'])
    pool = list(dict.fromkeys(hr_pool + rr_pool))
    max_hold = float(hr_meta['extraction_config']['max_hold_sec'])
    args.hr_window_sec = float(hr_meta['hr_config']['estimation_window_sec'])
    args.rr_min_window_sec = float(rr_meta['rr_config']['min_window_sec'])
    confidence_csv = (project_path(cfg, args.confidence_dir)/'roi_confidence.csv' if args.confidence_dir
                      else Path(hr_meta['inputs']['confidence']))
    output = project_path(cfg, args.output_dir) if args.output_dir else rr_dir/'vitals_monitor_videos'
    output.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest.load(project_path(cfg, args.manifest))
    records = manifest.df
    if args.videos:
        records = records[records.video_id.isin(args.videos)]
    elif args.split != 'all':
        records = records[records.split == args.split]
    hr_windows = pd.read_csv(hr_dir/'hr_windows.csv')
    hr_windows = hr_windows[(hr_windows.method == args.hr_method) & (hr_windows.strategy == args.strategy)]
    signal_strategy = 'dynamic_fusion' if args.strategy in MULTISITE_STRATEGIES else args.strategy
    rr_windows = pd.read_csv(rr_dir/'rr_windows.csv')
    rr_windows = rr_windows[(rr_windows.method == args.rr_method) & (rr_windows.strategy == signal_strategy)]
    pulse_signals = pd.read_csv(hr_dir/'pulse_signals.csv', low_memory=False)
    resp_signals = pd.read_csv(rr_dir/'resp_signals.csv', low_memory=False)
    confidence = pd.read_csv(confidence_csv, usecols=lambda c: c in ('video_id', 'frame_index', 'roi_name', 'roi_status',
                                                                     'polygon', 'bbox'), low_memory=False)
    confidence = confidence[confidence.roi_name.isin(pool)]

    written = 0
    for _, record in records.iterrows():
        video = str(record.video_id)
        pulse_frames = pulse_signals[pulse_signals.video_id == video].sort_values('frame_index')
        resp_frames = resp_signals[resp_signals.video_id == video].sort_values('frame_index')
        if pulse_frames.empty and resp_frames.empty:
            print(f'{video}: no Module 2 output, skipped')
            continue
        frames_n = max(len(pulse_frames), len(resp_frames))
        video_hr = hr_windows[hr_windows.video_id == video].sort_values('window_end_sec')
        video_rr = rr_windows[rr_windows.video_id == video].sort_values('window_end_sec')
        rows = confidence[confidence.video_id == video]
        geometry = {(row['roi_name'], int(row['frame_index'])): roi_polygon(row) for row in rows.to_dict('records')}
        fps = float(manifest_fps(record, video_hr, video_rr))
        hold = hold_indices(rows.frame_index.unique(), frames_n, fps, max_hold)
        times = np.arange(frames_n) / fps
        pulse = column(pulse_frames, f'pulse_{args.hr_method}_{signal_strategy}', frames_n)
        breath = column(resp_frames, f'resp_{args.rr_method}_{signal_strategy}', frames_n)
        hr_columns = {name: column(pulse_frames, f'weight_{name}', frames_n) for name in hr_pool}
        rr_columns = {name: column(resp_frames, f'weight_{name}', frames_n) for name in rr_pool}
        capture = cv2.VideoCapture(str(manifest.resolve_video_path(root, record)))
        writer, target = None, output/f'{video}_vitals_monitor.mp4'
        try:
            for index in range(frames_n):
                ok, image = capture.read()
                if not ok:
                    break
                u = image.shape[0] * args.scale / 720 * 1.25
                source = int(hold[index])
                rois = {name: geometry.get((name, source)) if source >= 0 else None for name in pool}
                hr_weights = {name: float(np.nan_to_num(values[index])) for name, values in hr_columns.items()}
                rr_weights = {name: float(np.nan_to_num(values[index])) for name, values in rr_columns.items()}
                left, hr_best, rr_best = draw_video(image, rois, hr_weights, rr_weights, args.scale, u)
                t = times[index]
                hr_recent = (times > t - args.hr_waveform_sec) & (times <= t)
                rr_recent = (times > t - args.rr_waveform_sec) & (times <= t)
                panel = draw_panel(left.shape[0], max(int(left.shape[0] * .95), int(640*u)), u, dict(video_id=video, t=t),
                                   current_window(video_hr, t), current_window(video_rr, t),
                                   (times[hr_recent], pulse[hr_recent]), (times[rr_recent], breath[rr_recent]),
                                   hr_weights, rr_weights, hr_best, rr_best, args)
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


def column(frames, name, n):
    values = np.full(n, np.nan)
    if name in frames:
        data = frames[name].to_numpy(dtype=float)[:n]
        values[:len(data)] = data
    return values


def manifest_fps(record, *window_tables):
    for table in window_tables:
        if not table.empty:
            return table.fps.iloc[0]
    return record.get('fps', 30.)


if __name__ == '__main__':
    raise SystemExit(main())
