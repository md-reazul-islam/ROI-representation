"""Module 2 (RR branch): camera respiration rate from the Module 1 dynamic ROIs.

Per video: per native frame, camera-compensated optical flow and grey level inside every respiratory
ROI (Module 1 Phase 8 geometry, held between processed frames) -> respiratory signal per ROI (FLOW:
principal-axis breathing displacement; INTENSITY: grey level) -> per window (growing 8 -> 15 s, 1 s
step) spectral fusion across ROIs -> respiration rate, SNR, gross-motion-aware SQI -> SQI-weighted
Kalman smoothing (the same filter as HR).

Fusion strategies (as for HR, for the ablation); ROIs are fused at the spectrum level:
  dynamic_fusion     proposed: w_r ∝ softmax(Q_r / T) x 10^(SNR_r / 10);
  confidence_fusion  Module 1 prior only;   snr_fusion  respiratory SNR only;
  top1               the ROI with the highest Module 1 confidence;   pool_mean  equal weights;
  infant_box         the whole Phase 6 infant box (no anatomical ROI).
Every window also reports the thoraco-abdominal phase (module2.cardiorespiratory): chest and abdominal FLOW
displacements on a shared axis, cross-spectral phase at the FLOW / dynamic_fusion RR (0 deg synchronous,
180 deg paradoxical) and its coherence.
No reference RR is used here; see evaluate_rr.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.video import VideoReader
from module1.roi import ROI_NAMES
from module2.cardiorespiratory import (ABDOMEN_ROIS, CHEST_ROIS, CardioRespConfig, shared_axis_displacements,
                                       thoraco_abdominal_phase)
from module2.extraction import (ExtractionConfig, INFANT_BOX, box_polygon, fusion_weights, hold_indices, number,
                                parse_json, roi_polygon)
from module2.respiration import (RESPIRATORY_POOL, RR_METHODS, RRConfig, align_signs, background_flow, estimate_rr,
                                 flow_frame, fuse_spectra, polygon_mask, resize_gray, respiratory_signal, rr_windows,
                                 signal_quality, window_spectrum)
from module2.rppg import kalman_smooth


STRATEGIES = ('dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box')
PRIMARY = ('FLOW', 'dynamic_fusion')
MAX_GOODNESS_DB = 40.


def build_config(section, cls, overrides=None):
    from dataclasses import fields
    values = {field.name: section[field.name] for field in fields(cls) if field.name in section}
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def normalize(weights):
    total = weights.sum()
    return weights / total if total > 0 else weights * 0.


def window_weights(prior, goodness, equal, available):
    """Per-strategy ROI weights (k,) for one window (same definitions as the HR branch)."""
    top = np.zeros_like(prior)
    if prior.sum() > 0:
        top[int(prior.argmax())] = 1.
    return dict(dynamic_fusion=normalize(prior * goodness), confidence_fusion=normalize(prior),
                snr_fusion=normalize(goodness * available), top1=top, pool_mean=normalize(equal))


def weighted_signal(signals, weights):
    """Sum_r w_r s_r / Sum_r w_r over finite samples; NaN where nothing is usable."""
    weights = np.broadcast_to(weights, signals.shape)
    finite = np.isfinite(signals)
    w = np.where(finite, weights, 0.)
    total = w.sum(axis=1)
    fused = (np.where(finite, signals, 0.) * w).sum(axis=1)
    return np.divide(fused, total, out=np.full(len(total), np.nan), where=total > 0)


def read_video(path, names, by_roi, boxes, hold, rr):
    """Per frame: ROI velocity (n, m, 2), grey level (n, m), Q (n, m-1) and gross motion (n, box heights/frame)."""
    velocity, intensity, q, gross = [], [], [], []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'Cannot decode video: {path}')
    previous = None
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            gray, scale = resize_gray(image, rr.flow_width)
            flow = flow_frame(previous, gray) if previous is not None else np.zeros(gray.shape + (2,), np.float32)
            previous = gray
            source = int(hold[index]) if index < len(hold) else -1
            box = boxes.get(source) if source >= 0 else None
            box_mask, box_height = None, None
            if box:
                x1, y1, x2, y2 = map(float, box)
                pad_x, pad_y = .1 * (x2 - x1), .1 * (y2 - y1)
                box_mask = polygon_mask(gray.shape, box_polygon([x1-pad_x, y1-pad_y, x2+pad_x, y2+pad_y]), scale)
                box_height = (y2 - y1) * scale
            flow = flow - background_flow(flow, box_mask, rr)
            if box_mask is not None and box_mask.any() and box_height and index > 0:
                gross.append(float(np.linalg.norm(flow[box_mask], axis=1).mean()) / box_height)
            else:
                gross.append(np.nan)
            frame_velocity, frame_intensity, frame_q = [], [], []
            for name in names:
                if name == INFANT_BOX:
                    polygon, row = (box_polygon(box) if box else None), {}
                else:
                    row = by_roi[name].get(source, {}) if source >= 0 else {}
                    polygon = roi_polygon(row) if row else None
                    frame_q.append(number(row.get('q')) if row else None)
                mask = polygon_mask(gray.shape, polygon, scale) if polygon is not None else None
                if mask is None or mask.sum() < 4:
                    frame_velocity.append((np.nan, np.nan))
                    frame_intensity.append(np.nan)
                    continue
                frame_velocity.append(tuple(flow[mask].mean(axis=0)) if index > 0 else (0., 0.))
                frame_intensity.append(float(gray[mask].mean()))
            velocity.append(frame_velocity)
            intensity.append(frame_intensity)
            q.append([np.nan if v is None else v for v in frame_q])
            index += 1
    finally:
        capture.release()
    return (np.asarray(velocity, float), np.asarray(intensity, float), np.asarray(q, float),
            np.asarray(gross, float))


def window_taa(velocity, pool, usable_flow, goodness, rate, fps, rr, cr):
    """Thoraco-abdominal phase for one window: the best breathing chest ROI against the abdomen."""
    def best(names):
        candidates = [pool.index(name) for name in names if name in pool and usable_flow[pool.index(name)]]
        return max(candidates, key=lambda r: goodness[r]) if candidates else None
    chest, abdomen = best(CHEST_ROIS), best(ABDOMEN_ROIS)
    empty = dict(taa_phase_deg=None, taa_coherence=None, taa_chest_roi=None, taa_abdomen_roi=None)
    if chest is None or abdomen is None or rate is None:
        return empty
    pair = shared_axis_displacements(velocity[:, chest], velocity[:, abdomen], fps, rr, cr)
    if pair is None:
        return empty
    phase, coherence = thoraco_abdominal_phase(*pair, fps, rate, cr)
    return dict(taa_phase_deg=phase, taa_coherence=coherence, taa_chest_roi=pool[chest], taa_abdomen_roi=pool[abdomen])


def process_video(video, split, path, rows, recovery_rows, rr, ex, cr=None):
    cr = cr or CardioRespConfig()
    meta = VideoReader(path).metadata()
    if not meta.opened_successfully or meta.native_fps <= 0:
        raise ValueError(f'Cannot decode video: {path}')
    fps = float(meta.native_fps)
    pool = list(ex.roi_pool)
    k = len(pool)
    by_roi = {name: {} for name in pool}
    for row in rows:
        if row['roi_name'] in by_roi:
            by_roi[row['roi_name']][int(row['frame_index'])] = row
    processed = sorted({int(row['frame_index']) for row in rows})
    frame_count = max(meta.frame_count, processed[-1] + 1 if processed else 0)
    hold = hold_indices(processed, frame_count, fps, ex.max_hold_sec)
    boxes = {int(row['frame_index']): parse_json(row.get('final_infant_bbox')) for row in recovery_rows}
    names = pool + [INFANT_BOX]
    velocity, intensity, q, gross = read_video(path, names, by_roi, boxes, hold, rr)
    n = len(velocity)
    if n == 0:
        raise ValueError(f'{video}: no frames decoded')
    gross = gross * fps  # box heights per second

    traces = dict(FLOW=lambda r: velocity[:, r], INTENSITY=lambda r: intensity[:, r])
    signals, raw = {}, {}
    for method in RR_METHODS:
        pairs = [respiratory_signal(traces[method](r), fps, method, rr, return_raw=True) for r in range(len(names))]
        signals[method] = np.stack([z for z, _ in pairs], axis=1)
        raw[method] = np.stack([r for _, r in pairs], axis=1)
    usable = np.isfinite(signals['FLOW'][:, :k]) & np.isfinite(q)
    prior = fusion_weights(q, usable, ex, fps)
    equal = fusion_weights(np.zeros((n, k)), usable, ExtractionConfig(weight_temperature=1., weight_smoothing_sec=0.), fps)
    moving = np.where(np.isfinite(gross), gross > rr.gross_motion_box_per_sec, np.nan)

    window_rows = []
    display = {(m, s): np.zeros((n, k)) for m in RR_METHODS for s in STRATEGIES if s != 'infant_box'}
    coverage = np.zeros(n)
    for start, stop in rr_windows(n, fps, rr):
        coverage[start:stop] += 1
        duration = (stop - start) / fps
        prior_w, equal_w = prior[start:stop].mean(axis=0), equal[start:stop].mean(axis=0)
        available = (usable[start:stop].mean(axis=0) >= rr.min_valid_fraction).astype(float)
        motion = float(np.nanmean(moving[start:stop])) if np.isfinite(moving[start:stop]).any() else None
        base = dict(video_id=video, split=split, window_start_sec=start/fps, window_end_sec=stop/fps,
                    window_center_sec=(start+stop)/2/fps, window_sec=duration, fps=fps)
        amplitude = np.array([np.nanstd(raw['FLOW'][start:stop, r]) if np.isfinite(raw['FLOW'][start:stop, r]).any()
                              else 0. for r in range(len(names))])
        first_row = len(window_rows)
        flow_state = None
        for method in RR_METHODS:
            segment = signals[method][start:stop]
            # FLOW: an ROI that barely moves carries no breathing, whatever its normalized spectrum looks like.
            moves = amplitude >= rr.min_flow_amplitude_px if method == 'FLOW' else np.ones(len(names), bool)
            spectra = [window_spectrum(segment[:, r], fps, rr) if (r == k or available[r]) and moves[r] else None
                       for r in range(len(names))]
            available_m = available * moves[:k]
            snrs = [estimate_rr(*item, duration, rr)[1] if item is not None else None for item in spectra]
            goodness = np.array([10 ** (min(s, MAX_GOODNESS_DB) / 10) if s is not None and not math.isnan(s) else 0.
                                 for s in snrs[:k]])
            strategies = window_weights(prior_w * available_m, goodness, equal_w * available_m, available_m)
            for strategy in STRATEGIES:
                if strategy == 'infant_box':
                    weights, fused, top = None, spectra[k], INFANT_BOX
                else:
                    weights = strategies[strategy]
                    display[(method, strategy)][start:stop] += weights
                    fused = fuse_spectra(spectra[:k], weights)
                    top = pool[int(weights.argmax())] if weights.sum() > 0 else None
                rate, snr = estimate_rr(*fused, duration, rr) if fused is not None else (None, None)
                top_index = k if strategy == 'infant_box' else (pool.index(top) if top else None)
                sqi = signal_quality(snr, motion, rr) if rate is not None else 0.
                window_rows.append(dict(
                    base, method=method, strategy=strategy, top_roi=top,
                    roi_weights=json.dumps({} if weights is None else
                                           {pool[r]: round(float(w), 4) for r, w in enumerate(weights) if w > 0}),
                    rr_brpm=rate, snr_db=snr, gross_motion_fraction=motion, sqi=sqi,
                    top_roi_flow_amplitude_px=float(amplitude[top_index]) if top_index is not None else None,
                    reliable=bool(rate is not None and sqi >= rr.sqi_threshold)))
                if (method, strategy) == PRIMARY:
                    flow_state = (available_m > 0, goodness, rate)
        if flow_state is not None:
            taa = window_taa(velocity[start:stop], pool, *flow_state, fps, rr, cr)
            for row in window_rows[first_row:]:
                row.update(taa)

    windows = pd.DataFrame(window_rows)
    if not windows.empty:
        windows['rr_kalman_brpm'] = np.nan
        for _, group in windows.groupby(['method', 'strategy'], sort=False):
            smoothed = kalman_smooth(group.window_end_sec.to_numpy(), group.rr_brpm.to_numpy(dtype=float),
                                     group.sqi.to_numpy(), rr)
            windows.loc[group.index, 'rr_kalman_brpm'] = np.array([np.nan if v is None else v for v in smoothed], dtype=float)

    # Per-frame display weights (mean over covering windows) and the sign-aligned breathing waveform.
    frame_table = dict(video_id=video, frame_index=np.arange(n), timestamp_sec=np.arange(n) / fps,
                       gross_motion_box_per_sec=gross)
    covered = coverage[:, None] > 0
    primary_weights = np.where(covered, display[PRIMARY] / np.maximum(coverage, 1)[:, None], prior)
    for r, name in enumerate(pool):
        frame_table[f'weight_{name}'] = primary_weights[:, r]
    top_frame = np.where(primary_weights.sum(axis=1) > 0, primary_weights.argmax(axis=1), -1)
    frame_table['top_roi'] = [pool[i] if i >= 0 else '' for i in top_frame]
    for method in RR_METHODS:
        for strategy in STRATEGIES:
            if strategy == 'infant_box':
                waveform = signals[method][:, k]
            else:
                weights = np.where(covered, display[(method, strategy)] / np.maximum(coverage, 1)[:, None], prior)
                reference = int(weights.sum(axis=0).argmax()) if weights.sum() > 0 else None
                waveform = weighted_signal(signals[method][:, :k] * align_signs(signals[method][:, :k], reference), weights)
            frame_table[f'resp_{method}_{strategy}'] = waveform
    info = dict(video_id=video, split=split, fps=fps, frames=n, duration_sec=n/fps, processed_frames=len(processed),
                usable_roi_frame_fraction=float(usable.any(axis=1).mean()) if k else 0.)
    return pd.DataFrame(frame_table), windows, info


def summarize(windows):
    rows = []
    for (video, split, method, strategy), group in windows.groupby(['video_id', 'split', 'method', 'strategy'], sort=False):
        reliable = group[group.reliable]
        rows.append(dict(video_id=video, split=split, method=method, strategy=strategy, windows=len(group),
                         reliable_windows=len(reliable), reliable_fraction=len(reliable)/len(group) if len(group) else 0.,
                         rr_median_brpm_reliable=reliable.rr_brpm.median() if len(reliable) else None,
                         rr_median_brpm_all=group.rr_brpm.median(), rr_kalman_last_brpm=group.rr_kalman_brpm.iloc[-1],
                         snr_median_db=group.snr_db.median(), sqi_median=group.sqi.median(),
                         gross_motion_fraction_mean=group.gross_motion_fraction.mean(),
                         taa_phase_median_deg=group.taa_phase_deg.median() if 'taa_phase_deg' in group else None,
                         taa_coherence_median=group.taa_coherence.median() if 'taa_coherence' in group else None))
    return pd.DataFrame(rows)


def plot_video(target, frames, windows, rr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    method, strategy = PRIMARY
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                                                 gridspec_kw=dict(height_ratios=[1, 1.3, .7]))
    top.plot(frames.timestamp_sec, frames[f'resp_{method}_{strategy}'], color='#2a9fd6', lw=1.2)
    top.set_ylabel('breathing (z)')
    top.set_title(f'{frames.video_id.iloc[0]}: camera breathing signal ({method}, confidence x SNR dynamic ROI fusion)')
    colors = dict(dynamic_fusion='#2a9fd6', confidence_fusion='#dd8452', snr_fusion='#c44e52', top1='#55a868',
                  pool_mean='#8172b3', infant_box='#8c8c8c')
    for name, color in colors.items():
        part = windows[(windows.method == method) & (windows.strategy == name)]
        middle.plot(part.window_end_sec, part.rr_brpm, '-', color=color, lw=1.6 if name == strategy else .8,
                    alpha=1 if name == strategy else .55, label=name)
    part = windows[(windows.method == method) & (windows.strategy == strategy)]
    middle.plot(part.window_end_sec, part.rr_kalman_brpm, '--', color='k', lw=1.2, label='proposed + Kalman')
    bad = part[~part.reliable]
    middle.plot(bad.window_end_sec, bad.rr_brpm, 'x', color='k', label=f'SQI < {rr.sqi_threshold:g}')
    middle.set_ylim(rr.rr_min_bpm, rr.rr_max_bpm)
    middle.set_ylabel('RR (breaths/min)')
    middle.legend(fontsize=7, ncol=4, loc='upper right')
    bottom.plot(part.window_end_sec, part.sqi, color='#2a9fd6', label='SQI')
    bottom.plot(part.window_end_sec, part.gross_motion_fraction, color='#8c8c8c', lw=1, label='gross motion fraction')
    bottom.axhline(rr.sqi_threshold, color='k', ls=':', lw=1)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel('SQI')
    bottom.set_xlabel('time (s, window end)')
    bottom.legend(fontsize=7, loc='lower right')
    for axis in (top, middle, bottom):
        axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(target, dpi=130)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 2 RR: camera respiration rate from Module 1 dynamic ROIs.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--confidence-dir', help='Module 1 Phase 10 output (default: confidence.output_dir; '
                                                 'use outputs/confidence_sam2 for the SAM 2 branch)')
    parser.add_argument('--recovery', help='Phase 6 recovery_results.csv (default: recovery.output_dir)')
    parser.add_argument('--output-dir', default='outputs/rr')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='all')
    parser.add_argument('--videos', nargs='+', help='Explicit video IDs (overrides --split)')
    parser.add_argument('--roi-pool', nargs='+', help='RR candidate ROIs (default: neck, shoulders, chest, abdomen); '
                                                      '"all" = all 17 regions')
    parser.add_argument('--window-sec', type=float, help='Full RR analysis window (default 15 s)')
    parser.add_argument('--sqi-threshold', type=float, help='Reliability threshold on the SQI (default 0.5)')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = cfg.get('rr', {})
    root = Path(cfg['_project_root'])
    pool = None
    if args.roi_pool:
        pool = tuple(ROI_NAMES) if args.roi_pool == ['all'] else tuple(args.roi_pool)
        unknown = set(pool) - set(ROI_NAMES)
        if unknown:
            raise ValueError(f'Unknown ROI names: {sorted(unknown)}')
    rr = build_config(settings, RRConfig, dict(estimation_window_sec=args.window_sec, sqi_threshold=args.sqi_threshold))
    ex = build_config(settings, ExtractionConfig, dict(roi_pool=pool or settings.get('roi_pool') or RESPIRATORY_POOL))
    cr = build_config(cfg.get('cardiorespiratory', {}), CardioRespConfig)
    confidence_dir = project_path(cfg, args.confidence_dir or cfg.get('confidence', {}).get('output_dir', 'outputs/confidence'))
    recovery_path = project_path(cfg, args.recovery or
                                 str(Path(cfg.get('recovery', {}).get('output_dir', 'outputs/recovery'))/'recovery_results.csv'))
    output = project_path(cfg, args.output_dir)
    for path in (confidence_dir/'roi_confidence.csv', recovery_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Run Module 1 (Phases 6-10) first.')
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError(f'{output} is not empty; use --output-dir for a new experiment')

    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    records = manifest.df
    if args.videos:
        missing = set(args.videos) - set(records.video_id)
        if missing:
            raise ValueError(f'Unknown video IDs: {sorted(missing)}')
        records = records[records.video_id.isin(args.videos)]
    elif args.split != 'all':
        records = records[records.split == args.split]
    if records.empty:
        raise ValueError('No videos selected')

    columns = ['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'bbox', 'q']
    confidence = pd.read_csv(confidence_dir/'roi_confidence.csv', usecols=lambda c: c in columns, low_memory=False)
    confidence = confidence[confidence.roi_name.isin(ex.roi_pool)]
    recovery = pd.read_csv(recovery_path, usecols=lambda c: c in ('video_id', 'frame_index', 'final_infant_bbox'), low_memory=False)
    by_video = {video: group.to_dict('records') for video, group in confidence.groupby('video_id')}
    recovery_by_video = {video: group.to_dict('records') for video, group in recovery.groupby('video_id')}

    output.mkdir(parents=True, exist_ok=True)
    (output/'plots').mkdir()
    frame_tables, window_tables, infos = [], [], []
    for _, record in records.iterrows():
        video = str(record.video_id)
        if video not in by_video:
            print(f'{video}: no Module 1 respiratory ROIs, skipped', flush=True)
            continue
        frames, windows, info = process_video(video, str(record.split), manifest.resolve_video_path(root, record),
                                              by_video[video], recovery_by_video.get(video, []), rr, ex, cr)
        frame_tables.append(frames)
        window_tables.append(windows)
        infos.append(info)
        if windows.empty:
            print(f'{video}: {info["frames"]} frames, too short for an RR window (< {rr.min_window_sec:g} s)', flush=True)
            continue
        primary = windows[(windows.method == PRIMARY[0]) & (windows.strategy == PRIMARY[1])]
        reliable = primary[primary.reliable]
        summary = f'median RR {reliable.rr_brpm.median():.1f} breaths/min' if len(reliable) else 'no reliable window'
        print(f'{video}: {info["frames"]} frames, {len(primary)} windows, {len(reliable)} reliable, {summary}', flush=True)
        if not args.no_plots:
            plot_video(output/'plots'/f'{video}_rr.png', frames, windows, rr)
    windows = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    if windows.empty:
        raise ValueError('No RR windows were produced (videos too short or no usable respiratory ROI).')

    pd.concat(frame_tables).to_csv(output/'resp_signals.csv', index=False)
    windows.to_csv(output/'rr_windows.csv', index=False)
    summary = summarize(windows)
    summary.to_csv(output/'rr_summary.csv', index=False)
    overview = (summary.groupby(['method', 'strategy'])
                .agg(videos=('video_id', 'count'), reliable_fraction=('reliable_fraction', 'mean'),
                     rr_median_brpm=('rr_median_brpm_reliable', 'median'), snr_median_db=('snr_median_db', 'median'),
                     sqi_median=('sqi_median', 'median'))
                .reset_index())
    overview.to_csv(output/'rr_overview.csv', index=False)
    metadata = dict(phase='MODULE_2_RR', run_status='COMPLETE', videos=len(infos), windows=len(windows),
                    primary_result=dict(method=PRIMARY[0], strategy=PRIMARY[1], display='rr_kalman_brpm'),
                    methods=list(RR_METHODS), strategies=list(STRATEGIES),
                    rr_config=asdict(rr), extraction_config=asdict(ex), cardiorespiratory_config=asdict(cr),
                    taa_definition='per window: chest (best of UPPER/LOWER_CHEST by FLOW SNR) and ABDOMEN band-passed FLOW '
                                   'displacements projected on their shared principal axis; |phase| of the Welch '
                                   'cross-spectrum and magnitude-squared coherence at the FLOW / dynamic_fusion RR; '
                                   '0 deg synchronous, 180 deg paradoxical; heuristic, not clinically validated',
                    fusion_definition='window spectrum = sum_r w_r P_r / sum_r w_r (normalized ROI spectra; phase-invariant, '
                                      'robust to paradoxical chest/abdomen motion); dynamic_fusion w_r ∝ prior_r * goodness_r; '
                                      'prior_r = window mean of softmax_r(Q_r(t)/weight_temperature); goodness_r = 10^(SNR_r/10)',
                    sqi_definition='sigmoid((SNR - snr_threshold_db)/snr_slope_db) * (1 - fraction of window frames with '
                                   'camera-compensated infant-box flow > gross_motion_box_per_sec)',
                    reliability_definition=f'SQI >= {rr.sqi_threshold}; heuristic, calibrate on validation data with a reference RR',
                    references=['Villarroel et al., npj Digit. Med. 2019 (NICU camera HR and RR)',
                                'Tarassenko et al., Physiol. Meas. 2014 (pixel-intensity RR)',
                                'Lorato et al., Sensors 2021 (neonatal optical-flow respiration)',
                                'Farneback, SCIA 2003 (dense optical flow)', 'arXiv:2207.11886 (neonatal ROI comparison)'],
                    reference_rr_used=False,
                    inputs=dict(confidence=str(confidence_dir/'roi_confidence.csv'), recovery=str(recovery_path),
                                manifest=str(manifest_path), manifest_sha256=sha256_file(manifest_path)),
                    video_info=infos, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), opencv_version=cv2.__version__)
    (output/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print(overview.to_string(index=False))
    print(f'Done: {len(infos)} video(s) -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
