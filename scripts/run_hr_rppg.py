"""Module 2 (HR branch): camera heart rate from the Module 1 dynamic ROIs.

Per video: per native frame, mean skin RGB inside every candidate ROI (Module 1 Phase 8 geometry,
held between processed frames) -> pulse per ROI (POS / CHROM / GREEN) -> per 8 s window (1 s step)
fusion across ROIs -> heart rate, SNR, motion-aware SQI ->
SQI-weighted Kalman smoothing. The window HR is the interpolated FFT peak by default; the AR best model
(Villarroel et al. 2019) is always reported alongside it (hr_ar_bpm).

Fusion strategies (all on the same traces, for the ablation):
  dynamic_fusion     proposed: weight_r = prior_r x goodness_r, where prior_r is the softmax of the
                     Module 1 confidence Q and goodness_r the ROI's own spectral SNR (linear), i.e.
                     maximal-ratio combining (Kumar et al., DistancePPG 2015) with a Module 1 prior;
  confidence_fusion  Module 1 prior only (no signal evidence);
  snr_fusion         goodness only (DistancePPG-style, no Module 1 confidence);
  top1               the single ROI with the highest Module 1 confidence;
  pool_mean          equal weights over usable ROIs;
  infant_box         whole Phase 6 infant box (no anatomical ROI).
Multi-site / cross-vital extensions (module2.cardiorespiratory), on the same ROI pulses and weights:
  consensus_fusion          dynamic-fusion weights inside each anatomical site; each site votes with its
                            spectral peak and the HR supported by the most body sites wins;
  cardiorespiratory_fusion  consensus_fusion + respiratory-harmonic-aware peak selection using the RR of
                            the motion branch (only with --rr-windows from run_rr.py).
With --neural-dir (run_neural_rppg.py), the neural pulses (NEURAL_SRC, NEURAL_TTA, NEURAL_TTA_UNGUIDED) are
added as pulse methods next to POS / CHROM / GREEN and go through every strategy above.
No reference HR is used here; see evaluate_hr.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
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
from module2.cardiorespiratory import CardioRespConfig, consensus_hr, nearest_rr, roi_spectrum
from module2.extraction import (ExtractionConfig, FACE_POOL, INFANT_BOX, box_polygon, fusion_weights,
                                hold_indices, mean_rgb, number, parse_json, roi_polygon)
from module2.rppg import (ESTIMATORS, METHODS, HRConfig, _usable, analysis_windows, estimate_hr, extract_pulse,
                          kalman_smooth, signal_quality, snr_db, window_snr)


STRATEGIES = ('dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box')
MULTISITE_STRATEGIES = ('consensus_fusion', 'cardiorespiratory_fusion')
PRIMARY = ('POS', 'dynamic_fusion')


def build_config(section, cls, overrides=None):
    values = {field.name: section[field.name] for field in fields(cls) if field.name in section}
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def weighted_pulse(pulses, weights):
    """Sum_r w_r z_r / Sum_r w_r over ROIs with a finite pulse; NaN where nothing is usable.

    weights: (n, k) per sample or (k,) constant over the segment.
    """
    weights = np.broadcast_to(weights, pulses.shape)
    finite = np.isfinite(pulses)
    w = np.where(finite, weights, 0.)
    total = w.sum(axis=1)
    fused = (np.where(finite, pulses, 0.) * w).sum(axis=1)
    return np.divide(fused, total, out=np.full(len(total), np.nan), where=total > 0)


def normalize(weights):
    total = weights.sum()
    return weights / total if total > 0 else weights * 0.


def window_weights(prior, goodness, equal, available):
    """Per-strategy ROI weights (k,) for one window."""
    top = np.zeros_like(prior)
    if prior.sum() > 0:
        top[int(prior.argmax())] = 1.
    return dict(dynamic_fusion=normalize(prior * goodness), confidence_fusion=normalize(prior),
                snr_fusion=normalize(goodness * available), top1=top, pool_mean=normalize(equal))


def nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def read_video(path, names, by_roi, boxes, hold, ex):
    rgb, q, skin, motion, pixels = [], [], [], [], []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'Cannot decode video: {path}')
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            source = int(hold[index]) if index < len(hold) else -1
            values = []
            for name in names:
                if name == INFANT_BOX:
                    box = boxes.get(source) if source >= 0 else None
                    polygon, row = (box_polygon(box) if box else None), {}
                else:
                    row = by_roi[name].get(source, {}) if source >= 0 else {}
                    polygon = roi_polygon(row) if row else None
                value, count = mean_rgb(image, polygon, ex) if polygon is not None else (None, 0)
                values.append((value if value is not None else (np.nan,)*3,
                               *(number(row.get(field)) if row else None
                                 for field in ('q', 'visible_skin_proportion', 'motion_magnitude')), count))
            rgb.append([v[0] for v in values])
            q.append([v[1] for v in values])
            skin.append([v[2] for v in values])
            motion.append([v[3] for v in values])
            pixels.append([v[4] for v in values])
            index += 1
    finally:
        capture.release()
    as_float = lambda rows: np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    return np.asarray(rgb, float), as_float(q), as_float(skin), as_float(motion), np.asarray(pixels)


def multisite_row(strategy, segment, weights, pool, fps, hr, cr, rr_bpm, window_motion):
    """Window result of a consensus strategy; SNR and SQI are those of the dynamic-fusion pulse at the
    consensus HR, so reliability is comparable with the other strategies, plus multi-site agreement."""
    spectra = [roi_spectrum(segment[:, r], fps, hr) if weights[r] > 0 else None for r in range(len(pool))]
    result = consensus_hr(spectra, pool, weights, fps, hr, cr, rr_bpm)
    x = weighted_pulse(segment, weights)
    clean, bpm = _usable(x, hr), result['hr_bpm']
    snr = float(snr_db(clean, fps, bpm, hr)) if bpm is not None and clean is not None else None
    sqi = signal_quality(snr, window_motion, hr) if snr is not None else 0.
    agreed = result['consensus_support'] >= min(2, result['consensus_sites'])
    return dict(strategy=strategy, top_roi=pool[int(weights.argmax())] if weights.sum() > 0 else None,
                roi_weights=json.dumps({pool[r]: round(float(w), 4) for r, w in enumerate(weights) if w > 0}),
                valid_fraction=float(np.mean(np.isfinite(x))), hr_bpm=bpm, hr_fft_bpm=bpm, hr_ar_bpm=None,
                snr_db=snr, motion=window_motion, sqi=sqi,
                reliable=bool(bpm is not None and sqi >= hr.sqi_threshold and agreed),
                consensus_sites=result['consensus_sites'], consensus_abstained=result['consensus_abstained'],
                consensus_support=result['consensus_support'],
                rr_reference_brpm=rr_bpm, harmonic_ambiguous=result['harmonic_ambiguous'])


def load_neural_pulses(path, names, n):
    """{method: (n, len(names)) pulse} from a run_neural_rppg.py file, columns matched by ROI name."""
    data = np.load(path, allow_pickle=False)
    stored = list(data['names'])
    missing = set(names) - set(stored)
    if missing:
        raise ValueError(f'{path}: neural pulses lack ROIs {sorted(missing)}')
    columns = [stored.index(name) for name in names]
    out = {}
    for method in data['methods']:
        values = np.full((n, len(names)), np.nan)
        part = data[method][:n, columns]
        values[:len(part)] = part
        out[str(method)] = values
    return out


def process_video(video, split, path, rows, recovery_rows, hr, ex, cr=None, rr_table=None, neural_path=None):
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
    rgb, q, skin, motion, pixels = read_video(path, names, by_roi, boxes, hold, ex)
    if len(rgb) == 0:
        raise ValueError(f'{video}: no frames decoded')
    n = len(rgb)

    pulses = {method: np.stack([extract_pulse(rgb[:, r], fps, method, hr) for r in range(len(names))], axis=1)
              for method in METHODS}
    if neural_path is not None:
        pulses.update(load_neural_pulses(neural_path, names, n))
    methods = list(pulses)
    usable = (np.isfinite(pulses['POS'][:, :k]) & np.isfinite(q[:, :k]) &
              (np.nan_to_num(skin[:, :k], nan=0.) >= ex.min_visible_skin))
    prior = fusion_weights(q[:, :k], usable, ex, fps)
    equal = fusion_weights(np.zeros((n, k)), usable, ExtractionConfig(weight_temperature=1., weight_smoothing_sec=0.), fps)

    window_rows = []
    display = {(m, s): np.zeros((n, k)) for m in methods for s in STRATEGIES if s != 'infant_box'}
    coverage = np.zeros(n)
    for start, stop in analysis_windows(n, fps, hr):
        coverage[start:stop] += 1
        prior_w, equal_w = prior[start:stop].mean(axis=0), equal[start:stop].mean(axis=0)
        available = (usable[start:stop].mean(axis=0) >= hr.min_valid_fraction).astype(float)
        roi_motion = np.array([nanmean(motion[start:stop, r]) if np.isfinite(motion[start:stop, r]).any() else np.nan
                               for r in range(k)])
        base = dict(video_id=video, split=split, window_start_sec=start/fps, window_end_sec=stop/fps,
                    window_center_sec=(start+stop)/2/fps, fps=fps, estimator=hr.estimator)
        for method in methods:
            segment = pulses[method][start:stop]
            snrs = [window_snr(segment[:, r], fps, hr) if available[r] else None for r in range(k)]
            goodness = np.array([10 ** (s / 10) if s is not None and np.isfinite(s) else 0. for s in snrs])
            strategies = window_weights(prior_w * available, goodness, equal_w * available, available)
            for strategy in STRATEGIES:
                if strategy == 'infant_box':
                    x, weights = segment[:, k], None
                    window_motion = nanmean(motion[start:stop, :k])
                    top = INFANT_BOX
                else:
                    weights = strategies[strategy]
                    display[(method, strategy)][start:stop] += weights
                    x = weighted_pulse(segment[:, :k], weights)
                    used = weights > 0
                    window_motion = (float(np.nansum(weights[used] * np.nan_to_num(roi_motion[used], nan=0.)))
                                     if used.any() else None)
                    top = pool[int(weights.argmax())] if weights.sum() > 0 else None
                estimates = {name: estimate_hr(x, fps, hr, estimator=name) for name in ESTIMATORS}
                bpm, snr = estimates[hr.estimator]
                sqi = signal_quality(snr, window_motion, hr) if bpm is not None else 0.
                window_rows.append(dict(
                    base, method=method, strategy=strategy, top_roi=top,
                    roi_weights=json.dumps({} if weights is None else
                                           {pool[r]: round(float(w), 4) for r, w in enumerate(weights) if w > 0}),
                    valid_fraction=float(np.mean(np.isfinite(x))), hr_bpm=bpm,
                    hr_fft_bpm=estimates['fft'][0], hr_ar_bpm=estimates['ar_best'][0], snr_db=snr,
                    motion=window_motion, sqi=sqi, reliable=bool(bpm is not None and sqi >= hr.sqi_threshold)))
            dynamic = strategies['dynamic_fusion']
            used = dynamic > 0
            dynamic_motion = (float(np.nansum(dynamic[used] * np.nan_to_num(roi_motion[used], nan=0.)))
                              if used.any() else None)
            rr_bpm = nearest_rr(rr_table, base['window_center_sec'], cr) if rr_table is not None else None
            for strategy in MULTISITE_STRATEGIES:
                if strategy == 'cardiorespiratory_fusion' and rr_table is None:
                    continue
                window_rows.append(dict(base, method=method, **multisite_row(
                    strategy, segment[:, :k], dynamic, pool, fps, hr, cr,
                    rr_bpm if strategy == 'cardiorespiratory_fusion' else None, dynamic_motion)))

    # SQI-weighted causal Kalman smoothing per method x strategy (time = window end, as a monitor sees it).
    windows = pd.DataFrame(window_rows)
    if not windows.empty:
        windows['hr_kalman_bpm'] = np.nan
        for _, group in windows.groupby(['method', 'strategy'], sort=False):
            smoothed = kalman_smooth(group.window_end_sec.to_numpy(), group.hr_bpm.to_numpy(dtype=float),
                                     group.sqi.to_numpy(), hr)
            windows.loc[group.index, 'hr_kalman_bpm'] = np.array([np.nan if v is None else v for v in smoothed], dtype=float)

    # Per-frame display weights: average of the weights of all windows covering the frame
    # (Module 1 prior where no window covers it).
    frame_table = dict(video_id=video, frame_index=np.arange(n), timestamp_sec=np.arange(n) / fps)
    covered = coverage[:, None] > 0
    primary_weights = np.where(covered, display[PRIMARY] / np.maximum(coverage, 1)[:, None], prior)
    for r, name in enumerate(pool):
        frame_table[f'weight_{name}'] = primary_weights[:, r]
        frame_table[f'prior_{name}'] = prior[:, r]
    top_frame = np.where(primary_weights.sum(axis=1) > 0, primary_weights.argmax(axis=1), -1)
    frame_table['top_roi'] = [pool[i] if i >= 0 else '' for i in top_frame]
    for method in methods:
        for strategy in STRATEGIES:
            if strategy == 'infant_box':
                pulse = pulses[method][:, k]
            else:
                weights = np.where(covered, display[(method, strategy)] / np.maximum(coverage, 1)[:, None], prior)
                pulse = weighted_pulse(pulses[method][:, :k], weights)
            frame_table[f'pulse_{method}_{strategy}'] = pulse
    traces = dict(video_id=video, frame_index=np.arange(n), timestamp_sec=np.arange(n) / fps)
    for r, name in enumerate(names):
        for c, channel in enumerate('rgb'):
            traces[f'{channel}_{name}'] = rgb[:, r, c]
        traces[f'pixels_{name}'] = pixels[:, r]
    info = dict(video_id=video, split=split, fps=fps, frames=n, duration_sec=n/fps,
                processed_frames=len(processed), usable_roi_frame_fraction=float(usable.any(axis=1).mean()))
    return pd.DataFrame(frame_table), pd.DataFrame(traces), windows, info


def summarize(windows):
    rows = []
    for (video, split, method, strategy), group in windows.groupby(['video_id', 'split', 'method', 'strategy'], sort=False):
        reliable = group[group.reliable]
        rows.append(dict(video_id=video, split=split, method=method, strategy=strategy, windows=len(group),
                         reliable_windows=len(reliable), reliable_fraction=len(reliable)/len(group) if len(group) else 0.,
                         hr_median_bpm_reliable=reliable.hr_bpm.median() if len(reliable) else None,
                         hr_median_bpm_all=group.hr_bpm.median(), hr_kalman_last_bpm=group.hr_kalman_bpm.iloc[-1],
                         snr_median_db=group.snr_db.median(), sqi_median=group.sqi.median()))
    return pd.DataFrame(rows)


def plot_video(target, frames, windows, hr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    method, strategy = PRIMARY
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                                                 gridspec_kw=dict(height_ratios=[1, 1.3, .7]))
    top.plot(frames.timestamp_sec, frames[f'pulse_{method}_{strategy}'], color='#d6589f', lw=1)
    top.set_ylabel('pulse (z)')
    top.set_title(f'{frames.video_id.iloc[0]}: camera pulse ({method}, confidence x SNR weighted dynamic ROI fusion)')
    colors = dict(dynamic_fusion='#d6589f', confidence_fusion='#dd8452', snr_fusion='#4c72b0', top1='#55a868',
                  pool_mean='#8172b3', infant_box='#8c8c8c', consensus_fusion='#c44e52',
                  cardiorespiratory_fusion='#937860')
    for name, color in colors.items():
        part = windows[(windows.method == method) & (windows.strategy == name)]
        middle.plot(part.window_end_sec, part.hr_bpm, '-', color=color, lw=1.6 if name == strategy else .8,
                    alpha=1 if name == strategy else .55, label=name)
    part = windows[(windows.method == method) & (windows.strategy == strategy)]
    middle.plot(part.window_end_sec, part.hr_kalman_bpm, '--', color='k', lw=1.2, label='proposed + Kalman')
    bad = part[~part.reliable]
    middle.plot(bad.window_end_sec, bad.hr_bpm, 'x', color='k', label=f'SQI < {hr.sqi_threshold:g}')
    middle.set_ylim(hr.hr_min_bpm, hr.hr_max_bpm)
    middle.set_ylabel('HR (bpm)')
    middle.legend(fontsize=7, ncol=4, loc='upper right')
    bottom.plot(part.window_end_sec, part.sqi, color='#d6589f')
    bottom.axhline(hr.sqi_threshold, color='k', ls=':', lw=1)
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel('SQI')
    bottom.set_xlabel('time (s, window end)')
    for axis in (top, middle, bottom):
        axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(target, dpi=130)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 2 HR: camera heart rate from Module 1 dynamic ROIs.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--confidence-dir', help='Module 1 Phase 10 output (default: confidence.output_dir; '
                                                 'use outputs/confidence_sam2 for the SAM 2 branch)')
    parser.add_argument('--recovery', help='Phase 6 recovery_results.csv (default: recovery.output_dir)')
    parser.add_argument('--output-dir', default='outputs/hr_rppg')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='all')
    parser.add_argument('--videos', nargs='+', help='Explicit video IDs (overrides --split)')
    parser.add_argument('--roi-pool', nargs='+', help='HR candidate ROIs (default: all 17 regions); "face" = face regions only')
    parser.add_argument('--estimator', choices=ESTIMATORS, help='Primary HR estimator (default fft; both are always reported)')
    parser.add_argument('--window-sec', type=float, help='HR analysis window (default 8 s)')
    parser.add_argument('--sqi-threshold', type=float, help='Reliability threshold on the SQI (default 0.5)')
    parser.add_argument('--skin-mode', choices=('adaptive', 'fixed', 'none'))
    parser.add_argument('--neural-dir', help='run_neural_rppg.py output: adds the neural pulse methods')
    parser.add_argument('--rr-windows', help='rr_windows.csv of run_rr.py: enables cardiorespiratory_fusion '
                                             '(respiratory-harmonic-aware HR)')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = cfg.get('hr', {})
    root = Path(cfg['_project_root'])
    pool = None
    if args.roi_pool:
        pool = FACE_POOL if args.roi_pool == ['face'] else tuple(args.roi_pool)
        unknown = set(pool) - set(ROI_NAMES)
        if unknown:
            raise ValueError(f'Unknown ROI names: {sorted(unknown)}')
    hr = build_config(settings, HRConfig, dict(estimation_window_sec=args.window_sec, sqi_threshold=args.sqi_threshold,
                                               estimator=args.estimator))
    ex = build_config(settings, ExtractionConfig, dict(roi_pool=pool, skin_mode=args.skin_mode))
    cr = build_config(cfg.get('cardiorespiratory', {}), CardioRespConfig)
    rr_by_video, rr_path = None, None
    if args.rr_windows:
        rr_path = project_path(cfg, args.rr_windows)
        if not rr_path.is_file():
            raise FileNotFoundError(f'Missing input: {rr_path}. Run run_rr.py first.')
        rr = pd.read_csv(rr_path, usecols=['video_id', 'method', 'strategy', 'window_center_sec', 'rr_brpm', 'sqi'])
        rr = rr[(rr.method == 'FLOW') & (rr.strategy == 'dynamic_fusion') & rr.rr_brpm.notna() & (rr.sqi >= cr.min_rr_sqi)]
        rr_by_video = {str(video): group.reset_index(drop=True) for video, group in rr.groupby('video_id')}
    neural_dir = project_path(cfg, args.neural_dir) if args.neural_dir else None
    if neural_dir is not None and not neural_dir.is_dir():
        raise FileNotFoundError(f'Missing input: {neural_dir}. Run run_neural_rppg.py first.')

    def neural_path(video):
        if neural_dir is None:
            return None
        path = neural_dir/f'{re.sub(r"[^A-Za-z0-9_.-]", "_", video)}_neural_pulses.npz'
        if not path.is_file():
            print(f'{video}: no neural pulses in {neural_dir}', flush=True)
            return None
        return path
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

    columns = ['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'bbox', 'q',
               'visible_skin_proportion', 'motion_magnitude']
    confidence = pd.read_csv(confidence_dir/'roi_confidence.csv', usecols=lambda c: c in columns, low_memory=False)
    confidence = confidence[confidence.roi_name.isin(ex.roi_pool)]
    recovery = pd.read_csv(recovery_path, usecols=lambda c: c in ('video_id', 'frame_index', 'final_infant_bbox'), low_memory=False)
    by_video = {video: group.to_dict('records') for video, group in confidence.groupby('video_id')}
    recovery_by_video = {video: group.to_dict('records') for video, group in recovery.groupby('video_id')}

    output.mkdir(parents=True, exist_ok=True)
    (output/'plots').mkdir()
    frame_tables, trace_tables, window_tables, infos = [], [], [], []
    for _, record in records.iterrows():
        video = str(record.video_id)
        if video not in by_video:
            print(f'{video}: no Module 1 ROIs, skipped', flush=True)
            continue
        frames, traces, windows, info = process_video(video, str(record.split), manifest.resolve_video_path(root, record),
                                                      by_video[video], recovery_by_video.get(video, []), hr, ex, cr,
                                                      None if rr_by_video is None else
                                                      rr_by_video.get(video, pd.DataFrame(columns=['window_center_sec', 'rr_brpm'])),
                                                      neural_path(video))
        frame_tables.append(frames)
        trace_tables.append(traces)
        window_tables.append(windows)
        infos.append(info)
        if windows.empty:
            print(f'{video}: {info["frames"]} frames, too short for an HR window', flush=True)
            continue
        primary = windows[(windows.method == PRIMARY[0]) & (windows.strategy == PRIMARY[1])]
        reliable = primary[primary.reliable]
        summary = f'median HR {reliable.hr_bpm.median():.1f} bpm' if len(reliable) else 'no reliable window'
        print(f'{video}: {info["frames"]} frames, {len(primary)} windows, {len(reliable)} reliable, {summary}', flush=True)
        if not args.no_plots:
            plot_video(output/'plots'/f'{video}_hr.png', frames, windows, hr)
    windows = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    if windows.empty:
        raise ValueError('No HR windows were produced (videos too short or no usable ROI).')

    pd.concat(frame_tables).to_csv(output/'pulse_signals.csv', index=False)
    pd.concat(trace_tables).to_csv(output/'rgb_traces.csv', index=False)
    windows.to_csv(output/'hr_windows.csv', index=False)
    summary = summarize(windows)
    summary.to_csv(output/'hr_summary.csv', index=False)
    overview = (summary.groupby(['method', 'strategy'])
                .agg(videos=('video_id', 'count'), reliable_fraction=('reliable_fraction', 'mean'),
                     hr_median_bpm=('hr_median_bpm_reliable', 'median'), snr_median_db=('snr_median_db', 'median'),
                     sqi_median=('sqi_median', 'median'))
                .reset_index())
    overview.to_csv(output/'hr_overview.csv', index=False)
    metadata = dict(phase='MODULE_2_HR_RPPG', run_status='COMPLETE', videos=len(infos), windows=len(windows),
                    primary_result=dict(method=PRIMARY[0], strategy=PRIMARY[1], estimator=hr.estimator,
                                        display='hr_kalman_bpm'),
                    methods=sorted(set(windows.method)), strategies=sorted(set(windows.strategy)),
                    neural_dir=None if neural_dir is None else str(neural_dir),
                    hr_config=asdict(hr), extraction_config=asdict(ex), cardiorespiratory_config=asdict(cr),
                    multisite_definition='consensus_fusion: dynamic-fusion weights within each anatomical site '
                                         '(head, trunk, arms, legs); each site votes with its spectral peak, most '
                                         'agreeing sites win, ties by fusion-weighted spectral power; '
                                         'reliable also requires >= min(2, sites) sites agreeing within '
                                         'site_agreement_bpm. cardiorespiratory_fusion: additionally moves an HR peak '
                                         'lying on k x RR (k in rr_harmonics) to the strongest non-harmonic peak if it '
                                         'has >= harmonic_competitor_ratio of its power, else flags harmonic_ambiguous; '
                                         'sites whose own peak is an ambiguous harmonic abstain from the vote',
                    rr_reference=None if rr_path is None else dict(path=str(rr_path), sha256=sha256_file(rr_path),
                                                                   source='FLOW / dynamic_fusion windows'),
                    fusion_definition='window pulse = sum_r w_r z_r / sum_r w_r; dynamic_fusion w_r ∝ prior_r * goodness_r; '
                                      'prior_r = window mean of softmax_r(Q_r(t)/weight_temperature) (Module 1 Phase 10 Q); '
                                      'goodness_r = 10^(SNR_r/10) of the ROI pulse in the window (maximal-ratio combining)',
                    sqi_definition='sigmoid((SNR - snr_threshold_db)/snr_slope_db) * (1 - weighted Module 1 motion_magnitude)',
                    reliability_definition=f'SQI >= {hr.sqi_threshold}; heuristic, calibrate on validation data with a reference HR',
                    references=['Villarroel et al., npj Digit. Med. 2019 (NICU: 8 s/1 s windows, 1.5-4.5 Hz, AR best model, '
                                'SQI-weighted Kalman)', 'Wang et al., IEEE TBME 2017 (POS)',
                                'de Haan & Jeanne, IEEE TBME 2013 (CHROM, SNR)', 'Kumar et al., DistancePPG 2015 (MRC)',
                                'Tarvainen et al., IEEE TBME 2002 (detrending)'],
                    reference_hr_used=False, rr_calculated=False,
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
