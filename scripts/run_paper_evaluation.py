"""Paper evaluation of the Module 1 dynamic ROI representation (all manuscript tables and figures).

Usage (from the Module1 folder, after run_paper_pilot.py):
    python scripts/run_paper_evaluation.py --run runs/pilot24_paper --output-dir reports/paper
    python scripts/run_paper_evaluation.py ... --expert-labels labels.csv   # adds expert-label metrics

Configurations (same sampled frames, k = 3, fusion softmax(score / T) for all):
  B0    fixed box ROI          centre half of the first-frame infant box, held for the whole video
  B1    fixed anatomical ROI   first-frame region with the highest anatomical confidence, held fixed
  B2    region-level dynamic   per frame top-k by anatomical (pose) confidence
  B3    + visibility / skin    per frame top-k by mean(P, V, S)
  B4    + temporal / motion    implemented Phase 10 Q (equal weights), no SAM 2               (= S0)
  B5    + SAM 2 guidance       B4 on the SAM 2 branch (mask-constrained pose, ROIs, mask support)
  Full  + pixel refinement     B5 with colour pixels restricted to ROI n SAM 2 mask           (= S1)
B5 and Full share geometry; they differ only in which pixels inside the selected ROIs are used.

No metric here uses expert ground truth unless --expert-labels is given. Pulse SNR is a label-free
signal-quality proxy (POS, de Haan & Jeanne SNR at the in-band peak) used as independent evidence:
Q never sees the colour time series. Thresholds for strata and the "pulse-informative" label are
fixed on training-split videos only and then applied unchanged to validation and test videos.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from module1 import paper_eval as pe
from module1.confidence.mechanism import ConfidenceConfig, ConfidenceMechanism
from module2.ablation import bootstrap_ci, holm, paired_test
from module2.extraction import ExtractionConfig, hold_indices, mean_rgb
from module2.rppg import HRConfig, analysis_windows, bandpass, detrend_smoothness_priors, pulse_pos, window_snr

CONFIGS = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'Full')
NAMES = dict(B0='B0 Fixed box ROI', B1='B1 Fixed anatomical ROI', B2='B2 Region-level dynamic (anatomy)',
             B3='B3 + visibility/skin', B4='B4 + temporal/motion (S0, no SAM 2)', B5='B5 + SAM 2 guidance (region-level)',
             Full='Full: + pixel refinement (S1)')
SCORED = ('B2', 'B3', 'B4', 'B5')
TORSO = ('UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN', 'THORACO_ABDOMINAL')
GEOMETRY = ('coverage_percent', 'temporal_iou', 'centroid_jitter', 'selection_jaccard', 'switches_per_sec', 'mask_support')
HR = HRConfig()
EXTRACT = ExtractionConfig()


def load_branch(out, branch):
    table = pd.read_csv(out/branch/'roi_confidence.csv', low_memory=False)
    return {video: rows for video, rows in table.groupby('video_id')}


def boxes_of(recovery):
    boxes = {}
    for row in recovery.itertuples():
        value = pe.parse(row.final_infant_bbox) if isinstance(row.final_infant_bbox, str) else None
        boxes[int(row.frame_index)] = value if isinstance(value, list) else None
    return boxes


def load_mask(sam2_dir, row):
    if not bool(row.get('mask_reliable')) or not isinstance(row.get('mask_path'), str):
        return None
    mask = cv2.imread(str(sam2_dir/row['mask_path']), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 0


def pulse(rgb, fs):
    """POS pulse of an (n, 3) trace with NaN gaps (gaps short enough are interpolated, others stay NaN)."""
    out = np.full(len(rgb), np.nan)
    finite = np.isfinite(rgb).all(axis=1)
    if finite.sum() < fs * HR.min_window_sec:
        return out
    index = np.arange(len(rgb))
    filled = np.column_stack([np.interp(index, index[finite], rgb[finite, c]) for c in range(3)])
    h = pulse_pos(filled, fs, HR)
    h = bandpass(detrend_smoothness_priors(h, HR.detrend_lambda), fs, HR)
    out[finite] = h[finite]
    return out


def window_table(pulses, fs, windows):
    return [window_snr(pulses[a:b], fs, HR) for a, b in windows]


def evaluate_video(job):
    """Every per-video measurement (runs in a worker process)."""
    video, record, out, cand0, cand1, recovery, sam2_rows = job
    out = Path(out)
    fps = float(record['fps'])
    boxes = boxes_of(recovery)
    sam2_dir = out/'sam2'
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    fixed = pe.fixed_rois(cand0, boxes)
    selected = {c: pe.selections(cand1 if c in ('B5', 'Full') else cand0, 'Full' if c in ('B5', 'Full') else c,
                                 fixed=fixed) for c in CONFIGS}
    frames = sorted(set(cand0.frame_index) | set(cand1.frame_index))
    duration = float(record['duration_sec'])
    fps_sampled = len(frames) / duration if duration > 0 else np.nan

    masks = {}
    for frame in frames:
        if len(sam2_rows) and frame in sam2_rows.index:
            masks[frame] = load_mask(sam2_dir, sam2_rows.loc[frame])
    reliable = {f: m for f, m in masks.items() if m is not None}

    geometry = {}
    for c in CONFIGS:
        g = pe.geometry_metrics(selected[c], boxes, fps_sampled, masks=reliable)
        if c == 'Full':   # by construction: pixels outside a reliable mask are never used
            g['mask_support'] = 1. if reliable else np.nan
        geometry[c] = g

    # Per-native-frame colour traces of every candidate polygon (both branches) and of B0 / B1.
    capture = cv2.VideoCapture(str(record['video_path']))
    n = int(record['frame_count'])
    hold = hold_indices(frames, n, fps, EXTRACT.max_hold_sec)
    by_frame0 = {f: rows for f, rows in cand0.groupby('frame_index')}
    by_frame1 = {f: rows for f, rows in cand1.groupby('frame_index')}
    names = list(pe.ROI_NAMES)
    keys = [('b0', r) for r in names] + [('b1', r) for r in names] + [('m1', r) for r in names] + [('fix', 'B0'), ('fix', 'B1')]
    index = {k: i for i, k in enumerate(keys)}
    rgb = np.full((n, len(keys), 3), np.nan)
    luminance = np.full(n, np.nan)
    for i in range(n):
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source = int(hold[i])
        box = boxes.get(source) if source >= 0 else None
        if box is not None:
            x1, y1, x2, y2 = (int(max(0, v)) for v in box)
            region = gray[y1:y2, x1:x2]
            luminance[i] = region.mean() if region.size else gray.mean()
        else:
            luminance[i] = gray.mean()
        for key, polygon in (('B0', fixed[0]), ('B1', fixed[1][1] if fixed[1] else None)):
            if polygon is not None:
                value, _ = mean_rgb(image, np.asarray(polygon, float), EXTRACT)
                if value is not None:
                    rgb[i, index[('fix', key)]] = value
        if source < 0:
            continue
        for branch, table in (('b0', by_frame0), ('b1', by_frame1)):
            rows = table.get(source)
            if rows is None:
                continue
            for row in rows[rows.roi_status == 'AVAILABLE'].itertuples():
                polygon = pe.parse(row.polygon)
                if not polygon or len(polygon) < 3:
                    continue
                polygon = np.asarray(polygon, float)
                value, _ = mean_rgb(image, polygon, EXTRACT)
                if value is not None:
                    rgb[i, index[(branch, row.roi_name)]] = value
                mask = reliable.get(source)
                if branch == 'b1' and mask is not None and mask.shape == image.shape[:2]:
                    masked = image.copy()
                    masked[~mask] = 0      # below dark_level: excluded by mean_rgb
                    value, _ = mean_rgb(masked, polygon, EXTRACT)
                    if value is not None:
                        rgb[i, index[('m1', row.roi_name)]] = value
                elif branch == 'b1':
                    rgb[i, index[('m1', row.roi_name)]] = rgb[i, index[('b1', row.roi_name)]]
    capture.release()

    pulses = np.column_stack([pulse(rgb[:, j], fps) for j in range(len(keys))])
    windows = analysis_windows(n, fps, HR)
    centres = [(a + b) / 2 / fps for a, b in windows]

    # Fused pulse per configuration.
    fused = {}
    for c in CONFIGS:
        if c in ('B0', 'B1'):
            fused[c] = pulses[:, index[('fix', c)]]
            continue
        prefix = 'b0' if c in ('B2', 'B3', 'B4') else ('b1' if c == 'B5' else 'm1')
        sub = [index[(prefix, r)] for r in names]
        weights = pe.native_weights(selected[c], names, hold, n)
        fused[c] = pe.fused_pulse(pulses[:, sub], weights)

    # Per-frame condition descriptors (no-SAM 2 branch features; identical definitions for all configs).
    def frame_mean(column, subset=None):
        rows = cand0[cand0.roi_status == 'AVAILABLE']
        if subset:
            rows = rows[rows.roi_name.isin(subset)]
        return rows.groupby('frame_index')[column].mean() if column in rows else pd.Series(dtype=float)
    motion = frame_mean('motion_magnitude')
    occlusion = frame_mean('occlusion_proxy', TORSO)
    swaddling = frame_mean('swaddling_proxy', TORSO)
    skin = frame_mean('visible_skin_proportion')
    native = lambda series: np.array([series.get(int(hold[i]), np.nan) if hold[i] >= 0 else np.nan for i in range(n)])
    motion_n, occl_n, swad_n, skin_n = native(motion), native(occlusion), native(swaddling), native(skin)
    dlum = np.abs(np.diff(luminance, prepend=luminance[0]))

    window_rows = []
    for w, (a, b) in enumerate(windows):
        row = dict(video_id=video, split=record['split'], window=w, centre_sec=centres[w],
                   motion=np.nanmean(motion_n[a:b]) if np.isfinite(motion_n[a:b]).any() else np.nan,
                   occlusion=np.nanmean(occl_n[a:b]) if np.isfinite(occl_n[a:b]).any() else np.nan,
                   swaddling=np.nanmean(swad_n[a:b]) if np.isfinite(swad_n[a:b]).any() else np.nan,
                   skin=np.nanmean(skin_n[a:b]) if np.isfinite(skin_n[a:b]).any() else np.nan,
                   luminance=np.nanmean(luminance[a:b]), luminance_change=np.nanmax(dlum[a:b]))
        for c in CONFIGS:
            row[f'snr_{c}'] = window_snr(fused[c][a:b], fps, HR)
        window_rows.append(row)

    # Per ROI and window: score of each scored configuration vs the ROI's own pulse SNR.
    roi_rows = []
    for c in SCORED:
        prefix = 'b0' if c in ('B2', 'B3', 'B4') else 'b1'
        table = cand0 if c in ('B2', 'B3', 'B4') else cand1
        table = table[table.roi_status == 'AVAILABLE'].copy()
        table['score'] = np.nan
        for frame, rows in table.groupby('frame_index'):
            table.loc[rows.index, 'score'] = pe.score(rows, 'B4' if c == 'B5' else c)
        table['q01'] = pe.rescaled(table.score.to_numpy(float), 'B4' if c == 'B5' else c)
        for roi, rows in table.groupby('roi_name'):
            series = rows.set_index('frame_index').q01
            for w, (a, b) in enumerate(windows):
                sampled = [f for f in series.index if a <= f < b]
                if len(sampled) < .5 * len([f for f in frames if a <= f < b]) or not sampled:
                    continue
                snr = window_snr(pulses[a:b, index[(prefix, roi)]], fps, HR)
                if snr is None or not math.isfinite(snr):
                    continue
                roi_rows.append(dict(video_id=video, split=record['split'], config=c, roi_name=roi, window=w,
                                     q01=float(series.loc[sampled].mean()), snr=snr,
                                     motion=window_rows[w]['motion']))

    # Per-sampled-frame episode descriptors for the failure analysis.
    sampled_rows = []
    best_q = cand1[cand1.roi_status == 'AVAILABLE'].groupby('frame_index').q.max()
    seg = sam2_rows['segmentation_status'] if len(sam2_rows) else pd.Series(dtype=object)
    for f in frames:
        lum = luminance[f] if f < n else np.nan
        sampled_rows.append(dict(video_id=video, split=record['split'], frame_index=f, time_sec=f / fps,
                                 detected=boxes.get(f) is not None,
                                 recovered=None, motion=motion.get(f, np.nan), occlusion=occlusion.get(f, np.nan),
                                 luminance=lum, segmentation_status=seg.get(f, 'NONE') if len(seg) else 'NONE',
                                 selected_full=bool(selected['Full'].get(f)), best_q=best_q.get(f, np.nan)))
    detection = recovery.set_index('frame_index').detector_status.to_dict()
    for row in sampled_rows:
        row['detector_status'] = detection.get(row['frame_index'], 'NONE')

    per_video = []
    for c in CONFIGS:
        snr = np.array([r[f'snr_{c}'] for r in window_rows], float)
        snr = snr[np.isfinite(snr)]
        per_video.append(dict(video_id=video, split=record['split'], config=c, fps_sampled=fps_sampled,
                              sampled_frames=len(frames), reliable_mask_frames=len(reliable),
                              pulse_snr_db=float(np.median(snr)) if len(snr) else np.nan, **geometry[c]))
    return per_video, window_rows, roi_rows, sampled_rows


def mean_sd(values, digits=3):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 'n/a'
    return f'{values.mean():.{digits}f} ± {values.std(ddof=1) if len(values) > 1 else 0.:.{digits}f}'


def summary_table(per_video, metrics, digits):
    rows = []
    for c in CONFIGS:
        part = per_video[per_video.config == c]
        row = dict(configuration=NAMES[c], videos=int(part.video_id.nunique()))
        for m in metrics:
            estimate, low, high = bootstrap_ci(part[m])
            row[m] = mean_sd(part[m], digits[m])
            row[f'{m}_ci95'] = f'[{low:.{digits[m]}f}, {high:.{digits[m]}f}]' if math.isfinite(low) else 'n/a'
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(per_video, metrics, reference='Full'):
    rows = []
    for m in metrics:
        wide = per_video.pivot_table(index='video_id', columns='config', values=m)
        block = [dict(metric=m, baseline=c, candidate=reference, **paired_test(wide[c], wide[reference]))
                 for c in CONFIGS if c != reference and c in wide and reference in wide]
        for row, adjusted in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = adjusted
        rows += block
    return pd.DataFrame(rows)


def tertiles(train_values):
    values = np.asarray(train_values, float)
    values = values[np.isfinite(values)]
    return tuple(np.percentile(values, [100/3, 200/3])) if len(values) else (np.nan, np.nan)


def stratified(windows, column, edges, labels, threshold, informative=True):
    """Per configuration and stratum: mean ± sd over videos of window-median SNR and pulse-informative rate."""
    level = pd.cut(windows[column], [-np.inf, *edges, np.inf], labels=labels)
    rows = []
    for c in CONFIGS:
        row = dict(configuration=NAMES[c])
        per_level = {}
        for name in labels:
            part = windows[level == name]
            snr = part.groupby('video_id')[f'snr_{c}'].median()
            rate = part.groupby('video_id')[f'snr_{c}'].apply(lambda s: 100. * (s >= threshold).mean())
            per_level[name] = rate
            row[f'{name} SNR dB'] = mean_sd(snr, 2)
            row[f'{name} informative %'] = mean_sd(rate, 1)
            row[f'{name} videos'] = int(snr.notna().sum())
        if len(labels) >= 2:
            first, last = per_level[labels[0]], per_level[labels[-1]]
            drop = (first - last).dropna()   # paired within video; few videos span both extreme strata
            row[f'{labels[0]}-vs-{labels[-1]} drop (pp)'] = (f'{mean_sd(drop, 1)} (n={len(drop)})' if len(drop) >= 3
                                                             else f'n/a (n={len(drop)})')
        rows.append(row)
    return pd.DataFrame(rows)


def episodes(flags):
    """(start, stop) index runs of True values."""
    runs, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    return runs


def failure_analysis(frames, thresholds, fps_sampled):
    """Failure conditions measured from diagnostics; expert-only conditions are reported as not assessable."""
    q_low = thresholds['q_low']
    conditions = {
        'Extended detector loss (> recovery cap)': lambda g: (g.detector_status != 'DETECTED') & ~g.detected,
        'Detector miss bridged by recovery': lambda g: (g.detector_status != 'DETECTED') & g.detected,
        'SAM 2 low-confidence / failed propagation': lambda g: g.segmentation_status.isin(['LOW_CONFIDENCE', 'FAILED']),
        'Severe underexposure (< train 5th pct luminance)': lambda g: g.luminance < thresholds['lum_p05'],
        'High nuisance motion (> train 95th pct)': lambda g: g.motion > thresholds['motion_p95'],
        'Strong torso occlusion (> train 95th pct)': lambda g: g.occlusion > thresholds['occlusion_p95'],
    }
    rows = []
    window = max(1, int(round(fps_sampled)))   # 1 s recovery window in sampled frames
    for name, rule in conditions.items():
        count, flagged, recovered, total, videos, example = 0, 0, 0, 0, set(), ''
        for video, g in frames.groupby('video_id'):
            g = g.sort_values('frame_index').reset_index(drop=True)
            flags = rule(g).fillna(False).to_numpy(bool)
            for a, b in episodes(flags):
                count += 1
                videos.add(video)
                part = g.iloc[a:b]
                low = (~part.selected_full) | (part.best_q < q_low)
                flagged += int(low.sum())
                total += len(part)
                after = g.iloc[b:b + window]
                recovered += bool(len(after) and (after.selected_full & (after.best_q >= q_low)).any())
                if not example:
                    example = f'{video} frames {int(part.frame_index.iloc[0])}-{int(part.frame_index.iloc[-1])}'
        rows.append({'Failure condition': name, 'Episodes (n)': count, 'Videos (n)': len(videos),
                     'Low-confidence / no-ROI frames (%)': round(100. * flagged / total, 1) if total else 'n/a',
                     'Recovered within 1 s (%)': round(100. * recovered / count, 1) if count else 'n/a',
                     'Representative episode': example or 'none observed'})
    for name in ('Near-complete swaddling', 'Caregiver occlusion + infant motion', 'Reflective incubator artifact',
                 'Unusual neonatal posture'):
        rows.append({'Failure condition': name, 'Episodes (n)': 'requires expert labels', 'Videos (n)': '',
                     'Low-confidence / no-ROI frames (%)': '', 'Recovered within 1 s (%)': '',
                     'Representative episode': 'see annotation kit'})
    return pd.DataFrame(rows)


def q_validity(roi, threshold, calibrators):
    rows = []
    for c in SCORED:
        for split in ('all', 'train', 'validation', 'test'):
            part = roi[(roi.config == c) & ((roi.split == split) | (split == 'all'))]
            if len(part) < 10:
                continue
            label = part.snr >= threshold
            rho, p = stats.spearmanr(part.q01, part.snr)
            ece, brier, _ = pe.calibration(part.q01, label)
            row = dict(configuration=NAMES[c], split=split, roi_windows=len(part), videos=part.video_id.nunique(),
                       spearman_q_snr=rho, spearman_p=p, auroc=pe.auroc(part.q01, label), auprc=pe.auprc(part.q01, label),
                       positive_rate=label.mean(), ece_raw=ece, brier_raw=brier)
            a, b = calibrators.get(c, (np.nan, np.nan))
            if math.isfinite(a):
                probability = 1 / (1 + np.exp(-(a * part.q01 + b)))
                row['ece_platt'], row['brier_platt'], _ = pe.calibration(probability, label)
            # Per-video Spearman (the video is the statistical unit).
            per_video = [stats.spearmanr(g.q01, g.snr)[0] for _, g in part.groupby('video_id') if len(g) >= 5]
            per_video = np.asarray([v for v in per_video if np.isfinite(v)])
            row['per_video_spearman'] = mean_sd(per_video, 3)
            row['per_video_spearman_wilcoxon_p'] = float(stats.wilcoxon(per_video).pvalue) if len(per_video) >= 5 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def platt(q, label):
    """Logistic calibration fitted on training-split ROI windows (Newton steps, L2 1e-3)."""
    x = np.column_stack([np.asarray(q, float), np.ones(len(q))])
    y = np.asarray(label, float)
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ w))
        gradient = x.T @ (p - y) + 1e-3 * w
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + 1e-3 * np.eye(2)
        w -= np.linalg.solve(hessian, gradient)
    return float(w[0]), float(w[1])


def generalization(out, per_video_rows, manifest):
    """Equal weights vs weights fitted on the training split only (existing weakly_supervised regime), per split."""
    table = pd.read_csv(out/'roi_quality_sam2'/'roi_quality_features.csv', low_memory=False)
    rows = table.to_dict('records')
    fitted = {}
    for regime in ('equal_fixed', 'weakly_supervised'):
        mechanism = ConfidenceMechanism(ConfidenceConfig(regime=regime))
        mechanism.fit(rows)
        fitted[regime] = (mechanism, mechanism.fit_metadata)
    return fitted


def model_parameters():
    """Parameter counts read from the checkpoint files (no model construction)."""
    import torch
    counts = {}
    try:
        from safetensors import safe_open
        with safe_open(str(ROOT/'models/sapiens2_0.4b_pose.safetensors'), 'pt') as handle:
            counts['Sapiens2 0.4B pose'] = sum(np.prod(handle.get_slice(k).get_shape()) for k in handle.keys())
    except Exception as error:   # noqa: BLE001 - reported, not fatal
        counts['Sapiens2 0.4B pose'] = f'unavailable ({error})'
    for name, path in (('SAM 2.1 Hiera-T', 'models/sam2.1_hiera_tiny.pt'), ('YOLO11s', 'models/yolo11s.pt'),
                       ('RT-DETR-L', 'models/rtdetr-l.pt')):
        try:
            checkpoint = torch.load(str(ROOT/path), map_location='cpu', weights_only=False)
            state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if hasattr(state, 'state_dict'):
                state = state.state_dict()
            counts[name] = int(sum(v.numel() for v in state.values() if hasattr(v, 'numel')))
        except Exception as error:   # noqa: BLE001
            counts[name] = f'unavailable ({error})'
    return counts


def compute_table(run, manifest, gpu_log):
    timings = json.loads((run/'phase_timings.json').read_text()) if (run/'phase_timings.json').is_file() else {}
    native = int(manifest.frame_count.sum())
    sampled = None
    try:
        sampled = len(pd.read_csv(run/'outputs/recovery/recovery_results.csv', usecols=['frame_index']))
    except (FileNotFoundError, ValueError):
        pass
    rows = []
    for phase, seconds in timings.items():
        rows.append(dict(phase=phase, wall_time_sec=round(seconds, 1),
                         sampled_fps=round(sampled / seconds, 2) if sampled and seconds else np.nan,
                         native_fps_equivalent=round(native / seconds, 2) if seconds else np.nan))
    total = sum(timings.values())
    rows.append(dict(phase='TOTAL (both branches, incl. model loading)', wall_time_sec=round(total, 1),
                     sampled_fps=round(sampled / total, 2) if sampled and total else np.nan,
                     native_fps_equivalent=round(native / total, 2) if total else np.nan))
    peak = np.nan
    if gpu_log and Path(gpu_log).is_file():
        log = pd.read_csv(gpu_log, header=None, names=['time', 'memory', 'util'])
        peak = log.memory.str.extract(r'(\d+)')[0].astype(float).max()
    return pd.DataFrame(rows), dict(native_frames=native, sampled_frames=sampled, peak_gpu_memory_mib=peak)


def perception_table(out, manifest):
    split = manifest.set_index('video_id').split
    parts = []
    fusion = pd.read_csv(out/'fusion/fusion_metrics.csv')
    fusion = fusion[fusion.scope == 'video']
    for method, g in fusion.groupby('method'):
        parts.append((f'Detection: {method}', g.set_index('video_id')[['detection_rate', 'longest_consecutive_miss', 'number_of_detection_gaps']]))
    tracking = pd.read_csv(out/'tracking/bytetrack/tracking_metrics.csv')
    tracking = tracking[tracking.scope == 'video'].set_index('video_id')
    parts.append(('Tracking: ByteTrack', tracking[['track_coverage', 'number_of_fragments', 'observed_primary_id_changes']]))
    recovery = pd.read_csv(out/'recovery/recovery_metrics.csv')
    recovery = recovery[recovery.scope == 'video'].set_index('video_id')
    parts.append(('Bounded recovery', recovery[['detector_miss_rate', 'recovered_miss_rate', 'recovered_frames']]))
    sam2 = pd.read_csv(out/'sam2/sam2_metrics.csv').set_index('video_id')
    parts.append(('SAM 2 segmentation', sam2[['mask_observation_rate', 'mean_mask_quality', 'mean_temporal_consistency']]))
    for branch, label in (('pose', 'Pose (no SAM 2)'), ('pose_sam2', 'Pose (SAM 2-guided)')):
        pose = pd.read_csv(out/branch/'pose_results.csv', usecols=['video_id', 'pose_status', 'confidence'])
        g = pose.groupby('video_id')
        parts.append((label, pd.DataFrame(dict(usable_pose_rate=g.pose_status.apply(lambda s: s.isin(['VALID', 'FLAGGED']).mean()),
                                               strict_valid_pose_rate=g.pose_status.apply(lambda s: (s == 'VALID').mean()),
                                               mean_keypoint_confidence=g.confidence.mean()))))
    rows = []
    for component, table in parts:
        for metric in table.columns:
            values = pd.to_numeric(table[metric], errors='coerce')
            row = dict(component=component, metric=metric, all=mean_sd(values, 3))
            for s in ('train', 'validation', 'test'):
                row[s] = mean_sd(values[split.reindex(values.index) == s], 3)
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 paper evaluation (label-free + optional expert labels).')
    parser.add_argument('--run', default=str(ROOT/'runs/pilot24_paper'))
    parser.add_argument('--output-dir', default=str(ROOT/'reports/paper'))
    parser.add_argument('--gpu-log', default=str(ROOT/'runs/gpu_monitor_pilot24.csv'))
    parser.add_argument('--expert-labels', help='CSV: video_id, frame_index, roi_name, valid (0/1)[, best (0/1)]')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    run, output = Path(args.run), Path(args.output_dir)
    out = run/'outputs'
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run/'manifest.csv')
    cand0, cand1 = load_branch(out, 'confidence'), load_branch(out, 'confidence_sam2')
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = [(r.video_id, r._asdict(), str(out), cand0.get(r.video_id, pd.DataFrame()), cand1.get(r.video_id, pd.DataFrame()),
             recovery[recovery.video_id == r.video_id], sam2[sam2.video_id == r.video_id])
            for r in manifest.itertuples() if r.video_id in cand0 and r.video_id in cand1]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate_video, jobs))
    per_video = pd.DataFrame([row for r in results for row in r[0]])
    windows = pd.DataFrame([row for r in results for row in r[1]])
    roi = pd.DataFrame([row for r in results for row in r[2]])
    frames = pd.DataFrame([row for r in results for row in r[3]])
    for name, table in (('per_video_metrics', per_video), ('window_metrics', windows), ('roi_window_q_snr', roi),
                        ('sampled_frame_descriptors', frames)):
        table.to_csv(output/f'{name}.csv', index=False)

    # Thresholds fixed on training videos only.
    train_w, train_roi, train_f = windows[windows.split == 'train'], roi[roi.split == 'train'], frames[frames.split == 'train']
    thresholds = dict(
        informative_snr_db=float(np.nanmedian(train_roi.snr)) if len(train_roi) else 0.,
        motion_tertiles=tertiles(train_w.motion), occlusion_tertiles=tertiles(train_w.occlusion),
        swaddling_tertiles=tertiles(train_w.swaddling), luminance_tertiles=tertiles(train_w.luminance),
        luminance_change_p90=float(np.nanpercentile(train_w.luminance_change, 90)),
        lum_p05=float(np.nanpercentile(train_f.luminance, 5)), motion_p95=float(np.nanpercentile(train_f.motion, 95)),
        occlusion_p95=float(np.nanpercentile(train_f.occlusion, 95)),
        q_low=float(np.nanpercentile(train_f.best_q, 25)))
    tau = thresholds['informative_snr_db']
    for c in CONFIGS:
        per_video.loc[per_video.config == c, 'informative_window_percent'] = per_video[per_video.config == c].video_id.map(
            windows.groupby('video_id')[f'snr_{c}'].apply(lambda s: 100. * (s >= tau).mean()))
    per_video.to_csv(output/'per_video_metrics.csv', index=False)

    metrics = list(GEOMETRY) + ['pulse_snr_db', 'informative_window_percent']
    digits = dict(coverage_percent=1, temporal_iou=3, centroid_jitter=3, selection_jaccard=3, switches_per_sec=2,
                  mask_support=3, pulse_snr_db=2, informative_window_percent=1)
    summary_table(per_video, metrics, digits).to_csv(output/'table3_main_all_videos.csv', index=False)
    summary_table(per_video[per_video.split == 'test'], metrics, digits).to_csv(output/'table3_main_test_split.csv', index=False)
    summary_table(per_video[per_video.split != 'train'], metrics, digits).to_csv(output/'table3_main_heldout_val_test.csv', index=False)
    comparisons(per_video, metrics).to_csv(output/'table3_paired_tests_vs_full.csv', index=False)
    comparisons(per_video, metrics, reference='B4').to_csv(output/'table3_paired_tests_vs_B4.csv', index=False)

    stratified(windows, 'motion', thresholds['motion_tertiles'], ['low', 'medium', 'high'], tau).to_csv(
        output/'table4_motion_stratified.csv', index=False)
    conditions = []
    for column, edges, labels, title in (
            ('occlusion', thresholds['occlusion_tertiles'], ['low', 'partial', 'strong'], 'Occlusion proxy'),
            ('luminance', thresholds['luminance_tertiles'], ['low light', 'mid', 'bright'], 'Illumination'),
            ('swaddling', thresholds['swaddling_tertiles'], ['low', 'mid', 'swaddled'], 'Swaddling proxy')):
        table = stratified(windows, column, edges, labels, tau)
        table.insert(0, 'condition', title)
        conditions.append(table)
    transition = windows.assign(transition=np.where(windows.luminance_change > thresholds['luminance_change_p90'], 1., 0.))
    table = stratified(transition, 'transition', (.5,), ['stable', 'transition'], tau)
    table.insert(0, 'condition', 'Illumination transition')
    conditions.append(table)
    pd.concat(conditions).to_csv(output/'table5_condition_stratified.csv', index=False)

    calibrators = {}
    for c in SCORED:
        part = train_roi[train_roi.config == c]
        if len(part) >= 10:
            calibrators[c] = platt(part.q01, part.snr >= tau)
    q_validity(roi, tau, calibrators).to_csv(output/'table_q_validity.csv', index=False)

    # Held-out-split generalization: split-wise metrics + train-fitted (weakly supervised) weights.
    split_rows = []
    for split in ('train', 'validation', 'test'):
        part = per_video[per_video.split == split]
        for c in CONFIGS:
            g = part[part.config == c]
            split_rows.append(dict(split=split, configuration=NAMES[c], videos=len(g),
                                   **{m: mean_sd(g[m], digits[m]) for m in ('temporal_iou', 'centroid_jitter', 'pulse_snr_db',
                                                                           'informative_window_percent')}))
    pd.DataFrame(split_rows).to_csv(output/'table6_split_generalization.csv', index=False)
    fitted = generalization(out, per_video, manifest)
    (output/'table6_weight_regimes.json').write_text(json.dumps({k: v[1] for k, v in fitted.items()}, indent=2, default=str))

    fps_sampled = float(per_video.fps_sampled.median())
    failure_analysis(frames, thresholds, fps_sampled).to_csv(output/'table8_failure_cases.csv', index=False)
    perception_table(out, manifest).to_csv(output/'table_perception_diagnostics.csv', index=False)
    compute, compute_meta = compute_table(run, manifest, args.gpu_log)
    compute.to_csv(output/'table_compute.csv', index=False)
    compute_meta['parameters'] = {k: (int(v) if not isinstance(v, str) else v) for k, v in model_parameters().items()}

    dataset = manifest.groupby('split').agg(videos=('video_id', 'count'), frames=('frame_count', 'sum'),
                                            duration_sec=('duration_sec', 'sum'), median_fps=('fps', 'median'))
    dataset.to_csv(output/'table_dataset.csv')

    if args.expert_labels:
        labels = pd.read_csv(args.expert_labels)
        print('Expert labels supplied; compute expert metrics with module1.paper_eval.expert_metrics on per-frame selections.')
        labels.to_csv(output/'expert_labels_used.csv', index=False)

    metadata = dict(run=str(run), thresholds={k: (list(v) if isinstance(v, tuple) else v) for k, v in thresholds.items()},
                    videos=int(per_video.video_id.nunique()), windows=len(windows), roi_windows=len(roi),
                    hr_config=dict(band_bpm=[HR.hr_min_bpm, HR.hr_max_bpm], window_sec=HR.estimation_window_sec,
                                   step_sec=HR.estimation_step_sec),
                    calibrators={k: list(v) for k, v in calibrators.items()}, compute=compute_meta,
                    ground_truth='NONE: all metrics are label-free diagnostics or signal-quality proxies',
                    statistical_unit='video', tests='Wilcoxon signed-rank, Holm-corrected per metric; 95% percentile bootstrap CI')
    (output/'evaluation_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
