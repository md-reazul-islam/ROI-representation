"""HR branch of Module 2 v2.

Pixels: Module 1 v2 refined pixels (pixel polygon of the region, i.e. 2-point boxes read as rectangles, inside the
reliable SAM 2 infant mask, adaptive skin), per native frame, geometry held between sampled frames (<= 0.25 s).
Fusion: the Module 1 v2 stabilized selection and fusion weights (no same-window re-selection by SNR).
Per 8 s window (1 s step): HR = interpolated FFT peak (90-270 bpm), SNR, SQI; pulse-present test against
block-shuffled surrogates of the same pixels fused with the same weights; split-half agreement of the left and
right halves of every selected region (primary null: block shuffling; see analyse); Kalman track over validated windows only.
"""
import json
import math

import cv2
import numpy as np

from module1 import paper_eval as pe
from module1_v2 import baselines as bl
from module1_v2.selection import NAMES
from module2.extraction import hold_indices, mean_rgb
from module2.rppg import analysis_windows, estimate_hr, kalman_smooth, signal_quality
from hr_rr_m1v2 import validity as va

SIDES = ('', 'L', 'R')
SIDES_TB = ('', 'L', 'R', 'T', 'B')     # + top / bottom halves (gate2_mode: fallback, pair top_bottom)


def unit_index(sides=SIDES):
    return {(r, s): i for i, (r, s) in enumerate((r, s) for r in NAMES for s in sides)}


def clip_tb(polygon, side):
    """Part of a polygon above ('T') or below ('B') its own horizontal midline: Module 1 v2's clip_half on swapped
    axes, so the two halves are disjoint pixel sets exactly like the left / right halves."""
    p = np.asarray(polygon, float)
    if len(p) < 3:
        return None
    half = bl.clip_half(p[:, ::-1], 'L' if side == 'T' else 'R')
    return None if half is None else half[:, ::-1]


def half_polygon(polygon, side):
    if side == '':
        return polygon
    return bl.clip_half(polygon, side) if side in ('L', 'R') else clip_tb(polygon, side)


def extract(record, rows, masks, extract_config, hold, counts=None, sides=SIDES):
    """(n, 17 * len(sides), 3) mean skin colour per region and half, per native frame (counts: optional skin pixels)."""
    index = unit_index(sides)
    n = int(record['frame_count'])
    rgb = np.full((n, len(index), 3), np.nan)
    by_frame = {int(f): g for f, g in rows[rows.available].groupby('frame_index')}
    capture = cv2.VideoCapture(str(record['video_path']))
    for i in range(n):
        ok, image = capture.read()
        if not ok:
            break
        source = int(hold[i])
        part = by_frame.get(source) if source >= 0 else None
        if part is None:
            continue
        mask = masks.get(source)
        if mask is not None and mask.shape == image.shape[:2]:
            image = image.copy()
            image[~mask] = 0                    # below dark_level: excluded by mean_rgb (pixel refinement)
        for row in part.itertuples():
            polygon = json.loads(row.pixel_polygon) if isinstance(row.pixel_polygon, str) else None
            if not polygon:
                continue
            for side, poly in ((s, half_polygon(polygon, s)) for s in sides):
                if poly is None or len(poly) < 3:
                    continue
                value, count = mean_rgb(image, np.asarray(poly, float), extract_config)
                if counts is not None:
                    counts[i, index[(row.roi_name, side)]] = count
                if value is not None:
                    rgb[i, index[(row.roi_name, side)]] = value
    capture.release()
    return rgb


def native_weights(rows, hold, n):
    fusion = {int(f): [(r.roi_name, r.weight, None, None) for r in g.itertuples() if r.weight > 0]
              for f, g in rows[rows.selected].groupby('frame_index')}
    return pe.native_weights(fusion, NAMES, hold, n)


def analyse(record, rows, masks, cfg, hr_config, extract_config, pulse, seed, waves=None):
    """Window table of one video. With `waves` (dict), the fused pulse at the native frame rate is stored in it.
    cfg['region_set']: 'module1' (default: the Module 1 v2 selection and weights) or 'hr_specific' (hr_specific.py)."""
    if cfg.get('region_set', 'module1') == 'hr_specific':
        from hr_rr_m1v2 import hr_specific
        return hr_specific.analyse(record, rows, masks, cfg, hr_config, extract_config, pulse, seed, extract, unit_index, waves)
    fps, n = float(record['fps']), int(record['frame_count'])
    frames = sorted(rows.frame_index.unique())
    hold = hold_indices(frames, n, fps, extract_config.max_hold_sec)
    rgb = extract(record, rows, masks, extract_config, hold)
    index = unit_index()
    full = [index[(r, '')] for r in NAMES]
    pulses = np.column_stack([pulse(rgb[:, j], fps) for j in range(rgb.shape[1])])
    W = native_weights(rows, hold, n)
    fused = {s: pe.fused_pulse(pulses[:, [index[(r, s)] for r in NAMES]], W) for s in SIDES}
    windows = analysis_windows(n, fps, hr_config)
    motion = rows[rows.selected].groupby('frame_index').motion_magnitude.mean()
    motion_native = np.array([motion.get(int(hold[i]), np.nan) if hold[i] >= 0 else np.nan for i in range(n)])

    # surrogate nulls on the same pixels, fused with the same weights. Primary: block shuffling in time (destroys the
    # rhythm, keeps the colour distribution); validated on synthetic video (true pulse passes, static patch does not).
    # Diagnostic only: per-channel phase randomization, which keeps every channel's spectrum and therefore also keeps
    # a genuine pulse line, so it rejects real pulses (it tests cross-channel coherence, not pulse presence).
    used = [j for j, r in zip(full, NAMES) if W[:, NAMES.index(r)].any()]
    block = max(1, int(round(cfg.get('shuffle_block_sec', .25) * fps)))
    vid = int(record['video_id'][-3:]) if record['video_id'][-3:].isdigit() else 0
    thresholds = {}
    for kind in ('shuffle', 'phase'):
        null = []
        count = cfg['surrogates'] if kind == cfg.get('null', 'shuffle') else cfg.get('diagnostic_surrogates', 19)
        for s in range(count):
            rng = np.random.default_rng([seed, vid, s, kind == 'phase'])
            sur = np.full((n, len(NAMES)), np.nan)
            for j, r in zip(full, NAMES):
                if j in used:
                    trace = bl.phase_surrogate(rgb[:, j], rng) if kind == 'phase' else bl.shuffle_surrogate(rgb[:, j], rng, block)
                    sur[:, NAMES.index(r)] = pulse(trace, fps)
            f = pe.fused_pulse(sur, W)
            for a, b in windows:
                _, value = estimate_hr(f[a:b], fps, hr_config)
                null.append(value if value is not None else np.nan)
        null = np.asarray(null, float)
        thresholds[kind] = np.nanpercentile(null, cfg['null_percentile']) if np.isfinite(null).any() else np.nan
    threshold = thresholds[cfg.get('null', 'shuffle')]

    out = []
    for w, (a, b) in enumerate(windows):
        hr, snr = estimate_hr(fused[''][a:b], fps, hr_config)
        hl, _ = estimate_hr(fused['L'][a:b], fps, hr_config)
        hrr, _ = estimate_hr(fused['R'][a:b], fps, hr_config)
        mot = np.nanmean(motion_native[a:b]) if np.isfinite(motion_native[a:b]).any() else None
        sqi = signal_quality(snr, mot, hr_config) if snr is not None else 0.
        is_present = snr is not None and math.isfinite(threshold) and snr > threshold
        agreement = bl.agree(hl, hrr, cfg['agreement_bpm'])
        out.append(dict(window=w, window_start_sec=a / fps, window_end_sec=b / fps, window_center_sec=(a + b) / 2 / fps,
                        hr_bpm=hr, snr_db=snr, sqi=sqi, motion=mot, null_threshold_db=threshold, pulse_present=is_present,
                        pulse_present_phase_null=bool(snr is not None and math.isfinite(thresholds['phase']) and snr > thresholds['phase']),
                        hr_left_bpm=hl, hr_right_bpm=hrr, halves_agree=agreement, label=va.label(is_present, agreement)))
    kal = va.smoothed([r['window_end_sec'] for r in out], [r['hr_bpm'] for r in out], [r['label'] for r in out],
                      [r['sqi'] for r in out], hr_config, kalman_smooth)
    for r, k in zip(out, kal):
        r['hr_kalman_bpm'] = k
    if waves is not None:
        waves.update(pulse=fused[''], pulse_fps=fps)
    return out
