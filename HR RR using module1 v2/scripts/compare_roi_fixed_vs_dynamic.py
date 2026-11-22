"""Fixed vs dynamic ROI comparison on the SAME HR chain, reported per video condition, with or without a reference HR.

Only the ROI differs between methods. Every method gets the same pixel rule (adaptive skin, saturation / dark
exclusion), the same POS pulse, the same 8 s / 1 s windows, the same gate 1 (SNR above the 95th percentile of
block-shuffled surrogates of the same pixels and procedure), gate 2 (two disjoint halves agree within 6 bpm) and the
same Kalman track over validated windows. D5 and D5_HR call the unchanged HR branch (hr_rr_m1v2.hr.analyse), so their
windows are identical to the infant-notebook pipeline.

Methods (configs/roi_comparison.yaml)
  B0          fixed: centre 50 % of the first infant box, whole video     (fixed box, no tracking)
  B0_WIN      piecewise fixed: centre 50 % of the infant box at the START of each 8 s window, held for that window
  B1          fixed: first-frame best anatomical region, whole video      (fixed anatomical box)
  FACE_TRACK  tracked face box: bounding rectangle of the FACE region per sampled frame (standard rPPG practice)
  M0          whole infant mask x skin inside the tracked infant box      (no anatomy, no scoring)
  D5          Module 1 v2 dynamic ROI, stabilized top-3 selection and weights (selection may change inside a window)
  D5_WIN      D5 regions tracked every frame, but selection and weights FROZEN per 8 s window (window-mean D5 weights,
              top 3). D5_WIN - D5 = effect of switching inside a window. D5's colour-free selection uses no pulse
              information, so freezing it with the window's own weights does not leak the outcome.
  D5_HR       Module 1 v2 dynamic ROI + HR-specific regions (infant notebook; gate 2 fallback by default)
  BG          NEGATIVE CONTROL: frame-corner background patches (no skin filter). Expected gate-1 rate ~5 %.

Effects separated in report.md (paired per video, per condition)
  switching   D5_WIN - D5         tracking   B0_WIN - B0 and FACE_TRACK - B0
  selection   D5_HR - D5_WIN      overall    D5_HR - B0
  Diagnostic groups diag:d5_switched / diag:d5_no_switch split windows by whether D5 switched regions inside them.

Conditions (per window, automatic, no labels needed; optional manual labels add 'manual:<name>' groups)
  moving, camera_motion, posture_change, covered_low_skin, occluded_few_regions, illumination_change, dark, blurry,
  caregiver (if detect_caregiver_contact.py was run), clean (no automatic flag), all.

Without a reference: coverage (% reported), gate-1 pass %, gate-1 pass above the background control, median SNR,
split-half agreement, HR stability. These say whether a method measures a PULSE, not whether its HR is RIGHT.
With --reference-hr (csv: video_id,timestamp_sec,hr_bpm): MAE, RMSE, % within +-5 bpm, bias, Pearson r, on the
reported windows and on all windows, next to coverage (a method can look accurate by reporting less).

Usage (from the Module1 root)
  python "HR RR using module1 v2/scripts/compare_roi_fixed_vs_dynamic.py" --run "runs/finalcheck HR RR_25a9a4f1"
  python ... --run runs/ubfc_flat_38d8f553 --hr-band 45,180 \
      --reference-hr "HR RR using module1 v2/runs/ubfc_flat_38d8f553/ubfc_reference/reference_hr_oximeter.csv"
  options: --conditions labels.csv (video_id,start_sec,end_sec,condition)  --videos ID ...  --methods B0,D5_HR,BG
           --reference-offset-sec 0.0  --workers 2  --out-name roi_comparison  --resummarize
Needs: Module 1 base run (runs/<run>) and module1_v2/runs/<run>/dynamic_roi_v2_D5/dynamic_roi_v2.csv (the infant
notebook makes both). Writes HR RR using module1 v2/runs/<run>/<out-name>/.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy import stats

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(p))
import run_paper_evaluation as rpe                                  # noqa: E402  (Module 1 pixel / pulse helpers)
from module1 import paper_eval as pe                                # noqa: E402
from module1_v2 import baselines as bl                              # noqa: E402
from module1_v2.motion import background_mask                       # noqa: E402
from module1_v2.selection import NAMES                              # noqa: E402
from module2.ablation import bootstrap_ci, holm, paired_test        # noqa: E402
from module2.extraction import hold_indices, mean_rgb               # noqa: E402
from module2.rppg import HRConfig, analysis_windows, estimate_hr, kalman_smooth, signal_quality  # noqa: E402
from hr_rr_m1v2 import hr as hr_branch, validity as va              # noqa: E402

HR_CFG = yaml.safe_load((M2/'configs/hr_rr_m1v2.yaml').read_text())
CMP = yaml.safe_load((M2/'configs/roi_comparison.yaml').read_text())
BG_DILATION = yaml.safe_load((ROOT/'module1_v2/configs/module1_v2.yaml').read_text())['phase1']['background_dilation_px']
SIDES = ('', 'L', 'R')
CORNERS = ('TL', 'TR', 'BL', 'BR')
CORNER_FRACTION, BG_MIN_PIXELS = .2, 200          # as the Phase 2 negative control
TRACE_METHODS = ('B0', 'B1', 'FACE_TRACK', 'M0', 'BG')
ALL_METHODS = TRACE_METHODS + ('B0_WIN', 'D5', 'D5_WIN', 'D5_HR')
REPORTED = ('validated', 'validated_one_side')
HR = HRConfig()


def set_band(band):
    """Adult / infant HR search band, also applied to the POS band-pass (as run_vitals_v2_adult.py)."""
    global HR
    if band:
        low, high = band
        HR = dataclasses.replace(HRConfig(), hr_min_bpm=low, hr_max_bpm=high)
        rpe.HR = HR


# ------------------------------------------------------------------ pixels of the trace methods + frame conditions
def unit_keys():
    keys = [(m, s) for m in ('B0', 'B1', 'FACE_TRACK', 'M0') for s in SIDES] + [('BG', c) for c in CORNERS]
    return {k: i for i, k in enumerate(keys)}


def bounding_rect(polygon):
    p = np.asarray(polygon, float)
    (x1, y1), (x2, y2) = p.min(axis=0), p.max(axis=0)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] if x2 > x1 and y2 > y1 else None


def centre_box(box, fraction=.5):
    """pe.fixed_rois B0 geometry: centre `fraction` of an infant box."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * (1 - fraction) / 2, (y2 - y1) * (1 - fraction) / 2
    return [[x1 + dx, y1 + dy], [x2 - dx, y1 + dy], [x2 - dx, y2 - dy], [x1 + dx, y2 - dy]]


def window_boxes(boxes, hold, windows):
    """B0_WIN geometry: B0 box placed on the infant box at the start of every window (first box inside it, else None)."""
    sampled = np.array(sorted(f for f, b in boxes.items() if b is not None), int)
    out = []
    for a, b in windows:
        box = boxes.get(int(hold[a])) if hold[a] >= 0 else None
        if box is None:
            inside = sampled[(sampled >= a) & (sampled < b)]
            box = boxes[int(inside[0])] if len(inside) else None
        out.append(centre_box(box) if box is not None else None)
    return out


def extract_traces(record, rows, cand, recovery, masks, hold, windows=(), win_box=False):
    """(n, units, 3) mean colour of the trace methods, plus per-frame subject luminance and sharpness.
    win_box: also B0_WIN, one (segment, 3 sides, 3) trace per window over the window +- b0_win_context_sec (the box is
    held over that segment; the context only lets POS / detrending / band-pass settle, HR uses the window itself)."""
    n, index = int(record['frame_count']), unit_keys()
    rgb = np.full((n, len(index), 3), np.nan)
    lum = np.full(n, np.nan)
    sharp = np.full(n, np.nan)
    boxes = rpe.boxes_of(recovery)
    b0, b1 = pe.fixed_rois(cand, boxes)
    fixed = {'B0': b0, 'B1': b1[1] if b1 else None}
    face = {int(r.frame_index): bounding_rect(json.loads(r.pixel_polygon)) for r in
            rows[(rows.roi_name == 'FACE') & rows.available & rows.pixel_polygon.notna()].itertuples()}
    background = {}
    use_mask = CMP.get('fixed_methods_use_mask', False)
    fps = float(record['fps'])
    pad = int(round(CMP.get('b0_win_context_sec', 2.) * fps))
    segs = [(max(0, a - pad), min(n, b + pad)) for a, b in windows] if win_box else []
    seg_start, seg_end = np.array([s for s, _ in segs], int), np.array([e for _, e in segs], int)
    win_poly = window_boxes(boxes, hold, windows) if win_box else []
    win_rgb = [np.full((e - s0, len(SIDES), 3), np.nan) for s0, e in segs]

    def side_values(image, polygon, box_ok=False):
        if polygon is None or len(polygon) < (2 if box_ok else 3):
            return {}
        out = {}
        for side in SIDES:
            poly = polygon if side == '' else (bl.clip_half(polygon, side) if len(polygon) >= 3 else None)
            if poly is None or len(poly) < 3:
                continue
            value, _ = mean_rgb(image, np.asarray(poly, float), rpe.EXTRACT)
            if value is not None:
                out[side] = value
        return out

    def put(image, polygon, method, i, box_ok=False):
        for side, value in side_values(image, polygon, box_ok).items():
            rgb[i, index[(method, side)]] = value

    capture = cv2.VideoCapture(str(record['video_path']))
    for i in range(n):
        ok, image = capture.read()
        if not ok:
            break
        source = int(hold[i])
        mask = masks.get(source) if source >= 0 else None
        box = boxes.get(source) if source >= 0 else None
        masked = image
        if mask is not None and mask.shape == image.shape[:2]:
            masked = image.copy()
            masked[~mask] = 0                                  # below dark_level: excluded by mean_rgb
        plain = masked if use_mask else image
        for name, polygon in fixed.items():
            put(plain, polygon, name, i, box_ok=True)
        if win_box:
            for w in np.flatnonzero((seg_start <= i) & (seg_end > i)):
                for side, value in side_values(plain, win_poly[w], box_ok=True).items():
                    win_rgb[w][i - seg_start[w], SIDES.index(side)] = value
        if source < 0:
            continue
        put(plain, face.get(source), 'FACE_TRACK', i)
        if box is not None:
            put(masked, bl.box_polygon(box), 'M0', i)
        if source not in background:
            background[source] = background_mask(image.shape[:2], mask, box, BG_DILATION)
        bgm = background[source]
        if bgm is not None:
            h, w = image.shape[:2]
            ph, pw = int(round(h * CORNER_FRACTION)), int(round(w * CORNER_FRACTION))
            for name, (y0, x0) in zip(CORNERS, ((0, 0), (0, w - pw), (h - ph, 0), (h - ph, w - pw))):
                crop = image[y0:y0 + ph, x0:x0 + pw]
                bright = crop.max(axis=2)
                ok_px = bgm[y0:y0 + ph, x0:x0 + pw] & (bright < rpe.EXTRACT.saturation_level) & (bright > rpe.EXTRACT.dark_level)
                if ok_px.sum() >= BG_MIN_PIXELS:
                    b, g, r = crop[ok_px].astype(float).mean(axis=0)
                    rgb[i, index[('BG', name)]] = (r, g, b)
        # subject luminance / sharpness (inside the infant box; whole frame if no box)
        h, w = image.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3])))
        else:
            x1, y1, x2, y2 = 0, 0, w, h
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        lum[i] = float(gray.mean())
        if i % CMP['sharpness_every_n_frames'] == 0:
            width = CMP['sharpness_width_px']
            small = cv2.resize(gray, (width, max(8, int(round(gray.shape[0] * width / gray.shape[1])))))
            sharp[i] = float(cv2.Laplacian(small, cv2.CV_64F).var())
    capture.release()
    return rgb, lum, sharp, index, (segs, win_rgb)


def method_columns(method, index):
    if method == 'BG':
        return {'': [index[('BG', c)] for c in CORNERS], 'L': [index[('BG', c)] for c in ('TL', 'BL')],
                'R': [index[('BG', c)] for c in ('TR', 'BR')]}
    return {s: [index[(method, s)]] for s in SIDES}


def mean_trace(pulses, cols):
    block = pulses[:, cols]
    ok = np.isfinite(block)
    out = np.full(len(block), np.nan)
    any_ok = ok.any(axis=1)
    out[any_ok] = np.where(ok, block, 0.).sum(axis=1)[any_ok] / ok.sum(axis=1)[any_ok]
    return out


def analyse_trace_method(method, rgb, index, windows, fps, motion_native, seed, video):
    """Same window procedure as hr_rr_m1v2.hr.analyse (module1 path), for one fixed / tracked / control ROI."""
    cfg = HR_CFG['hr']
    cols = method_columns(method, index)
    pulses = np.full(rgb.shape[:2], np.nan)
    used = sorted({j for c in cols.values() for j in c})
    for j in used:
        pulses[:, j] = rpe.pulse(rgb[:, j], fps)
    trace = {s: mean_trace(pulses, c) for s, c in cols.items()}
    block = max(1, int(round(cfg.get('shuffle_block_sec', .25) * fps)))
    null = []
    for s in range(cfg['surrogates']):
        rng = np.random.default_rng([seed, zlib.crc32(video.encode()), s, zlib.crc32(method.encode())])
        sur = np.full(rgb.shape[:2], np.nan)
        for j in cols['']:
            sur[:, j] = rpe.pulse(bl.shuffle_surrogate(rgb[:, j], rng, block), fps)
        f = mean_trace(sur, cols[''])
        for a, b in windows:
            _, v = estimate_hr(f[a:b], fps, HR)
            null.append(v if v is not None else np.nan)
    values = [(*estimate_hr(trace[''][a:b], fps, HR), estimate_hr(trace['L'][a:b], fps, HR)[0],
               estimate_hr(trace['R'][a:b], fps, HR)[0]) for a, b in windows]
    return finish(values, null, windows, fps, motion_native)


def finish(values, null, windows, fps, motion_native, extra=None):
    """Labels (gate 1 vs the pooled surrogate null, gate 2 left / right) and Kalman, as hr_rr_m1v2.hr.analyse.
    values: per window (hr, snr, hr_left, hr_right)."""
    cfg = HR_CFG['hr']
    null = np.asarray(null, float)
    threshold = np.nanpercentile(null, cfg['null_percentile']) if np.isfinite(null).any() else np.nan
    out = []
    for w, ((a, b), (hr, snr, hl, hrr)) in enumerate(zip(windows, values)):
        mot = np.nanmean(motion_native[a:b]) if np.isfinite(motion_native[a:b]).any() else None
        present = snr is not None and math.isfinite(threshold) and snr > threshold
        agreement = bl.agree(hl, hrr, cfg['agreement_bpm'])
        out.append(dict(window=w, hr_bpm=hr, snr_db=snr, sqi=signal_quality(snr, mot, HR) if snr is not None else 0.,
                        null_threshold_db=threshold, pulse_present=present, halves_agree=agreement,
                        label=va.label(present, agreement), **(extra[w] if extra else {})))
    kal = va.smoothed([b / fps for _, b in windows], [r['hr_bpm'] for r in out], [r['label'] for r in out],
                      [r['sqi'] for r in out], HR, kalman_smooth)
    for r, k in zip(out, kal):
        r['hr_kalman_bpm'] = k
    return out


def analyse_b0_win(segs, win_rgb, windows, fps, motion_native, seed, video):
    """B0_WIN: per window its own held box; pulse over the window +- context, HR / gates on the window."""
    cfg = HR_CFG['hr']
    block = max(1, int(round(cfg.get('shuffle_block_sec', .25) * fps)))
    values, null = [], []
    for w, ((a, b), (s0, _), seg) in enumerate(zip(windows, segs, win_rgb)):
        p = [rpe.pulse(seg[:, k], fps)[a - s0:b - s0] for k in range(len(SIDES))]
        hr, snr = estimate_hr(p[0], fps, HR)
        values.append((hr, snr, estimate_hr(p[1], fps, HR)[0], estimate_hr(p[2], fps, HR)[0]))
        for s in range(cfg['surrogates']):
            rng = np.random.default_rng([seed, zlib.crc32(video.encode()), s, zlib.crc32(b'B0_WIN'), w])
            y = rpe.pulse(bl.shuffle_surrogate(seg[:, 0], rng, block), fps)[a - s0:b - s0]
            _, v = estimate_hr(y, fps, HR)
            null.append(v if v is not None else np.nan)
    return finish(values, null, windows, fps, motion_native)


def frozen_weights(W, windows, k):
    """Per window: mean D5 weight of every region over the window, top k kept, renormalized (constant in the window)."""
    out = []
    for a, b in windows:
        m = W[a:b].mean(axis=0)
        keep = [j for j in np.argsort(-m, kind='stable')[:k] if m[j] > 0]
        v = np.zeros(W.shape[1])
        v[keep] = m[keep]
        out.append(v / v.sum() if v.sum() > 0 else v)
    return out


def d5_switches(rows, windows):
    """Changes of the D5 selected region set between consecutive sampled frames inside each window."""
    sets = rows[rows.selected].groupby('frame_index').roi_name.apply(frozenset)
    frames = np.array(sets.index, int)
    out = []
    for a, b in windows:
        seq = list(sets[(frames >= a) & (frames < b)])
        out.append(int(sum(x != y for x, y in zip(seq, seq[1:]))))
    return out


def analyse_d5_win(record, rows, masks, hold, windows, fps, motion_native, seed, video):
    """D5_WIN: D5 pixels and tracking (hr_rr_m1v2.hr.extract), selection + weights frozen per window."""
    cfg = HR_CFG['hr']
    n = int(record['frame_count'])
    rgb = hr_branch.extract(record, rows, masks, rpe.EXTRACT, hold)
    index = hr_branch.unit_index()
    W = hr_branch.native_weights(rows, hold, n)
    frozen = frozen_weights(W, windows, CMP.get('d5_win_top_k', 3))
    used = sorted({j for v in frozen for j in np.flatnonzero(v)})
    cols = {s: [index[(NAMES[j], s)] for j in used] for s in SIDES}
    pulses = {s: np.column_stack([rpe.pulse(rgb[:, c], fps) for c in cols[s]]) if used else np.full((n, 0), np.nan)
              for s in SIDES}
    values, extra = [], []
    for (a, b), v in zip(windows, frozen):
        wv = np.tile(v[used], (b - a, 1))
        f = {s: pe.fused_pulse(pulses[s][a:b], wv) if used else np.full(b - a, np.nan) for s in SIDES}
        hr, snr = estimate_hr(f[''], fps, HR)
        values.append((hr, snr, estimate_hr(f['L'], fps, HR)[0], estimate_hr(f['R'], fps, HR)[0]))
        extra.append(dict(regions=json.dumps({NAMES[j]: round(float(v[j]), 3) for j in np.flatnonzero(v)})))
    block = max(1, int(round(cfg.get('shuffle_block_sec', .25) * fps)))
    null = []
    for s in range(cfg['surrogates']):
        rng = np.random.default_rng([seed, zlib.crc32(video.encode()), s, zlib.crc32(b'D5_WIN')])
        sur = np.column_stack([rpe.pulse(bl.shuffle_surrogate(rgb[:, c], rng, block), fps) for c in cols['']]) \
            if used else np.full((n, 0), np.nan)
        for (a, b), v in zip(windows, frozen):
            _, val = estimate_hr(pe.fused_pulse(sur[a:b], np.tile(v[used], (b - a, 1))), fps, HR) if used else (None, None)
            null.append(val if val is not None else np.nan)
    return finish(values, null, windows, fps, motion_native, extra)


# ------------------------------------------------------------------ window conditions
def window_conditions(record, rows, recovery, windows, fps, lum, sharp):
    cand = CMP['hr_candidates']
    boxes = rpe.boxes_of(recovery)
    av = rows[rows.available]
    per_frame = pd.DataFrame(index=sorted(rows.frame_index.unique()))
    per_frame['motion_body'] = av.groupby('frame_index').motion_magnitude.mean()
    g = rows.groupby('frame_index')[['global_dx', 'global_dy']].first()
    per_frame['motion_camera_px'] = np.hypot(g.global_dx, g.global_dy)
    hr_rows = av[av.roi_name.isin(cand)]
    per_frame['skin_visible'] = hr_rows.groupby('frame_index').visible_skin_proportion.mean()
    per_frame['regions_available'] = hr_rows.groupby('frame_index').roi_name.nunique().reindex(per_frame.index).fillna(0) / len(cand)
    box = pd.DataFrame({f: b for f, b in boxes.items() if b is not None}, index=['x1', 'y1', 'x2', 'y2']).T
    out = []
    for w, (a, b) in enumerate(windows):
        part = per_frame[(per_frame.index >= a) & (per_frame.index < b)]
        bx = box[(box.index >= a) & (box.index < b)] if len(box) else box
        posture = np.nan
        if len(bx) >= 2:
            cx, cy = (bx.x1 + bx.x2) / 2, (bx.y1 + bx.y2) / 2
            diag = np.hypot(bx.x2 - bx.x1, bx.y2 - bx.y1).median()
            aspect = np.log(((bx.x2 - bx.x1) / (bx.y2 - bx.y1).clip(lower=1)).clip(lower=1e-3))
            shift = np.hypot(cx - cx.iloc[0], cy - cy.iloc[0]).max() / max(diag, 1.)
            posture = float(max(shift, (aspect - aspect.iloc[0]).abs().max()))
        lw = lum[a:b]
        level = np.nanmean(lw) if np.isfinite(lw).any() else np.nan
        change = (np.nanmax(lw) - np.nanmin(lw)) / max(level, 1.) if np.isfinite(lw).sum() > 1 else np.nan
        sw = sharp[a:b]
        out.append(dict(window=w, window_start_sec=a / fps, window_end_sec=b / fps, window_center_sec=(a + b) / 2 / fps,
                        motion_body=part.motion_body.mean(), motion_camera_px=part.motion_camera_px.mean(),
                        posture_index=posture, skin_visible=part.skin_visible.mean(),
                        regions_available=part.regions_available.mean(), illumination_level=level,
                        illumination_change=change, sharpness=np.nanmedian(sw) if np.isfinite(sw).any() else np.nan))
    return out


# ------------------------------------------------------------------ one video
def video_job(job):
    record, rows, cand, recovery, sam2_rows, out, methods, band = job
    set_band(band)
    started = time.perf_counter()
    video, fps, n = record['video_id'], float(record['fps']), int(record['frame_count'])
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    masks = {}
    for f in rows.frame_index.unique():
        if len(sam2_rows) and f in sam2_rows.index:
            m = rpe.load_mask(Path(out)/'sam2', sam2_rows.loc[f])
            if m is not None:
                masks[int(f)] = m
    hold = hold_indices(sorted(rows.frame_index.unique()), n, fps, rpe.EXTRACT.max_hold_sec)
    windows = analysis_windows(n, fps, HR)
    rgb, lum, sharp, index, (segs, win_rgb) = extract_traces(record, rows, cand, recovery, masks, hold, windows,
                                                              win_box='B0_WIN' in methods)
    motion = rows[rows.selected].groupby('frame_index').motion_magnitude.mean()      # SQI motion, as hr.analyse
    motion_native = np.array([motion.get(int(hold[i]), np.nan) if hold[i] >= 0 else np.nan for i in range(n)])
    table = {}
    for m in methods:
        if m in TRACE_METHODS:
            res = analyse_trace_method(m, rgb, index, windows, fps, motion_native, HR_CFG['seed'], video)
        elif m == 'B0_WIN':
            res = analyse_b0_win(segs, win_rgb, windows, fps, motion_native, HR_CFG['seed'], video)
        elif m == 'D5_WIN':
            res = analyse_d5_win(record, rows, masks, hold, windows, fps, motion_native, HR_CFG['seed'], video)
        else:
            cfg = copy.deepcopy(HR_CFG['hr'])
            cfg['region_set'] = 'hr_specific' if m == 'D5_HR' else 'module1'
            if m == 'D5_HR':
                cfg['gate2_mode'] = CMP.get('d5_hr_gate2_mode', 'strict')
            res = hr_branch.analyse(record, rows, masks, cfg, HR, rpe.EXTRACT, rpe.pulse, HR_CFG['seed'])
        table[m] = res
    cond = window_conditions(record, rows, recovery, windows, fps, lum, sharp)
    for c, k in zip(cond, d5_switches(rows, windows)):
        c['d5_switches_in_window'] = k
    wide = []
    for w, c in enumerate(cond):
        row = dict(video_id=video, split=record['split'], **c)
        for m, res in table.items():
            r = res[w]
            row.update({f'{k}_{m}': r.get(k) for k in ('hr_bpm', 'snr_db', 'pulse_present', 'halves_agree', 'label',
                                                       'hr_kalman_bpm', 'null_threshold_db')})
            if m == 'D5_WIN':
                row['regions_D5_WIN'] = r.get('regions')
            if m == 'D5_HR':
                row['regions_D5_HR'] = r.get('hr_regions')
        wide.append(row)
    return wide, time.perf_counter() - started


# ------------------------------------------------------------------ conditions, reference, metrics
def flag_conditions(w, manual, caregiver):
    thresholds = {}
    flags = []
    for name, spec in CMP['conditions'].items():
        x = w[spec['measure']].astype(float)
        thr = spec.get('threshold')
        if thr is None:
            thr = float(np.nanpercentile(x, 100 / 3 if spec['hard'] == 'low' else 200 / 3)) if x.notna().any() else np.nan
        thresholds[name] = dict(measure=spec['measure'], hard=spec['hard'], threshold=thr,
                                source='fixed' if spec.get('threshold') is not None else 'tertile')
        w[f'cond_{name}'] = (x < thr) if spec['hard'] == 'low' else (x > thr)
        flags.append(f'cond_{name}')
    if caregiver is not None:
        w['cond_caregiver'] = False
        for r in caregiver.itertuples():
            s0 = getattr(r, 'pad_start_sec', r.start_sec) if pd.notna(getattr(r, 'pad_start_sec', np.nan)) else r.start_sec
            s1 = getattr(r, 'pad_end_sec', r.end_sec) if pd.notna(getattr(r, 'pad_end_sec', np.nan)) else r.end_sec
            hit = (w.video_id == r.video_id) & (w.window_end_sec > s0) & (w.window_start_sec < s1)
            w.loc[hit, 'cond_caregiver'] = True
        flags.append('cond_caregiver')
    w['cond_clean'] = ~w[flags].any(axis=1)
    if manual is not None:
        length = (w.window_end_sec - w.window_start_sec).clip(lower=1e-6)
        names = sorted(manual.condition.astype(str).unique())
        for c in names:
            col = f'manual_{c}'
            w[col] = False
            for r in manual[manual.condition.astype(str) == c].itertuples():
                overlap = (np.minimum(w.window_end_sec, r.end_sec) - np.maximum(w.window_start_sec, r.start_sec)).clip(lower=0)
                w.loc[(w.video_id == r.video_id) & (overlap / length >= CMP['manual_label_min_overlap']), col] = True
        w['manual_unlabelled'] = ~w[[f'manual_{c}' for c in names]].any(axis=1)
    return w, thresholds


def groups(w):
    out = {'all': np.ones(len(w), bool), 'clean': w.cond_clean.to_numpy(bool)}
    for c in w.columns:
        if c.startswith('cond_') and c != 'cond_clean':
            out[c[5:]] = w[c].to_numpy(bool)
        elif c.startswith('manual_'):
            out['manual:' + c[7:]] = w[c].to_numpy(bool)
    if 'd5_switches_in_window' in w:      # diagnostic (not a video condition; not used for 'clean')
        out['diag:d5_switched'] = w.d5_switches_in_window.fillna(0).to_numpy() > 0
        out['diag:d5_no_switch'] = ~out['diag:d5_switched']
    return out


def attach_reference(w, path, offset):
    ref = pd.read_csv(path)
    ref = ref.assign(timestamp_sec=ref.timestamp_sec + offset)
    w['hr_ref'] = np.nan
    w['ref_source'] = ''
    for video, idx in w.groupby('video_id').groups.items():
        r = ref[ref.video_id == video].sort_values('timestamp_sec')
        if not len(r):
            continue
        t, v = r.timestamp_sec.to_numpy(), r.hr_bpm.to_numpy(float)
        for i in idx:
            a, b, c = w.at[i, 'window_start_sec'], w.at[i, 'window_end_sec'], w.at[i, 'window_center_sec']
            inside = (t >= a) & (t < b) & np.isfinite(v)
            if inside.any():
                w.at[i, 'hr_ref'], w.at[i, 'ref_source'] = float(v[inside].mean()), 'window_mean'
            else:
                j = int(np.clip(np.searchsorted(t, c), 0, len(t) - 1))
                j = min((k for k in (j - 1, j) if 0 <= k < len(t)), key=lambda k: abs(t[k] - c))
                if abs(t[j] - c) <= CMP['reference']['nearest_max_sec'] and np.isfinite(v[j]):
                    w.at[i, 'hr_ref'], w.at[i, 'ref_source'] = float(v[j]), 'nearest'
    return w


def metrics(part, m, has_ref):
    n = len(part)
    if not n:
        return {}
    lab = part[f'label_{m}']
    rep = lab.isin(REPORTED).to_numpy()
    snr = pd.to_numeric(part[f'snr_db_{m}'], errors='coerce')
    agree = part[f'halves_agree_{m}'].dropna()
    hr = pd.to_numeric(part[f'hr_bpm_{m}'], errors='coerce').to_numpy(float)
    jumps = []
    for _, p in part.assign(_hr=hr, _rep=rep).groupby('video_id'):
        p = p[p._rep].sort_values('window')
        consecutive = p.window.diff() == 1
        jumps += list(p._hr.diff().abs()[consecutive])
    out = dict(windows=n, reported_percent=100 * rep.mean(),
               gate1_percent=100 * part[f'pulse_present_{m}'].astype(bool).mean(),
               snr_median_db=float(snr.median()) if snr.notna().any() else np.nan,
               agreement_percent=100 * agree.astype(bool).mean() if len(agree) else np.nan,
               agreement_testable_percent=100 * len(agree) / n,
               hr_jump_median_bpm=float(np.median(jumps)) if jumps else np.nan)
    if 'pulse_present_BG' in part:
        out['gate1_above_BG_pp'] = out['gate1_percent'] - 100 * part['pulse_present_BG'].astype(bool).mean()
    if has_ref:
        ref = part.hr_ref.to_numpy(float)
        tol = CMP['reference']['tolerance_bpm']
        for subset, keep in (('reported', rep), ('all', np.ones(n, bool))):
            ok = keep & np.isfinite(ref) & np.isfinite(hr)
            e = hr[ok] - ref[ok]
            out.update({f'n_{subset}_with_ref': int(ok.sum()),
                        f'mae_{subset}': float(np.abs(e).mean()) if ok.any() else np.nan,
                        f'rmse_{subset}': float(np.sqrt((e ** 2).mean())) if ok.any() else np.nan,
                        f'within{tol:g}_{subset}_percent': float(100 * (np.abs(e) <= tol).mean()) if ok.any() else np.nan,
                        f'bias_{subset}': float(e.mean()) if ok.any() else np.nan,
                        f'pearson_r_{subset}': float(stats.pearsonr(hr[ok], ref[ok])[0]) if ok.sum() > 2 and
                        np.std(ref[ok]) > 0 and np.std(hr[ok]) > 0 else np.nan})
        out['coverage_with_ref_percent'] = 100 * out['n_reported_with_ref'] / max(1, int(np.isfinite(ref).sum()))
    return out


PAIRED_METRICS = ('reported_percent', 'gate1_above_BG_pp', 'snr_median_db', 'agreement_percent', 'hr_jump_median_bpm')
PAIRED_REF = ('mae_reported', 'within5_reported_percent', 'mae_all', 'coverage_with_ref_percent')
LOWER_IS_BETTER = ('hr_jump_median_bpm', 'mae_reported', 'mae_all', 'rmse_reported', 'rmse_all')


def summarize(w, methods, has_ref, target):
    gs = groups(w)
    pooled, per_video = [], []
    for g, keep in gs.items():
        part = w[keep]
        for m in methods:
            pooled.append(dict(group=g, method=m, videos=part.video_id.nunique(), **metrics(part, m, has_ref)))
            for video, p in part.groupby('video_id'):
                per_video.append(dict(group=g, method=m, video_id=video, **metrics(p, m, has_ref)))
    pooled, per_video = pd.DataFrame(pooled), pd.DataFrame(per_video)
    pooled.to_csv(target/'summary_by_condition.csv', index=False)
    per_video.to_csv(target/'per_video_by_condition.csv', index=False)
    names = [x for x in PAIRED_METRICS if x in per_video] + ([x for x in PAIRED_REF if x in per_video] if has_ref else [])
    tests = []
    for g in gs:
        pv = per_video[per_video.group == g]
        for ref_m in dict.fromkeys([CMP['baseline'], CMP['dynamic']]):
            if ref_m not in methods:
                continue
            for metric in names:
                wide = pv.pivot_table(index='video_id', columns='method', values=metric)
                block = []
                for m in methods:
                    if m in (ref_m, 'BG') or m not in wide or ref_m not in wide:
                        continue
                    d = (wide[m] - wide[ref_m]).to_numpy(float)
                    est, lo, hi = bootstrap_ci(d, n=CMP['bootstrap'], seed=CMP['seed'])
                    block.append(dict(group=g, metric=metric, reference=ref_m, method=m,
                                      better_if='lower' if metric in LOWER_IS_BETTER else 'higher',
                                      mean_diff_method_minus_reference=est, ci95_low=lo, ci95_high=hi,
                                      **paired_test(wide[ref_m], wide[m])))
                for row, adj in zip(block, holm([b['p_value'] for b in block])):
                    row['p_holm'] = adj
                tests += block
    tests = pd.DataFrame(tests)
    tests.to_csv(target/'paired_tests_by_condition.csv', index=False)
    effects = []
    for g in gs:
        pv = per_video[per_video.group == g]
        for metric in names:
            wide = pv.pivot_table(index='video_id', columns='method', values=metric)
            block = []
            for effect, (m, ref_m) in CMP['effects'].items():
                if m not in wide or ref_m not in wide:
                    continue
                est, lo, hi = bootstrap_ci((wide[m] - wide[ref_m]).to_numpy(float), n=CMP['bootstrap'], seed=CMP['seed'])
                block.append(dict(group=g, metric=metric, effect=effect, method=m, minus=ref_m,
                                  better_if='lower' if metric in LOWER_IS_BETTER else 'higher',
                                  mean_diff=est, ci95_low=lo, ci95_high=hi, **paired_test(wide[ref_m], wide[m])))
            for row, adj in zip(block, holm([b['p_value'] for b in block])):
                row['p_holm'] = adj
            effects += block
    effects = pd.DataFrame(effects)
    effects.to_csv(target/'effects_by_condition.csv', index=False)
    return pooled, per_video, tests, effects


def md_table(df, index=True):
    """Markdown table without the optional 'tabulate' dependency."""
    d = df.reset_index() if index else df
    fmt = lambda v: '' if v is None or (isinstance(v, float) and not math.isfinite(v)) else (f'{v:.2f}' if isinstance(v, float) else str(v))
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    return '\n'.join([head, '|' + '---|' * len(d.columns)] + ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in d.itertuples(index=False)])


def write_report(target, pooled, tests, effects, thresholds, meta, has_ref, methods):
    key = ['reported_percent', 'gate1_percent', 'gate1_above_BG_pp', 'snr_median_db', 'agreement_percent']
    if has_ref:
        tol = CMP['reference']['tolerance_bpm']
        key = ['reported_percent', 'mae_reported', f'within{tol:g}_reported_percent', 'mae_all', 'gate1_above_BG_pp']
    lines = ['# Fixed vs dynamic ROI comparison', '',
             f"Run: `{meta['run']}` | videos: {meta['videos']} | windows: {meta['windows']} | HR band: {meta['hr_band_bpm']} bpm "
             f"| reference: {meta['reference_hr'] or 'none (pulse-evidence metrics only)'}", '',
             'Same pixels rule, POS, 8 s / 1 s windows, gate 1 (block-shuffle null), gate 2 (split halves), Kalman for every '
             'method; only the ROI differs. BG is the negative control (expected gate-1 rate ~5 %), not a method.', '',
             '**Reading guide.** Without a reference, a higher `gate1_above_BG_pp` / `reported_percent` means the ROI carries '
             'more pulse evidence than background, NOT that its HR is correct. Only MAE against a reference shows that. '
             'Always read coverage (`reported_percent`) next to MAE.', '',
             'Gate 2 needs both halves of the ROI to pass the shared pixel rule (>= 10 % skin pixels, >= 20 px). A large box '
             '(M0, B0) whose one half is mostly hair, clothes or bedding therefore stays `present` (gate 2 not testable); '
             'see `agreement_testable_percent` in summary_by_condition.csv.', '',
             '## Condition thresholds', '', '| condition | measure | hard when | threshold | source | windows flagged |',
             '|---|---|---|---|---|---|']
    counts = pooled[pooled.method == methods[0]].set_index('group').windows
    for name, t in thresholds.items():
        lines.append(f"| {name} | {t['measure']} | {t['hard']} | {t['threshold']:.4g} | {t['source']} | {int(counts.get(name, 0))} |")
    lines += ['', 'Tertile thresholds are relative to THIS run (the worst third of its windows). For a frozen study set fixed '
              'thresholds in configs/roi_comparison.yaml.', '']
    for metric in key:
        if metric not in pooled:
            continue
        tab = pooled.pivot_table(index='group', columns='method', values=metric).reindex(columns=methods)
        tab.insert(0, 'windows', counts.reindex(tab.index))
        lines += [f'## {metric} (pooled windows, by condition)', '', md_table(tab), '']
    if len(effects):
        main = f'mae_reported' if has_ref else 'gate1_above_BG_pp'
        show = ['all', 'clean', 'moving', 'posture_change', 'covered_low_skin', 'occluded_few_regions',
                'illumination_change', 'diag:d5_switched', 'diag:d5_no_switch']
        lines += ['## Effects: switching, tracking, selection', '',
                  'Per-video paired difference (method minus comparison), mean with bootstrap 95 % CI; Holm within '
                  'group x metric. '
                  + ('Lower MAE is better: a negative difference favours the first method.' if has_ref else
                     'Higher gate-1 pass above background is better (pulse evidence only, not HR accuracy).'), '']
        for metric in dict.fromkeys([main, 'reported_percent']):
            e = effects[(effects.metric == metric) & effects.group.isin(show)].copy()
            if not len(e):
                continue
            e['difference [95 % CI]'] = [f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]' if math.isfinite(lo) else
                                        (f'{d:+.2f}' if math.isfinite(d) else '') for d, lo, hi in
                                        zip(e.mean_diff, e.ci95_low, e.ci95_high)]
            e['contrast'] = e.effect + ' (' + e.method + ' - ' + e.minus + ')'
            tab = e.pivot_table(index='group', columns='contrast', values='difference [95 % CI]', aggfunc='first')
            tab = tab.reindex([g for g in show if g in tab.index])
            lines += [f'### {metric}', '', md_table(tab), '']
        lines += ['Full table with n, Wilcoxon p and p_holm: effects_by_condition.csv. `diag:d5_switched` = windows in which '
                  'D5 changed its region set; if D5_WIN beats D5 mainly there, switching is what hurts D5.', '']
    if len(tests):
        sig = tests[(tests.p_holm < .05)]
        lines += ['## Paired per-video tests (Wilcoxon, Holm within group x metric x reference)', '',
                  f'{len(tests)} comparisons, {len(sig)} with p_holm < 0.05. Full table: paired_tests_by_condition.csv. '
                  'With few videos the tests have little power; read the bootstrap CIs.', '']
        if len(sig):
            lines += [md_table(sig[['group', 'metric', 'reference', 'method', 'mean_diff_method_minus_reference', 'ci95_low',
                           'ci95_high', 'n', 'p_holm']], index=False), '']
    (target/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--module1-config', default='D5')
    ap.add_argument('--videos', nargs='+')
    ap.add_argument('--methods', default=','.join(CMP['methods']))
    ap.add_argument('--reference-hr')
    ap.add_argument('--reference-offset-sec', type=float, default=0.)
    ap.add_argument('--conditions', help='manual labels csv: video_id,start_sec,end_sec,condition')
    ap.add_argument('--hr-band', default=None, help='low,high bpm (default: infant band of HRConfig); adults: 45,180')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out-name', default='roi_comparison')
    ap.add_argument('--resummarize', action='store_true', help='reuse window_results.csv; redo conditions/metrics only')
    a = ap.parse_args(argv)
    band = tuple(float(v) for v in a.hr_band.split(',')) if a.hr_band else None
    set_band(band)
    methods = [m.strip() for m in a.methods.split(',') if m.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise SystemExit(f'unknown methods {sorted(unknown)}')
    run = (ROOT/a.run).resolve()
    target = M2/'runs'/run.name/a.out_name
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if a.resummarize:
        w = pd.read_csv(target/'window_results.csv')
    else:
        src = ROOT/'module1_v2/runs'/run.name/f'dynamic_roi_v2_{a.module1_config}'/'dynamic_roi_v2.csv'
        if not src.is_file():
            raise SystemExit(f'Missing {src}: run module1_v2/scripts/export_dynamic_roi_v2.py (the infant notebook does) first')
        rows = pd.read_csv(src, low_memory=False)
        manifest = pd.read_csv(run/'manifest.csv')
        if a.videos:
            manifest = manifest[manifest.video_id.isin(a.videos)]
        out = run/'outputs'
        cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                           usecols=['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'anatomical_confidence'])
        recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
        sam = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
        jobs = [(r._asdict(), rows[rows.video_id == r.video_id], cand[cand.video_id == r.video_id],
                 recovery[recovery.video_id == r.video_id], sam[sam.video_id == r.video_id], str(out), methods, band)
                for r in manifest.itertuples()]
        with ProcessPoolExecutor(a.workers) as pool:
            results = list(pool.map(video_job, jobs))
        w = pd.DataFrame([row for r in results for row in r[0]])
        w.to_csv(target/'window_results.csv', index=False)
    for c in [c for c in w.columns if c.startswith(('cond_', 'manual_'))] + ['hr_ref', 'ref_source']:
        w = w.drop(columns=c, errors='ignore')
    manual = pd.read_csv(a.conditions) if a.conditions else None
    cg_path = M2/'runs'/run.name/'caregiver'/'caregiver_segments.csv'
    caregiver = pd.read_csv(cg_path) if cg_path.is_file() else None
    w, thresholds = flag_conditions(w, manual, caregiver)
    has_ref = bool(a.reference_hr)
    if has_ref:
        w = attach_reference(w, a.reference_hr, a.reference_offset_sec)
        has_ref = bool(w.hr_ref.notna().any())
        if not has_ref:
            print('WARNING: no reference sample matched any window (video_id / time offset?); reporting without reference')
    w.to_csv(target/'window_results_with_conditions.csv', index=False)
    pooled, _, tests, effects = summarize(w, methods, has_ref, target)
    meta = dict(run=str(run), videos=int(w.video_id.nunique()), windows=len(w), methods=methods,
                hr_band_bpm=[HR.hr_min_bpm, HR.hr_max_bpm], reference_hr=a.reference_hr if has_ref else None,
                reference_offset_sec=a.reference_offset_sec, manual_conditions=a.conditions,
                caregiver_segments=str(cg_path) if caregiver is not None else None, condition_thresholds=thresholds,
                comparison_config=CMP, hr_config=HR_CFG['hr'], wall_time_sec=time.perf_counter() - started)
    (target/'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    write_report(target, pooled, tests, effects, thresholds, meta, has_ref, methods)
    show = ['reported_percent', 'gate1_above_BG_pp', 'snr_median_db'] + (['mae_reported', 'mae_all'] if has_ref else [])
    print(pooled[pooled.group.isin(['all', 'clean', 'moving', 'posture_change', 'covered_low_skin'])]
          [['group', 'method', 'windows'] + [c for c in show if c in pooled]].round(2).to_string(index=False))
    print('\nwrote', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
