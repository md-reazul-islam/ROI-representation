"""Annotated HR video: the camera frame with the HR regions used, the system HR next to the reference HR, and the
reference PPG waveform scrolling above the system's camera pulse signal.

Everything shown is read from a finished run (hr_windows.csv, waveforms/<video>.npz, the Module 1 v2 representation)
and the dataset's ground truth file; nothing is recomputed except a display band-pass of the finger PPG.

    python "HR RR using module1 v2/scripts/annotate_hr_video.py" --results <OUT_V2> --video-id NEW_subject1 \
        --ground-truth data/ubfc/subject1/ground_truth.txt --reference-hr <REF>/reference_hr_ppg.csv
Writes <results>/annotated/<video_id>_annotated.mp4 (any player) and .webm (VS Code / browser), plus a PNG of 3 stills.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
W, TOP, BOTTOM = 1280, 480, 300
SCROLL_SEC = 8.0
COL = dict(validated=(60, 170, 60), present=(0, 140, 255), not_assessable=(150, 150, 150),
           caregiver_intervention=(180, 80, 180))
WORDS = dict(validated='VALIDATED (reported)', present='pulse present, halves disagree (not reported)',
             not_assessable='not assessable (fails surrogate test)', caregiver_intervention='caregiver contact (excluded)')
REF_C, SYS_C, INK, MUTED, GRID = (40, 40, 200), (200, 110, 30), (30, 30, 30), (110, 110, 110), (225, 225, 225)


def text(img, s, xy, scale=.55, color=INK, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bandpass(x, fs, low_bpm, high_bpm):
    b, a = butter(3, [low_bpm / 60 / (fs / 2), min(high_bpm / 60 / (fs / 2), .95)], 'band')
    return filtfilt(b, a, x - np.mean(x))


def load_ground_truth(path, duration):
    rows = [np.array(l.split(), float) for l in Path(path).read_text().strip().splitlines()]
    ppg, t = rows[0], rows[2]
    if t[-1] > 5 * duration:                          # some copies store ms
        t = t / 1000.
    t = t - t[0]
    fs = 30.
    grid = np.arange(0, t[-1], 1 / fs)
    return grid, np.interp(grid, t, ppg), fs


def polygons(results, video_id):
    meta = json.loads((results/'metadata.json').read_text())
    src = Path(meta['representation'])
    if not src.is_file():
        return {}
    parts = []
    for chunk in pd.read_csv(src, usecols=['video_id', 'frame_index', 'roi_name', 'available', 'pixel_polygon'], chunksize=100000):
        parts.append(chunk[(chunk.video_id == video_id) & chunk.available.astype(bool)])
    d = pd.concat(parts)
    out = {}
    for r in d.itertuples():
        if isinstance(r.pixel_polygon, str):
            pts = np.asarray(json.loads(r.pixel_polygon), float)
            if pts.ndim == 2 and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            if pts.ndim == 2 and len(pts) >= 3:
                out[(int(r.frame_index), r.roi_name)] = pts.astype(np.int32).reshape(-1, 1, 2)
    return out


def timeline_base(win, ref, duration, w, h):
    """Static HR-vs-time panel; returns the image and a function mapping (t, bpm) to pixels."""
    img = np.full((h, w, 3), 255, np.uint8)
    vals = np.concatenate([win.hr_bpm.dropna().values, ref.hr_bpm.dropna().values])
    lo, hi = np.floor(np.nanmin(vals) / 10) * 10 - 5, np.ceil(np.nanmax(vals) / 10) * 10 + 5
    L, R, T, B = 52, w - 12, 28, h - 30
    X = lambda t: int(L + (R - L) * t / duration)
    Y = lambda v: int(B - (B - T) * (v - lo) / (hi - lo))
    for v in np.arange(np.ceil(lo / 10) * 10, hi + 1e-9, 10):
        cv2.line(img, (L, Y(v)), (R, Y(v)), GRID, 1)
        text(img, f'{v:.0f}', (8, Y(v) + 4), .4, MUTED)
    for s in range(0, int(duration) + 1, 10):
        text(img, f'{s}s', (X(s) - 8, h - 10), .4, MUTED)
    text(img, 'HR over the whole video (bpm)', (L, 18), .5, INK)
    r = ref.dropna().sort_values('timestamp_sec')
    pts = np.array([[X(t), Y(v)] for t, v in zip(r.timestamp_sec, r.hr_bpm)], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, REF_C, 2, cv2.LINE_AA)
    for row in win.dropna(subset=['hr_bpm']).itertuples():
        c = (X(row.window_center_sec), Y(row.hr_bpm))
        if row.label == 'validated':
            cv2.circle(img, c, 4, COL['validated'], -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 4, COL.get(row.label, MUTED), 1, cv2.LINE_AA)
    # legend
    cv2.line(img, (R - 310, 14), (R - 290, 14), REF_C, 2); text(img, 'reference', (R - 285, 18), .4, INK)
    cv2.circle(img, (R - 205, 14), 4, COL['validated'], -1); text(img, 'system validated', (R - 196, 18), .4, INK)
    cv2.circle(img, (R - 66, 14), 4, MUTED, 1); text(img, 'rejected', (R - 57, 18), .4, INK)
    return img, X, (T, B)


def lane(img, y0, y1, t_axis, x, colour, label):
    L, R = 60, W - 20
    cv2.rectangle(img, (L, y0), (R, y1), GRID, 1)
    text(img, label, (L + 6, y0 + 16), .48, colour)
    if len(x) < 2 or not np.isfinite(x).any():
        text(img, 'no signal', (L + 6, (y0 + y1) // 2), .5, MUTED)
        return
    x = np.nan_to_num(x)
    s = np.std(x) or 1.
    z = np.clip((x - np.mean(x)) / s, -3, 3)
    xs = L + (R - L) * (t_axis - t_axis[0]) / max(t_axis[-1] - t_axis[0], 1e-9)
    mid, amp = (y0 + y1 + 14) / 2, (y1 - y0 - 24) / 6
    pts = np.stack([xs, mid - z * amp], 1).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='HR output folder (with hr_windows.csv and waveforms/)')
    ap.add_argument('--video-id', required=True)
    ap.add_argument('--ground-truth', required=True, help='UBFC ground_truth.txt (line 1 = PPG waveform)')
    ap.add_argument('--reference-hr', required=True, help='reference HR CSV (video_id, timestamp_sec, hr_bpm)')
    ap.add_argument('--video', help='video file (default: from the run manifest)')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    results = Path(a.results)
    meta = json.loads((results/'metadata.json').read_text())
    low, high = meta.get('hr_band_bpm', [45., 180.])
    win = pd.read_csv(results/'hr_windows.csv')
    win = win[win.video_id == a.video_id].sort_values('window_center_sec').reset_index(drop=True)
    assert len(win), f'{a.video_id} not in hr_windows.csv'
    ref = pd.read_csv(a.reference_hr)
    ref = ref[ref.video_id == a.video_id]
    video = a.video or pd.read_csv(Path(meta['run'])/'manifest.csv').set_index('video_id').video_path[a.video_id]
    video = Path(video) if Path(video).is_absolute() else ROOT/video
    z = np.load(results/'waveforms'/f'{a.video_id}.npz')
    pulse, pfs = z['pulse'].astype(float), float(z['pulse_fps'])
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    gt_t, gt_ppg, gt_fs = load_ground_truth(a.ground_truth, duration)
    gt_bp = bandpass(gt_ppg, gt_fs, low, high)
    p_t = np.arange(len(pulse)) / pfs
    polys = polygons(results, a.video_id)
    out = Path(a.out) if a.out else results/'annotated'
    out.mkdir(parents=True, exist_ok=True)
    tl, X, (tT, tB) = timeline_base(win, ref, duration, 640, TOP - 200)
    size = (W, TOP + BOTTOM)
    writers = [cv2.VideoWriter(str(out/f'{a.video_id}_annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size),
               cv2.VideoWriter(str(out/f'{a.video_id}_annotated.webm'), cv2.VideoWriter_fourcc(*'VP80'), fps, size)]
    stills_at = {int(n * f) for f in (.25, .5, .75)}
    stills = []
    centres = win.window_center_sec.values
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        k = int(np.argmin(np.abs(centres - t)))
        w = win.iloc[k]
        label = w.label if isinstance(w.label, str) else 'not_assessable'
        rr = ref.iloc[(ref.timestamp_sec - t).abs().argsort().iloc[0]] if len(ref) else None
        ref_hr = rr.hr_bpm if rr is not None and abs(rr.timestamp_sec - t) <= 1. else np.nan
        canvas = np.full((size[1], W, 3), 255, np.uint8)
        # --- video with the HR regions of the current window
        img = cv2.resize(frame, (640, TOP))
        sx, sy = 640 / frame.shape[1], TOP / frame.shape[0]
        regions = json.loads(w.hr_regions) if isinstance(w.hr_regions, str) else {}
        for name, weight in regions.items():
            p = polys.get((i, name))
            if p is None:
                continue
            p = (p * [sx, sy]).astype(np.int32)
            cv2.polylines(img, [p], True, COL[label], 2, cv2.LINE_AA)
            c = p.reshape(-1, 2).min(0)
            tag, org = f'{name} {weight:.2f}', (int(c[0]), max(int(c[1]) - 6, 16))
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
            cv2.rectangle(img, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2, org[1] + 3), (255, 255, 255), -1)
            text(img, tag, org, .45, COL[label], 1)
        cv2.rectangle(img, (0, 0), (300, 24), (255, 255, 255), -1)
        text(img, f'{a.video_id}   t = {t:5.1f} s', (6, 17), .5, INK)
        canvas[:TOP, :640] = img
        # --- numbers
        P = canvas[:200, 640:]
        text(P, 'Reference HR (finger PPG)', (20, 32), .55, REF_C)
        text(P, '--' if not np.isfinite(ref_hr) else f'{ref_hr:5.1f} bpm', (20, 78), 1.25, REF_C, 2)
        text(P, 'System HR (camera)', (340, 32), .55, SYS_C)
        sys_hr = w.hr_bpm if np.isfinite(w.hr_bpm) else np.nan
        text(P, '--' if not np.isfinite(sys_hr) else f'{sys_hr:5.1f} bpm', (340, 78), 1.25,
             SYS_C if label == 'validated' else MUTED, 2)
        cv2.rectangle(P, (20, 98), (620, 132), COL[label], -1)
        text(P, f'window {int(w.window)}: {WORDS.get(label, label)}', (30, 121), .55, (255, 255, 255), 1)
        diff = sys_hr - ref_hr if np.isfinite(sys_hr) and np.isfinite(ref_hr) else np.nan
        text(P, ('error = ' + (f'{diff:+.1f} bpm' if np.isfinite(diff) else 'n/a')) +
             (f'    SNR {w.snr_db:.2f} dB vs null {w.null_threshold_db:.2f} dB' if np.isfinite(w.snr_db) else ''),
             (20, 160), .52, INK)
        hl, hr_ = w.hr_left_bpm, w.hr_right_bpm
        text(P, f'left half {hl:.1f} / right half {hr_:.1f} bpm (must agree within 6)' if np.isfinite(hl) and np.isfinite(hr_)
             else 'left / right halves: one half has no HR', (20, 186), .48, MUTED)
        # --- HR timeline with cursor
        T = tl.copy()
        cv2.line(T, (X(t), tT), (X(t), tB), (0, 0, 220), 1)
        canvas[200:TOP, 640:] = T
        # --- scrolling signals (last SCROLL_SEC seconds)
        t0 = max(0., t - SCROLL_SEC)
        g = (gt_t >= t0) & (gt_t <= t)
        q = (p_t >= t0) & (p_t <= t)
        y = TOP + 8
        text(canvas, f'Signals, last {SCROLL_SEC:.0f} s (each scaled to its own range; right edge = now)', (60, y + 12), .5, INK)
        lane(canvas, y + 22, y + 22 + 126, gt_t[g] if g.sum() > 1 else np.array([0.]), gt_bp[g],
             REF_C, 'reference: finger PPG waveform (band-passed to the HR band)')
        lane(canvas, y + 156, y + 156 + 126, p_t[q] if q.sum() > 1 else np.array([0.]), pulse[q],
             SYS_C, 'system: camera pulse (rPPG) from the HR regions')
        for wr in writers:
            wr.write(canvas)
        if i in stills_at:
            stills.append(canvas.copy())
    cap.release()
    for wr in writers:
        wr.release()
    if stills:
        cv2.imwrite(str(out/f'{a.video_id}_annotated_stills.png'), np.vstack(stills))
    print('wrote', out/f'{a.video_id}_annotated.mp4')
    print('wrote', out/f'{a.video_id}_annotated.webm')
    print('wrote', out/f'{a.video_id}_annotated_stills.png')


if __name__ == '__main__':
    main()
