"""HR + RR monitor videos for the v2 pipeline (Module 1 v2 dynamic ROI + Module 2 v2 vitals).

What the overlay shows, and how it differs from the v1 monitor (scripts/render_vitals_monitor.py, unchanged):
  left   the frame with the SAM 2 infant-mask outline (pixel refinement), the Module 1 v2 stabilized HR regions
         (red; the selected set, labelled with its stabilized fusion weight) and the RR regions (cyan; all
         chest / abdomen candidates used by Module 2 v2, labelled with their window weight)
  right  HR and RR are shown ONLY when the last finished window is validated (beats the block-shuffle null of the
         same data AND two disjoint body parts agree); otherwise '--' with the reason (not assessable / not
         confirmed). Waveforms are the Module 2 v2 fused pulse and breathing signals; a strip shows the window labels.
  caregiver (optional, --caregiver <results>/caregiver from apply_caregiver_labels.py): a purple banner while a caregiver
         touches the infant, and windows overlapping the contact show '--' with 'EXCLUDED: caregiver intervention'
         (purple cells in the label strip)
Usage: python "HR RR using module1 v2/scripts/render_monitor_v2.py --run runs/final34_paper --module1-config D5 [--videos VID_001]
       [--caregiver "HR RR using module1 v2/runs/<run>/<config>/caregiver"]
Output: HR RR using module1 v2/runs/<run>/<config>/monitor/<video>_monitor_v2.mp4 (mp4v; play in VLC)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(p))
import render_vitals_monitor as v1                               # noqa: E402  (v1 drawing helpers, read-only)
from module2.extraction import hold_indices                      # noqa: E402
from run_paper_evaluation import EXTRACT, load_mask               # noqa: E402

HR_COLOR, RR_COLOR, MASK_COLOR = v1.HR_COLOR, v1.RR_COLOR, (235, 235, 235)
GOOD, LOW, MUTED, WHITE, PANEL = v1.GOOD, v1.LOW, v1.MUTED, v1.WHITE, v1.PANEL
LABEL_COLOR = {'validated': (110, 220, 110), 'present': (0, 190, 255), 'not_assessable': (90, 90, 90),
               'caregiver_intervention': (200, 60, 170), 'validated_one_side': (200, 200, 60)}
CARE = LABEL_COLOR['caregiver_intervention']


def done_window(windows, t):
    done = windows[windows.window_end_sec <= t + 1e-9]
    return None if done.empty else done.iloc[-1]


def status(window, kind):
    """(value text, value colour, status line, status colour)."""
    if window is None:
        return '--', MUTED, 'acquiring signal (first window not finished)', MUTED
    rate, kal = ('hr_bpm', 'hr_kalman_bpm') if kind == 'HR' else ('rr_brpm', 'rr_kalman_brpm')
    if bool(window.get('caregiver_intervention', False)):
        return '--', CARE, 'EXCLUDED: caregiver intervention (window overlaps caregiver contact)', CARE
    snr = window.snr_db
    thr = window.null_threshold_db
    snr_txt = 'NA' if pd.isna(snr) else f'{snr:.1f} dB'
    thr_txt = 'NA' if pd.isna(thr) else f'{thr:.1f} dB'
    if kind == 'HR':
        parts = ('left', window.hr_left_bpm, 'right', window.hr_right_bpm, 'bpm')
    else:
        parts = ('chest', window.rr_chest_brpm, 'abdomen', window.rr_abdomen_brpm, '/min')
    split = (f'{parts[0]} {parts[1]:.0f} / {parts[2]} {parts[3]:.0f} {parts[4]}'
             if pd.notna(parts[1]) and pd.notna(parts[3]) else f'{parts[0]}/{parts[2]} not testable')
    if kind == 'HR' and isinstance(window.get('pair_used'), str) and window.pair_used != 'none' and window.get('gate2_mode') == 'fallback':
        a_, b_ = window.pair_used.split('_')
        split = f'{a_} {window.hr_a:.0f} / {b_} {window.hr_b:.0f} bpm'
    if window.label == 'validated':
        value = window[kal] if pd.notna(window[kal]) else window[rate]
        return f'{value:.0f}', GOOD, f'VALIDATED: SNR {snr_txt} > null {thr_txt}; {split}', GOOD
    if window.label == 'validated_one_side':
        value = window[kal] if pd.notna(window[kal]) else window[rate]
        c = LABEL_COLOR['validated_one_side']
        return f'{value:.0f}', c, f'ONE SIDE ONLY: SNR {snr_txt} > null {thr_txt}; no second body part visible', c
    if window.label == 'present':
        return '--', LOW, f'NOT CONFIRMED: SNR {snr_txt} > null {thr_txt}, but {split}', LOW
    if pd.isna(snr):
        why = 'no usable pulse signal' if kind == 'HR' else 'no usable chest/abdomen motion (covered or out of view)'
        return '--', MUTED, f'NOT ASSESSABLE: {why}', MUTED
    return '--', MUTED, f'NOT ASSESSABLE: SNR {snr_txt} <= null {thr_txt}', MUTED


def vital(panel, y, u, margin, width, title, unit, window, kind, times, wave, t, span, wave_label, wave_color, windows):
    v1.text(panel, title, (margin, y), .62*u, WHITE, max(1, int(1.5*u)))
    value, value_color, line, line_color = status(window, kind)
    scale, thickness = 1.9*u, max(2, int(4*u))
    (nw, nh), _ = cv2.getTextSize(value, v1.FONT, scale, thickness)
    v1.text(panel, value, (margin, y + int(12*u) + nh), scale, value_color, thickness)
    v1.text(panel, unit, (margin, y + int(34*u) + nh), .55*u, WHITE, max(1, int(1.3*u)))
    x0 = margin + max(nw, int(150*u)) + int(25*u)
    box = (x0, y - int(16*u), width - margin, y + int(36*u) + nh)
    recent = (times > t - span) & (times <= t)
    v1.waveform(panel, box, times[recent], wave[recent], t, span, wave_color, u)
    v1.text(panel, wave_label, (x0, box[3] + int(17*u)), .42*u, MUTED, max(1, int(u)))
    # label strip: one cell per finished window (green validated, orange not confirmed, grey not assessable)
    strip_y = box[3] + int(26*u)
    done = windows[windows.window_end_sec <= t + 1e-9]
    if len(windows):
        total = max(windows.window_end_sec.max(), 1e-6)
        for r in done.itertuples():
            xa = int(x0 + (r.window_end_sec - 1) / total * (box[2] - x0))
            xb = int(x0 + r.window_end_sec / total * (box[2] - x0))
            cv2.rectangle(panel, (xa, strip_y), (max(xa + 1, xb), strip_y + int(6*u)),
                          LABEL_COLOR[getattr(r, 'label_with_caregiver', r.label)], -1)
    y = box[3] + int(48*u)
    v1.text(panel, line, (margin, y), .46*u, line_color, max(1, int(1.2*u)))
    return y


def weights_column(panel, x, y, u, width, title, weights, color):
    v1.text(panel, title, (x, y), .45*u, WHITE, max(1, int(1.3*u)))
    ranked = [n for n in sorted(weights, key=lambda n: -weights[n]) if weights[n] > 0][:3]
    for n in ranked:
        y += int(24*u)
        name_w, bar_w = int(width * .58), int(width * .22)
        v1.text(panel, n, (x, y), .42*u, color, max(1, int(1.2*u)))
        cv2.rectangle(panel, (x + name_w, y - int(11*u)), (x + name_w + bar_w, y), (55, 50, 48), -1)
        cv2.rectangle(panel, (x + name_w, y - int(11*u)), (x + name_w + int(bar_w*min(weights[n], 1.)), y), color, -1)
        v1.text(panel, f'{weights[n]:.2f}', (x + name_w + bar_w + int(6*u), y), .42*u, color, max(1, int(1.2*u)))


def draw_left(image, mask, hr_rois, rr_rois, scale, u):
    frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else image.copy()
    if mask is not None and mask.shape == image.shape[:2]:
        m = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, MASK_COLOR, max(1, int(1.2*u)), cv2.LINE_AA)
    for rois, color, kind, below in ((rr_rois, RR_COLOR, 'RR', True), (hr_rois, HR_COLOR, 'HR', False)):
        if not rois:
            continue
        best = max(rois, key=lambda n: rois[n][1])
        for name, (poly, w) in rois.items():
            pts = np.rint(np.asarray(poly, float) * scale).astype(np.int32)
            if name == best:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                frame = cv2.addWeighted(overlay, .25, frame, .75, 0)
            cv2.polylines(frame, [pts], True, color, max(2, int((3 if name == best else 1.5)*u)), cv2.LINE_AA)
        pts = np.rint(np.asarray(rois[best][0], float) * scale).astype(np.int32)
        v1.label_box(frame, pts, f'{kind}: {best} w={rois[best][1]:.2f}', color, u, below=below)
    return frame


def render(video, record, rows, hr, rr, waves, sam2_rows, out_dir, target, contact=None):
    fps, n = float(record['fps']), int(record['frame_count'])
    frames = sorted(rows.frame_index.unique())
    hold = hold_indices(frames, n, fps, EXTRACT.max_hold_sec)
    by_frame = {int(f): g for f, g in rows.groupby('frame_index')}
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    times = np.arange(n) / fps
    pulse = waves.get('pulse', np.full(n, np.nan))
    pulse = (pulse - np.nanmean(pulse)) / (np.nanstd(pulse) or 1.) if np.isfinite(pulse).any() else pulse
    if 'breath' in waves:
        bf = waves['breath_frames'] / fps
        breath = np.interp(times, bf, np.nan_to_num(waves['breath'])) if len(bf) > 1 else np.full(n, np.nan)
    else:
        breath = np.full(n, np.nan)
    rr_rois_names = set()
    for w in rr.rr_weights if len(rr) else []:
        rr_rois_names |= set(json.loads(w))
    cap = cv2.VideoCapture(str(record['video_path']))
    writer = None
    mask_cache = {}
    try:
        for i in range(n):
            ok, image = cap.read()
            if not ok:
                break
            scale = float(np.clip(720. / image.shape[0], .5, 2.))
            u = image.shape[0] * scale / 720 * 1.25
            source = int(hold[i])
            part = by_frame.get(source) if source >= 0 else None
            t = times[i]
            rr_window = done_window(rr, t) if len(rr) else None
            rr_w = json.loads(rr_window.rr_weights) if rr_window is not None and isinstance(rr_window.rr_weights, str) else {}
            hr_rois, rr_rois, hr_w = {}, {}, {}
            hr_window = done_window(hr, t) if len(hr) else None
            specific = 'hr_regions' in hr.columns           # HR-specific region set: regions chosen per window
            spec_w = json.loads(hr_window.hr_regions) if specific and hr_window is not None and isinstance(hr_window.hr_regions, str) else {}
            if part is not None:
                for r in part.itertuples():
                    poly = json.loads(r.pixel_polygon) if isinstance(r.pixel_polygon, str) else None
                    if not poly or not r.available:
                        continue
                    if specific:
                        if spec_w.get(r.roi_name, 0) > 0:
                            hr_rois[r.roi_name] = (poly, spec_w[r.roi_name])
                            hr_w[r.roi_name] = spec_w[r.roi_name]
                    elif r.selected and r.weight > 0:
                        hr_rois[r.roi_name] = (poly, r.weight)
                        hr_w[r.roi_name] = r.weight
                    if r.roi_name in rr_w:
                        rr_rois[r.roi_name] = (poly, rr_w[r.roi_name])
            mask = None
            if source >= 0 and len(sam2_rows) and source in sam2_rows.index:
                if source not in mask_cache:
                    mask_cache = {source: load_mask(Path(out_dir)/'sam2', sam2_rows.loc[source])}
                mask = mask_cache[source]
            left = draw_left(image, mask, hr_rois, rr_rois, scale, u)
            if contact is not None and len(contact) and ((contact.start_sec <= t) & (t <= contact.end_sec)).any():
                bar = int(34*u)
                cv2.rectangle(left, (0, 0), (left.shape[1], bar), CARE, -1)
                v1.text(left, 'CAREGIVER CONTACT: vital signs excluded', (int(12*u), int(24*u)), .6*u, WHITE, max(1, int(1.5*u)))
            h = left.shape[0]
            width = max(int(h * .95), int(640*u))
            panel = np.full((h, width, 3), PANEL, np.uint8)
            margin = int(22*u)
            y = int(36*u)
            cv2.circle(panel, (margin + int(7*u), y - int(7*u)), int(7*u), GOOD, -1, cv2.LINE_AA)
            v1.text(panel, 'Camera-derived vital signs v2 (HR + RR)', (margin + int(22*u), y), .6*u, WHITE, max(1, int(1.5*u)))
            v1.text(panel, f'{video}   t = {t:.1f} s   Module 1 v2 dynamic ROI + ' + ('HR-specific regions' if specific else 'Module 2 v2'), (margin, y + int(26*u)),
                    .44*u, MUTED, max(1, int(u)))
            y = int(92*u)
            y = vital(panel, y, u, margin, width, 'Heart rate', 'bpm', done_window(hr, t) if len(hr) else None, 'HR',
                      times, pulse, t, 6., 'pulse: POS on ROI n mask n skin (last 6 s)',
                      v1.PULSE_COLOR, hr)
            y += int(34*u)
            y = vital(panel, y, u, margin, width, 'Respiration rate', 'breaths/min', rr_window, 'RR', times, breath, t, 15.,
                      'breathing: chest/abdomen motion (last 15 s)', v1.BREATH_COLOR, rr)
            y += int(30*u)
            cw = (width - 3*margin) // 2
            if y + int(72*u) < h - int(40*u):
                weights_column(panel, margin, y, u, cw, 'HR regions (HR-specific set)' if specific else 'HR regions (v2 stabilized weights)', hr_w, HR_COLOR)
                weights_column(panel, 2*margin + cw, y, u, cw, 'RR regions (v2 window weights)', rr_w, RR_COLOR)
            v1.text(panel, ('RED = HR regions (HR-specific, from Module 1 v2 candidates)' if specific else 'RED = HR regions (Module 1 v2)') + '  CYAN = RR regions  WHITE = SAM 2 infant mask',
                    (margin, h - int(34*u)), .38*u, MUTED, max(1, int(u)))
            v1.text(panel, 'values shown only for validated windows; research prototype, not for clinical use',
                    (margin, h - int(14*u)), .38*u, MUTED, max(1, int(u)))
            canvas = np.hstack([left, panel])
            if writer is None:
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot open a video writer for {target}')
            writer.write(canvas)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--module1-config', default='D5')
    ap.add_argument('--results', help='Module 2 v2 result folder (default HR RR using module1 v2/runs/<run>/<config>)')
    ap.add_argument('--videos', nargs='+')
    ap.add_argument('--caregiver', help='folder from apply_caregiver_labels.py (<results>/caregiver): adds caregiver labels')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    res = Path(a.results).resolve() if a.results else M2/'runs'/run.name/a.module1_config
    table = pd.read_csv(ROOT/'module1_v2/runs'/run.name/f'dynamic_roi_v2_{a.module1_config}'/'dynamic_roi_v2.csv', low_memory=False)
    hr = pd.read_csv(res/'hr_windows.csv') if (res/'hr_windows.csv').is_file() else pd.DataFrame(columns=['video_id'])
    rr = pd.read_csv(res/'rr_windows.csv') if (res/'rr_windows.csv').is_file() else pd.DataFrame(columns=['video_id'])
    segs = None
    if a.caregiver:
        care = Path(a.caregiver).resolve()
        hr, rr = (pd.read_csv(care/f'{k}_windows_caregiver.csv') if (care/f'{k}_windows_caregiver.csv').is_file() else t
                  for k, t in (('hr', hr), ('rr', rr)))
        seg_path = M2/'runs'/run.name/'caregiver'/'caregiver_segments.csv'
        segs = pd.read_csv(seg_path) if seg_path.is_file() else None
    manifest = pd.read_csv(run/'manifest.csv')
    videos = a.videos or sorted(set(hr.video_id) | set(rr.video_id))
    sam = pd.read_csv(run/'outputs/sam2/sam2_masks.csv', low_memory=False)
    target_dir = res/'monitor'
    target_dir.mkdir(exist_ok=True)
    for video in videos:
        record = manifest[manifest.video_id == video].iloc[0].to_dict()
        wave_path = res/'waveforms'/f'{video}.npz'
        waves = dict(np.load(wave_path)) if wave_path.is_file() else {}
        target = target_dir/f'{video}_monitor_v2.mp4'
        render(video, record, table[table.video_id == video], hr[hr.video_id == video].sort_values('window_end_sec'),
               rr[rr.video_id == video].sort_values('window_end_sec'), waves, sam[sam.video_id == video],
               run/'outputs', target, None if segs is None else segs[segs.video_id == video])
        print(video, '->', target, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
