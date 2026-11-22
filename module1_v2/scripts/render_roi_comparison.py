"""Side-by-side dynamic-ROI comparison video: Module 1 v1 (per-frame top-k, = D0) vs Module 1 v2 (default D5).

Both selections are computed here from the SAME Module 1 candidates of one run, so the only difference is the
selection method. v2 uses the stabilizer settings of a frozen tuning log (default: final34).
  left   frame with the v1 selected regions (orange); the top region filled, companions outlined; a selected
         2-point box, which carries no pixels in v1, is drawn dashed grey and labelled 'no pixels'
  right  the same frame with the v2 selected regions (blue)
  below  timeline of the whole video: one tick per change of the top region (v1 top row, v2 bottom row), a cursor
         at the current time, and switches per second (whole video and so far)
Also writes <video>_v1_vs_v2.png (three frames + timeline) for figures.
Usage: python module1_v2/scripts/render_roi_comparison.py --run runs/<run> [--config D5] [--videos NEW_1]
       [--tuning-log module1_v2/runs/final34_paper/tuning_log.json] [--video-path <file if it moved>]
Output: module1_v2/runs/<run>/roi_comparison_<config>/<video>_v1_vs_v2.mp4 (mp4v; play in VLC)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
import run_v2_evaluation as ev                       # noqa: E402
from module1 import paper_eval as pe                 # noqa: E402
from module2.extraction import hold_indices          # noqa: E402
from run_paper_evaluation import EXTRACT             # noqa: E402
from module1_v2 import selection as sel              # noqa: E402

V1_COLOR, V2_COLOR, DEAD_COLOR = (52, 104, 235), (214, 120, 42), (150, 150, 150)   # BGR: orange, blue, grey
INK, MUTED, BG = (235, 235, 235), (150, 150, 150), (24, 20, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FEATURES = ['visibility_keypoint', 'motion_nuisance', 'motion_total', 'motion_residual', 'respiratory_ratio']


def text(img, s, org, scale, color, th=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, th, cv2.LINE_AA)


def selections(cand, frames, flags, fps):
    geometry, _, _, _ = sel.select_video(cand, frames, flags, fps, 3, 1., None)
    out = {}
    for f in frames:
        rows = geometry.get(f) or []
        out[f] = [(name, w, poly) for name, w, poly, _ in rows]
    return out


def top_switch_times(selected, frames, fps_native):
    top = [selected[f][0][0] if selected[f] else None for f in frames]
    return np.array([frames[i] / fps_native for i in range(1, len(top)) if top[i] and top[i - 1] and top[i] != top[i - 1]])


def draw_panel(image, regions, color, title, scale, u, box_mode, x0=0):
    frame = cv2.resize(image[:, x0:], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for k, (name, w, poly) in enumerate(regions):
        pix = sel.pixel_polygon(poly, box_mode)
        dead = pix is None
        shape = sel.pixel_polygon(poly, 'rectangle') if dead else pix
        if shape is None:
            continue
        pts = np.rint((np.asarray(shape, float) - [x0, 0]) * scale).astype(np.int32)
        if dead:                                     # selected but carries no pixels (v1 2-point box)
            for a, b in zip(pts, np.roll(pts, -1, 0)):
                for t0 in np.linspace(0, 1, 12, endpoint=False)[::2]:
                    p0, p1 = a + (b - a) * t0, a + (b - a) * (t0 + 1 / 12)
                    cv2.line(frame, tuple(int(x) for x in p0), tuple(int(x) for x in p1), DEAD_COLOR, max(2, int(2*u)), cv2.LINE_AA)
            text(frame, f'{name}: no pixels', (pts[:, 0].min(), pts[:, 1].max() + int(16*u)), .45*u, DEAD_COLOR)
            continue
        if k == 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            frame = cv2.addWeighted(overlay, .35, frame, .65, 0)
        cv2.polylines(frame, [pts], True, color, max(2, int((3 if k == 0 else 1.6)*u)), cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], int(56*u)), BG, -1)
    for k, line in enumerate(title.split('|')):
        text(frame, line.strip(), (int(8*u), int((22 + 24 * k)*u)), .55*u, color, max(1, int(1.4*u)))
    y = frame.shape[0] - int(12*u) - int(22*u) * (len(regions) - 1)
    for k, (name, w, poly) in enumerate(regions):
        tag = ' (no pixels)' if sel.pixel_polygon(poly, box_mode) is None else ''
        text(frame, f"{'TOP ' if k == 0 else '    '}{name}  w={w:.2f}{tag}", (int(10*u), y), .45*u, color if not tag else DEAD_COLOR)
        y += int(22*u)
    return frame


def timeline(width, u, duration, t, s1, s2, rate1, rate2):
    h = int(150*u)
    strip = np.full((h, width, 3), BG, np.uint8)
    x0, x1 = int(150*u), width - int(20*u)
    rows = [(int(45*u), s1, V1_COLOR, 'v1 top-k', rate1), (int(100*u), s2, V2_COLOR, 'v2 stabilized', rate2)]
    for y, sw, color, name, rate in rows:
        cv2.line(strip, (x0, y), (x1, y), (70, 66, 62), 1)
        for ts in sw:
            x = int(x0 + ts / duration * (x1 - x0))
            cv2.line(strip, (x, y - int(14*u)), (x, y + int(14*u)), color, max(1, int(u)))
        so_far = (sw <= t).sum() / max(t, 1e-6)
        text(strip, name, (int(10*u), y + int(5*u)), .5*u, color, max(1, int(1.3*u)))
        text(strip, f'{rate:.2f} switches/s (whole video)   {so_far:.2f}/s so far', (x0, y - int(18*u)), .42*u, color)
    xc = int(x0 + min(t, duration) / duration * (x1 - x0))
    cv2.line(strip, (xc, int(20*u)), (xc, h - int(18*u)), INK, max(1, int(1.5*u)))
    text(strip, f't = {t:.1f} s', (min(xc + 4, width - int(90*u)), h - int(4*u)), .42*u, INK)
    text(strip, 'tick = change of the top region', (x0, h - int(4*u)), .38*u, MUTED)
    return strip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', default='D5')
    ap.add_argument('--tuning-log', default=str(V2/'runs/final34_paper/tuning_log.json'))
    ap.add_argument('--videos', nargs='+')
    ap.add_argument('--video-path', help='use this file when the manifest path no longer exists (one video)')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    out, work = run/'outputs', V2/'runs'/run.name
    configs = ev.configurations(json.loads(Path(a.tuning_log).read_text())['chosen'])
    flags1, flags2 = configs['D0'], configs[a.config]
    target = work/f'roi_comparison_{a.config}'
    target.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False, usecols=ev.CAND_COLUMNS)
    feat = pd.read_csv(work/'features/roi_features_median.csv', usecols=['video_id', 'frame_index', 'roi_name'] + FEATURES)
    cand = cand.merge(feat, on=['video_id', 'frame_index', 'roi_name'], how='left')
    base = out/'confidence/roi_confidence.csv'
    frames0 = pd.read_csv(base if base.is_file() else out/'confidence_sam2/roi_confidence.csv',
                          usecols=['video_id', 'frame_index'], low_memory=False)
    manifest = pd.read_csv(run/'manifest.csv')
    summary = []
    for r in manifest.itertuples():
        if a.videos and r.video_id not in a.videos:
            continue
        path = Path(r.video_path)
        if not path.is_file():
            if not a.video_path:
                raise SystemExit(f'{path} not found; pass --video-path')
            path = Path(a.video_path)
        c = cand[cand.video_id == r.video_id]
        frames = sorted(c.frame_index.unique())
        fps_sampled = len(set(frames0[frames0.video_id == r.video_id].frame_index) | set(frames)) / float(r.duration_sec)
        s1, s2 = selections(c, frames, flags1, fps_sampled), selections(c, frames, flags2, fps_sampled)
        fps, n = float(r.fps), int(r.frame_count)
        sw1, sw2 = top_switch_times(s1, frames, fps), top_switch_times(s2, frames, fps)
        duration = float(r.duration_sec)
        rate1, rate2 = len(sw1) / duration, len(sw2) / duration
        hold = hold_indices(frames, n, fps, EXTRACT.max_hold_sec)
        cap = cv2.VideoCapture(str(path))
        # crop black side bars (pillarboxed portrait clips) once, from a few frames
        probe = []
        for k in np.linspace(0, n - 1, 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, im = cap.read()
            if ok:
                probe.append(im.max(axis=(0, 2)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cols = np.flatnonzero(np.max(probe, axis=0) > 20) if probe else np.array([])
        xa, xb = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, None)
        writer = None
        stills, picks = [], [int(n * .2), int(n * .5), int(n * .8)]
        mp4 = target/f'{r.video_id}_v1_vs_v2.mp4'
        for i in range(n):
            ok, image = cap.read()
            if not ok:
                break
            image = image[:, :xb] if xb else image
            target_h = 720. if (image.shape[1] - xa) / image.shape[0] < .8 else 540.   # portrait content: taller panels
            scale = float(np.clip(target_h / image.shape[0], .3, 2.))
            u = image.shape[0] * scale / 540
            src = int(hold[i])
            left = draw_panel(image, s1.get(src, []) if src >= 0 else [], V1_COLOR,
                              f'Module 1 v1: per-frame top-k | {rate1:.2f} switches/s', scale, u, flags1.get('box_roi_pixels', 'skip'), xa)
            right = draw_panel(image, s2.get(src, []) if src >= 0 else [], V2_COLOR,
                               f'Module 1 v2 ({a.config}): stabilized | {rate2:.2f} switches/s', scale, u,
                               flags2.get('box_roi_pixels', 'skip'), xa)
            gap = np.full((left.shape[0], int(6*u), 3), BG, np.uint8)
            top = np.hstack([left, gap, right])
            width = max(top.shape[1], int(980*u))
            if top.shape[1] < width:                  # pad narrow (portrait) panels so the timeline stays readable
                pad = (width - top.shape[1]) // 2
                top = cv2.copyMakeBorder(top, 0, 0, pad, width - top.shape[1] - pad, cv2.BORDER_CONSTANT, value=BG)
            canvas = np.vstack([top, timeline(top.shape[1], u, duration, i / fps, sw1, sw2, rate1, rate2)])
            if writer is None:
                writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if i in picks:
                stills.append(top)
        cap.release()
        if writer is not None:
            writer.release()
        if stills:
            strip = timeline(stills[0].shape[1], stills[0].shape[0] / 540, duration, duration, sw1, sw2, rate1, rate2)
            cv2.imwrite(str(target/f'{r.video_id}_v1_vs_v2.png'), np.vstack(stills + [strip]))
        summary.append(dict(video_id=r.video_id, duration_sec=duration, v1_switches_per_sec=rate1, v2_switches_per_sec=rate2,
                            v1_dead_selection_percent=100 * float(np.mean([sel.pixel_polygon(p_, 'skip') is None
                                                                           for f in frames for _, _, p_ in s1[f]] or [np.nan])),
                            v2_dead_selection_percent=100 * float(np.mean([sel.pixel_polygon(p_, flags2.get('box_roi_pixels', 'skip')) is None
                                                                           for f in frames for _, _, p_ in s2[f]] or [np.nan])),
                            video=str(mp4), still=str(target/f'{r.video_id}_v1_vs_v2.png')))
        print(f'{r.video_id}: v1 {rate1:.2f} switches/s, v2 {rate2:.2f} switches/s -> {mp4}', flush=True)
    if summary:
        table = pd.DataFrame(summary)
        old = target/'switch_rates.csv'
        if old.is_file():
            prev = pd.read_csv(old)
            table = pd.concat([prev[~prev.video_id.isin(table.video_id)], table], ignore_index=True)
        table.to_csv(old, index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
