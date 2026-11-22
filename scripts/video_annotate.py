"""Render annotated videos of the final Module 1 output.

Each processed (sampled) frame shows the Phase 6 infant box, all Phase 8 candidate
ROIs, and the Phase 10 dynamically selected ROIs with their confidence Q, softmax
weight alpha and rank. The highest-confidence ROI (the top-1 dynamic ROI) is marked
in red. A side panel lists the ranked ROIs. Frames are enlarged (--scale) so the
text stays readable. Only frames that the pipeline processed are written, at the
processing FPS; nothing is interpolated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest
from module1.roi import ROI_NAMES


STRATEGIES = ('top1', 'top2', 'topk', 'confidence_weighted_fusion')
FONT = cv2.FONT_HERSHEY_SIMPLEX
BEST_COLOR = (0, 0, 255)  # BGR red: reserved for the highest-confidence ROI
# One fixed BGR color per region (no reds, so red always means "best ROI").
ROI_COLORS = dict(zip(ROI_NAMES, [
    (255, 255, 0), (0, 255, 255), (0, 215, 255), (80, 255, 160), (80, 200, 80), (255, 180, 0),
    (255, 120, 60), (255, 0, 255), (200, 60, 255), (0, 255, 128), (255, 255, 160), (180, 255, 255),
    (200, 160, 255), (160, 255, 255), (128, 255, 0), (255, 200, 120), (255, 0, 128),
]))
STATUS_COLORS = {'DETECTED': (255, 255, 255), 'RECOVERED': (0, 165, 255)}


def parse_json(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == '':
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def put_text(canvas, text, origin, scale=.6, color=(255, 255, 255), thickness=2, background=None):
    """Outlined text kept fully inside the image; optional filled background box."""
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x = int(min(max(origin[0], 3), canvas.shape[1] - width - 3))
    y = int(min(max(origin[1], height + 3), canvas.shape[0] - baseline - 3))
    if background is not None:
        cv2.rectangle(canvas, (x-3, y-height-4), (x+width+3, y+baseline+2), background, -1)
        cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
        return
    cv2.putText(canvas, text, (x, y), FONT, scale, (0, 0, 0), thickness+3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def roi_points(row, scale):
    polygon = parse_json(row.get('polygon')) or []
    if len(polygon) >= 3:
        return np.rint(np.asarray(polygon, dtype=float) * scale).astype(np.int32)
    bbox = parse_json(row.get('bbox'))
    if bbox:
        x1, y1, x2, y2 = (np.asarray(bbox, dtype=float) * scale).tolist()
        return np.rint([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32)
    return None


def draw_frame(image, rois, selection, recovery, info, show_candidates, scale):
    frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1 else image.copy()
    height = frame.shape[0]
    by_name = {row['roi_name']: row for row in rois}
    scored = sorted((row for row in rois if number(row.get('q')) is not None),
                    key=lambda row: (-number(row['q']), row['roi_name']))
    best = scored[0]['roi_name'] if scored else None
    selected = [name for name, _ in selection if name in by_name]

    # Phase 6 infant localization.
    status = recovery.get('final_status') if recovery else None
    infant_box = parse_json(recovery.get('final_infant_bbox')) if recovery else None
    if infant_box:
        color = STATUS_COLORS.get(status, (200, 200, 200))
        x1, y1, x2, y2 = np.rint(np.asarray(infant_box, dtype=float) * scale).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        confidence = number(recovery.get('final_infant_confidence'))
        put_text(frame, f'infant {status} {confidence:.2f}' if confidence is not None else f'infant {status}',
                 (x1+4, y1-8), .75, color)
    else:
        put_text(frame, 'NO INFANT BOX (MISSED)', (10, 30), .8, (255, 255, 255), 2, background=BEST_COLOR)

    # Non-selected candidates: thin grey outlines.
    if show_candidates:
        for row in rois:
            points = roi_points(row, scale)
            if row.get('roi_status') == 'AVAILABLE' and row['roi_name'] not in selected and points is not None:
                cv2.polylines(frame, [points], True, (150, 150, 150), 1, cv2.LINE_AA)
    # Other selected ROIs in their own color; the best ROI last, in red, on top.
    for rank, name in enumerate(selected, 1):
        if name == best:
            continue
        points = roi_points(by_name[name], scale)
        if points is None:
            continue
        color = ROI_COLORS.get(name, (0, 255, 0))
        cv2.polylines(frame, [points], True, color, 3, cv2.LINE_AA)
        centroid = parse_json(by_name[name].get('centroid'))
        q = number(by_name[name].get('q'))
        if centroid:
            put_text(frame, f'#{rank} {name} Q={q:.2f}', (centroid[0]*scale - 60, centroid[1]*scale), .75, color)
    if best is not None:
        points = roi_points(by_name[best], scale)
        if points is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], BEST_COLOR)
            frame = cv2.addWeighted(overlay, .25, frame, .75, 0)
            cv2.polylines(frame, [points], True, BEST_COLOR, 4, cv2.LINE_AA)
            centroid = parse_json(by_name[best].get('centroid'))
            if centroid:
                put_text(frame, f'BEST #1 {best} Q={number(by_name[best]["q"]):.2f}',
                         (centroid[0]*scale - 80, centroid[1]*scale), .85, (255, 255, 255), 2, background=BEST_COLOR)

    # Side panel: header and ranked list of all scored ROIs.
    panel_width = int(330 * scale)
    panel = np.full((height, panel_width, 3), 28, dtype=np.uint8)
    text, row_height = .45 * scale, int(17 * scale)
    y = row_height + 4
    for line in (f"{info['video_id']} ({info['split']})",
                 f"frame {info['frame_index']}   t = {info['timestamp_sec']:.2f} s",
                 f"strategy: {info['strategy']}"):
        put_text(panel, line, (12, y), text * 1.05)
        y += row_height
    y += row_height // 2
    put_text(panel, 'rank  ROI              Q     alpha', (12, y), text * .95, (190, 190, 190))
    bar_x, bar_width = int(panel_width * .78), int(panel_width * .19)
    for rank, row in enumerate(scored, 1):
        y += row_height
        if y > height - 3 * row_height:
            break
        name, q, alpha = row['roi_name'], number(row['q']), number(row.get('alpha')) or 0.
        if name == best:
            color = BEST_COLOR
            cv2.rectangle(panel, (4, y - row_height + 5), (panel_width - 4, y + 6), (60, 60, 110), -1)
        elif name in selected:
            color = ROI_COLORS.get(name, (0, 255, 0))
        else:
            color = (150, 150, 150)
        marker = '*' if name in selected else ' '
        put_text(panel, f'{marker}{rank:>2}  {name[:15]:<15} {q:5.2f}  {alpha:.2f}', (8, y), text * .95, color)
        bar = int(round(min(max(alpha, 0.), 1.) * bar_width))
        cv2.rectangle(panel, (bar_x, y - int(row_height * .6)), (bar_x + bar, y), color, -1)
    if not scored:
        put_text(panel, 'no scored ROI on this frame', (12, y + row_height), text, BEST_COLOR)
    put_text(panel, 'RED = highest confidence (top-1 ROI)', (12, height - row_height - 6), text * .9, BEST_COLOR)
    put_text(panel, '* = selected     bar = alpha', (12, height - 8), text * .9, (190, 190, 190))
    return np.hstack([frame, panel])


def open_writer(path, fps, size):
    # MPEG-4 Part 2 is available in every OpenCV build (pip wheels lack an H.264 encoder).
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open a video writer for {path}')
    return writer, 'mp4v'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Render annotated Module 1 output videos (infant box + dynamic ROIs + confidence).')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--confidence-dir', help='Phase 10 output directory (default: confidence.output_dir)')
    parser.add_argument('--recovery', help='Phase 6 recovery_results.csv (default: recovery.output_dir)')
    parser.add_argument('--strategy', choices=STRATEGIES, default='topk')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--videos', nargs='+', help='Explicit video IDs, e.g. VID_006 VID_011 (overrides --split)')
    parser.add_argument('--fps', type=float, help='Output FPS (default: processing.target_fps)')
    parser.add_argument('--scale', type=float, default=2.0, help='Enlarge frames for readable text (default 2)')
    parser.add_argument('--hide-candidates', action='store_true', help='Do not draw non-selected candidate ROIs')
    parser.add_argument('--output-dir', default='outputs/annotated_videos')
    args = parser.parse_args(argv)
    if args.scale <= 0:
        raise ValueError('--scale must be positive')

    cfg = load_config(args.config)
    root = Path(cfg['_project_root'])
    confidence_dir = project_path(cfg, args.confidence_dir or cfg.get('confidence', {}).get('output_dir', 'outputs/confidence'))
    recovery_path = project_path(cfg, args.recovery or
                                 str(Path(cfg.get('recovery', {}).get('output_dir', 'outputs/recovery'))/'recovery_results.csv'))
    output = project_path(cfg, args.output_dir)
    fps = args.fps or float(cfg['processing']['target_fps'])
    for path in (confidence_dir/'roi_confidence.csv', confidence_dir/'dynamic_roi.csv', recovery_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Run Phases 6-10 first.')

    confidence = pd.read_csv(confidence_dir/'roi_confidence.csv', low_memory=False)
    dynamic = pd.read_csv(confidence_dir/'dynamic_roi.csv', low_memory=False)
    dynamic = dynamic[dynamic.selection_strategy == args.strategy]
    recovery = pd.read_csv(recovery_path, low_memory=False)
    manifest = DatasetManifest.load(project_path(cfg, args.manifest))

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

    output.mkdir(parents=True, exist_ok=True)
    rois_by_frame = {key: group.to_dict('records')
                     for key, group in confidence.groupby(['video_id', 'frame_index'])}
    selection_by_frame = {}
    for row in dynamic.to_dict('records'):
        names = parse_json(row['selected_roi_names']) or []
        weights = parse_json(row['fusion_weights']) or [None]*len(names)
        selection_by_frame[(row['video_id'], int(row['frame_index']))] = list(zip(names, weights))
    recovery_by_frame = {(row['video_id'], int(row['frame_index'])): row for row in recovery.to_dict('records')}

    written = []
    for _, record in records.iterrows():
        video = str(record.video_id)
        frames = sorted(int(f) for v, f in rois_by_frame if v == video)
        if not frames:
            print(f'{video}: no Phase 10 frames, skipped')
            continue
        path = manifest.resolve_video_path(root, record)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        target = output/f'{video}_{record.split}_{args.strategy}.mp4'
        writer, codec, current_index = None, None, 0
        try:
            for frame_index in frames:
                image = None
                while current_index <= frame_index:
                    ok, decoded = capture.read()
                    if not ok:
                        raise ValueError(f'{video}: decoding failed before frame {frame_index}')
                    current_index += 1
                    if current_index - 1 == frame_index:
                        image = decoded
                rois = rois_by_frame[(video, frame_index)]
                info = dict(video_id=video, split=record.split, frame_index=frame_index,
                            timestamp_sec=float(rois[0]['timestamp_sec']), strategy=args.strategy)
                canvas = draw_frame(image, rois, selection_by_frame.get((video, frame_index), []),
                                    recovery_by_frame.get((video, frame_index)), info, not args.hide_candidates, args.scale)
                if writer is None:
                    writer, codec = open_writer(target, fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        written.append(target)
        print(f'{video}: {len(frames)} frames -> {target} ({codec}, {fps:g} fps)')
    print(f'Done: {len(written)} video(s) in {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
