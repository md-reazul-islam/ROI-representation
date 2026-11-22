"""Caregiver contact detection: when does an adult touch the infant or take over the infant region?

Add-on of "HR RR using module1 v2". It does not change Module 1 v2, module2_v2 or any HR / RR computation; it only
produces a time line of caregiver contact that apply_caregiver_labels.py turns into window labels.

Cues per analysed frame (default 10 frames/s), relative to the Module 1 SAM 2 infant mask:
  hand  (contact)  COCO pose (YOLO11s-pose) on the full frame. Every person that is not the infant contributes its
                   wrists; hand point = wrist + 0.35 x (wrist - elbow). Contact when a hand point lies within 2 % of
                   the frame diagonal of the infant mask.
  mask  (contact)  SAM 2 mask growth: the infant mask area jumps above 1.40 x the median of the clean masks of the
                   previous 10 s. This is what happens when SAM 2 absorbs a caregiver's hands / arms into the
                   "infant" mask (seen on NEW_10). A shrink below 0.60 x is only reported (mask_shrink): it also
                   happens when SAM 2 corrects an over-segmentation (seen on NEW_1104, no caregiver).
  body  (reported) COCO segmentation (YOLO11s-seg): adult silhouette touching the infant mask. Not used for contact by
                   default, because in a 2-D image an adult standing behind the infant looks the same as touching.
Which detection is the infant: highest IoU (>= 0.30) with the infant box; a pose whose confident shoulders / hips lie
>= 50 % inside the infant mask; a silhouette >= 60 % inside the infant mask. An adult must have >= 25 % of its box
outside the (5 % enlarged) infant box. Adult hand points within 1 % of the diagonal of an infant wrist are dropped.
While the SAM 2 mask is flagged as changed, the last clean mask is the infant reference.
Per-frame contact is smoothed (majority of 5 frames), merged into episodes (gaps <= 1 s, >= 0.3 s long) and padded
by 1 s on both sides. All settings: configs/caregiver.yaml (fixed a priori, not tuned on evaluation videos).

Usage:  python "HR RR using module1 v2/scripts/detect_caregiver_contact.py" --run runs/<RUN> [--videos NEW_10]
        [--no-preview] [--device 0]
Writes: HR RR using module1 v2/runs/<RUN>/caregiver/
          caregiver_frames.csv     one row per analysed frame (adults, hands, distances, mask change, contact)
          caregiver_segments.csv   contact episodes: start/end and padded start/end (= excluded from evaluation)
          caregiver_summary.csv    per video: % adult in view, % contact, episodes, excluded seconds
          metadata.json            settings, model files (sha256), timing
          preview/<video>_caregiver.mp4   infant reference (white), adults (orange), hand points (red = contact)
Limitations: 2-D contact (a hand hovering over the infant counts); a caregiver not detected as a person (only a few
fingers in view) is missed unless SAM 2 absorbs it; no ground-truth contact labels yet (check the preview video).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
CFG = yaml.safe_load((M2/'configs/caregiver.yaml').read_text())
ORANGE, WHITE, RED, YELLOW, PURPLE = (0, 140, 255), (235, 235, 235), (40, 40, 230), (0, 220, 255), (200, 60, 170)
TORSO, WRISTS = (5, 6, 11, 12), ((9, 7), (10, 8))                 # COCO: shoulders + hips; (wrist, elbow) left / right


def box_of(value):
    try:
        b = json.loads(value) if isinstance(value, str) else None
        return np.asarray(b, float) if b is not None and len(b) == 4 else None
    except (ValueError, TypeError):
        return None


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def outside_fraction(box, ref, enlarge=.05):
    w, h = ref[2] - ref[0], ref[3] - ref[1]
    r = (ref[0] - enlarge * w, ref[1] - enlarge * h, ref[2] + enlarge * w, ref[3] + enlarge * h)
    ix = max(0., min(box[2], r[2]) - max(box[0], r[0]))
    iy = max(0., min(box[3], r[3]) - max(box[1], r[1]))
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return 1. - ix * iy / area


def mask_change_flags(rows, fps):
    """Per SAM 2 row: area ratio to the clean baseline, changed flag and index of the reference (last clean) row."""
    c = CFG['mask_change']
    areas = rows.mask_area.astype(float).to_numpy()
    usable = rows.mask_reliable.astype(bool).to_numpy() & np.isfinite(areas) & (areas > 0)
    t = rows.frame_index.to_numpy() / fps
    init = usable & (t <= t[0] + c['init_sec']) if len(t) else usable
    clean_t, clean_a = list(t[init]), list(areas[init])
    ratio, changed, ref = np.full(len(rows), np.nan), np.zeros(len(rows), bool), np.full(len(rows), -1)
    shrink = np.zeros(len(rows), bool)
    last_clean, flag_start = -1, None
    for i in range(len(rows)):
        if not usable[i]:
            ref[i] = last_clean
            continue
        recent = [a for tt, a in zip(clean_t, clean_a) if t[i] - c['baseline_sec'] <= tt <= t[i]]
        base = np.median(recent) if recent else (np.median(clean_a) if clean_a else areas[i])
        ratio[i] = areas[i] / base
        bad = ratio[i] > c['growth_ratio']                             # growth: something entered the mask
        shrink[i] = ratio[i] < c['shrink_ratio']                       # reported only (also SAM 2 self-correction)
        if bad:
            flag_start = t[i] if flag_start is None else flag_start
            if t[i] - flag_start > c['max_flag_sec']:                  # lasting change: re-framing, new baseline
                bad, flag_start = False, None
                clean_t, clean_a = [], []
        else:
            flag_start = None
        changed[i] = bad
        if not bad:
            clean_t.append(t[i])
            clean_a.append(areas[i])
            last_clean = i
        ref[i] = i if not bad else last_clean
    return ratio, changed, shrink, ref


class InfantRegion:
    """Infant reference per frame: nearest SAM 2 mask (<= 1 s), or the last clean one while the mask is changed;
    the Module 1 final infant box as a rectangle when no reliable mask exists."""

    def __init__(self, sam_rows, sam2_dir, shape, margin, fps):
        self.rows = sam_rows.sort_values('frame_index').reset_index(drop=True)
        self.frames = self.rows.frame_index.to_numpy()
        self.ratio, self.changed, self.shrink, self.ref = (mask_change_flags(self.rows, fps) if len(self.rows)
                                                            else ([], [], [], []))
        self.dir, self.shape, self.margin = Path(sam2_dir), shape, margin
        self.cache = {}

    def _load(self, j):
        if j in self.cache:
            return self.cache[j]
        row = self.rows.iloc[j]
        box = box_of(row.get('final_infant_bbox'))
        mask, source = None, 'none'
        if bool(row.get('mask_reliable')) and isinstance(row.get('mask_path'), str):
            m = cv2.imread(str(self.dir/row['mask_path']), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == self.shape:
                mask, source = m > 0, 'sam2'
        if mask is None and box is not None:
            mask = np.zeros(self.shape, bool)
            x1, y1, x2, y2 = np.clip(np.rint(box), 0, [self.shape[1], self.shape[0]] * 2).astype(int)
            mask[y1:y2, x1:x2] = True
            source = 'box'
        value = None
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask)
            mbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
            dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)   # distance to the infant mask
            value = dict(mask=mask, box=mbox if source == 'sam2' else box, dist=dist, area=int(mask.sum()), source=source)
        self.cache = {j: value}                                        # frames are read in order: keep one
        return value

    def get(self, frame, fps):
        if not len(self.frames):
            return None, dict(mask_ratio=np.nan, mask_changed=False, mask_shrink=False)
        i = np.searchsorted(self.frames, frame, side='right') - 1
        cands = [j for j in (i, i + 1) if 0 <= j < len(self.frames)]
        j = min(cands, key=lambda k: abs(self.frames[k] - frame))
        info = dict(mask_ratio=float(self.ratio[j]) if np.isfinite(self.ratio[j]) else np.nan,
                    mask_changed=bool(self.changed[j]), mask_shrink=bool(self.shrink[j]))
        if abs(self.frames[j] - frame) > fps:                          # nothing within 1 s
            return None, dict(mask_ratio=np.nan, mask_changed=False, mask_shrink=False)
        k = int(self.ref[j])
        if k < 0 or abs(self.frames[k] - frame) > 30 * fps:            # no clean reference within 30 s
            region = self._load(j)
            info['reference'] = 'current_changed' if info['mask_changed'] else 'current'
        else:
            region = self._load(k)
            info['reference'] = 'current' if k == j else 'last_clean'
        return region, info


def people(result, region, diag):
    """Infant index (or None), adult indices and confident keypoints of a pose / segmentation result."""
    if result.boxes is None or not len(result.boxes) or region is None:
        return None, [], None
    boxes = result.boxes.xyxy.cpu().numpy()
    kps = result.keypoints.data.cpu().numpy() if getattr(result, 'keypoints', None) is not None else None
    ious = [iou(b, region['box']) for b in boxes]
    infant = int(np.argmax(ious)) if max(ious) >= CFG['infant']['iou_infant'] else None
    h, w = region['mask'].shape
    adults = []
    for i, b in enumerate(boxes):
        if i == infant or outside_fraction(b, region['box']) < CFG['infant']['min_outside_fraction']:
            continue
        if kps is not None:
            torso = [k for k in kps[i][list(TORSO)] if k[2] >= CFG['contact']['wrist_conf']]
            inside = [region['mask'][int(np.clip(k[1], 0, h - 1)), int(np.clip(k[0], 0, w - 1))] for k in torso]
            if inside and np.mean(inside) >= CFG['infant']['torso_inside_fraction']:
                continue                                               # torso on the infant: a duplicate of the infant
        adults.append(i)
    return infant, adults, (boxes, kps)


def hand_points(kps, adults, infant):
    c = CFG['contact']
    own = []
    if infant is not None:
        own = [kps[infant][wr, :2] for wr, _ in WRISTS if kps[infant][wr, 2] >= c['wrist_conf']]
    out = []
    for i in adults:
        k = kps[i]
        for wrist, elbow in WRISTS:
            if k[wrist, 2] < c['wrist_conf']:
                continue
            p = k[wrist, :2].copy()
            if k[elbow, 2] >= c['wrist_conf']:
                p = p + c['hand_extension'] * (k[wrist, :2] - k[elbow, :2])
            out.append((float(p[0]), float(p[1]), k[wrist, :2]))
    return out, own


def analyse_frame(image, region, info, pose_model, seg_model, device):
    m = CFG['models']
    h, w = image.shape[:2]
    diag = float(np.hypot(w, h))
    margin = CFG['contact']['margin_diag_fraction'] * diag
    rec = dict(mask_source='none' if region is None else region['source'], reference=info.get('reference', 'none'),
               mask_ratio=info['mask_ratio'], mask_change=info['mask_changed'], mask_shrink=info['mask_shrink'], n_adults_pose=0, n_adults_seg=0,
               n_hands=0, hand_min_dist_px=np.nan, hand_contact=False, body_overlap_px=0, body_overlap_fraction=0.,
               body_contact=False)
    draw = dict(hands=[], adult_boxes=[], silhouettes=[])
    kw = dict(imgsz=m['imgsz'], conf=m['person_conf'], classes=[0], device=device, verbose=False)
    p = pose_model(image, **kw)[0]
    s = seg_model(image, **kw)[0]
    # cue 1: adult hands near the infant mask
    infant, adults, data = people(p, region, diag)
    if adults and data[1] is not None:
        rec['n_adults_pose'] = len(adults)
        draw['adult_boxes'] = [data[0][i].tolist() for i in adults]
        hands, own = hand_points(data[1], adults, infant)
        own_tol = CFG['infant']['own_wrist_diag_fraction'] * diag
        dists = []
        for x, y, wrist in hands:
            if any(np.hypot(*(wrist - o)) <= own_tol for o in own):
                continue                                               # the infant's own wrist
            inside = 0 <= x < w and 0 <= y < h
            d = float(region['dist'][int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]) if inside else np.nan
            dists.append(d)
            draw['hands'].append((x, y, inside and d <= margin))
        rec['n_hands'] = len(dists)
        finite = [d for d in dists if np.isfinite(d)]
        if finite:
            rec['hand_min_dist_px'] = min(finite)
            rec['hand_contact'] = bool(min(finite) <= margin)
    # cue 3 (reported): adult silhouette touching the infant mask
    if region is not None and s.masks is not None and s.boxes is not None and len(s.boxes):
        _, sadults, _ = people(s, region, diag)
        b = CFG['body']
        near = region['dist'] <= b['margin_diag_fraction'] * diag
        best = 0
        for i in sadults:
            if i >= len(s.masks.xy) or len(s.masks.xy[i]) < 3:
                continue
            sil = np.zeros((h, w), np.uint8)
            cv2.fillPoly(sil, [np.rint(s.masks.xy[i]).astype(np.int32)], 1)
            sil = sil.astype(bool)
            n = int(sil.sum())
            if not n or (sil & region['mask']).sum() / n >= CFG['infant']['infant_mask_fraction']:
                continue                                               # mostly inside the infant mask: the infant
            rec['n_adults_seg'] += 1
            draw['silhouettes'].append(s.masks.xy[i])
            best = max(best, int((sil & near).sum()))
        rec['body_overlap_px'] = best
        rec['body_overlap_fraction'] = best / max(region['area'], 1)
        rec['body_contact'] = bool(best >= max(b['overlap_min_fraction'] * region['area'], b['overlap_min_px']))
    rec['adult_in_view'] = rec['n_adults_pose'] > 0 or rec['n_adults_seg'] > 0
    body = rec['body_contact'] and CFG['body']['triggers_contact']
    rec['contact_raw'] = rec['hand_contact'] or rec['mask_change'] or body
    rec['contact_source'] = '+'.join(n for n, v in (('hand', rec['hand_contact']), ('mask', rec['mask_change']),
                                                     ('body', body)) if v)
    rec['hands'] = json.dumps([[round(x, 1), round(y, 1)] for x, y, _ in draw['hands']])
    rec['adult_boxes'] = json.dumps([[round(v, 1) for v in bb] for bb in draw['adult_boxes']])
    return rec, draw


def segments(frames, fps_analysed, duration):
    """Smoothed contact -> episodes and padded episodes (excluded from evaluation)."""
    t = CFG['temporal']
    f = frames.sort_values('timestamp_sec')
    smooth = f.contact_raw.astype(float).rolling(t['smooth_frames'], center=True, min_periods=1).mean() > .5
    frames.loc[f.index, 'contact'] = smooth.to_numpy()
    half = .5 / fps_analysed
    times, on = f.timestamp_sec.to_numpy(), smooth.to_numpy()
    raw, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            raw.append([max(0., times[i] - half), min(duration, times[j] + half)])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= t['merge_gap_sec']:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in (s for s in merged if s[1] - s[0] >= t['min_segment_sec']):
        pa, pb = max(0., a - t['pad_sec']), min(duration, b + t['pad_sec'])
        if out and pa <= out[-1]['pad_end_sec']:                       # padded episodes touching: one exclusion span
            out[-1].update(end_sec=b, pad_end_sec=pb)
        else:
            out.append(dict(start_sec=a, end_sec=b, pad_start_sec=pa, pad_end_sec=pb))
    return out


def preview_frame(image, region, rec, draw, t):
    scale = min(1., 960. / image.shape[1])
    frame = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1 else image.copy()
    if draw['silhouettes']:
        overlay = frame.copy()
        for poly in draw['silhouettes']:
            cv2.fillPoly(overlay, [np.rint(np.asarray(poly) * scale).astype(np.int32)], ORANGE)
        frame = cv2.addWeighted(overlay, .30, frame, .70, 0)
    for b in draw['adult_boxes']:
        x1, y1, x2, y2 = (np.asarray(b) * scale).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 2)
    if region is not None:
        m = cv2.resize(region['mask'].astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, WHITE, 2, cv2.LINE_AA)
    for x, y, touching in draw['hands']:
        cv2.circle(frame, (int(x * scale), int(y * scale)), 9, RED if touching else YELLOW, -1, cv2.LINE_AA)
    ratio = '' if pd.isna(rec['mask_ratio']) else f'  mask x{rec["mask_ratio"]:.2f}'
    cv2.putText(frame, f't={t:.1f}s  adults={max(rec["n_adults_pose"], rec["n_adults_seg"])}  hands={rec["n_hands"]}'
                f'{ratio}  ref={rec["reference"]}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, WHITE, 2, cv2.LINE_AA)
    if rec['contact_raw']:
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), PURPLE, -1)
        cv2.putText(frame, f'CAREGIVER CONTACT ({rec["contact_source"]}; this frame, before smoothing)',
                    (10, frame.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, .6, WHITE, 2, cv2.LINE_AA)
    return frame


def process_video(record, sam_rows, sam2_dir, pose_model, seg_model, device, preview_path):
    cap = cv2.VideoCapture(str(record['video_path']))
    if not cap.isOpened() or not int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise FileNotFoundError(f"cannot read the video {record['video_path']} (moved? use --video-root <folder>)")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(record['fps'])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(record['frame_count'])
    step = max(1, int(round(fps / CFG['sampling']['fps'])))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    regions = InfantRegion(sam_rows, sam2_dir, (h, w), CFG['contact']['margin_diag_fraction'] * np.hypot(w, h), fps)
    writer, rows = None, []
    try:
        for i in range(n):
            if not cap.grab():
                break
            if i % step:
                continue
            ok, image = cap.retrieve()
            if not ok:
                break
            t = i / fps
            region, info = regions.get(i, fps)
            rec, draw = analyse_frame(image, region, info, pose_model, seg_model, device)
            rows.append(dict(video_id=record['video_id'], frame_index=i, timestamp_sec=t, **rec))
            if preview_path is not None:
                frame = preview_frame(image, region, rec, draw, t)
                if writer is None:
                    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*'mp4v'), fps / step,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    frames = pd.DataFrame(rows)
    duration = n / fps
    segs = segments(frames, fps / step, duration) if len(frames) else []
    if 'contact' not in frames:
        frames['contact'] = False
    frames['contact'] = frames.contact.astype(bool)
    for s in segs:
        s['video_id'] = record['video_id']
        part = frames[(frames.timestamp_sec >= s['start_sec']) & (frames.timestamp_sec <= s['end_sec']) & frames.contact_raw]
        s['source'] = ','.join(sorted({x for v in part.contact_source for x in v.split('+') if x}))
    pct = lambda col: 100 * frames[col].astype(bool).mean() if len(frames) else np.nan   # noqa: E731
    summary = dict(video_id=record['video_id'], duration_sec=duration, analysed_frames=len(frames), analysed_fps=fps / step,
                   adult_in_view_percent=pct('adult_in_view'), hand_contact_percent=pct('hand_contact'),
                   mask_change_percent=pct('mask_change'), body_contact_percent=pct('body_contact'),
                   contact_percent=pct('contact'), episodes=len(segs),
                   contact_sec=sum(s['end_sec'] - s['start_sec'] for s in segs),
                   excluded_sec=sum(s['pad_end_sec'] - s['pad_start_sec'] for s in segs),
                   mask_sam2_percent=100 * frames.mask_source.eq('sam2').mean() if len(frames) else np.nan,
                   no_infant_region_percent=100 * frames.mask_source.eq('none').mean() if len(frames) else np.nan)
    summary['excluded_percent'] = 100 * summary['excluded_sec'] / duration if duration else np.nan
    return frames, segs, summary


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--videos', nargs='+')
    ap.add_argument('--device', default=None, help="'0' for the first GPU, 'cpu' (default: GPU when available)")
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--video-root', help='folder to look for the videos when the manifest path no longer exists')
    a = ap.parse_args(argv)
    from ultralytics import YOLO                                       # imported here: heavy
    import torch
    device = a.device or ('0' if torch.cuda.is_available() else 'cpu')
    run = (ROOT/a.run).resolve()
    target = M2/'runs'/run.name/'caregiver'
    target.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run/'manifest.csv')
    if a.videos:
        manifest = manifest[manifest.video_id.isin(a.videos)]
    sam = pd.read_csv(run/'outputs/sam2/sam2_masks.csv', low_memory=False)
    paths = {k: ROOT/CFG['models'][k] for k in ('pose', 'seg')}
    for p in paths.values():
        if not p.is_file():
            raise SystemExit(f'Missing model {p}: download https://github.com/ultralytics/assets/releases/download/v8.3.0/{p.name}')
    pose_model, seg_model = YOLO(str(paths['pose'])), YOLO(str(paths['seg']))
    started = time.perf_counter()
    all_frames, all_segs, summaries = [], [], []
    for record in manifest.to_dict('records'):
        if a.video_root and not Path(record['video_path']).is_file():
            record['video_path'] = str((ROOT/a.video_root/record['filename']).resolve())
        preview = None
        if not a.no_preview:
            (target/'preview').mkdir(exist_ok=True)
            preview = target/'preview'/f"{record['video_id']}_caregiver.mp4"
        t0 = time.perf_counter()
        frames, segs, summary = process_video(record, sam[sam.video_id == record['video_id']], run/'outputs/sam2',
                                              pose_model, seg_model, device, preview)
        summary['wall_time_sec'] = time.perf_counter() - t0
        all_frames.append(frames)
        all_segs += segs
        summaries.append(summary)
        print(f"{record['video_id']}: {summary['analysed_frames']} frames, adult in view {summary['adult_in_view_percent']:.0f}%, "
              f"contact {summary['contact_percent']:.0f}%, {summary['episodes']} episode(s), excluded "
              f"{summary['excluded_sec']:.1f} s ({summary['wall_time_sec']:.0f} s)", flush=True)
    frames = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    segs = pd.DataFrame(all_segs, columns=['video_id', 'start_sec', 'end_sec', 'pad_start_sec', 'pad_end_sec', 'source'])
    summary = pd.DataFrame(summaries)
    for name, table in (('caregiver_frames.csv', frames), ('caregiver_segments.csv', segs), ('caregiver_summary.csv', summary)):
        path = target/name
        if a.videos and path.is_file():                                 # keep the other videos' rows
            old = pd.read_csv(path)
            table = pd.concat([old[~old.video_id.isin(manifest.video_id)], table], ignore_index=True)
        table.to_csv(path, index=False)
    meta = dict(run=str(run), config=CFG, device=device, wall_time_sec=time.perf_counter() - started,
                models={k: dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for k, p in paths.items()},
                videos=list(manifest.video_id))
    (target/'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    print(summary.round(1).to_string(index=False))
    print(segs.round(2).to_string(index=False) if len(segs) else 'no caregiver contact episodes')
    print('->', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
