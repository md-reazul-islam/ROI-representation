"""Figure 1 of the Module 1 v2 manuscript: the 8-stage architecture, illustrated with real data of one video
(frames, fused box, SAM 2 mask, mask-constrained pose and ROIs, selection switches of v1 vs v2, refined pixels).

Usage: python module1_v2/manuscript/make_figure1.py --name final34_paper [--video VID_010] [--out ...png]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
from module1 import paper_eval as pe                       # noqa: E402
from module2.extraction import shrink, skin_pixels         # noqa: E402
from run_paper_evaluation import EXTRACT, boxes_of         # noqa: E402
from run_v2_evaluation import CAND_COLUMNS, configurations  # noqa: E402
from module1_v2 import selection as sel                    # noqa: E402

INK, INK2 = '#1b2a4a', '#3d4a5c'
FILL = ['#eaf3fb', '#e9f6ee', '#fdf5dc', '#f1ecfa', '#e6f4f7', '#eef0fb', '#fcebf1', '#e9f6ee']
EDGE = ['#7fb0d9', '#7cc49a', '#e3c15a', '#b49ad8', '#6fb8c9', '#8f9de0', '#e08aa8', '#7cc49a']
COCO = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4)]
LIMB = ['#e8a33a', '#e8a33a', '#3aa0e8', '#3aa0e8', '#9b59b6', '#2ecc71', '#2ecc71', '#9b59b6', '#e8743a', '#e8743a',
        '#3a6ee8', '#3a6ee8', '#d81b60', '#d81b60', '#d81b60', '#d81b60']


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    return image if ok else None


def crop(image, box, pad=.08):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy))
    return image[y1:y2, x1:x2], (x1, y1)


def square_crop(image, box, pad=.1, size=360):
    """Square crop centred on the box (enlarged by pad), clipped to the image, resized to size x size."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = min(side, w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a = int(np.clip(cx - side / 2, 0, w - side))
    b = int(np.clip(cy - side / 2, 0, h - side))
    part = image[b:b + int(side), a:a + int(side)]
    return cv2.resize(part, (size, size))


def square(image, size=360, fill=255):
    h, w = image.shape[:2]
    s = size / max(h, w)
    image = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))))
    out = np.empty((size, size, 3), np.uint8)
    out[:] = fill
    y, x = (size - image.shape[0]) // 2, (size - image.shape[1]) // 2
    out[y:y + image.shape[0], x:x + image.shape[1]] = image
    return out


def hexrgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def whole(image, box, fill, pad=.06, size=420):
    """The full infant box (portrait or landscape), letterboxed in `fill` to a square."""
    return square(rgb(crop(image, box, pad)[0]), size, fill)


def load(name, video):
    run = ROOT/'runs'/name
    out = run/'outputs'
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=CAND_COLUMNS + ['keypoints_path'])
    features = pd.read_csv(V2/'runs'/name/'features/roi_features_median.csv')
    cand = cand.merge(features[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                                'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    rec = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'], low_memory=False)
    log = json.loads((V2/'runs'/name/'tuning_log.json').read_text())
    return dict(run=run, out=out, manifest=manifest, cand=cand, rec=rec, sam=sam, frames0=frames0,
                configs=configurations(log['chosen']), head=headline(name))


def headline(name):
    """Headline configuration recorded in reports/<run>/headline.json (falls back to D3, the stabilizer alone)."""
    path = V2/'reports'/name/'headline.json'
    return json.loads(path.read_text())['headline'] if path.is_file() else 'D3'


def video_data(D, video):
    r = D['manifest'].loc[video]
    cand = D['cand'][D['cand'].video_id == video]
    frames = sorted(set(D['frames0'][D['frames0'].video_id == video].frame_index) | set(cand.frame_index))
    fps_sampled = len(frames) / float(r.duration_sec)
    sel_frames = sorted(cand.frame_index.unique())
    out = {}
    for cid in ('D0', D['head']):
        geometry, _, _, _ = sel.select_video(cand, sel_frames, D['configs'][cid], fps_sampled, 3, 1., None)
        out[cid] = geometry
    return r, cand, sel_frames, out


def anchors(geometry, frames):
    top = []
    for f in frames:
        rows = geometry.get(f) or []
        top.append(max(rows, key=lambda x: x[1])[0] if rows else None)
    return top


def pick_frame(D, video, cand, geometry):
    sam = D['sam'][(D['sam'].video_id == video) & (D['sam'].mask_reliable == True)]  # noqa: E712
    counts = cand[cand.roi_status == 'AVAILABLE'].groupby('frame_index').size()
    good = [f for f in sam.frame_index if counts.get(f, 0) >= 12 and geometry.get(f)]
    return good[len(good) // 2]


def panels(D, video, condition_videos):
    r, cand, frames, geo = video_data(D, video)
    f = pick_frame(D, video, cand, geo[D['head']])
    image = read(r.video_path, f)
    box = boxes_of(D['rec'][D['rec'].video_id == video]).get(f)
    srow = D['sam'][(D['sam'].video_id == video) & (D['sam'].frame_index == f)].iloc[0]
    mask = cv2.imread(str(D['out']/'sam2'/srow.mask_path), cv2.IMREAD_GRAYSCALE) > 0
    rows = cand[(cand.frame_index == f) & (cand.roi_status == 'AVAILABLE')]
    kp = json.loads((D['out']/'pose_sam2'/rows.keypoints_path.dropna().iloc[0]).read_text())['keypoints']
    P = {'head': D['head']}

    # 1 video frames (stack of three)
    n = int(r.frame_count)
    P['frames'] = [whole(read(r.video_path, i), box, (248, 251, 254)) for i in (int(n * .2), int(n * .5), f)]
    # 2 localization
    loc = image.copy()
    cv2.rectangle(loc, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (60, 170, 60), max(3, image.shape[1] // 150))
    P['loc'] = whole(loc, box, hexrgb(FILL[1]), .12)
    # 3 SAM 2 mask
    m = np.zeros_like(image)
    m[mask] = (235, 240, 245)
    P['mask'] = whole(m, box, (0, 0, 0), .12)
    # 4 pose + ROIs
    pose = image.copy()
    overlay = pose.copy()
    palette = plt.get_cmap('tab20')
    for i, row in enumerate(rows.itertuples()):
        poly = pe.parse(row.polygon)
        if poly and len(poly) >= 3:
            c = tuple(int(255 * v) for v in palette(i % 20)[:3][::-1])
            cv2.polylines(overlay, [np.int32(poly)], True, c, max(3, image.shape[1] // 150))
    pose = cv2.addWeighted(overlay, .9, pose, .1, 0)
    pts = {k['index']: (k['x'], k['y'], k['confidence']) for k in kp}
    th = max(4, image.shape[1] // 110)
    for (a, b), color in zip(COCO, LIMB):
        if a in pts and b in pts and min(pts[a][2], pts[b][2]) >= .5:
            c = tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))
            cv2.line(pose, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), c, th)
    for k, (x, y, c) in pts.items():
        if c >= .5 and k < 17:
            cv2.circle(pose, (int(x), int(y)), th + 2, (255, 255, 255), -1)
            cv2.circle(pose, (int(x), int(y)), th + 1, (40, 40, 200), -1)
    P['pose'] = whole(pose, box, hexrgb(FILL[3]))
    P['n_rois'] = len(rows)
    # 5 conditions: infant crops from other videos
    P['conditions'] = []
    for v, label in condition_videos:
        rv = D['manifest'].loc[v]
        boxes = boxes_of(D['rec'][D['rec'].video_id == v])
        keys = [k for k in sorted(boxes) if boxes[k] is not None]
        k = keys[len(keys) // 2]
        P['conditions'].append((rgb(square_crop(read(rv.video_path, k), boxes[k], .02, 220)), label))
    # 6 selection switches, v1 vs v2
    t = np.asarray(frames, float) / float(r.fps)
    P['switch'] = {}
    for cid in ('D0', D['head']):
        top = anchors(geo[cid], frames)
        P['switch'][cid] = [t[i] for i in range(1, len(top)) if top[i] and top[i - 1] and top[i] != top[i - 1]]
    P['duration'] = float(r.duration_sec)
    # 7 refined pixels of the v2 selection at frame f
    chosen = geo[D['head']][f]
    masked = image.copy()
    masked[~mask] = 0
    vis = (image * .35).astype(np.uint8)
    union = np.zeros(image.shape[:2], bool)
    for name, _, polygon, _ in chosen:
        polygon = sel.pixel_polygon(polygon, 'rectangle')      # v2 pixel rule (D3 reads 2-point boxes as rectangles)
        if polygon is None:
            continue
        poly = shrink(np.asarray(polygon, float), EXTRACT.roi_shrink)
        x1, y1 = np.floor(poly.min(0)).astype(int)
        x2, y2 = np.ceil(poly.max(0)).astype(int)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, image.shape[1]), min(y2, image.shape[0])
        part = masked[y1:y2, x1:x2]
        inside = np.zeros(part.shape[:2], np.uint8)
        cv2.fillPoly(inside, [np.rint(poly - [x1, y1]).astype(np.int32)], 1)
        bright = part.max(2)
        usable = inside.astype(bool) & (bright < EXTRACT.saturation_level) & (bright > EXTRACT.dark_level)
        union[y1:y2, x1:x2] |= skin_pixels(part, usable, EXTRACT)
        cv2.polylines(vis, [np.int32(poly)], True, (255, 255, 255), max(2, image.shape[1] // 400))
    vis[union] = (0.45 * image[union] + 0.55 * np.array([90, 220, 120])).astype(np.uint8)
    P['pixels'] = whole(vis, box, hexrgb(FILL[6]))
    # 8 output: crops of the selected ROIs with their weights
    P['output'] = []
    for name, weight, polygon, _ in sorted(chosen, key=lambda x: -x[1]):
        polygon = sel.pixel_polygon(polygon, 'rectangle')
        if polygon is not None:
            pts_ = np.asarray(polygon, float)
            P['output'].append((rgb(square_crop(image, (*pts_.min(0), *pts_.max(0)), .15, 200)),
                                name.replace('_', ' ').lower(), weight))
    P['frame'], P['video'] = f, video
    return P


def draw(P, out, stats):
    fig = plt.figure(figsize=(16, 9.9), dpi=150)
    fig.patch.set_facecolor('white')
    W, H = 16, 9.9
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    ax.text(W / 2, H - .35, 'NeoSTAR: Temporally Stabilized, Motion-Aware, Pixel-Refined Dynamic ROI Representation',
            ha='center', va='center', fontsize=17, weight='bold', color=INK)
    bw, bh, gap = 3.55, 3.95, .42
    x0 = (W - 4 * bw - 3 * gap) / 2
    tops = [H - .75, H - .75 - bh - .5]
    spec = [
        ('1  Newborn RGB Video', ['Validated frames (SHA-256)', 'Native timestamps', 'Sampled at ≤ 20 FPS'], False),
        ('2  Infant Localization', ['YOLO11s + RT-DETR-L', 'OR detection fusion', 'ByteTrack + bounded recovery'], False),
        ('3  SAM 2 Infant Mask', ['Box-prompted video mask', 'Temporal propagation', 'Mask reliability gating'], False),
        ('4  Pose & Anatomy', ['Sapiens2, 308 keypoints', 'Mask-constrained pose', f'17 anatomical ROIs'], False),
        ('5  Condition-Aware\n    Characterization', ["Pose P, keypoint visibility V′", 'Skin S, temporal overlap C',
                                                    'Camera-compensated motion Mₙ', 'Blur, light, swaddling; subject check'], True),
        ('6  Confidence &\n    Temporal Stabilization', ["Q′ = 0.2(P+V′+S+C) − 0.2M", 'EMA + hysteresis',
                                                        'Viterbi anchor (offline / 1-s lag)',
                                                        '4 settings chosen on training videos'], True),
        ('7  Pixel Refinement\n    & Fusion', ['ROI ∩ SAM 2 mask ∩ adaptive skin', 'Smoothed, cross-faded weights',
                                              'Rule-based, no trained model'], True),
        ('8  Dynamic ROI\n    Representation', ['Selected ROIs + weights', 'Refined pixel set per ROI',
                                                'Condition & motion metadata'], False),
    ]
    boxes = []
    for i, (title, bullets, new) in enumerate(spec):
        row, col = divmod(i, 4)
        x, top = x0 + col * (bw + gap), tops[row]
        y = top - bh
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=.12', fc=FILL[i], ec=EDGE[i],
                                    lw=1.6))
        ax.text(x + .15, top - .18, title, ha='left', va='top', fontsize=11.5, weight='bold', color=INK, linespacing=1.1)
        if new:
            ax.add_patch(FancyBboxPatch((x + bw - .62, top - .36), .5, .24, boxstyle='round,pad=0,rounding_size=.06',
                                        fc='#2a78d6', ec='none'))
            ax.text(x + bw - .37, top - .24, 'v2', ha='center', va='center', fontsize=9, weight='bold', color='white')
        for j, line in enumerate(bullets):
            ax.text(x + .2, y + .2 + (len(bullets) - 1 - j) * .27, '•  ' + line, ha='left', va='bottom', fontsize=8.8,
                    color=INK2)
        boxes.append((x, y, bw, bh))

    def inset(i, rect):
        x, y, w, h = boxes[i]
        ix, iy, iw, ih = rect
        return fig.add_axes([(x + ix) / W, (y + iy) / H, iw / W, ih / H])

    def show(a, img):
        a.imshow(img)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color('#b9c2cf')

    img_y, img_h = 1.18, 2.05
    for k, img in enumerate(P['frames']):                                   # 1
        show(inset(0, (.45 + .42 * k, img_y + .45 - .22 * k, 1.75, 1.75)), img)
    show(inset(1, (.75, img_y, 2.05, img_h)), P['loc'])                      # 2
    show(inset(2, (.75, img_y, 2.05, img_h)), P['mask'])                     # 3
    show(inset(3, (.75, img_y, 2.05, img_h)), P['pose'])                     # 4
    for k, (img, label) in enumerate(P['conditions']):                       # 5
        cx = .45 + (k % 2) * 1.55
        cy = 2.38 - (k // 2) * 1.02
        show(inset(4, (cx, cy, .82, .82)), img)
        ax.text(boxes[4][0] + cx + .41, boxes[4][1] + cy - .04, label, fontsize=7.5, color=INK2, ha='center', va='top')
    a = inset(5, (.3, img_y + .5, bw - .55, img_h - .6))                     # 6
    for yv, cid, color, name in ((1, 'D0', '#eb6834', 'v1 per-frame top-k'), (0, P['head'], '#2a78d6', f"v2 stabilized ({P['head']})")):
        a.vlines(P['switch'][cid], yv - .32, yv + .32, color=color, lw=1)
        a.text(P['duration'] * .01, yv + .42, f"{name}: {stats[cid]:.2f} switches/s (all videos)", fontsize=7.3,
               color=color, va='bottom')
    a.set_xlim(0, P['duration'])
    a.set_ylim(-.5, 1.85)
    a.set_yticks([])
    a.set_xlabel(f"time (s), top-region switches in {P['video']}", fontsize=7.5, color=INK2)
    a.tick_params(labelsize=7, colors=INK2, length=2)
    for s in ('top', 'right', 'left'):
        a.spines[s].set_visible(False)
    a.set_facecolor('none')
    show(inset(6, (.75, img_y, 2.05, img_h)), P['pixels'])                   # 7
    for k, (img, name, weight) in enumerate(P['output'][:3]):                # 8
        a = inset(7, (.35, img_y + 1.42 - k * .72, .68, .68))
        show(a, img)
        ax.text(boxes[7][0] + 1.2, boxes[7][1] + img_y + 1.42 - k * .72 + .34, f'{name}\nfusion weight α = {weight:.2f}',
                fontsize=8.3, color=INK2, va='center', linespacing=1.3)

    arrow = dict(arrowstyle='-|>', mutation_scale=18, color='#2f5ea8', lw=2.2)
    for i in (0, 1, 2, 4, 5, 6):
        x, y, w, h = boxes[i]
        ax.add_patch(FancyArrowPatch((x + w + .04, y + h / 2), (x + w + gap - .04, y + h / 2), **arrow))
    x3, y3, w3, _ = boxes[3]
    x4, y4, w4, h4 = boxes[4]
    ax.add_patch(FancyArrowPatch((x3 + w3 / 2, y3 - .03), (x4 + w4 / 2, y4 + h4 + .03),
                                 connectionstyle='angle,angleA=-90,angleB=180,rad=8', **arrow))
    ax.text(W / 2, .27, 'localization → SAM 2 mask → pose → 17 anatomical ROIs → condition terms → stabilized selection → '
            'pixel refinement → dynamic ROI\n(v2 = changed relative to the v1 baseline; no trained model, only 4 stabilizer '
            'settings chosen on training videos)', ha='center', va='center', fontsize=9.5, color=INK2, linespacing=1.5)
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)


def make(name, out, video='VID_010', conditions=None):
    D = load(name, video)
    conditions = conditions or [('VID_031', 'caregiver hand'), ('VID_021', 'swaddling'), ('VID_024', 'caption overlay'),
                                ('VID_002', 'clinician in view')]
    P = panels(D, video, conditions)
    pv = pd.read_csv(V2/'reports'/name/'phase1/per_video_metrics.csv')
    stats = {c: float(pv.loc[pv.config == c, 'switches_per_sec'].mean()) for c in ('D0', P['head'])}
    draw(P, out, stats)
    return dict(video=video, frame=P['frame'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--video', default='VID_010')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    target = Path(a.out) if a.out else V2/'reports'/a.name/'manuscript_figures/fig1_architecture.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    print(make(a.name, target, a.video), '->', target)
