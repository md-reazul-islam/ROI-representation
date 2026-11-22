"""Pre-registered occluder design (../PREREGISTRATION.md): one simulated hand at a fixed face-relative position.

Shared by make_occlusion_videos.py (writes the videos) and occ_report.py (re-renders the hand mask to measure how much
of the fixed box / face regions it actually covered). DataAugment's hand renderer and noise are imported read only.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'dataaugment'))
from src.occlusion import render_hand  # noqa: E402
from src.video_utils import smooth_noise  # noqa: E402

SUBJECTS = (1, 3, 4)
# palm centre relative to the face-box centre (dx in face widths, dy in face heights; image left = negative x) and the
# finger direction (degrees counter-clockwise, 0 = fingers up). Fingers point away from the face centre, as a hand
# reaching in from that side would, so each position covers its own area as far as a hand shape allows.
POSITIONS = {
    'centre':      (0.00, 0.05, 180),   # nose and mid-cheeks; fingers over mouth / chin
    'forehead':    (0.00, -0.30, 0),    # forehead; fingers above the head
    'left_cheek':  (-0.28, 0.12, 90),   # image-left cheek; fingers point outwards
    'right_cheek': (0.28, 0.12, -90),   # image-right cheek
    'lower_face':  (0.00, 0.40, 180),   # mouth and chin; fingers over the neck
}
SIZES = {'small': 0.30, 'large': 0.45}  # palm width / face width
SKIN_BGR = (118, 148, 196)              # DataAugment's hand colour
SHADOW_STRENGTH, SHADOW_OFFSET, SHADOW_BLUR = 0.25, (6, 10), 7   # DataAugment's hand shadow
CLEAN_LEAD_SEC = 10.0                   # first 10 s unoccluded
ON_SEC, OFF_SEC = (10.0, 14.0), (4.0, 8.0)   # episode lengths, uniform, drawn per subject
FADE_SEC = 0.3
JITTER_POS, JITTER_ROT, JITTER_HZ = 0.05, 8.0, 0.08   # +- face sizes, +- degrees, low-pass cut-off
SEED = 20261116


def schedule(n, fps, subject):
    """Hand visibility 0..1 per frame (same for every position and size of one subject: common random numbers)."""
    rng = np.random.default_rng([SEED, subject, 0])
    t = np.arange(n) / fps
    vis = np.zeros(n)
    s = CLEAN_LEAD_SEC
    episodes = []
    while s < t[-1]:
        on = rng.uniform(*ON_SEC)
        e = min(s + on, t[-1])
        episodes.append((round(s, 3), round(e, 3)))
        up = np.clip((t - s) / FADE_SEC, 0, 1)
        down = np.clip((e - t) / FADE_SEC, 0, 1)
        vis = np.maximum(vis, np.minimum(up, down) * ((t >= s) & (t <= e)))
        s = e + rng.uniform(*OFF_SEC)
    return vis, episodes


def jitter(n, fps, subject):
    rng = np.random.default_rng([SEED, subject, 1])
    return {k: smooth_noise(n, fps, JITTER_HZ, rng) for k in ('dx', 'dy', 'rot')}


class Hand:
    def __init__(self, face, size, position, W, H, subject):
        self.W, self.H = W, H
        palm = SIZES[size] * face['w']
        self.rgb, self.alpha, self.anchor = render_hand(palm, SKIN_BGR, np.random.default_rng([SEED, subject, 2]))
        dx, dy, self.angle = POSITIONS[position]
        self.cx = face['x'] + face['w'] * (0.5 + dx)
        self.cy = face['y'] + face['h'] * (0.5 + dy)
        self.fw, self.fh = face['w'], face['h']

    def placement(self, i, jit):
        return (self.cx + JITTER_POS * self.fw * jit['dx'][i], self.cy + JITTER_POS * self.fh * jit['dy'][i],
                self.angle + JITTER_ROT * jit['rot'][i])

    def mask(self, x, y, ang, rgb=False):
        M = cv2.getRotationMatrix2D(self.anchor, float(ang), 1.0)
        M[0, 2] += x - self.anchor[0]
        M[1, 2] += y - self.anchor[1]
        a = cv2.warpAffine(self.alpha, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
        if not rgb:
            return a
        return a, cv2.warpAffine(self.rgb, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def apply(self, frame, i, vis, jit):
        if vis <= 0:
            return frame, None
        x, y, ang = self.placement(i, jit)
        a, rgb = self.mask(x, y, ang, rgb=True)
        a = a * vis
        S = np.float32([[1, 0, SHADOW_OFFSET[0]], [0, 1, SHADOW_OFFSET[1]]])
        sh = cv2.GaussianBlur(cv2.warpAffine(a, S, (self.W, self.H)), (0, 0), SHADOW_BLUR) * SHADOW_STRENGTH
        out = frame.astype(np.float32) * (1 - sh)[:, :, None]
        out = out * (1 - a[:, :, None]) + rgb * a[:, :, None]
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), a


def face_regions(face):
    """Descriptive face sub-regions (x1, y1, x2, y2) from the Haar face box; image left / right."""
    x, y, w, h = face['x'], face['y'], face['w'], face['h']
    r = lambda a, b, c, d: (x + a * w, y + b * h, x + c * w, y + d * h)
    return {'forehead': r(.2, .05, .8, .3), 'centre': r(.33, .35, .67, .7), 'left_cheek': r(.08, .38, .33, .75),
            'right_cheek': r(.67, .38, .92, .75), 'lower_face': r(.25, .72, .75, 1.0), 'face_box': r(0, 0, 1, 1)}


def covered(a, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    sub = a[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    return float(sub.mean()) if sub.size else np.nan
