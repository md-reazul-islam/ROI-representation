"""Phase 1c: motion decomposition.

v1 penalizes the total Farneback flow inside a region, which mixes camera motion, caregiver motion and
infant motion, including breathing. Here:
  1. global (camera) motion is estimated from background pixels only (outside the dilated SAM 2 infant
     mask, or outside the expanded infant box when no mask is reliable): the median flow vector, or a
     RANSAC homography fitted to background flow correspondences;
  2. residual flow = flow - global flow; per region: mean residual magnitude and mean residual vector;
  3. per region, the periodic share rho of the residual-velocity power in the respiratory band
     (0.5-1.5 Hz) over a trailing (causal) window;
  4. nuisance = (1 - exp(-m_residual / scale)) * (1 - rho) on thoraco-abdominal regions (periodic breathing
     motion is not penalized there but stored as `respiratory_ratio`), and 1 - exp(-m_residual / scale)
     elsewhere.
Farneback parameters are the v1 ones, so `motion_total` reproduces v1 `motion_magnitude` definitions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FARNEBACK = (.5, 3, 15, 3, 5, 1.2, 0)   # v1 roi_quality.features._motion


def dense_flow(previous_gray, gray):
    return cv2.calcOpticalFlowFarneback(previous_gray, gray, None, *FARNEBACK)


def background_mask(shape, mask=None, box=None, dilation=15, box_margin=.1):
    """True on background pixels; None if no infant localization is available."""
    h, w = shape
    if mask is not None and mask.shape == (h, w) and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        return ~cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    if box is not None:
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * box_margin, (y2 - y1) * box_margin
        out = np.ones((h, w), bool)
        out[max(0, int(y1 - my)):min(h, int(math.ceil(y2 + my))), max(0, int(x1 - mx)):min(w, int(math.ceil(x2 + mx)))] = False
        return out
    return None


def global_flow(flow, background, mode='median', min_fraction=.01, step=8):
    """Per-pixel camera/global flow field (H, W, 2) and a status string."""
    h, w = flow.shape[:2]
    zero = np.zeros_like(flow)
    if background is None or background.mean() < min_fraction:
        return zero, 'NO_BACKGROUND'
    if mode == 'median':
        vector = np.median(flow[background], axis=0)
        return np.broadcast_to(vector, flow.shape).copy(), 'MEDIAN'
    if mode == 'homography':
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        keep = background[ys, xs]
        src = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
        if len(src) < 8:
            return zero, 'NO_BACKGROUND'
        dst = src + flow[ys[keep], xs[keep]]
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.)
        if H is None:
            vector = np.median(flow[background], axis=0)
            return np.broadcast_to(vector, flow.shape).copy(), 'MEDIAN_FALLBACK'
        gy, gx = np.mgrid[0:h, 0:w]
        points = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(points, H).reshape(h, w, 2)
        return moved - np.stack([gx, gy], -1), 'HOMOGRAPHY'
    raise ValueError(f'unknown global motion mode {mode}')


def polygon_mask(shape, polygon):
    canvas = np.zeros(shape, np.uint8)
    cv2.fillPoly(canvas, [np.rint(np.asarray(polygon, float)).astype(np.int32)], 1)
    return canvas.astype(bool)


def region_motion(flow, residual, region):
    """Mean total magnitude, mean residual magnitude and mean residual vector inside a boolean region."""
    if region is None or not region.any():
        return np.nan, np.nan, (np.nan, np.nan)
    total = float(np.mean(np.linalg.norm(flow[region], axis=1)))
    vectors = residual[region]
    return total, float(np.mean(np.linalg.norm(vectors, axis=1))), tuple(float(v) for v in vectors.mean(0))


def bounded(value, scale):
    return float(1 - math.exp(-value / scale)) if math.isfinite(value) else np.nan


def periodic_ratio(vectors, fs, band=(.5, 1.5), floor_hz=.1, min_samples=None):
    """Share of residual-velocity power in `band` along the principal motion axis. NaN if too few samples."""
    v = np.asarray(vectors, float)
    v = v[np.isfinite(v).all(1)] if v.ndim == 2 else v
    if min_samples is None:
        min_samples = max(8, int(4 * fs))
    if len(v) < min_samples:
        return np.nan
    v = v - v.mean(0)
    _, _, axes = np.linalg.svd(v, full_matrices=False)
    x = v @ axes[0]
    x = x * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    usable = freqs >= floor_hz
    total = power[usable].sum()
    if total <= 0:
        return 0.
    return float(power[(freqs >= band[0]) & (freqs <= band[1])].sum() / total)


def nuisance(residual_bounded, ratio, respiratory_region):
    """Penalized motion: aperiodic residual motion. Unknown periodicity counts as fully aperiodic."""
    if not math.isfinite(residual_bounded):
        return np.nan
    if respiratory_region:
        return residual_bounded * (1 - (ratio if math.isfinite(ratio) else 0.))
    return residual_bounded
