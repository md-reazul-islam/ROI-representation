"""Turn Module 1 dynamic ROIs into per-native-frame RGB traces and fusion weights.

Module 1 scores ROIs on sampled frames (processing.target_fps). rPPG needs every native frame,
so each native frame reuses the geometry of the latest processed frame at or before it (zero-order
hold), but never for longer than max_hold_sec. Pixels are restricted to a shrunken ROI polygon
(less boundary jitter), to unsaturated, non-dark pixels and to skin:
  'adaptive' (default)  pixels inside a broad YCrCb skin range whose chromaticity lies within
                        skin_mad_k robust deviations of the ROI's own median (rejects tubes, tape,
                        blanket edges; adapts to neonatal, jaundiced or phototherapy-lit skin);
  'fixed'               the broad YCrCb range only;  'none'  no skin filtering.
The candidate pool is every Module 1 region by default: in a neonatal study the abdomen gave the
lowest HR error of all regions (arXiv:2207.11886, 2022), so the pool is not restricted to the face.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import cv2
import numpy as np

from module1.roi import ROI_NAMES


FACE_POOL = ('FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE', 'FACE')
DEFAULT_POOL = tuple(ROI_NAMES)
INFANT_BOX = 'INFANT_BOX'


@dataclass(frozen=True)
class ExtractionConfig:
    roi_pool: tuple = DEFAULT_POOL
    max_hold_sec: float = .25
    roi_shrink: float = .85
    skin_mode: str = 'adaptive'
    skin_cr_range: tuple = (128., 185.)
    skin_cb_range: tuple = (65., 135.)
    skin_mad_k: float = 3.
    min_skin_fraction: float = .1
    min_pixels: int = 20
    saturation_level: int = 250
    dark_level: int = 15
    # Prior weights: softmax(Q / temperature) over usable ROIs of the pool. Module 1 uses a softmax
    # temperature of 1 over Q in [-0.2, 0.8], which is nearly uniform; HR fusion uses a sharper one.
    weight_temperature: float = .1
    min_visible_skin: float = .2
    weight_smoothing_sec: float = 1.


def parse_json(value, default=None):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == '':
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return default


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def roi_polygon(row):
    if row.get('roi_status') != 'AVAILABLE':
        return None
    polygon = parse_json(row.get('polygon'), []) or []
    if len(polygon) >= 3:
        return np.asarray(polygon, dtype=float)
    bbox = parse_json(row.get('bbox'))
    if bbox:
        x1, y1, x2, y2 = map(float, bbox)
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return None


def box_polygon(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def shrink(polygon, factor):
    center = polygon.mean(axis=0)
    return center + (polygon - center) * factor


def hold_indices(processed_frames, frame_count, fps, max_hold_sec):
    """For every native frame, the processed frame whose geometry it reuses (or -1)."""
    processed = np.asarray(sorted(processed_frames), dtype=int)
    native = np.arange(frame_count)
    if processed.size == 0:
        return np.full(frame_count, -1)
    position = np.searchsorted(processed, native, side='right') - 1
    source = np.where(position >= 0, processed[np.clip(position, 0, None)], processed[0])
    too_far = np.abs(native - source) > max_hold_sec * fps + 1e-9
    return np.where(too_far, -1, source)


def skin_pixels(crop, usable, config):
    """Boolean mask of skin pixels among `usable` pixels of a BGR crop."""
    if config.skin_mode == 'none':
        return usable
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (usable & (cr >= config.skin_cr_range[0]) & (cr <= config.skin_cr_range[1]) &
            (cb >= config.skin_cb_range[0]) & (cb <= config.skin_cb_range[1]))
    if config.skin_mode == 'adaptive' and skin.sum() >= config.min_pixels:
        for channel in (cr, cb):
            values = channel[skin]
            median = np.median(values)
            spread = 1.4826 * np.median(np.abs(values - median)) + 1.
            skin &= np.abs(channel - median) <= config.skin_mad_k * spread
    return skin


def mean_rgb(image, polygon, config):
    """Mean (R, G, B) of usable skin pixels inside the polygon, plus the pixel count; None if unusable."""
    height, width = image.shape[:2]
    polygon = shrink(polygon, config.roi_shrink)
    x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, 0
    crop = image[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon - [x1, y1]).astype(np.int32)], 1)
    inside = mask.astype(bool)
    brightest = crop.max(axis=2)
    usable = inside & (brightest < config.saturation_level) & (brightest > config.dark_level)
    selected = skin_pixels(crop, usable, config)
    count = int(selected.sum())
    if count < max(config.min_pixels, config.min_skin_fraction * inside.sum()):
        return None, count
    b, g, r = crop[selected].astype(float).mean(axis=0)
    return (r, g, b), count


def fusion_weights(scores, usable, config, fps):
    """Per-frame softmax weights over ROIs, smoothed in time and renormalized.

    scores: (n, k) Module 1 confidence Q (NaN when unavailable); usable: (n, k) bool.
    Returns (n, k) weights; rows with no usable ROI are all zero.
    """
    scores = np.where(usable & np.isfinite(scores), scores, -np.inf)
    best = scores.max(axis=1, keepdims=True)
    safe = np.where(np.isfinite(best), best, 0.)
    raw = np.where(np.isfinite(scores), np.exp((scores - safe) / config.weight_temperature), 0.)
    total = raw.sum(axis=1, keepdims=True)
    weights = np.divide(raw, total, out=np.zeros_like(raw), where=total > 0)
    length = max(1, int(round(config.weight_smoothing_sec * fps)))
    if length > 1:
        kernel = np.ones(length) / length
        weights = np.stack([np.convolve(column, kernel, mode='same') for column in weights.T], axis=1)
        weights = np.where(usable, weights, 0.)
        total = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    return weights
