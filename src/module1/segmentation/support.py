"""Pixel overlap and keypoint support from a verified original-frame mask."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np


def _json(value):
    if value is None or value == "":
        return None
    return json.loads(value) if isinstance(value, str) else value


def landmark_mask_support(mask, keypoints):
    """One nullable inside-mask value for every original 308-keypoint location."""
    if len(keypoints) not in (0, 308):
        raise ValueError("Expected 308 ordered keypoints")
    result = []
    for index, point in enumerate(keypoints):
        x, y = point.get("x"), point.get("y")
        inside = None
        if mask is not None and x is not None and y is not None:
            x, y = float(x), float(y)
            if math.isfinite(x) and math.isfinite(y):
                px, py = int(math.floor(x)), int(math.floor(y))
                inside = bool(0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px])
        result.append((index, inside))
    return result


def roi_mask_support(mask, roi_row, keypoints, *, min_support_ratio=0.5):
    empty = dict(roi_inside_mask_ratio=None, roi_mask_overlap=None,
                 visible_infant_pixel_ratio=None, landmark_inside_mask_ratio=None,
                 SAM2_mask_support=None, mask_supported_roi_status="NO_MASK")
    if roi_row["roi_status"] != "AVAILABLE":
        empty["mask_supported_roi_status"] = "UNAVAILABLE"
        return empty
    if mask is None:
        return empty
    region = np.zeros(mask.shape, dtype=np.uint8)
    polygon = _json(roi_row.get("polygon")) or []
    if len(polygon) >= 3:
        points = np.rint(np.asarray(polygon, dtype=float)).astype(np.int32)
        cv2.fillPoly(region, [points], 1)
    else:
        box = _json(roi_row.get("bbox"))
        if box is None:
            return empty
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        region[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = 1
    roi_pixels = region.astype(bool)
    roi_area = int(np.count_nonzero(roi_pixels))
    mask_area = int(np.count_nonzero(mask))
    if roi_area == 0 or mask_area == 0:
        return empty
    overlap = int(np.count_nonzero(roi_pixels & mask))
    inside = overlap/roi_area
    support_ids = _json(roi_row.get("support_keypoint_indices")) or []
    point_values = dict(landmark_mask_support(mask, keypoints))
    valid = [point_values[int(index)] for index in support_ids
             if int(index) in point_values and point_values[int(index)] is not None]
    landmark_ratio = sum(valid)/len(valid) if valid else None
    return dict(roi_inside_mask_ratio=inside,
                roi_mask_overlap=overlap/(roi_area+mask_area-overlap),
                visible_infant_pixel_ratio=overlap/mask_area,
                landmark_inside_mask_ratio=landmark_ratio,
                SAM2_mask_support=inside,
                mask_supported_roi_status="SUPPORTED" if inside >= min_support_ratio else "OUTSIDE_MASK")
