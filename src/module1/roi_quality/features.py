from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ROIQualityConfig:
    skin_cr_center: float = 150.
    skin_cb_center: float = 110.
    skin_cr_scale: float = 35.
    skin_cb_scale: float = 25.
    blur_scale: float = 500.
    motion_scale_pixels: float = 20.
    landmark_velocity_scale: float = 300.
    centroid_displacement_scale: float = 300.
    area_change_scale: float = .5

    def __post_init__(self):
        for name in ('skin_cr_scale', 'skin_cb_scale', 'blur_scale', 'motion_scale_pixels',
                     'landmark_velocity_scale', 'centroid_displacement_scale', 'area_change_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('skin_cr_center', 'skin_cb_center'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')


def _bounded_exponential(value, scale):
    if value is None or not math.isfinite(float(value)):
        return None
    return float(1. - math.exp(-max(0., float(value)) / scale))


def _parse_json(value, default):
    if value is None or value == '':
        return default
    import json
    parsed = json.loads(value) if isinstance(value, str) else value
    return parsed if parsed is not None else default


def _polygon_from_row(row):
    polygon = _parse_json(row.get('polygon'), [])
    if len(polygon) >= 3:
        return np.asarray(polygon, dtype=np.float32)
    bbox = _parse_json(row.get('bbox'), None)
    if bbox is None:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _mask(shape, polygon):
    if polygon is None or len(polygon) < 3:
        return None
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    clipped = np.rint(polygon).astype(np.int32)
    cv2.fillPoly(mask, [clipped], 1)
    return mask.astype(bool)


def _roi_iou(shape, first, second):
    first_mask, second_mask = _mask(shape, first), _mask(shape, second)
    if first_mask is None or second_mask is None:
        return None
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(first_mask, second_mask).sum() / union)


class ROIQualityCalculator:
    """Continuous image/pose descriptors; deliberately no final ROI confidence score."""

    def __init__(self, keypoint_names, config=None):
        if len(keypoint_names) != 308:
            raise ValueError('Expected the official 308 Sapiens2 keypoint names')
        self.names = tuple(keypoint_names)
        self.config = config or ROIQualityConfig()
        self.name_to_index = {name: i for i, name in enumerate(self.names)}

    def _pixels(self, image, polygon):
        mask = _mask(image.shape, polygon)
        if mask is None or not mask.any():
            return None, None
        return mask, image[mask]

    def _appearance(self, image, polygon):
        mask, pixels = self._pixels(image, polygon)
        if pixels is None:
            return dict(visible_skin_proportion=None, occlusion_proxy=None,
                        blur_score=None, illumination_score=None, saturation_score=None)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[mask].astype(np.float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[mask].astype(np.float32)
        cfg = self.config
        cr_score = np.exp(-.5*((ycrcb[:, 1]-cfg.skin_cr_center)/cfg.skin_cr_scale)**2)
        cb_score = np.exp(-.5*((ycrcb[:, 2]-cfg.skin_cb_center)/cfg.skin_cb_scale)**2)
        skin = np.sqrt(cr_score * cb_score)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian[mask]))
        luminance = ycrcb[:, 0] / 255.
        saturation = hsv[:, 1] / 255.
        illumination = np.clip(1. - 2.*np.abs(luminance-.5), 0., 1.)
        saturation_quality = np.clip(1. - np.abs(saturation-.45)/.55, 0., 1.)
        visible = float(np.mean(skin))
        return dict(visible_skin_proportion=visible, occlusion_proxy=float(1.-visible),
                    blur_score=_bounded_exponential(variance, cfg.blur_scale),
                    illumination_score=float(np.mean(illumination)),
                    saturation_score=float(np.mean(saturation_quality)))

    def _motion(self, previous_image, image, polygon):
        if previous_image is None:
            return dict(motion_magnitude_pixels=None, motion_magnitude=None)
        if previous_image.shape != image.shape:
            raise ValueError('Consecutive ROI frames must have identical dimensions')
        previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        mask = _mask(image.shape, polygon)
        if mask is None or not mask.any():
            return dict(motion_magnitude_pixels=None, motion_magnitude=None)
        raw = float(np.mean(magnitude[mask]))
        return dict(motion_magnitude_pixels=raw,
                    motion_magnitude=_bounded_exponential(raw, self.config.motion_scale_pixels))

    def _pose_temporal(self, current_keypoints, previous_keypoints, support_indices, dt):
        if previous_keypoints is None or dt is None or dt <= 0:
            return dict(landmark_velocity_pixels_per_second=None, landmark_velocity=None)
        current = {int(p['index']): p for p in current_keypoints if p.get('x') is not None and p.get('y') is not None}
        previous = {int(p['index']): p for p in previous_keypoints if p.get('x') is not None and p.get('y') is not None}
        velocities = []
        for index in support_indices:
            if index not in current or index not in previous:
                continue
            a, b = current[index], previous[index]
            if not all(math.isfinite(float(a.get(field))) and math.isfinite(float(b.get(field))) for field in ('x', 'y')):
                continue
            velocities.append(math.hypot(float(a['x'])-float(b['x']), float(a['y'])-float(b['y'])) / dt)
        if not velocities:
            return dict(landmark_velocity_pixels_per_second=None, landmark_velocity=None)
        raw = float(np.mean(velocities))
        return dict(landmark_velocity_pixels_per_second=raw,
                    landmark_velocity=_bounded_exponential(raw, self.config.landmark_velocity_scale))

    @staticmethod
    def _row_area(row):
        bbox = _parse_json(row.get('bbox'), None)
        if bbox is None:
            return None
        return max(0., float(bbox[2])-float(bbox[0])) * max(0., float(bbox[3])-float(bbox[1]))

    def _temporal(self, image, row, previous_row, dt):
        if previous_row is None or dt is None or dt <= 0:
            return dict(roi_centroid_displacement_pixels=None, roi_centroid_displacement=None,
                        roi_area_change=None, roi_area_variation=None,
                        consecutive_roi_iou=None, temporal_consistency=None)
        current_centroid = _parse_json(row.get('centroid'), None)
        previous_centroid = _parse_json(previous_row.get('centroid'), None)
        if current_centroid is None or previous_centroid is None:
            displacement = None
        else:
            displacement = math.hypot(float(current_centroid[0])-float(previous_centroid[0]),
                                      float(current_centroid[1])-float(previous_centroid[1])) / dt
        current_area, previous_area = self._row_area(row), self._row_area(previous_row)
        if current_area is None or previous_area is None or previous_area <= 0:
            area_change = None
        else:
            area_change = abs(current_area-previous_area)/previous_area
        iou = _roi_iou(image.shape, _polygon_from_row(row), _polygon_from_row(previous_row))
        return dict(roi_centroid_displacement_pixels=displacement,
                    roi_centroid_displacement=_bounded_exponential(displacement, self.config.centroid_displacement_scale),
                    roi_area_change=area_change,
                    roi_area_variation=_bounded_exponential(area_change, self.config.area_change_scale),
                    consecutive_roi_iou=iou, temporal_consistency=iou)

    def _swaddling(self, keypoints):
        relevant = [i for i, name in enumerate(self.names)
                    if any(token in name.lower() for token in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'heel', 'toe'))]
        if not relevant:
            return None
        scores = []
        for point in keypoints:
            if int(point.get('index', -1)) in relevant:
                value = point.get('confidence')
                scores.append(float(value) if value is not None and math.isfinite(float(value)) else 0.)
        if not scores:
            return None
        return float(1.-np.mean(np.clip(scores, 0., 1.)))

    def calculate(self, image, row, previous_image=None, previous_row=None,
                  keypoints=None, previous_keypoints=None, timestamp=None, previous_timestamp=None):
        if row.get('roi_status') != 'AVAILABLE':
            return dict(quality_status='UNAVAILABLE', visible_skin_proportion=None, occlusion_proxy=None,
                        blur_score=None, illumination_score=None, saturation_score=None,
                        motion_magnitude_pixels=None, motion_magnitude=None,
                        landmark_velocity_pixels_per_second=None, landmark_velocity=None,
                        roi_centroid_displacement_pixels=None, roi_centroid_displacement=None,
                        roi_area_change=None, roi_area_variation=None, consecutive_roi_iou=None,
                        temporal_consistency=None, swaddling_proxy=None,
                        quality_scoring_implemented=False)
        polygon = _polygon_from_row(row)
        if polygon is None:
            return dict(quality_status='UNAVAILABLE', quality_scoring_implemented=False)
        current_keypoints = keypoints or []
        dt = None if timestamp is None or previous_timestamp is None else float(timestamp)-float(previous_timestamp)
        support = _parse_json(row.get('support_keypoint_indices'), [])
        result = dict(quality_status='AVAILABLE', **self._appearance(image, polygon))
        result.update(self._motion(previous_image, image, polygon))
        result.update(self._pose_temporal(current_keypoints, previous_keypoints, support, dt))
        result.update(self._temporal(image, row, previous_row, dt))
        result['swaddling_proxy'] = self._swaddling(current_keypoints)
        result['quality_scoring_implemented'] = False
        return result
