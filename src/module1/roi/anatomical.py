from dataclasses import dataclass
import json
import math

import numpy as np


ROI_NAMES = (
    'HEAD', 'FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE',
    'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'UPPER_CHEST', 'LOWER_CHEST',
    'ABDOMEN', 'THORACO_ABDOMINAL', 'LEFT_ARM', 'RIGHT_ARM', 'LEFT_LEG', 'RIGHT_LEG',
)


@dataclass(frozen=True)
class ROIResult:
    roi_name: str
    status: str
    geometry_type: str | None
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float] | None
    centroid: tuple[float, float] | None
    width: float | None
    height: float | None
    bbox_normalized: tuple[float, float, float, float] | None
    polygon_bbox_normalized: tuple[tuple[float, float], ...]
    centroid_bbox_normalized: tuple[float, float] | None
    centroid_torso_normalized: tuple[float, float] | None
    support_keypoint_indices: tuple[int, ...]
    support_keypoint_names: tuple[str, ...]
    anatomical_confidence: float | None
    unavailable_reason: str | None = None

    def as_record(self, video_id, split, frame_index, timestamp_sec, pose_status):
        return dict(
            video_id=video_id, split=split, frame_index=frame_index,
            timestamp_sec=timestamp_sec, pose_status=pose_status,
            roi_name=self.roi_name, roi_status=self.status,
            geometry_type=self.geometry_type,
            polygon=json.dumps([list(point) for point in self.polygon]),
            bbox=json.dumps(list(self.bbox)) if self.bbox is not None else None,
            centroid=json.dumps(list(self.centroid)) if self.centroid is not None else None,
            width=self.width, height=self.height,
            bbox_normalized=json.dumps(list(self.bbox_normalized)) if self.bbox_normalized is not None else None,
            polygon_bbox_normalized=json.dumps([list(point) for point in self.polygon_bbox_normalized]),
            centroid_bbox_normalized=json.dumps(list(self.centroid_bbox_normalized)) if self.centroid_bbox_normalized else None,
            centroid_torso_normalized=json.dumps(list(self.centroid_torso_normalized)) if self.centroid_torso_normalized else None,
            support_keypoint_indices=json.dumps(list(self.support_keypoint_indices)),
            support_keypoint_names=json.dumps(list(self.support_keypoint_names)),
            anatomical_confidence=self.anatomical_confidence,
            unavailable_reason=self.unavailable_reason,
        )


def _finite_point(point):
    return (point is not None and len(point) == 2 and
            all(math.isfinite(float(value)) for value in point))


def _convex_hull(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 2:
        return tuple(points)

    def cross(origin, a, b):
        return ((a[0]-origin[0]) * (b[1]-origin[1]) -
                (a[1]-origin[1]) * (b[0]-origin[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


class AnatomicalCoordinateSystem:
    """Maps original-pixel points into box and optional torso-relative coordinates."""

    def __init__(self, bbox, torso=None):
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError('bbox must contain four finite values')
        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('bbox must have positive area')
        self.bbox = (x1, y1, x2, y2)
        self.width, self.height = x2-x1, y2-y1
        self.torso = self._validate_torso(torso) if torso is not None else None

    @staticmethod
    def _validate_torso(torso):
        if len(torso) != 4 or any(not _finite_point(point) for point in torso):
            raise ValueError('torso must contain four finite corner points')
        left_top, right_top, right_bottom, left_bottom = tuple(tuple(map(float, point)) for point in torso)
        if right_top[0] <= left_top[0] or right_bottom[0] <= left_bottom[0]:
            raise ValueError('torso horizontal edges must have positive width')
        if (right_top[1] + left_top[1]) / 2 >= (right_bottom[1] + left_bottom[1]) / 2:
            raise ValueError('torso top must be above torso bottom')
        return left_top, right_top, right_bottom, left_bottom

    def bbox_normalize(self, point):
        if not _finite_point(point):
            return None
        return ((float(point[0])-self.bbox[0])/self.width,
                (float(point[1])-self.bbox[1])/self.height)

    def torso_normalize(self, point):
        if self.torso is None or not _finite_point(point):
            return None
        lt, rt, rb, lb = self.torso
        top_y, bottom_y = (lt[1]+rt[1])/2, (lb[1]+rb[1])/2
        v = (float(point[1])-top_y)/(bottom_y-top_y)
        left_x = lt[0] + v*(lb[0]-lt[0])
        right_x = rt[0] + v*(rb[0]-rt[0])
        if right_x <= left_x:
            return None
        return ((float(point[0])-left_x)/(right_x-left_x), v)

    def geometry(self, points):
        hull = _convex_hull(points)
        if not hull:
            return None
        xs, ys = zip(*hull)
        bbox = (min(xs), min(ys), max(xs), max(ys))
        centroid = (sum(xs)/len(xs), sum(ys)/len(ys)) if len(hull) < 3 else self._polygon_centroid(hull)
        bbox_norm = tuple(value for point in ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
                          for value in self.bbox_normalize(point))
        normalized_polygon = tuple(self.bbox_normalize(point) for point in hull)
        normalized_centroid = self.bbox_normalize(centroid)
        torso_centroid = self.torso_normalize(centroid)
        return hull, bbox, centroid, bbox[2]-bbox[0], bbox[3]-bbox[1], bbox_norm, normalized_polygon, normalized_centroid, torso_centroid

    @staticmethod
    def _polygon_centroid(polygon):
        area_twice = sum(polygon[i][0]*polygon[(i+1) % len(polygon)][1]
                         - polygon[(i+1) % len(polygon)][0]*polygon[i][1]
                         for i in range(len(polygon)))
        if abs(area_twice) < 1e-12:
            return (sum(point[0] for point in polygon)/len(polygon),
                    sum(point[1] for point in polygon)/len(polygon))
        cx = sum((polygon[i][0]+polygon[(i+1) % len(polygon)][0]) *
                 (polygon[i][0]*polygon[(i+1) % len(polygon)][1] - polygon[(i+1) % len(polygon)][0]*polygon[i][1])
                 for i in range(len(polygon))) / (3*area_twice)
        cy = sum((polygon[i][1]+polygon[(i+1) % len(polygon)][1]) *
                 (polygon[i][0]*polygon[(i+1) % len(polygon)][1] - polygon[(i+1) % len(polygon)][0]*polygon[i][1])
                 for i in range(len(polygon))) / (3*area_twice)
        return cx, cy


class AnatomicalROIMapper:
    """Dynamic candidate ROI generator; it does not score physiological signal quality."""

    def __init__(self, keypoint_names, min_keypoint_confidence=.5, min_support_points=2):
        if len(keypoint_names) != 308 or len(set(keypoint_names)) != 308:
            raise ValueError('Expected exactly 308 unique Sapiens2 keypoint names')
        if not math.isfinite(min_keypoint_confidence) or not 0 <= min_keypoint_confidence <= 1:
            raise ValueError('min_keypoint_confidence must be in [0,1]')
        if not isinstance(min_support_points, int) or min_support_points < 1:
            raise ValueError('min_support_points must be a positive integer')
        self.names = tuple(keypoint_names)
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.threshold = float(min_keypoint_confidence)
        self.min_support = min_support_points
        self.face_indices = self._face_indices()

    def _find(self, exact=(), contains=(), prefixes=()):
        output = []
        for i, name in enumerate(self.names):
            lower = name.lower()
            if name in exact or any(token in lower for token in contains) or any(lower.startswith(prefix) for prefix in prefixes):
                output.append(i)
        return output

    def _index(self, *names):
        return [self.name_to_index[name] for name in names if name in self.name_to_index]

    def _face_indices(self):
        return self._find(contains=('eye', 'eyebrow', 'eyelid', 'lash', 'nose', 'nostril',
                                    'mouth', 'lip', 'chin', 'glabella', 'labiomental', 'iris',
                                    'pupil', 'face-'), prefixes=('l_', 'r_'))[0:]

    def _valid_points(self, keypoints):
        points = {}
        for i, point in enumerate(keypoints):
            if isinstance(point, dict):
                x, y, confidence = point.get('x'), point.get('y'), point.get('confidence')
            else:
                x, y, confidence = point
            try:
                x, y, confidence = float(x), float(y), float(confidence)
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (x, y, confidence)) and confidence >= self.threshold:
                points[i] = (x, y, confidence)
        return points

    def _region_support(self, region, valid):
        names = self.names
        def idx(*items): return [i for i in items if i in valid]
        left_shoulder = self._index('left_shoulder', 'left_acromion')
        right_shoulder = self._index('right_shoulder', 'right_acromion')
        left_arm = self._index('left_shoulder', 'left_acromion', 'left_elbow', 'left_wrist')
        right_arm = self._index('right_shoulder', 'right_acromion', 'right_elbow', 'right_wrist')
        left_leg = self._index('left_hip', 'left_knee', 'left_ankle', 'left_heel', 'left_big_toe', 'left_small_toe')
        right_leg = self._index('right_hip', 'right_knee', 'right_ankle', 'right_heel', 'right_big_toe', 'right_small_toe')
        torso = self._index('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
        facial = self.face_indices + self._index('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'neck')
        facial = list(dict.fromkeys(facial))
        if region in ('HEAD', 'FACE'):
            return facial if region == 'HEAD' else [i for i in facial if i not in self._index('left_ear', 'right_ear', 'neck')]
        if region == 'UPPER_FACE':
            return self._find(contains=('eyebrow', 'eye', 'eyelid', 'lash', 'glabella', 'nose_root', 'face-'))
        if region == 'FOREHEAD':
            return self._find(contains=('eyebrow', 'glabella', 'nose_root', 'forehead'))
        if region == 'LEFT_CHEEK':
            return self._find(contains=('left_cheek', 'l_cheek')) or self._side_face(valid, 'left')
        if region == 'RIGHT_CHEEK':
            return self._find(contains=('right_cheek', 'r_cheek')) or self._side_face(valid, 'right')
        if region == 'NECK':
            return self._index('neck', 'left_shoulder', 'right_shoulder')
        if region == 'LEFT_SHOULDER': return left_shoulder + self._index('left_elbow')
        if region == 'RIGHT_SHOULDER': return right_shoulder + self._index('right_elbow')
        if region == 'LEFT_ARM': return left_arm
        if region == 'RIGHT_ARM': return right_arm
        if region == 'LEFT_LEG': return left_leg
        if region == 'RIGHT_LEG': return right_leg
        if region == 'THORACO_ABDOMINAL': return torso
        if region in ('UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN'):
            return torso
        return []

    def _side_face(self, valid, side):
        facial = [i for i in self.face_indices if i in valid]
        if not facial:
            return []
        nose = next((valid[i][0] for i in self._index('nose') if i in valid), None)
        if nose is None:
            nose = np.median([valid[i][0] for i in facial])
        values = [i for i in facial if (valid[i][0] <= nose if side == 'left' else valid[i][0] >= nose)]
        return values

    def _split_vertical(self, indices, valid, coordinate_system, ratio):
        if len(indices) < self.min_support:
            return indices
        ys = [valid[i][1] for i in indices if i in valid]
        if not ys:
            return []
        cutoff = min(ys) + ratio*(max(ys)-min(ys))
        return [i for i in indices if i in valid and (valid[i][1] <= cutoff)]

    def _support_for_region(self, region, valid, coordinate_system):
        indices = [i for i in self._region_support(region, valid) if i in valid]
        if region in ('THORACO_ABDOMINAL', 'UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN') and coordinate_system.torso is None:
            return []
        if region == 'FOREHEAD':
            if len(indices) < self.min_support:
                indices = self._split_vertical(self.face_indices, valid, coordinate_system, .35)
        elif region == 'UPPER_FACE':
            if len(indices) < self.min_support:
                indices = self._split_vertical(self.face_indices, valid, coordinate_system, .5)
        elif region in ('UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN'):
            if len(indices) >= self.min_support:
                ys = [valid[i][1] for i in indices]
                low, high = min(ys), max(ys)
                if region == 'UPPER_CHEST': indices = [i for i in indices if valid[i][1] <= low + .4*(high-low)]
                elif region == 'LOWER_CHEST': indices = [i for i in indices if low + .25*(high-low) <= valid[i][1] <= low + .7*(high-low)]
                else: indices = [i for i in indices if valid[i][1] >= low + .55*(high-low)]
        return list(dict.fromkeys(indices))

    def map_pose(self, bbox, keypoints):
        valid = self._valid_points(keypoints)
        torso_ids = self._index('left_shoulder', 'right_shoulder', 'right_hip', 'left_hip')
        torso = [valid[i][:2] for i in torso_ids if i in valid]
        torso_frame = None
        if len(torso) == 4:
            left_shoulder, right_shoulder, right_hip, left_hip = torso
            # An infant facing the camera mirrors anatomical left/right in the image, so also try
            # image-ordered corners. Rotated or collapsed anchors define no upright torso frame
            # and are treated like missing anchors.
            for corners in ((left_shoulder, right_shoulder, right_hip, left_hip),
                            (right_shoulder, left_shoulder, left_hip, right_hip)):
                try:
                    torso_frame = AnatomicalCoordinateSystem._validate_torso(corners)
                    break
                except ValueError:
                    continue
        coordinate_system = AnatomicalCoordinateSystem(bbox, torso_frame)
        results = []
        for region in ROI_NAMES:
            indices = self._support_for_region(region, valid, coordinate_system)
            if len(indices) < self.min_support:
                results.append(ROIResult(region, 'UNAVAILABLE', None, (), None, None, None, None, None, (), None, None,
                                         tuple(indices), tuple(self.names[i] for i in indices), None,
                                         'INSUFFICIENT_VALID_KEYPOINTS'))
                continue
            geometry = coordinate_system.geometry([valid[i][:2] for i in indices])
            if geometry is None:
                results.append(ROIResult(region, 'UNAVAILABLE', None, (), None, None, None, None, None, (), None, None,
                                         tuple(indices), tuple(self.names[i] for i in indices), None, 'INVALID_GEOMETRY'))
                continue
            hull, region_bbox, centroid, width, height, bbox_norm, norm_polygon, norm_centroid, torso_centroid = geometry
            if region in ('UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN') and coordinate_system.torso is not None:
                lt, rt, rb, lb = coordinate_system.torso
                top_y, bottom_y = (lt[1]+rt[1])/2, (lb[1]+rb[1])/2
                low, high = dict(UPPER_CHEST=(0, .35), LOWER_CHEST=(.25, .7), ABDOMEN=(.55, 1))[region]
                region_bbox = (min(lt[0], rt[0], lb[0], rb[0]), top_y + low*(bottom_y-top_y),
                               max(lt[0], rt[0], lb[0], rb[0]), top_y + high*(bottom_y-top_y))
                hull = ((region_bbox[0], region_bbox[1]), (region_bbox[2], region_bbox[1]),
                        (region_bbox[2], region_bbox[3]), (region_bbox[0], region_bbox[3]))
                centroid = ((region_bbox[0]+region_bbox[2])/2, (region_bbox[1]+region_bbox[3])/2)
                width, height = region_bbox[2]-region_bbox[0], region_bbox[3]-region_bbox[1]
                bbox_norm = tuple(value for point in ((region_bbox[0], region_bbox[1]), (region_bbox[2], region_bbox[3]))
                                  for value in coordinate_system.bbox_normalize(point))
                norm_polygon = tuple(coordinate_system.bbox_normalize(point) for point in hull)
                norm_centroid = coordinate_system.bbox_normalize(centroid)
                torso_centroid = coordinate_system.torso_normalize(centroid)
            results.append(ROIResult(region, 'AVAILABLE', 'POLYGON' if len(hull) >= 3 else 'BBOX', hull,
                                     region_bbox, centroid, width, height, bbox_norm, norm_polygon, norm_centroid,
                                     torso_centroid, tuple(indices), tuple(self.names[i] for i in indices),
                                     float(np.mean([valid[i][2] for i in indices])), None))
        return results
