import json

import cv2
import numpy as np
import pytest

from module1.roi_quality import ROIQualityCalculator, ROIQualityConfig
from test_roi import keypoint_names, points


def roi_row(name='FACE', x1=10, y1=10, x2=60, y2=60):
    return dict(roi_name=name, roi_status='AVAILABLE',
                polygon=json.dumps([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                bbox=json.dumps([x1, y1, x2, y2]), centroid=json.dumps([(x1+x2)/2, (y1+y2)/2]),
                support_keypoint_indices=json.dumps([0, 1, 2]))


def test_static_features_are_continuous_and_normalized():
    names = keypoint_names()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (80, 130, 180)
    calculator = ROIQualityCalculator(names)
    result = calculator.calculate(image, roi_row(), keypoints=points(names))
    for field in ('visible_skin_proportion', 'occlusion_proxy', 'blur_score',
                  'illumination_score', 'saturation_score', 'swaddling_proxy'):
        assert result[field] is not None and 0 <= result[field] <= 1
    assert result['quality_status'] == 'AVAILABLE'
    assert result['quality_scoring_implemented'] is False
    assert result['motion_magnitude'] is None
    assert result['pose_confidence'] if 'pose_confidence' in result else True


def test_temporal_features_use_actual_timestamp_and_are_bounded():
    names = keypoint_names()
    first = np.random.default_rng(1).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    second = np.roll(first, 2, axis=1)
    current = points(names)
    previous = points(names)
    for point in current[:3]:
        point['x'] += 4
    calculator = ROIQualityCalculator(names, ROIQualityConfig(landmark_velocity_scale=100))
    row = roi_row()
    result = calculator.calculate(second, row, first, row, current, previous, 1.0, .5)
    for field in ('motion_magnitude', 'landmark_velocity', 'roi_centroid_displacement',
                  'roi_area_variation', 'consecutive_roi_iou', 'temporal_consistency'):
        assert result[field] is not None and 0 <= result[field] <= 1
    assert result['landmark_velocity_pixels_per_second'] > 0
    assert result['consecutive_roi_iou'] == pytest.approx(1.)


def test_geometry_change_and_roi_iou_are_continuous():
    names = keypoint_names()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    calculator = ROIQualityCalculator(names)
    result = calculator.calculate(image, roi_row(x1=20, y1=20, x2=70, y2=70), image,
                                  roi_row(x1=10, y1=10, x2=60, y2=60), points(names), points(names), 1., 0.)
    assert 0 < result['consecutive_roi_iou'] < 1
    assert result['roi_area_change'] == pytest.approx(0.)
    assert result['roi_area_variation'] == pytest.approx(0.)
    assert result['roi_centroid_displacement_pixels'] > 0


def test_unavailable_roi_does_not_receive_fabricated_quality_values():
    names = keypoint_names()
    row = roi_row()
    row['roi_status'] = 'UNAVAILABLE'
    result = ROIQualityCalculator(names).calculate(np.zeros((100, 100, 3), np.uint8), row, keypoints=points(names))
    assert result['quality_status'] == 'UNAVAILABLE'
    assert result['visible_skin_proportion'] is None
    assert result['temporal_consistency'] is None


@pytest.mark.parametrize('values', [dict(blur_scale=0), dict(motion_scale_pixels=-1), dict(skin_cr_center=float('nan'))])
def test_quality_config_validation(values):
    with pytest.raises(ValueError): ROIQualityConfig(**values)
