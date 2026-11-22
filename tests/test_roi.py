import json

import numpy as np
import pytest

from module1.roi import AnatomicalCoordinateSystem, AnatomicalROIMapper, ROI_NAMES


def keypoint_names():
    names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
             'left_hip', 'right_hip', 'left_knee', 'right_knee',
             'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
             'left_big_toe', 'right_big_toe', 'neck', 'left_acromion', 'right_acromion',
             'left_wrist', 'right_wrist']
    names += [f'face-{i}' for i in range(100)]
    names += [f'point-{i}' for i in range(308-len(names))]
    return tuple(names)


def points(names, confidence=.9):
    values = np.full((308, 2), np.nan)
    positions = {
        'nose': (50, 45), 'left_eye': (43, 40), 'right_eye': (57, 40),
        'left_ear': (35, 48), 'right_ear': (65, 48), 'neck': (50, 78),
        'left_shoulder': (25, 85), 'right_shoulder': (75, 85),
        'left_acromion': (27, 85), 'right_acromion': (73, 85),
        'left_elbow': (15, 115), 'right_elbow': (85, 115),
        'left_wrist': (10, 145), 'right_wrist': (90, 145),
        'left_hip': (35, 165), 'right_hip': (65, 165),
        'left_knee': (30, 230), 'right_knee': (70, 230),
        'left_ankle': (25, 295), 'right_ankle': (75, 295),
        'left_heel': (20, 305), 'right_heel': (80, 305),
        'left_big_toe': (30, 310), 'right_big_toe': (70, 310),
    }
    for i, name in enumerate(names):
        if name.startswith('face-'):
            j = int(name.split('-')[1])
            angle = 2*np.pi*j/100
            values[i] = [50+18*np.cos(angle), 48+25*np.sin(angle)]
        elif name in positions:
            values[i] = positions[name]
    return [dict(index=i, x=float(values[i, 0]), y=float(values[i, 1]), confidence=confidence)
            if np.isfinite(values[i]).all() else dict(index=i, x=None, y=None, confidence=0.)
            for i in range(308)]


def test_coordinate_system_bbox_and_torso_normalization():
    system = AnatomicalCoordinateSystem((100, 50, 300, 450),
                                        ((150, 150), (250, 150), (270, 350), (130, 350)))
    assert system.bbox_normalize((200, 250)) == pytest.approx((.5, .5))
    assert system.torso_normalize((200, 250)) == pytest.approx((.5, .5))
    geometry = system.geometry([(150, 150), (250, 150), (270, 350), (130, 350)])
    assert geometry[3:5] == pytest.approx((140, 200))
    assert geometry[7] == pytest.approx((.5, .5138888889))
    assert geometry[8] == pytest.approx((.5, .5277777778))


def test_dynamic_regions_and_normalized_coordinates():
    names = keypoint_names()
    mapper = AnatomicalROIMapper(names, min_keypoint_confidence=.5)
    result = {roi.roi_name: roi for roi in mapper.map_pose((0, 0, 100, 320), points(names))}
    assert set(result) == set(ROI_NAMES)
    for name in ('HEAD', 'FACE', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_CHEST', 'ABDOMEN', 'LEFT_ARM', 'RIGHT_LEG'):
        assert result[name].status == 'AVAILABLE'
        assert result[name].bbox_normalized is not None
        assert result[name].centroid_bbox_normalized is not None
        assert result[name].anatomical_confidence == pytest.approx(.9)
    assert result['UPPER_CHEST'].centroid_torso_normalized is not None
    assert result['UPPER_CHEST'].geometry_type == 'POLYGON'


def test_missing_or_low_confidence_anatomy_is_unavailable_not_invented():
    names = keypoint_names()
    data = points(names)
    for i, name in enumerate(names):
        if name in ('left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                    'left_heel', 'right_heel', 'left_big_toe', 'right_big_toe'):
            data[i]['confidence'] = .1
    result = {roi.roi_name: roi for roi in AnatomicalROIMapper(names).map_pose((0, 0, 100, 320), data)}
    assert result['LEFT_LEG'].status == 'UNAVAILABLE'
    assert result['RIGHT_LEG'].status == 'UNAVAILABLE'
    assert result['THORACO_ABDOMINAL'].status == 'UNAVAILABLE'
    assert result['THORACO_ABDOMINAL'].unavailable_reason == 'INSUFFICIENT_VALID_KEYPOINTS'
    assert result['HEAD'].status == 'AVAILABLE'


def test_no_torso_anchor_keeps_bbox_normalization_and_marks_torso_relative_missing():
    names = keypoint_names()
    data = points(names)
    for i, name in enumerate(names):
        if name in ('left_hip', 'right_hip'):
            data[i]['confidence'] = .1
    result = {roi.roi_name: roi for roi in AnatomicalROIMapper(names).map_pose((0, 0, 100, 320), data)}
    assert result['HEAD'].centroid_bbox_normalized is not None
    assert result['HEAD'].centroid_torso_normalized is None
    assert result['UPPER_CHEST'].status == 'UNAVAILABLE'


def test_invalid_inputs_and_configuration():
    names = keypoint_names()
    with pytest.raises(ValueError): AnatomicalROIMapper(names[:-1])
    with pytest.raises(ValueError): AnatomicalROIMapper(names, min_keypoint_confidence=2)
    with pytest.raises(ValueError): AnatomicalCoordinateSystem((0, 0, 0, 2))
    with pytest.raises(ValueError): AnatomicalCoordinateSystem((0, 0, 10, 10), ((0, 5), (5, 5), (5, 4), (0, 4)))


def test_result_record_contains_no_roi_score():
    names = keypoint_names()
    roi = AnatomicalROIMapper(names).map_pose((0, 0, 100, 320), points(names))[0]
    record = roi.as_record('v', 'train', 0, 0., 'ESTIMATED')
    assert 'roi_score' not in record
    assert record['anatomical_confidence'] == pytest.approx(.9)
    json.loads(record['polygon'])
