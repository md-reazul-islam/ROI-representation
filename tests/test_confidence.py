import json

import pytest

from module1.confidence import ConfidenceConfig, ConfidenceMechanism, ConfidenceWeights


def row(name='FACE', q=0.8, split='train'):
    return dict(video_id='v', split=split, frame_index=0, timestamp_sec=0., roi_name=name,
                roi_status='AVAILABLE', pose_confidence=q, visible_skin_proportion=q,
                occlusion_proxy=1-q, blur_score=q, illumination_score=q, saturation_score=q,
                motion_magnitude=1-q, landmark_velocity=q, roi_centroid_displacement=q,
                roi_area_variation=1-q, temporal_consistency=q, swaddling_proxy=1-q,
                bbox_normalized=json.dumps([0., 0., q, q]), centroid_bbox_normalized=json.dumps([q/2, q/2]),
                width=10*q, height=10*q, anatomical_confidence=q)


def test_equal_formula_and_softmax_selection():
    mechanism = ConfidenceMechanism()
    q, terms, imputed = mechanism.score(row(q=.8))
    assert q == pytest.approx(.2*(.8+.8+.8+.8-.2))
    assert terms == pytest.approx((.8, .8, .8, .8, .2))
    assert imputed == []
    probabilities = mechanism.softmax([0., 1., None])
    assert sum(probabilities) == pytest.approx(1.)
    assert probabilities[1] > probabilities[0] and probabilities[2] == 0
    assert mechanism.rank([.5, .5, None]) == [1, 2, None]


def test_temporal_and_motion_missing_use_explicit_neutral_imputation():
    value = row(q=.7)
    value['temporal_consistency'] = ''
    value['motion_magnitude'] = ''
    q, terms, imputed = ConfidenceMechanism().score(value)
    assert terms == pytest.approx((.7, .7, .7, 1., 0.))
    assert imputed == ['temporal', 'motion']
    assert q == pytest.approx(.7*.6 + .2)


def test_unavailable_roi_has_no_confidence_or_alpha():
    value = row(q=.8)
    value['roi_status'] = 'UNAVAILABLE'
    mechanism = ConfidenceMechanism()
    assert mechanism.score(value)[0] is None
    scores = [None, mechanism.score(row(q=.5))[0]]
    assert mechanism.softmax(scores) == pytest.approx([0., 1.])


def test_weights_normalize_and_validation_tuning_is_label_free():
    weights = ConfidenceWeights(2, 1, 1, 1, 1).normalized()
    assert sum(weights.as_dict().values()) == pytest.approx(1.)
    rows = [row(f'R{i}', q=.2+i*.04, split='validation') for i in range(17)]
    mechanism = ConfidenceMechanism(ConfidenceConfig(regime='validation_tuned', min_training_rows=10))
    fitted = mechanism.fit(rows)
    assert sum(fitted.as_dict().values()) == pytest.approx(1.)
    assert all(value >= 0 for value in fitted.as_dict().values())
    assert mechanism.fit_metadata['weak_supervision'] is True
    assert 'ground' not in mechanism.fit_metadata['target_definition']


def test_learned_regime_uses_training_rows_or_documented_fallback():
    rows = [row(f'R{i}', q=.2+i*.04, split='train') for i in range(17)]
    mechanism = ConfidenceMechanism(ConfidenceConfig(regime='weakly_supervised', min_training_rows=10))
    mechanism.fit(rows)
    assert mechanism.fit_metadata['training_source'] == 'training split'
    assert mechanism.weights.as_dict() != ConfidenceWeights(.2, .2, .2, .2, .2).as_dict()


def test_representation_is_numeric_and_tracks_missing_features():
    value = row(q=.8)
    value['temporal_consistency'] = ''
    vector, missing = ConfidenceMechanism.representation(value)
    assert all(isinstance(x, float) for x in vector)
    assert 'temporal_consistency' in missing
    assert len(vector) == 19


@pytest.mark.parametrize('values', [dict(regime='bad'), dict(top_k=0), dict(temperature=0), dict(iterations=0)])
def test_config_validation(values):
    with pytest.raises(ValueError): ConfidenceConfig(**values)
