from types import SimpleNamespace

import numpy as np
import pytest

from module1.data.sampler import SampledFrame
from module1.pose import PoseQualityConfig, Sapiens2PoseEstimator
from module1.pose.quality import PoseSanityChecker
from module1.pose.sapiens2 import Sapiens2Backend, load_pose_weights


def names():
    body = [f'{side}_{part}' for side in ('left', 'right')
            for part in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle')]
    return tuple(body + [f'point_{i}' for i in range(308-len(body))])


def prediction():
    xy = np.full((308, 2), 50.)
    xy[:6] = [[35, 25], [30, 40], [25, 55], [40, 55], [35, 70], [30, 85]]
    xy[6:12] = [[65, 25], [70, 40], [75, 55], [60, 55], [65, 70], [70, 85]]
    return xy, np.full(308, .9)


class FakeBackend:
    names = names()
    skeleton = ((0, 1), (1, 2))
    device = 'test_only'

    def __init__(self):
        self.xy, self.scores = prediction()
        self.visibility = None
        self.calls = []

    def predict(self, image, bbox):
        self.calls.append((image.copy(), bbox))
        return self.xy.copy(), self.scores.copy(), self.visibility


def frame(i=0, time=None):
    return SampledFrame(i, i/20 if time is None else time, np.zeros((100, 100, 3), np.uint8))


def test_adapter_contract_and_unchanged_box():
    backend = FakeBackend()
    estimator = Sapiens2PoseEstimator(backend=backend)
    result = estimator.estimate(frame(10), [-2., 5., 98., 100.])
    assert result.bbox == (-2., 5., 98., 100.)
    assert result.bbox == backend.calls[0][1]
    assert result.frame_index == 10 and result.timestamp == .5
    assert len(result.keypoints) == 308
    assert [p.index for p in result.keypoints] == list(range(308))
    assert (result.keypoints[0].x, result.keypoints[0].y) == (35, 25)
    assert all(p.visibility is None for p in result.keypoints)
    assert result.status == 'ESTIMATED'
    assert result.pose_quality == pytest.approx(.9)
    assert result.checks['temporal_plausibility'] == 'NOT_ASSESSED'


@pytest.mark.parametrize('kind', ['count', 'scores', 'nan', 'outside', 'low', 'collapse', 'visibility'])
def test_bad_predictions_fail_or_flag_without_fabrication(kind):
    backend = FakeBackend()
    if kind == 'count': backend.xy = backend.xy[:17]
    if kind == 'scores': backend.scores = backend.scores[:17]
    if kind == 'nan': backend.xy[0, 0] = np.nan
    if kind == 'outside': backend.xy[0, 0] = -1
    if kind == 'low': backend.scores[:] = .1
    if kind == 'collapse': backend.xy[:] = 50
    if kind == 'visibility': backend.visibility = np.ones(17)
    estimator = Sapiens2PoseEstimator(backend=backend)
    if kind in ('count', 'scores', 'visibility'):
        with pytest.raises(ValueError): estimator.estimate(frame(), [0, 0, 100, 100])
    else:
        result = estimator.estimate(frame(), [0, 0, 100, 100])
        assert result.status == 'FLAGGED' and result.pose_quality == 0
        assert len(result.keypoints) == 308


def test_visibility_preserved_and_raw_confidence_not_clipped():
    backend = FakeBackend()
    backend.visibility = np.linspace(0, 1, 308)
    backend.scores[:] = 1.2
    result = Sapiens2PoseEstimator(backend=backend).estimate(frame(), [0, 0, 100, 100])
    assert result.keypoints[-1].visibility == 1
    assert result.keypoints[0].confidence == 1.2
    assert result.confidence == pytest.approx(1.2)
    assert result.pose_quality == 1


def test_temporal_jump_gap_order_and_reset():
    backend = FakeBackend()
    estimator = Sapiens2PoseEstimator(backend=backend, quality_config=PoseQualityConfig(max_keypoint_speed=.1))
    estimator.estimate(frame(), [0, 0, 100, 100])
    backend.xy[:, 0] += 10
    result = estimator.estimate(frame(1), [0, 0, 100, 100])
    assert result.checks['temporal_plausibility'] == 'FAIL'
    assert result.pose_quality == 0
    result = estimator.estimate(frame(20), [0, 0, 100, 100])
    assert result.checks['temporal_reason'] == 'GAP_TOO_LONG'
    with pytest.raises(ValueError, match='increase'):
        estimator.estimate(frame(20), [0, 0, 100, 100])
    estimator.reset()
    assert estimator.estimate(frame(), [0, 0, 100, 100]).checks['temporal_reason'] == 'NO_PREVIOUS_VALID_POSE'


def test_sparse_body_is_not_anatomical_pass():
    backend = FakeBackend()
    backend.scores[:12] = 0
    result = Sapiens2PoseEstimator(backend=backend).estimate(frame(), [0, 0, 100, 100])
    assert result.checks['anatomical_plausibility'] == 'NOT_ASSESSED'
    assert result.status == 'INSUFFICIENT_EVIDENCE' and result.pose_quality == 0


def test_anatomical_bilateral_ratio_check():
    backend = FakeBackend()
    backend.xy[1] = backend.xy[0] + [1, 0]
    result = Sapiens2PoseEstimator(backend=backend).estimate(frame(), [0, 0, 100, 100])
    assert result.checks['anatomical_plausibility'] == 'FAIL'


def test_temporal_uses_actual_time_and_does_not_refresh_from_bad_pose():
    backend = FakeBackend()
    estimator = Sapiens2PoseEstimator(backend=backend, quality_config=PoseQualityConfig(max_keypoint_speed=.1))
    estimator.estimate(frame(), [0, 0, 100, 100])
    backend.xy[:, 0] += 2
    first = estimator.estimate(frame(1, .05), [0, 0, 100, 100])
    assert first.checks['temporal_plausibility'] == 'FAIL'
    second = estimator.estimate(frame(2, .25), [0, 0, 100, 100])
    assert second.checks['temporal_plausibility'] == 'PASS'
    assert second.checks['temporal_speed'] == pytest.approx(2 / np.hypot(100, 100) / .25)


@pytest.mark.parametrize('box', [[0, 0, 0, 20], [np.nan, 0, 20, 30], [120, 0, 140, 30], [1, 2, 3]])
def test_invalid_boxes_never_reach_model(box):
    backend = FakeBackend()
    with pytest.raises(ValueError): Sapiens2PoseEstimator(backend=backend).estimate(frame(), box)
    assert backend.calls == []


@pytest.mark.parametrize('values', [dict(confidence_threshold=-1), dict(max_keypoint_speed=0),
                                  dict(min_body_keypoints=0), dict(min_bone_length=1),
                                  dict(max_temporal_gap_seconds=float('nan'))])
def test_quality_configuration_validation(values):
    with pytest.raises(ValueError): PoseQualityConfig(**values)


def test_backend_missing_assets_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing Sapiens2 asset'):
        Sapiens2Backend(tmp_path, tmp_path/'model.py', tmp_path/'pose.safetensors')


def test_official_backend_decode_coordinate_restoration_and_flip():
    import torch

    backend = Sapiens2Backend.__new__(Sapiens2Backend)
    backend.torch = torch
    backend.dtype = torch.float32
    backend.meta = dict(flip_indices=list(range(308)))
    seen = []

    class Model:
        cfg = SimpleNamespace(val_cfg=dict(flip_test=True))
        def pipeline(self, data):
            seen.append(data)
            return dict(inputs=torch.zeros(1, 3, 8, 6), data_samples=dict(meta=dict(
                input_size=np.array([768, 1024]), bbox_center=np.array([[120, 150]]),
                bbox_scale=np.array([[60, 80]]))))
        def data_preprocessor(self, data): return data
        def __call__(self, inputs): return torch.ones(1, 308, 4, 3)

    backend.model = Model()
    backend.codec = SimpleNamespace(decode=lambda heatmap: (np.tile([[[384., 512.]]], (1, 308, 1)), np.ones((1, 308))*.8))
    xy, scores, visibility = backend.predict(frame().frame, (100, 100, 140, 200))
    np.testing.assert_allclose(xy, np.tile([120., 150.], (308, 1)))
    assert scores.shape == (308,) and visibility is None
    np.testing.assert_equal(seen[0]['bbox'], [[100, 100, 140, 200]])
    assert len(seen) == 1


@pytest.mark.parametrize('mutation', [None, 'missing', 'extra', 'shape'])
def test_streamed_weights_require_exact_keys_and_shapes(mutation):
    import torch

    model = torch.nn.Linear(3, 2)
    tensors = {'weight': torch.ones(2, 3), 'bias': torch.ones(2)}
    if mutation == 'missing': tensors.pop('bias')
    if mutation == 'extra': tensors['unexpected'] = torch.zeros(1)
    if mutation == 'shape': tensors['weight'] = torch.zeros(3, 2)
    initial = model.weight.detach().clone()

    class Handle:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return tensors.keys()
        def get_slice(self, name): return SimpleNamespace(get_shape=lambda: tensors[name].shape)
        def get_tensor(self, name): return tensors[name]

    def opener(path, **kwargs):
        assert kwargs['backend'] == 'pread'
        return Handle()

    if mutation:
        with pytest.raises(ValueError, match='checkpoint'):
            load_pose_weights(model, 'test-only', opener)
        torch.testing.assert_close(model.weight, initial)
    else:
        load_pose_weights(model, 'test-only', opener)
        torch.testing.assert_close(model.weight, tensors['weight'])
        torch.testing.assert_close(model.bias, tensors['bias'])


def test_bfloat16_model_keeps_original_float32_normalization(tmp_path, monkeypatch):
    import sys
    import torch
    from types import ModuleType
    import module1.pose.sapiens2 as adapter

    meta_path = tmp_path/'sapiens/pose/configs/_base_/keypoints308.py'
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text('# test metadata')
    config_path, checkpoint = tmp_path/'model.py', tmp_path/'model.safetensors'
    config_path.write_text('# test config')
    checkpoint.write_bytes(b'test-only')
    monkeypatch.syspath_prepend(str(tmp_path))
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.decode_head = torch.nn.Linear(2, 2)
    model.data_preprocessor = torch.nn.Module()
    original_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
    model.data_preprocessor.register_buffer('mean', original_mean.clone(), persistent=False)
    cfg = SimpleNamespace(num_keypoints=308, model={'backbone': {'type': 'Sapiens2', 'arch': 'sapiens2_0.4b'}},
                          codec={'type': 'UDPHeatmap'})
    attributes = {
        'safetensors': dict(safe_open=object()),
        'sapiens': {}, 'sapiens.engine': {}, 'sapiens.pose': {},
        'sapiens.engine.config': dict(Config=SimpleNamespace(fromfile=lambda path: cfg)),
        'sapiens.pose.models': dict(init_model=lambda *a, **kw: model),
        'sapiens.pose.datasets': dict(parse_pose_metainfo=lambda meta: {
            'keypoint_id2name': dict(enumerate(names())), 'skeleton_links': []}, UDPHeatmap=lambda **kw: None),
    }
    for name, values in attributes.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(adapter, 'load_pose_weights', lambda *a: None)
    backend = Sapiens2Backend(tmp_path, config_path, checkpoint, dtype='bfloat16')
    assert backend.model.backbone.weight.dtype == torch.bfloat16
    assert backend.model.decode_head.weight.dtype == torch.bfloat16
    assert backend.model.data_preprocessor.mean.dtype == torch.float32
    torch.testing.assert_close(backend.model.data_preprocessor.mean, original_mean, rtol=0, atol=0)
