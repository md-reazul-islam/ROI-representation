from pathlib import Path
import math
import sys

import numpy as np

from module1.data.sampler import SampledFrame
from .base import Keypoint, PoseResult
from .quality import PoseSanityChecker


def load_pose_weights(model, checkpoint, opener):
    """Strict key/shape validation with bounded checkpoint-loading memory."""
    import torch

    state = model.state_dict()
    with opener(str(checkpoint), framework='pt', device='cpu', backend='pread') as handle:
        if set(handle.keys()) != set(state):
            raise ValueError('Sapiens2 checkpoint keys do not exactly match the model')
        for name, target in state.items():
            if tuple(handle.get_slice(name).get_shape()) != tuple(target.shape):
                raise ValueError(f'Sapiens2 checkpoint shape mismatch: {name}')
        with torch.no_grad():
            for name, target in state.items():
                target.copy_(handle.get_tensor(name))


class Sapiens2Backend:
    """Official top-down preprocessing and UDP decoding, without a person detector."""

    def __init__(self, source_root, model_config, checkpoint, device='cpu', dtype='float32'):
        source_root, model_config, checkpoint = map(Path, (source_root, model_config, checkpoint))
        for path in (source_root / 'sapiens/pose/configs/_base_/keypoints308.py', model_config, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'Missing Sapiens2 asset: {path}. See README Phase 7 setup.')
        if checkpoint.suffix != '.safetensors':
            raise ValueError('Use an official Sapiens2 pose .safetensors checkpoint')
        sys.path.insert(0, str(source_root.resolve()))
        try:
            import torch
            from safetensors import safe_open
            from sapiens.engine.config import Config
            from sapiens.pose.models import init_model
            from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
        except ImportError as exc:
            raise RuntimeError('Sapiens2 dependencies are unavailable. Install requirements-pose.txt in a dedicated environment.') from exc
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu or a CUDA environment')
        if dtype not in ('float32', 'bfloat16'):
            raise ValueError('dtype must be float32 or bfloat16')
        self.dtype = getattr(torch, dtype)
        cfg = Config.fromfile(str(model_config))
        if (cfg.num_keypoints != 308 or cfg.model['backbone']['type'] != 'Sapiens2'
                or not cfg.model['backbone']['arch'].startswith('sapiens2_')):
            raise ValueError('Configuration must specify Sapiens2 with 308 keypoints')
        self.meta = parse_pose_metainfo(dict(from_file=str(source_root / 'sapiens/pose/configs/_base_/keypoints308.py')))
        self.names = tuple(self.meta['keypoint_id2name'][i] for i in range(308))
        self.skeleton = tuple(tuple(edge) for edge in self.meta['skeleton_links'])
        codec = dict(cfg.codec)
        if codec.pop('type') != 'UDPHeatmap':
            raise ValueError('Unsupported Sapiens2 pose codec')
        self.codec = UDPHeatmap(**codec)
        # Upstream init_model loads non-strictly. Require complete weights before inference.
        self.model = init_model(str(model_config), None, device='cpu')
        # Do not round the original normalization constants when casting model weights.
        for name, module in self.model.named_children():
            if name != 'data_preprocessor':
                module.to(dtype=self.dtype)
        load_pose_weights(self.model, checkpoint, safe_open)
        self.model.to(device).eval()
        self.torch = torch
        self.device = device

    def predict(self, image, bbox):
        data = self.model.pipeline(dict(img=image, bbox=np.asarray([bbox], dtype=np.float32),
                                        bbox_score=np.ones(1, dtype=np.float32)))
        data = self.model.data_preprocessor(data)
        inputs = data['inputs'].to(dtype=self.dtype)
        with self.torch.inference_mode():
            prediction = self.model(inputs)
            if self.model.cfg.val_cfg and self.model.cfg.val_cfg.get('flip_test', False):
                flipped = self.model(inputs.flip(-1)).flip(-1)
                prediction = (prediction + flipped[:, self.meta['flip_indices']]) / 2
        heatmaps = prediction.float().cpu().numpy()
        if heatmaps.ndim != 4 or heatmaps.shape[:2] != (1, 308):
            raise ValueError('Sapiens2 output must be one 308-channel heatmap tensor')
        xy, scores = self.codec.decode(heatmaps[0])
        meta = data['data_samples']['meta']
        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        # Official inverse top-down transform includes padding/aspect-ratio expansion.
        xy = xy / array(meta['input_size']) * array(meta['bbox_scale']) + array(meta['bbox_center']) - .5*array(meta['bbox_scale'])
        return np.asarray(xy[0]), np.asarray(scores[0]), None


class Sapiens2PoseEstimator:
    def __init__(self, source_root=None, model_config=None, checkpoint=None, device='cpu',
                 quality_config=None, *, backend=None, dtype='float32'):
        if backend is None:
            if any(value is None for value in (source_root, model_config, checkpoint)):
                raise ValueError('source_root, model_config and checkpoint are required')
            backend = Sapiens2Backend(source_root, model_config, checkpoint, device, dtype)
        self.backend = backend
        self.names = tuple(backend.names)
        self.skeleton = tuple(backend.skeleton)
        self.checker = PoseSanityChecker(self.names, quality_config)

    def reset(self):
        self.checker.reset()

    def estimate(self, frame: SampledFrame, bbox, mask=None) -> PoseResult:
        if not isinstance(frame, SampledFrame):
            raise TypeError('frame must be SampledFrame(frame_index, timestamp_sec, BGR_image)')
        image = frame.frame
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or min(image.shape[:2]) <= 0):
            raise ValueError('Expected a nonempty uint8 BGR image')
        if (not isinstance(frame.frame_index, int) or isinstance(frame.frame_index, bool) or frame.frame_index < 0
                or not math.isfinite(frame.timestamp_sec) or frame.timestamp_sec < 0):
            raise ValueError('Invalid frame metadata')
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('bbox must be finite positive-area xyxy')
        h, w = image.shape[:2]
        if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
            raise ValueError('bbox does not intersect the original image')
        if self.checker.last is not None and (frame.frame_index <= self.checker.last[0]
                                             or frame.timestamp_sec <= self.checker.last[1]):
            raise ValueError('Pose frames must increase strictly; reset between videos')
        # Preserve the input box, including any border overflow; never substitute a pose box.
        box = tuple(float(v) for v in box)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2] or mask.dtype != np.bool_ or not mask.any():
                raise ValueError('SAM 2 mask must be a nonempty original-frame boolean array')
            # Neutral ImageNet-mean BGR background keeps the official Sapiens2 model,
            # weights and top-down bbox geometry unchanged.
            image = image.copy()
            image[~mask] = (104, 116, 124)
        xy, scores, visibility = self.backend.predict(image, box)
        xy, scores = np.asarray(xy, dtype=float), np.asarray(scores, dtype=float)
        if visibility is not None:
            visibility = np.asarray(visibility, dtype=float)
            if visibility.shape != (308,) or not np.isfinite(visibility).all():
                raise ValueError('Invalid visibility output')
        confidence, quality, checks, status = self.checker.check(
            xy, scores, box, (w, h), frame.frame_index, frame.timestamp_sec)
        points = tuple(Keypoint(i, float(x), float(y), float(scores[i]),
                                float(visibility[i]) if visibility is not None else None)
                       for i, (x, y) in enumerate(xy))
        return PoseResult(points, confidence, box, quality, frame.timestamp_sec, frame.frame_index, checks, status)
