"""Box-prompted SAM 2 video segmentation; no infant detector lives here."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path

import cv2
import numpy as np


RELIABLE_STATES = frozenset(("INITIALIZED", "PROPAGATED", "REINITIALIZED"))
STATES = RELIABLE_STATES | {"LOW_CONFIDENCE", "FAILED", "DISABLED"}


@dataclass(frozen=True)
class SAM2Config:
    enabled: bool = False
    model_variant: str = "sam2.1_hiera_tiny"
    model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint: str = "models/sam2.1_hiera_tiny.pt"
    device: str = "auto"
    mask_threshold: float = 0.0
    chunk_frames: int = 96
    jpeg_quality: int = 100
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    min_prompt_confidence: float = 0.35
    max_frames_without_box: int = 10
    min_reprompt_interval: int = 3
    reinitialize_on_unreliable: bool = True
    min_mask_area_ratio: float = 0.002
    max_mask_area_ratio: float = 0.9
    min_detector_mask_consistency: float = 0.6
    min_temporal_consistency: float = 0.1
    min_mask_quality: float = 0.35
    boundary_tolerance_px: int = 3
    fallback_behavior: str = "bbox"
    save_annotated_frames: bool = True

    @classmethod
    def from_mapping(cls, values):
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def __post_init__(self):
        for name in ("enabled", "offload_video_to_cpu", "offload_state_to_cpu",
                     "reinitialize_on_unreliable", "save_annotated_frames"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, minimum in (("chunk_frames", 2), ("jpeg_quality", 1),
                              ("max_frames_without_box", 0), ("min_reprompt_interval", 1),
                              ("boundary_tolerance_px", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.jpeg_quality > 100:
            raise ValueError("jpeg_quality must be <= 100")
        for name in ("min_prompt_confidence", "min_mask_area_ratio",
                     "max_mask_area_ratio", "min_detector_mask_consistency",
                     "min_temporal_consistency", "min_mask_quality"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.min_mask_area_ratio >= self.max_mask_area_ratio:
            raise ValueError("mask area limits must increase")
        if not math.isfinite(self.mask_threshold):
            raise ValueError("mask_threshold must be finite")
        if not self.model_variant or not self.model_config or not self.checkpoint:
            raise ValueError("SAM 2 model variant, config and checkpoint are required")
        if not self.device or self.fallback_behavior != "bbox":
            raise ValueError("SAM 2 requires a device and bbox fallback")


def _native_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def binary_mask(logits, image_shape, threshold):
    """Return original-frame binary mask and original logits size."""
    scores = _native_array(logits)
    while scores.ndim > 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("SAM 2 returned invalid mask logits")
    height, width = image_shape[:2]
    source_shape = tuple(int(v) for v in scores.shape)
    if source_shape != (height, width):
        scores = cv2.resize(scores.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(scores > threshold, dtype=bool), source_shape


def _rectangle(shape, bbox):
    if bbox is None:
        return None
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Expected finite positive-area xyxy box")
    h, w = shape
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = np.clip((x1, x2), 0, w)
    y1, y2 = np.clip((y1, y2), 0, h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box does not intersect frame")
    rect = np.zeros((h, w), dtype=bool)
    rect[y1:y2, x1:x2] = True
    return rect


def _boundary_stability(current, previous, tolerance):
    kernel = np.ones((3, 3), np.uint8)
    a = cv2.morphologyEx(current.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    b = cv2.morphologyEx(previous.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    if not a.any() or not b.any():
        return None
    radius = 2*tolerance + 1
    dilation = np.ones((radius, radius), np.uint8)
    a_near = cv2.dilate(a.astype(np.uint8), dilation).astype(bool)
    b_near = cv2.dilate(b.astype(np.uint8), dilation).astype(bool)
    precision = np.count_nonzero(a & b_near) / np.count_nonzero(a)
    recall = np.count_nonzero(b & a_near) / np.count_nonzero(b)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0


def mask_measurements(mask, bbox=None, previous=None, config=None):
    """Geometric diagnostics, never segmentation accuracy."""
    config = config or SAM2Config()
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Mask must be a 2D boolean original-frame array")
    height, width = mask.shape
    area = int(np.count_nonzero(mask))
    if area == 0:
        raise ValueError("SAM 2 mask is empty")
    ys, xs = np.nonzero(mask)
    rect = _rectangle(mask.shape, bbox)
    inside = int(np.count_nonzero(mask & rect)) if rect is not None else None
    rect_area = int(np.count_nonzero(rect)) if rect is not None else None
    if previous is not None:
        previous = np.asarray(previous)
        if previous.shape != mask.shape or previous.dtype != np.bool_:
            raise ValueError("Previous mask uses a different coordinate system")
        intersection = int(np.count_nonzero(mask & previous))
        union = int(np.count_nonzero(mask | previous))
        temporal = intersection/union if union else None
        boundary = _boundary_stability(mask, previous, config.boundary_tolerance_px)
    else:
        temporal, boundary = None, None
    area_ratio = area/(height*width)
    consistency = inside/area if inside is not None else None
    box_iou = inside/(area+rect_area-inside) if inside is not None and area+rect_area-inside else None
    plausible = config.min_mask_area_ratio <= area_ratio <= config.max_mask_area_ratio
    diagnostics = [float(plausible)]
    diagnostics.extend(value for value in (consistency, temporal, boundary) if value is not None)
    quality = float(np.mean(diagnostics))
    reliable = (plausible and quality >= config.min_mask_quality
                and (consistency is None or consistency >= config.min_detector_mask_consistency)
                and (temporal is None or temporal >= config.min_temporal_consistency))
    return dict(mask_area=area, mask_area_ratio=area_ratio,
                mask_box_area_ratio=area/rect_area if rect_area else None,
                mask_bbox_x1=int(xs.min()), mask_bbox_y1=int(ys.min()),
                mask_bbox_x2=int(xs.max())+1, mask_bbox_y2=int(ys.max())+1,
                mask_centroid_x=float(xs.mean()+.5), mask_centroid_y=float(ys.mean()+.5),
                mask_quality=quality, temporal_consistency=temporal,
                detector_mask_consistency=consistency, mask_box_iou=box_iou,
                boundary_stability=boundary, mask_reliable=bool(reliable))


class SAM2VideoBackend:
    """Thin adapter over Meta's memory-bearing video predictor."""

    def __init__(self, config, project_root, *, predictor=None, builder=None, torch_module=None):
        self.config = config
        checkpoint = Path(project_root) / config.checkpoint
        if predictor is None:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SAM 2 checkpoint is missing: {checkpoint}")
            if torch_module is None:
                import torch as torch_module
            if builder is None:
                try:
                    from sam2.build_sam import build_sam2_video_predictor as builder
                except ImportError as exc:
                    raise RuntimeError("Install the official SAM 2 package for Phase 12") from exc
            device = ("cuda:0" if torch_module.cuda.is_available() else "cpu") if config.device == "auto" else config.device
            if device.startswith("cuda") and not torch_module.cuda.is_available():
                raise RuntimeError("CUDA requested for SAM 2 but unavailable")
            predictor = builder(config.model_config, str(checkpoint), device=device)
        else:
            device = config.device
        self.predictor = predictor
        self.device = device
        self.internal_image_size = getattr(predictor, "image_size", None)

    def init_state(self, frame_dir):
        return self.predictor.init_state(
            str(frame_dir), offload_video_to_cpu=self.config.offload_video_to_cpu,
            offload_state_to_cpu=self.config.offload_state_to_cpu)

    def prompt(self, state, frame_index, object_id, box):
        out_index, ids, logits = self.predictor.add_new_points_or_box(
            state, frame_idx=frame_index, obj_id=object_id, box=np.asarray(box, dtype=np.float32))
        if int(out_index) != frame_index:
            raise ValueError("SAM 2 prompt returned another frame")
        return self._for_object(ids, logits, object_id)

    def propagate(self, state, start_index):
        return self.predictor.propagate_in_video(state, start_frame_idx=start_index)

    def reset(self, state):
        self.predictor.reset_state(state)

    @staticmethod
    def _for_object(ids, logits, object_id):
        object_ids = [int(value) for value in ids]
        if object_id not in object_ids:
            raise ValueError("SAM 2 did not return the prompted infant object")
        return logits[object_ids.index(object_id)]


class TemporalSAM2Segmenter:
    """Process one bounded video chunk in sampled order, retaining SAM 2 memory."""

    def __init__(self, backend, config):
        self.backend = backend
        self.config = config

    @staticmethod
    def _base(row):
        return dict(video_id=row["video_id"], split=row["split"],
                    frame_index=row["frame_index"], timestamp_sec=row["timestamp_sec"],
                    track_id=row.get("track_id"), final_status=row["final_status"],
                    final_infant_bbox=row.get("final_infant_bbox"),
                    upstream_localization_confidence=row["final_infant_confidence"],
                    segmentation_status="FAILED", mask_path=None, failure_reason=None,
                    mask_area=None, mask_area_ratio=None, mask_box_area_ratio=None,
                    mask_bbox_x1=None, mask_bbox_y1=None, mask_bbox_x2=None, mask_bbox_y2=None,
                    mask_centroid_x=None, mask_centroid_y=None, mask_quality=None,
                    temporal_consistency=None, detector_mask_consistency=None,
                    mask_box_iou=None, boundary_stability=None, mask_reliable=False,
                    sam2_input_height=None, sam2_input_width=None,
                    original_height=None, original_width=None, mask_resized_to_original=False)

    def run_chunk(self, frame_dir, sampled_rows, *, restarted=False):
        state = self.backend.init_state(frame_dir)
        generator = None
        active = False
        object_id = 1
        active_track = None
        previous_reliable = None
        last_prompt = -10**9
        unboxed = 0
        try:
            for local_index, (row, image) in enumerate(sampled_rows):
                record = self._base(row)
                record["original_height"], record["original_width"] = image.shape[:2]
                bbox = row.get("bbox")
                promptable = bbox is not None and row["final_infant_confidence"] >= self.config.min_prompt_confidence
                if local_index and (row["frame_index"] <= sampled_rows[local_index-1][0]["frame_index"]
                                    or row["timestamp_sec"] <= sampled_rows[local_index-1][0]["timestamp_sec"]):
                    raise ValueError("SAM 2 frames must preserve original video order")
                if active and row.get("track_id") is not None and active_track is not None and row["track_id"] != active_track:
                    if generator is not None:
                        generator.close()
                    generator = None
                    self.backend.reset(state)
                    active, previous_reliable = False, None
                unboxed = 0 if promptable else unboxed + 1
                if active and unboxed > self.config.max_frames_without_box:
                    if generator is not None:
                        generator.close()
                    generator = None
                    self.backend.reset(state)
                    active, previous_reliable = False, None
                try:
                    if not active:
                        if not promptable:
                            record["failure_reason"] = "NO_RELIABLE_UPSTREAM_BOX"
                            yield record, None
                            continue
                        clipped = _rectangle(image.shape[:2], bbox)
                        ys, xs = np.nonzero(clipped)
                        prompt_box = (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))
                        raw = self.backend.prompt(state, local_index, object_id, prompt_box)
                        active = True
                        active_track = row.get("track_id")
                        last_prompt = local_index
                        status = "REINITIALIZED" if restarted else "INITIALIZED"
                        restarted = True
                    else:
                        if generator is None:
                            generator = iter(self.backend.propagate(state, local_index))
                        out_index, ids, logits = next(generator)
                        if int(out_index) != local_index:
                            raise ValueError("SAM 2 propagation changed frame ordering")
                        raw = self.backend._for_object(ids, logits, object_id)
                        status = "PROPAGATED"
                    mask, source_shape = binary_mask(raw, image.shape, self.config.mask_threshold)
                    record["sam2_input_height"], record["sam2_input_width"] = source_shape
                    record["mask_resized_to_original"] = source_shape != image.shape[:2]
                    if not mask.any():
                        raise ValueError("SAM 2 returned an empty mask")
                    measured = mask_measurements(mask, bbox, previous_reliable, self.config)
                    if (not measured["mask_reliable"] and status == "PROPAGATED" and promptable
                            and self.config.reinitialize_on_unreliable
                            and local_index-last_prompt >= self.config.min_reprompt_interval):
                        if generator is not None:
                            generator.close()
                        generator = None
                        self.backend.reset(state)
                        clipped = _rectangle(image.shape[:2], bbox)
                        ys, xs = np.nonzero(clipped)
                        prompt_box = (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))
                        raw = self.backend.prompt(state, local_index, object_id, prompt_box)
                        last_prompt = local_index
                        previous_reliable = None
                        mask, source_shape = binary_mask(raw, image.shape, self.config.mask_threshold)
                        record["sam2_input_height"], record["sam2_input_width"] = source_shape
                        record["mask_resized_to_original"] = source_shape != image.shape[:2]
                        if not mask.any():
                            raise ValueError("SAM 2 re-prompt returned an empty mask")
                        measured = mask_measurements(mask, bbox, None, self.config)
                        status = "REINITIALIZED"
                    record.update(measured)
                    record["segmentation_status"] = status if measured["mask_reliable"] else "LOW_CONFIDENCE"
                    if measured["mask_reliable"]:
                        previous_reliable = mask
                    yield record, mask
                except Exception as exc:
                    record["segmentation_status"] = "FAILED"
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    if generator is not None:
                        generator.close()
                    generator = None
                    self.backend.reset(state)
                    active, previous_reliable = False, None
                    yield record, None
        finally:
            if generator is not None:
                generator.close()
            self.backend.reset(state)
