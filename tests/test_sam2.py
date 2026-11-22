from types import SimpleNamespace

import numpy as np
import pytest

from module1.confidence import ConfidenceConfig, ConfidenceMechanism
from module1.data.sampler import SampledFrame
from module1.pose import Sapiens2PoseEstimator
from module1.segmentation.sam2 import (
    SAM2Config, SAM2VideoBackend, TemporalSAM2Segmenter, binary_mask, mask_measurements)
from module1.segmentation.support import landmark_mask_support, roi_mask_support
from test_pose import FakeBackend


def square(shape=(20, 20), box=(4, 4, 16, 16)):
    result = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = box
    result[y1:y2, x1:x2] = True
    return result


@pytest.mark.parametrize("bad", [
    dict(chunk_frames=1), dict(mask_threshold=float("nan")),
    dict(min_mask_area_ratio=.9, max_mask_area_ratio=.1),
    dict(device=""), dict(fallback_behavior="none"),
    dict(min_prompt_confidence=2),
])
def test_sam2_configuration_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        SAM2Config(**bad)


def test_sam2_configuration_is_optional_and_filters_roi_settings():
    config = SAM2Config.from_mapping(dict(enabled=False, chunk_frames=12,
                                          roi_confidence=dict(enabled=True)))
    assert not config.enabled and config.chunk_frames == 12
    assert config.fallback_behavior == "bbox"


def test_model_loading_uses_official_video_builder_and_checks_checkpoint(tmp_path):
    config = SAM2Config(device="auto")
    with pytest.raises(FileNotFoundError):
        SAM2VideoBackend(config, tmp_path)
    checkpoint = tmp_path/config.checkpoint
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test-only")
    seen = []
    predictor = SimpleNamespace(image_size=1024)
    def builder(model_config, path, device):
        seen.append((model_config, path, device))
        return predictor
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    backend = SAM2VideoBackend(config, tmp_path, builder=builder, torch_module=torch)
    assert backend.predictor is predictor and backend.device == "cuda:0"
    assert seen == [(config.model_config, str(checkpoint), "cuda:0")]
    with pytest.raises(RuntimeError, match="CUDA"):
        SAM2VideoBackend(SAM2Config(device="cuda:0"), tmp_path,
                         builder=builder,
                         torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_box_prompt_uses_original_xyxy_and_selects_infant_object():
    calls = []
    class Predictor:
        def add_new_points_or_box(self, state, frame_idx, obj_id, box):
            calls.append((frame_idx, obj_id, box.tolist()))
            return frame_idx, [9, 1], np.asarray([np.zeros((1, 4, 4)), np.ones((1, 4, 4))])
    adapter = SAM2VideoBackend(SAM2Config(), ".", predictor=Predictor())
    mask = adapter.prompt({}, 2, 1, (1, 2, 8, 9))
    assert calls == [(2, 1, [1., 2., 8., 9.])]
    assert mask.shape == (1, 4, 4) and np.all(mask == 1)


def test_mask_resize_area_bbox_centroid_and_consistency():
    logits = np.array([[[1., -1.], [-1., -1.]]])
    mask, source_shape = binary_mask(logits, (4, 4, 3), 0.)
    assert source_shape == (2, 2) and mask.shape == (4, 4)
    assert mask.sum() > 0
    exact = square()
    measured = mask_measurements(exact, (4, 4, 16, 16), exact)
    assert measured["mask_area"] == 144
    assert measured["mask_area_ratio"] == pytest.approx(.36)
    assert measured["mask_box_area_ratio"] == 1.
    assert (measured["mask_bbox_x1"], measured["mask_bbox_y1"],
            measured["mask_bbox_x2"], measured["mask_bbox_y2"]) == (4, 4, 16, 16)
    assert (measured["mask_centroid_x"], measured["mask_centroid_y"]) == (10., 10.)
    assert measured["detector_mask_consistency"] == 1.
    assert measured["temporal_consistency"] == 1.
    assert measured["boundary_stability"] == 1.
    assert measured["mask_reliable"]


def test_low_quality_mask_is_flagged_without_ground_truth():
    mask = square(box=(0, 0, 8, 8))
    measured = mask_measurements(mask, (12, 12, 20, 20), square(box=(12, 12, 20, 20)))
    assert measured["detector_mask_consistency"] == 0
    assert measured["temporal_consistency"] == 0
    assert not measured["mask_reliable"]
    with pytest.raises(ValueError, match="empty"):
        mask_measurements(np.zeros((20, 20), bool))


class FakeVideoPredictor:
    image_size = 1024

    def __init__(self, masks, *, fail_prompt=False):
        self.masks = masks
        self.fail_prompt = fail_prompt
        self.prompts = []
        self.propagations = []
        self.reset_count = 0

    def init_state(self, path, **kwargs):
        return {"count": len(self.masks), "kwargs": kwargs}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self.prompts.append(frame_idx)
        if self.fail_prompt:
            raise RuntimeError("prompt failure")
        good = square().astype(np.float32)*2-1
        return frame_idx, [obj_id], good[None, None]

    def propagate_in_video(self, state, start_frame_idx):
        self.propagations.append(start_frame_idx)
        for index in range(start_frame_idx, state["count"]):
            yield index, [1], self.masks[index].astype(np.float32)[None, None]*2-1

    def reset_state(self, state):
        self.reset_count += 1


def rows(count=3, *, boxes=None, tracks=None):
    boxes = boxes or [(4, 4, 16, 16)]*count
    tracks = tracks or [1]*count
    return [(dict(video_id="v", split="train", frame_index=i*2,
                  timestamp_sec=i*.1, track_id=tracks[i],
                  final_status="DETECTED" if boxes[i] is not None else "MISSED",
                  final_infant_bbox=list(boxes[i]) if boxes[i] is not None else None,
                  bbox=boxes[i], final_infant_confidence=.9 if boxes[i] is not None else 0.),
             np.zeros((20, 20, 3), np.uint8)) for i in range(count)]


def test_video_initializes_once_then_propagates_with_temporal_memory():
    predictor = FakeVideoPredictor([square()]*3)
    backend = SAM2VideoBackend(SAM2Config(), ".", predictor=predictor)
    generated = list(TemporalSAM2Segmenter(backend, SAM2Config()).run_chunk("unused", rows()))
    assert [record["segmentation_status"] for record, _ in generated] == [
        "INITIALIZED", "PROPAGATED", "PROPAGATED"]
    assert predictor.prompts == [0]
    assert predictor.propagations == [1]
    assert all(mask is not None for _, mask in generated)
    assert generated[1][0]["frame_index"] == 2 and generated[1][0]["timestamp_sec"] == .1


def test_unreliable_propagation_reinitializes_only_with_reliable_box():
    predictor = FakeVideoPredictor([square(), square(box=(0, 0, 5, 5)), square()])
    config = SAM2Config(min_reprompt_interval=1)
    backend = SAM2VideoBackend(config, ".", predictor=predictor)
    generated = list(TemporalSAM2Segmenter(backend, config).run_chunk("unused", rows()))
    assert [record["segmentation_status"] for record, _ in generated] == [
        "INITIALIZED", "REINITIALIZED", "PROPAGATED"]
    assert predictor.prompts == [0, 1]
    assert predictor.reset_count >= 1


def test_missing_box_propagates_then_falls_back_when_unreliable():
    predictor = FakeVideoPredictor([square(), square(), square(box=(0, 0, 5, 5))])
    config = SAM2Config(max_frames_without_box=2)
    backend = SAM2VideoBackend(config, ".", predictor=predictor)
    generated = list(TemporalSAM2Segmenter(backend, config).run_chunk(
        "unused", rows(boxes=[(4, 4, 16, 16), None, None])))
    assert generated[1][0]["segmentation_status"] == "PROPAGATED"
    assert generated[2][0]["segmentation_status"] == "LOW_CONFIDENCE"
    assert generated[2][0]["final_infant_bbox"] is None
    assert predictor.prompts == [0]


def test_prompt_failure_is_recorded_and_recovery_box_preserved():
    predictor = FakeVideoPredictor([square()]*2, fail_prompt=True)
    backend = SAM2VideoBackend(SAM2Config(), ".", predictor=predictor)
    generated = list(TemporalSAM2Segmenter(backend, SAM2Config()).run_chunk("unused", rows(2)))
    assert all(record["segmentation_status"] == "FAILED" and mask is None for record, mask in generated)
    assert all(record["final_infant_bbox"] == [4, 4, 16, 16] for record, _ in generated)


def test_pose_uses_mask_as_optional_spatial_constraint_without_changing_box():
    backend = FakeBackend()
    estimator = Sapiens2PoseEstimator(backend=backend)
    image = np.zeros((100, 100, 3), np.uint8)
    frame = SampledFrame(0, 0., image)
    baseline = estimator.estimate(frame, (0, 0, 100, 100))
    assert baseline.bbox == (0., 0., 100., 100.)
    assert np.all(backend.calls[0][0] == 0)
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    enhanced = estimator.estimate(SampledFrame(1, .05, image), (0, 0, 100, 100), mask=mask)
    assert enhanced.bbox == baseline.bbox
    assert np.array_equal(backend.calls[1][0][0, 0], [104, 116, 124])
    assert np.array_equal(backend.calls[1][0][30, 30], [0, 0, 0])
    assert np.all(image == 0)
    with pytest.raises(ValueError, match="mask"):
        estimator.estimate(SampledFrame(2, .1, image), (0, 0, 100, 100), mask=np.ones((10, 10), bool))


def test_landmark_and_roi_mask_support():
    mask = square()
    points = [dict(index=i, x=10., y=10., confidence=.9) for i in range(308)]
    points[1]["x"] = 1.
    support = landmark_mask_support(mask, points)
    assert len(support) == 308 and support[0] == (0, True) and support[1] == (1, False)
    roi = dict(roi_status="AVAILABLE", polygon="[[4,4],[15,4],[15,15],[4,15]]",
               bbox="[4,4,16,16]", support_keypoint_indices="[0,1]")
    result = roi_mask_support(mask, roi, points)
    assert result["roi_inside_mask_ratio"] == 1.
    assert result["roi_mask_overlap"] == 1.
    assert result["visible_infant_pixel_ratio"] == 1.
    assert result["landmark_inside_mask_ratio"] == .5
    assert result["mask_supported_roi_status"] == "SUPPORTED"
    assert roi_mask_support(None, roi, points)["mask_supported_roi_status"] == "NO_MASK"


def test_confidence_sam2_term_is_configurable_and_falls_back():
    row = dict(roi_status="AVAILABLE", pose_confidence=.8,
               visible_skin_proportion=.8, occlusion_proxy=.2,
               temporal_consistency=.8, motion_magnitude=.2,
               SAM2_mask_support=.5, SAM2_mask_quality=.8)
    baseline = ConfidenceMechanism().score(row)[0]
    enhanced = ConfidenceMechanism(ConfidenceConfig(sam2_enabled=True, sam2_weight=.2))
    assert enhanced.sam2_term(row) == pytest.approx(.4)
    assert enhanced.score(row)[0] == pytest.approx(baseline+.08)
    row["SAM2_mask_support"] = ""
    assert enhanced.score(row)[0] == baseline
    with pytest.raises(ValueError):
        ConfidenceConfig(sam2_enabled=True, sam2_weight=-.1)
