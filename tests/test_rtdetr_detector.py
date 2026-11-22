import numpy as np
import pytest
from module1.detection.base import Detection, BaseDetector
from module1.detection.rtdetr_detector import RTDETRDetector


class FakeTensor:
    """Mimics the .detach().cpu().numpy() chain ultralytics result tensors expose."""
    def __init__(self, array): self._array = np.asarray(array)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self._array


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy); self.conf = FakeTensor(conf); self.cls = FakeTensor(cls)
    def __len__(self): return len(self.xyxy._array)


class FakeResult:
    def __init__(self, boxes, names): self.boxes = boxes; self.names = names


def make_detector():
    return RTDETRDetector("models/not_loaded.pt", confidence=0.25, iou=0.45, imgsz=640, device="cpu", class_id=0)


def test_rtdetr_is_a_base_detector():
    assert issubclass(RTDETRDetector, BaseDetector)


def test_extract_candidates_empty_when_no_boxes():
    d = make_detector()
    result = FakeResult(None, {0: "infant_candidate"})
    assert d.extract_candidates(result) == []


def test_extract_candidates_sorted_by_confidence_descending():
    d = make_detector()
    boxes = FakeBoxes(
        xyxy=[[10, 10, 50, 90], [5, 5, 20, 20]],
        conf=[0.3, 0.9],
        cls=[0, 0],
    )
    result = FakeResult(boxes, {0: "infant_candidate"})
    candidates = d.extract_candidates(result)
    assert [c["confidence"] for c in candidates] == [0.9, 0.3]
    assert candidates[0]["class_name"] == "infant_candidate"
    assert candidates[0]["bbox_width"] == 15.0
    assert candidates[0]["bbox_height"] == 15.0


def test_detect_returns_detection_objects_with_frame_index_and_timestamp(monkeypatch):
    d = make_detector()
    boxes = FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.75], cls=[0])
    result = FakeResult(boxes, {0: "infant_candidate"})
    monkeypatch.setattr(d, "predict", lambda frame: result)

    detections = d.detect(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=42, timestamp=2.1)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.bbox == (1.0, 2.0, 3.0, 4.0)
    assert det.confidence == 0.75
    assert det.class_id == 0
    assert det.class_name == "infant_candidate"
    assert det.frame_index == 42
    assert det.timestamp == 2.1


def test_detect_returns_empty_list_when_no_candidates(monkeypatch):
    d = make_detector()
    result = FakeResult(None, {})
    monkeypatch.setattr(d, "predict", lambda frame: result)
    assert d.detect(np.zeros((4, 4, 3), dtype=np.uint8), frame_index=0, timestamp=0.0) == []


def test_load_raises_when_checkpoint_missing(monkeypatch):
    import sys, types
    fake_module = types.ModuleType("ultralytics")
    fake_module.RTDETR = object
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    d = make_detector()
    with pytest.raises(FileNotFoundError):
        d.load()


def test_detect_frame_only_and_prediction_options():
    from types import SimpleNamespace
    detector = make_detector()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    def predict(value, **kwargs):
        assert value is frame
        calls.append(kwargs)
        return [FakeResult(FakeBoxes([[1, 2, 4, 6]], [0.7], [0]), {0: 'person'})]

    detector._model = SimpleNamespace(predict=predict)
    detection = detector.detect(frame)[0]
    assert detection.frame_index == 0
    assert detection.timestamp == 0.0
    assert detection.class_name == 'person'
    assert calls == [dict(conf=0.25, iou=0.45, imgsz=640, classes=[0], verbose=False, device='cpu')]
