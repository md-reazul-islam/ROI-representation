import numpy as np
from module1.detection.base import Detection, BaseDetector
from module1.detection.rtdetr_detector import RTDETRDetector
from module1.detection.yolo_adapter import YOLODetectorAdapter
from module1.detection.yolo_detector import YOLODetector


def test_detection_to_dict_round_trip():
    det = Detection(bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.9, class_id=0,
                     class_name="infant_candidate", frame_index=7, timestamp=0.35)
    d = det.to_dict()
    assert d["x1"] == 1.0 and d["y2"] == 4.0
    assert d["confidence"] == 0.9
    assert d["frame_index"] == 7
    assert d["timestamp"] == 0.35


def test_both_detectors_are_base_detectors():
    assert issubclass(RTDETRDetector, BaseDetector)
    assert issubclass(YOLODetectorAdapter, BaseDetector)


def test_yolo_adapter_does_not_replace_yolo_detector():
    # YOLODetectorAdapter wraps YOLODetector via composition; the original
    # class remains untouched and independently usable.
    adapter = YOLODetectorAdapter("models/not_loaded.pt")
    assert isinstance(adapter.inner, YOLODetector)


def test_yolo_adapter_detect_conforms_to_shared_interface(monkeypatch):
    adapter = YOLODetectorAdapter("models/not_loaded.pt", class_id=0)
    fake_candidates = [{"class_id": 0, "class_name": "infant_candidate", "confidence": 0.5,
                         "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]
    monkeypatch.setattr(adapter.inner, "predict", lambda frame: object())
    monkeypatch.setattr(adapter.inner, "extract_candidates", lambda result: fake_candidates)

    detections = adapter.detect(np.zeros((5, 5, 3), dtype=np.uint8), frame_index=3, timestamp=0.15)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.bbox == (0.0, 0.0, 10.0, 10.0)
    assert det.frame_index == 3
    assert det.timestamp == 0.15


def test_rtdetr_and_yolo_adapter_return_same_detection_shape(monkeypatch):
    rtdetr = RTDETRDetector("models/not_loaded.pt")
    yolo = YOLODetectorAdapter("models/not_loaded.pt")
    fake_candidates = [{"class_id": 0, "class_name": "infant_candidate", "confidence": 0.8,
                         "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0}]
    monkeypatch.setattr(rtdetr, "predict", lambda frame: object())
    monkeypatch.setattr(rtdetr, "extract_candidates", lambda result: fake_candidates)
    monkeypatch.setattr(yolo.inner, "predict", lambda frame: object())
    monkeypatch.setattr(yolo.inner, "extract_candidates", lambda result: fake_candidates)

    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    rtdetr_detections = rtdetr.detect(frame, frame_index=1, timestamp=0.05)
    yolo_detections = yolo.detect(frame, frame_index=1, timestamp=0.05)

    assert [d.to_dict() for d in rtdetr_detections] == [d.to_dict() for d in yolo_detections]
