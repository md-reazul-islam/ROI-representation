from pathlib import Path
from module1.config import load_config, project_path
from module1.detection.yolo_detector import YOLODetector


def test_yolo_configuration_is_relative():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/module1.yaml")
    d = cfg["detection"]["yolo"]
    p = project_path(cfg, d["model_path"])
    assert p == root / "models/yolo11s.pt"
    assert d["class_id"] == 0
    assert d["confidence_threshold"] == 0.25


def test_detector_is_lazy():
    d = YOLODetector("models/not_loaded.pt")
    assert d._model is None
    assert d.class_id == 0
