from pathlib import Path
from module1.config import load_config, project_path
from module1.detection.rtdetr_detector import RTDETRDetector


def test_rtdetr_configuration_is_relative():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/module1.yaml")
    d = cfg["detection"]["rtdetr"]
    p = project_path(cfg, d["model_path"])
    assert p == root / "models/rtdetr-l.pt"
    assert d["class_id"] == 0
    assert d["confidence_threshold"] == 0.25


def test_rtdetr_and_yolo_configs_are_independent():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/module1.yaml")
    assert cfg["detection"]["yolo"]["model_path"] != cfg["detection"]["rtdetr"]["model_path"]


def test_detector_is_lazy():
    d = RTDETRDetector("models/not_loaded.pt")
    assert d._model is None
    assert d.class_id == 0
