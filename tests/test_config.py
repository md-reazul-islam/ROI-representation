from pathlib import Path
from module1.config import load_config

def test_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "module1.yaml")
    assert cfg["processing"]["target_fps"] == 20.0
    assert cfg["processing"]["preserve_native_fps"] is True
