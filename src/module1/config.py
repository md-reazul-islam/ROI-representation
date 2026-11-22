from pathlib import Path
import yaml

def load_config(path):
    path = Path(path).resolve()
    with path.open("r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent.resolve())
    return cfg

def project_path(cfg, configured_path):
    return Path(cfg["_project_root"]) / Path(configured_path)
