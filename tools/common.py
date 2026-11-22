"""Shared helpers for the repository tools: repository root and the dataset paths of config/paths.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/paths.yaml'
KEYS = ('neonatal_videos', 'ubfc_rppg', 'dataaugment_videos', 'derived_videos', 'models')


def paths():
    """Absolute paths of config/paths.yaml (None for an empty entry)."""
    cfg = yaml.safe_load(CONFIG.read_text()) or {}
    out = {}
    for k in KEYS:
        v = (cfg.get(k) or '').strip()
        out[k] = None if not v else (Path(v).expanduser() if Path(v).expanduser().is_absolute() else ROOT/v).resolve()
    return out


def write_paths(**values):
    cfg = yaml.safe_load(CONFIG.read_text()) or {}
    cfg.update({k: str(v) if v else '' for k, v in values.items() if k in KEYS})
    text = CONFIG.read_text().splitlines()
    out = []
    for line in text:                                   # keep the comments, replace the values
        key = line.split(':', 1)[0].strip()
        out.append(f"{key}: '{cfg[key]}'" if key in KEYS and not line.lstrip().startswith('#') else line)
    CONFIG.write_text('\n'.join(out) + '\n')
