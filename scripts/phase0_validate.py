from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from module1.config import load_config, project_path
from module1.data.manifest import build_manifest, validate_manifest
from module1.logging_utils import setup_logging

def main():
    cfg = load_config(ROOT / "configs" / "module1.yaml")
    logger = setup_logging(ROOT / "logs", cfg["logging"]["level"])

    video_dir = project_path(cfg, cfg["dataset"]["video_dir"]).resolve()
    logger.info("Dataset directory: %s", video_dir)
    logger.info("Dataset source: %s", cfg["dataset"]["source_name"])
    logger.info("Target processing FPS: %s", cfg["processing"]["target_fps"])

    df = build_manifest(video_dir, cfg["dataset"]["source_name"], cfg["dataset"]["extensions"])
    errors = validate_manifest(df)
    if errors:
        for e in errors:
            logger.error(e)
        raise SystemExit(1)

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    manifest_path = out / "dataset_manifest.csv"
    df.to_csv(manifest_path, index=False)

    metadata = {
        "dataset_source": cfg["dataset"]["source_name"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "video_count": int(len(df)),
        "processing_fps": cfg["processing"]["target_fps"],
        "native_fps_preserved": cfg["processing"]["preserve_native_fps"],
        "seed": cfg["project"]["seed"],
        "device": cfg["experiment"]["device"],
        "confidence_thresholds": cfg["experiment"]["confidence_thresholds"],
        "videos": df[["video_id","filename","input_fps","width","height","frame_count","duration_sec","sha256"]].to_dict("records"),
    }
    (ROOT / "logs" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Videos discovered: %d", len(df))
    logger.info("Manifest written: %s", manifest_path)
    logger.info("Phase 0 validation completed successfully.")

if __name__ == "__main__":
    main()
