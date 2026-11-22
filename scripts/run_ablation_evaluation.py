from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from module1.ablation import evaluate_paths
from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest
from module1.logging_utils import setup_logging


def _configured_path(cfg, value):
    if value is None or str(value).strip() == "":
        return None
    return project_path(cfg, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Phase 11 Module 1 ablation and quantitative evaluation.")
    parser.add_argument("--config", default=str(ROOT / "configs/module1.yaml"))
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--ground-truth")
    parser.add_argument("--fixed-roi")
    parser.add_argument("--quality-path")
    parser.add_argument("--dynamic-path")
    parser.add_argument("--detection-path")
    parser.add_argument("--recovery-path")
    parser.add_argument("--tracking-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--detection-method")
    parser.add_argument("--selection-strategy", choices=("top1", "top2", "topk", "confidence_weighted_fusion"))
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-visualizations", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = cfg.get("ablation", {})
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    errors = manifest.validate_schema()
    if errors:
        raise ValueError(f"Invalid manifest: {errors}")
    logger = setup_logging(project_path(cfg, cfg["logging"]["log_dir"]), cfg["logging"]["level"])
    output = project_path(cfg, args.output_dir or settings.get("output_dir", "reports"))
    values = {
        "quality_path": _configured_path(cfg, args.quality_path or settings.get("quality_path")),
        "dynamic_path": _configured_path(cfg, args.dynamic_path or settings.get("dynamic_path")),
        "fixed_path": _configured_path(cfg, args.fixed_roi or settings.get("fixed_roi_path")),
        "ground_truth_path": _configured_path(cfg, args.ground_truth or settings.get("ground_truth_path")),
        "detection_path": _configured_path(cfg, args.detection_path or settings.get("detection_path")),
        "recovery_path": _configured_path(cfg, args.recovery_path or settings.get("recovery_path")),
        "tracking_path": _configured_path(cfg, args.tracking_path or settings.get("tracking_path")),
    }
    results, statistics, tests, _ = evaluate_paths(
        manifest.df, output, **values,
        top_k=args.top_k or int(settings.get("top_k", 3)),
        selection_strategy=args.selection_strategy or settings.get("selection_strategy", "topk"),
        detection_method=args.detection_method or settings.get("detection_method", "or"),
        save_visualizations=not args.no_visualizations and bool(settings.get("save_visualizations", True)),
    )
    logger.info("Ablation complete: %d result rows, %d statistic rows, %d test rows", len(results), len(statistics), len(tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
