"""Compare the original and Phase 12 pipelines using identical upstream frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from module1.config import load_config, project_path
from module1.data.manifest import sha256_file
from module1.segmentation.ablation import compare_phase12


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired descriptive SAM 2 ablation.")
    parser.add_argument("--config", default=str(ROOT/"configs/module1.yaml"))
    parser.add_argument("--recovery", default="outputs/recovery/recovery_results.csv")
    parser.add_argument("--sam2-dir", default="outputs/sam2")
    for name in ("pose", "roi", "quality", "confidence"):
        parser.add_argument(f"--baseline-{name}-dir", default=f"outputs/{'roi_quality' if name == 'quality' else name}")
        parser.add_argument(f"--sam2-{name}-dir", default=f"outputs/{'roi_quality' if name == 'quality' else name}_sam2")
    parser.add_argument("--output-dir", default="reports/sam2_ablation")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    output = project_path(cfg, args.output_dir)
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError("SAM 2 ablation output directory is not empty")
    baseline = {name: project_path(cfg, getattr(args, f"baseline_{name}_dir"))
                for name in ("pose", "roi", "quality", "confidence")}
    enhanced = {name: project_path(cfg, getattr(args, f"sam2_{name}_dir"))
                for name in ("pose", "roi", "quality", "confidence")}
    recovery = project_path(cfg, args.recovery)
    sam2_dir = project_path(cfg, args.sam2_dir)
    frames, summary = compare_phase12(recovery, sam2_dir, baseline, enhanced)
    output.mkdir(parents=True)
    frames.to_csv(output/"sam2_ablation_frames.csv", index=False)
    summary.to_csv(output/"sam2_ablation_summary.csv", index=False)
    metadata = dict(phase="PHASE_12_SAM2_ABLATION", run_status="COMPLETE",
                    comparisons="same Recovery frame, split, timestamp and common upstream input hashes",
                    source_recovery_sha256=sha256_file(recovery),
                    source_sam2_sha256=sha256_file(sam2_dir/"sam2_masks.csv"),
                    baseline_directories={key: str(value) for key, value in baseline.items()},
                    sam2_directories={key: str(value) for key, value in enhanced.items()},
                    frame_count=len(frames),
                    interpretation="paired descriptive metrics only; differences do not establish improvement",
                    ground_truth_status="UNAVAILABLE", segmentation_accuracy="NOT_AVAILABLE",
                    timestamp_utc=datetime.now(timezone.utc).isoformat())
    (output/"experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
