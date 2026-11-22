from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from module1.config import load_config, project_path
from module1.logging_utils import setup_logging
from module1.data.manifest import DatasetManifest
from module1.data.validator import DatasetValidator
from module1.data.split import SubjectLeakageChecker, VideoSplitPolicy, VideoLeakageChecker


def main():
    cfg = load_config(ROOT / "configs/module1.yaml")
    logger = setup_logging(
        project_path(cfg, cfg["logging"]["log_dir"]),
        cfg["logging"]["level"],
    )

    authoritative = ROOT / "dataset_manifest.csv"
    if not authoritative.exists():
        raise FileNotFoundError("dataset_manifest.csv not found at project root.")

    manifest = DatasetManifest.load(authoritative)
    logger.info("Loaded authoritative manifest: %s", authoritative)
    logger.info("Manifest records: %d", len(manifest.df))

    # The supplied pilot ZIP contains exactly 24 physical videos named 1.mp4
    # through 24.mp4. The supplied manifest contains the original metadata and
    # SHA-256 values. SHA-256 resolution therefore links the physical files to
    # their authoritative manifest records without inventing identities.
    df = manifest.df.copy()
    resolved_paths = {}
    for _, row in df.iterrows():
        video_id = str(row["video_id"])
        resolved = manifest.resolve_video_path(ROOT, row)
        if resolved.exists():
            resolved_paths[video_id] = resolved

    # Official pilot split. The physical video order is the experimental unit.
    # Every frame extracted from a video inherits this split.
    if len(resolved_paths) == 24:
        policy = VideoSplitPolicy(train_end=18, val_end=21, total_videos=24)
        df = policy.apply(df, resolved_paths)
        logger.info("Applied pilot split: videos 1-18 train; 19-21 validation; 22-24 test")
    else:
        logger.warning(
            "Pilot split was not applied because %d/24 physical videos resolved.",
            len(resolved_paths),
        )

    working_manifest = DatasetManifest(df, authoritative)
    validator = DatasetValidator(working_manifest, ROOT)
    report = validator.run()

    out = project_path(cfg, cfg["outputs"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    validator.write_report(report, out / "dataset_validation_report.json")

    path_col = working_manifest.relative_path_column()
    df["project_relative_video_path"] = df[path_col].astype(str).map(
        lambda x: str(Path("data") / Path(x)).replace("\\", "/")
    )
    df["native_fps_preserved"] = True
    df["processing_fps"] = float(cfg["processing"]["target_fps"])
    df.to_csv(out / "resolved_manifest.csv", index=False)

    # Leakage audits are run on the FINAL resolved manifest after the fixed
    # pilot split is applied.
    video_leakage = VideoLeakageChecker.check(df)
    subject_leakage = SubjectLeakageChecker.check(df)
    leakage = {
        "video_level": video_leakage,
        "subject_level": subject_leakage,
        "frame_level_policy": "all_frames_inherit_source_video_split",
    }
    (ROOT / "logs" / "subject_leakage_check.json").write_text(
        json.dumps(leakage, indent=2), encoding="utf-8"
    )

    fps_col = "fps" if "fps" in df.columns else "input_fps"
    physical_video_count = len([
        p for p in (ROOT / "data" / "Dataset").rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ])
    split_counts = df["split"].astype(str).str.lower().value_counts().to_dict()
    stats = {
        "manifest_video_count": int(len(df)),
        "physical_video_count": int(physical_video_count),
        "video_count": int(len(df)),
        "processing_fps": float(cfg["processing"]["target_fps"]),
        "native_fps_preserved": True,
        "native_fps_values": sorted(
            pd.to_numeric(df[fps_col], errors="coerce").dropna().unique().tolist()
        ),
        "resolution_values": sorted(
            {f"{int(w)}x{int(h)}" for w, h in zip(df["width"], df["height"])}
        ),
        "split_counts": split_counts,
        "split_policy": "physical_video_order_1_18_train_19_21_validation_22_24_test",
        "split_unit": "video",
        "video_level_leakage_status": video_leakage["status"],
        "subject_level_leakage_status": subject_leakage["status"],
        "missing_infant_id_count": int(
            df["infant_id"].astype(str).str.upper().isin(
                ["", "NAN", "NONE", "UNKNOWN", "UNAVAILABLE"]
            ).sum()
        ),
        "missing_session_id_count": int(
            df["session_id"].astype(str).str.upper().isin(
                ["", "NAN", "NONE", "UNKNOWN", "UNAVAILABLE"]
            ).sum()
        ),
        "validation_status": report.status,
        "missing_file_count": len(report.missing_files),
        "unreadable_video_count": len(report.unreadable_videos),
        "metadata_mismatch_count": len(report.metadata_mismatches),
    }
    pd.DataFrame([stats]).to_csv(out / "dataset_statistics.csv", index=False)

    logger.info("Validation status: %s", report.status)
    logger.info("Videos in authoritative manifest: %d", len(df))
    logger.info("Final split counts: %s", split_counts)
    logger.info("Video-level leakage audit: %s", video_leakage["status"])
    logger.info("Subject-level leakage audit: %s", subject_leakage["status"])
    logger.info("Subject-independent evaluation: %s", report.subject_independent_evaluation)
    logger.info("Reports written under %s", out)

    if report.status == "FAIL":
        logger.error("PHASE 1 validation failed. See dataset_validation_report.json.")
        raise SystemExit(1)
    if video_leakage["status"] != "PASS":
        logger.error("PHASE 1 stopped because video-level leakage audit failed.")
        raise SystemExit(1)

    logger.info("PHASE 1 completed successfully.")


if __name__ == "__main__":
    main()
