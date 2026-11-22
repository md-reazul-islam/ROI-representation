"""Phase 12: SAM 2 temporal masks prompted only from Phase 6 infant boxes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.logging_utils import setup_logging
from module1.pose.inputs import load_recovery_frames
from module1.segmentation import SAM2Config, SAM2VideoBackend, TemporalSAM2Segmenter


def validate_inputs(cfg, manifest_path, recovery_path):
    manifest = DatasetManifest.load(manifest_path)
    frames = load_recovery_frames(recovery_path)
    if manifest.df.video_id.duplicated().any() or set(manifest.df.video_id) != {v for v, _ in frames}:
        raise ValueError("SAM 2 recovery/manifest video set mismatch")
    videos = []
    for _, record in manifest.df.iterrows():
        video = str(record.video_id)
        path = manifest.resolve_video_path(cfg["_project_root"], record)
        reader = VideoReader(path)
        meta = reader.metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f"Unreadable video: {video}")
        group = {index: value for (name, index), value in frames.items() if name == video}
        if any(row["split"] != record.split for row in group.values()):
            raise ValueError("SAM 2 recovery/manifest split mismatch")
        expected = FrameSampler(reader, float(cfg["processing"]["target_fps"])).selected_indices(meta)
        if set(expected) != set(group):
            raise ValueError("SAM 2 recovery frames differ from the original sampling timeline")
        if any(abs(row["timestamp_sec"]-index/meta.native_fps) > 1e-8 for index, row in group.items()):
            raise ValueError("SAM 2 recovery timestamps differ from the original timeline")
        for row in group.values():
            box = row["bbox"]
            if box is not None and (box[2] <= 0 or box[3] <= 0 or box[0] >= meta.width or box[1] >= meta.height):
                raise ValueError("Final infant bbox does not intersect original frame")
        videos.append((video, path, meta, group))
    return frames, videos


def annotate(image, row, mask):
    canvas = image.copy()
    if mask is not None:
        color = np.zeros_like(canvas)
        color[:, :, 1] = 210
        canvas[mask] = cv2.addWeighted(canvas, 0.55, color, 0.45, 0)[mask]
    bbox = row.get("final_infant_bbox")
    if bbox:
        values = json.loads(bbox) if isinstance(bbox, str) else bbox
        cv2.rectangle(canvas, tuple(round(v) for v in values[:2]),
                      tuple(round(v) for v in values[2:]), (255, 200, 20), 2)
    text = f"{row['video_id']} frame={row['frame_index']} SAM2={row['segmentation_status']}"
    cv2.putText(canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    return canvas


def metrics_table(rows):
    table = pd.DataFrame(rows)
    groups = [(str(video), split, group) for (video, split), group in table.groupby(["video_id", "split"], sort=False)]
    groups.append(("ALL", "all", table))
    result = []
    for video, split, group in groups:
        status = group["segmentation_status"]
        reliable = status.isin(("INITIALIZED", "PROPAGATED", "REINITIALIZED"))
        result.append(dict(video_id=video, split=split, total_frames=len(group),
                           reliable_mask_frames=int(reliable.sum()),
                           propagated_frames=int((status == "PROPAGATED").sum()),
                           low_confidence_frames=int((status == "LOW_CONFIDENCE").sum()),
                           failed_frames=int((status == "FAILED").sum()),
                           disabled_frames=int((status == "DISABLED").sum()),
                           mask_observation_rate=float(reliable.mean()),
                           mean_mask_quality=pd.to_numeric(group["mask_quality"], errors="coerce").mean(),
                           mean_temporal_consistency=pd.to_numeric(group["temporal_consistency"], errors="coerce").mean(),
                           mean_detector_mask_consistency=pd.to_numeric(group["detector_mask_consistency"], errors="coerce").mean(),
                           ground_truth_segmentation_accuracy="NOT_AVAILABLE"))
    return pd.DataFrame(result)


def main(argv=None, *, backend_factory=SAM2VideoBackend):
    parser = argparse.ArgumentParser(description="Box-prompted SAM 2 after Temporal Recovery.")
    parser.add_argument("--config", default=str(ROOT / "configs/module1.yaml"))
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--recovery")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-model", action="store_true",
                        help="Fail if SAM 2 cannot load; with --validate-only, check the model and GPU without writing outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--enable", action="store_true")
    mode.add_argument("--disable", action="store_true")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    cfg = load_config(args.config)
    settings = cfg.get("sam2", {})
    sam2_config = SAM2Config.from_mapping(settings)
    overrides = {}
    if args.enable or args.disable:
        overrides["enabled"] = args.enable
    for key in ("checkpoint", "model_config", "device"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if overrides:
        sam2_config = replace(sam2_config, **overrides)
    if args.require_model and not sam2_config.enabled:
        raise ValueError("--require-model needs SAM 2 enabled (use --enable)")
    root = Path(cfg["_project_root"])
    logger = setup_logging(project_path(cfg, cfg["logging"]["log_dir"]), cfg["logging"]["level"])
    manifest_path = project_path(cfg, args.manifest)
    recovery_path = project_path(cfg, args.recovery or settings.get("recovery_path", "outputs/recovery/recovery_results.csv"))
    hashes = {"manifest": sha256_file(manifest_path), "recovery": sha256_file(recovery_path)}
    frames, videos = validate_inputs(cfg, manifest_path, recovery_path)
    logger.info("SAM 2 input validation PASS: %d videos, %d original sampled frames", len(videos), len(frames))
    if args.validate_only and not args.require_model:
        return 0
    output = project_path(cfg, args.output_dir or settings.get("output_dir", "outputs/sam2"))
    if not args.validate_only and output.exists() and any(output.rglob("*")):
        raise FileExistsError("SAM 2 output directory is not empty; use --output-dir for a new run")
    backend = None
    backend_error = None
    if sam2_config.enabled:
        try:
            backend = backend_factory(sam2_config, root)
            logger.info("SAM 2 loaded: %s on %s", sam2_config.model_variant, backend.device)
        except Exception as exc:
            backend_error = f"{type(exc).__name__}: {exc}"
            if args.require_model:
                raise RuntimeError(f"SAM 2 model is required but unavailable: {backend_error}") from exc
            logger.error("SAM 2 unavailable; recording FAILED masks and preserving Phase 6 boxes: %s", backend_error)
    if args.validate_only:
        logger.info("SAM 2 model and device preflight PASS")
        return 0
    for folder in ("masks", "visualizations", "metadata"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    try:
        sam2_package_version = version("SAM-2")
    except PackageNotFoundError:
        sam2_package_version = "UNAVAILABLE"
    checkpoint_path = root / sam2_config.checkpoint
    metadata = dict(phase="PHASE_12_SAM2_TEMPORAL_SEGMENTATION", run_status="RUNNING",
                    mode="SAM2_ENABLED" if sam2_config.enabled else "SAM2_DISABLED",
                    configuration=asdict(sam2_config), total_input_frames=len(frames),
                    requested_frame_limit=args.max_frames, model_load_error=backend_error,
                    model_internal_image_size=getattr(backend, "internal_image_size", None),
                    sam2_package_version=sam2_package_version,
                    checkpoint_sha256=sha256_file(checkpoint_path) if sam2_config.enabled and checkpoint_path.is_file() else None,
                    device=getattr(backend, "device", None),
                    source_paths={"manifest": str(manifest_path), "recovery": str(recovery_path)},
                    source_hashes=hashes,
                    input_frame_encoding="temporary JPEG at configured quality, original dimensions, numeric sampled order",
                    mask_coordinate_system="original_video_frame_xy_pixels; xyxy mask bbox has exclusive x2/y2",
                    mask_transform="SAM 2 video logits are returned at video resolution; any mismatch is bilinearly resized before thresholding and recorded per frame",
                    mask_quality_definition="mean of area plausibility, mask-inside-recovery-box ratio, temporal mask IoU and tolerance boundary F1 when available; heuristic, not calibrated",
                    detector_mask_consistency_definition="fraction of mask pixels inside the original Temporal Recovery bbox",
                    ground_truth_status="UNAVAILABLE", segmentation_accuracy="NOT_AVAILABLE",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version())
    metadata_path = output / "metadata" / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    visualizations = 0
    limit_reached = False

    def publish(chunk, video, number):
        nonlocal visualizations
        if not chunk:
            return
        if not sam2_config.enabled or backend is None:
            generated = []
            for row, image in chunk:
                record = TemporalSAM2Segmenter._base(row)
                record["original_height"], record["original_width"] = image.shape[:2]
                record["segmentation_status"] = "DISABLED" if not sam2_config.enabled else "FAILED"
                record["failure_reason"] = backend_error
                generated.append((record, None))
        else:
            try:
                with TemporaryDirectory(prefix="sam2_frames_", dir=output) as temporary:
                    frame_dir = Path(temporary)
                    for local_index, (_, image) in enumerate(chunk):
                        target = frame_dir / f"{local_index:06d}.jpg"
                        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, sam2_config.jpeg_quality]):
                            raise OSError(f"Cannot write temporary SAM 2 frame: {target}")
                    generated = list(TemporalSAM2Segmenter(backend, sam2_config).run_chunk(
                        frame_dir, chunk, restarted=number > 0))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.error("%s chunk %d failed; using Phase 6 boxes: %s", video, number, error)
                generated = []
                for row, image in chunk:
                    record = TemporalSAM2Segmenter._base(row)
                    record["original_height"], record["original_width"] = image.shape[:2]
                    record["failure_reason"] = error
                    generated.append((record, None))
        if len(generated) != len(chunk):
            raise ValueError("SAM 2 did not account for every sampled frame")
        for (record, mask), (_, image) in zip(generated, chunk):
            name = f"{record['video_id']}_frame_{record['frame_index']:08d}"
            if mask is not None:
                target = output / "masks" / f"{name}.png"
                if not cv2.imwrite(str(target), mask.astype(np.uint8)*255):
                    raise OSError(f"Cannot write SAM 2 mask: {target}")
                record["mask_path"] = f"masks/{name}.png"
            rows.append(record)
            if not args.no_visualizations and sam2_config.save_annotated_frames:
                target = output / "visualizations" / f"{name}.jpg"
                if not cv2.imwrite(str(target), annotate(image, record, mask)):
                    raise OSError(f"Cannot write SAM 2 visualization: {target}")
                visualizations += 1

    try:
        for video, path, meta, group in videos:
            reader = VideoReader(path)
            chunk = []
            number = 0
            seen = set()
            for sampled in FrameSampler(reader, float(cfg["processing"]["target_fps"])).iter_sampled():
                row = group.get(sampled.frame_index)
                if row is None or abs(row["timestamp_sec"]-sampled.timestamp_sec) > 1e-8:
                    raise ValueError("SAM 2 decoded frame differs from validated recovery timeline")
                if sampled.frame.shape[:2] != (meta.height, meta.width):
                    raise ValueError("SAM 2 decoded frame changed original dimensions")
                seen.add(sampled.frame_index)
                chunk.append((row, sampled.frame))
                if len(chunk) >= sam2_config.chunk_frames:
                    publish(chunk, video, number)
                    number += 1
                    chunk = []
                if args.max_frames is not None and len(rows)+len(chunk) >= args.max_frames:
                    limit_reached = True
                    break
            publish(chunk, video, number)
            if not limit_reached and seen != set(group):
                raise ValueError("SAM 2 video decode ended before all sampled frames")
            logger.info("%s: %d SAM 2 frame records", video, len(seen))
            if limit_reached:
                break
        if hashes != {"manifest": sha256_file(manifest_path), "recovery": sha256_file(recovery_path)}:
            raise ValueError("SAM 2 inputs changed during the run")
        pd.DataFrame(rows).to_csv(output / "sam2_masks.csv", index=False)
        metrics_table(rows).to_csv(output / "sam2_metrics.csv", index=False)
        counts = dict(Counter(row["segmentation_status"] for row in rows))
        metadata.update(run_status=("PARTIAL" if len(rows) < len(frames) else
                                    "DEGRADED" if backend_error or counts.get("FAILED") else "COMPLETE"),
                        processed_frames=len(rows), visualization_count=visualizations,
                        status_counts=counts, chunk_frames=sam2_config.chunk_frames)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        metadata.update(run_status="FAILED", processed_frames=len(rows), error=str(exc))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    logger.info("SAM 2 run %s: %d frames", metadata["run_status"], len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
