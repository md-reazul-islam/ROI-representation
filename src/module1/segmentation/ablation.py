"""Paired Phase 12 descriptive comparison on identical Recovery frames."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from module1.data.manifest import sha256_file
from module1.pose.inputs import load_recovery_frames
from .inputs import load_sam2_table


def _table(path, key_columns):
    table = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if table.empty or not set((*key_columns, "split", "timestamp_sec")).issubset(table.columns):
        raise ValueError(f"Ablation input is empty or missing columns: {path}")
    keys = {}
    for row in table.to_dict("records"):
        for column in key_columns:
            if column == "frame_index":
                number = float(row[column])
                if not math.isfinite(number) or number < 0 or not number.is_integer():
                    raise ValueError(f"Invalid frame index in {path}")
        key = tuple(str(row[column]) if column in ("video_id", "roi_name") else int(float(row[column]))
                    for column in key_columns)
        if key in keys:
            raise ValueError(f"Duplicate ablation record in {path}")
        if not math.isfinite(float(row["timestamp_sec"])):
            raise ValueError(f"Invalid ablation timestamp in {path}")
        keys[key] = row
    return keys


def _aligned(a, b, label):
    if set(a) != set(b):
        raise ValueError(f"{label} baseline and SAM 2 frame sets differ")
    for key in a:
        if (a[key]["split"] != b[key]["split"] or
                abs(float(a[key]["timestamp_sec"])-float(b[key]["timestamp_sec"])) > 1e-8):
            raise ValueError(f"{label} baseline and SAM 2 timelines differ")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _verify_lineage(directories):
    links = (
        ("roi", "pose_results", "pose", "pose_results.csv"),
        ("quality", "roi_results", "roi", "candidate_rois.csv"),
        ("confidence", "quality_features", "quality", "roi_quality_features.csv"),
    )
    for downstream, source_name, upstream, filename in links:
        meta_path = Path(directories[downstream])/"experiment_metadata.json"
        if not meta_path.is_file():
            raise ValueError(f"Missing {downstream} provenance metadata")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = metadata.get("source_hashes", {}).get(source_name, {}).get("sha256")
        actual = sha256_file(Path(directories[upstream])/filename)
        if metadata.get("run_status") != "COMPLETE" or expected != actual:
            raise ValueError(f"{downstream} output does not match its {upstream} source")


def compare_phase12(recovery_path, sam2_dir, baseline_dirs, enhanced_dirs):
    recovery = load_recovery_frames(recovery_path)
    sam2, sam2_meta = load_sam2_table(sam2_dir)
    if set(recovery) != set(sam2):
        raise ValueError("SAM 2 and Recovery frame sets differ")
    for key in recovery:
        if (recovery[key]["split"] != sam2[key]["split"] or
                abs(recovery[key]["timestamp_sec"]-sam2[key]["timestamp_sec"]) > 1e-8):
            raise ValueError("SAM 2 and Recovery timelines differ")
    if (sam2_meta.get("run_status") not in ("COMPLETE", "DEGRADED")
            or sam2_meta.get("mode") != "SAM2_ENABLED"
            or sam2_meta.get("source_hashes", {}).get("recovery") != sha256_file(recovery_path)):
        raise ValueError("SAM 2 ablation requires a full segmentation run")
    base_meta = json.loads((Path(baseline_dirs["pose"])/"experiment_metadata.json").read_text(encoding="utf-8"))
    enhanced_meta = json.loads((Path(enhanced_dirs["pose"])/"experiment_metadata.json").read_text(encoding="utf-8"))
    if (base_meta.get("run_status") != "COMPLETE" or enhanced_meta.get("run_status") != "COMPLETE"
            or base_meta.get("sam2_mode", "DIRECT_RECOVERY_BOX") != "DIRECT_RECOVERY_BOX"
            or enhanced_meta.get("sam2_mode") != "MASK_CONSTRAINED"):
        raise ValueError("Ablation requires complete direct and mask-constrained pose runs")
    common = set(base_meta.get("input_hashes", {})) & set(enhanced_meta.get("input_hashes", {}))
    if not common or any(base_meta["input_hashes"][key] != enhanced_meta["input_hashes"][key] for key in common):
        raise ValueError("Pose runs did not use identical common upstream inputs")
    sam2_csv = (Path(sam2_dir)/"sam2_masks.csv").resolve()
    if not any(Path(path).resolve() == sam2_csv and digest == sha256_file(sam2_csv)
               for path, digest in enhanced_meta.get("input_hashes", {}).items()):
        raise ValueError("Enhanced pose run does not reference the supplied SAM 2 masks")
    _verify_lineage(baseline_dirs)
    _verify_lineage(enhanced_dirs)
    files = {"pose": ("pose_results.csv", ("video_id", "frame_index")),
             "roi": ("candidate_rois.csv", ("video_id", "frame_index", "roi_name")),
             "quality": ("roi_quality_features.csv", ("video_id", "frame_index", "roi_name")),
             "confidence": ("roi_confidence.csv", ("video_id", "frame_index", "roi_name"))}
    base, enhanced = {}, {}
    for name, (filename, keys) in files.items():
        base[name] = _table(Path(baseline_dirs[name])/filename, keys)
        enhanced[name] = _table(Path(enhanced_dirs[name])/filename, keys)
        _aligned(base[name], enhanced[name], name)
        if {key[:2] for key in base[name]} != set(recovery):
            raise ValueError(f"{name} frames differ from Temporal Recovery")
    grouped = {"baseline": {}, "sam2": {}}
    for label, source in (("baseline", base), ("sam2", enhanced)):
        for name in ("roi", "quality", "confidence"):
            by_frame = {}
            for entry_key, entry in source[name].items():
                by_frame.setdefault(entry_key[:2], []).append(entry)
            grouped[label][name] = by_frame
    per_frame = []
    for key in sorted(recovery):
        video, frame = key
        base_roi = grouped["baseline"]["roi"][key]
        enhanced_roi = grouped["sam2"]["roi"][key]
        if len(base_roi) != 17 or len(enhanced_roi) != 17:
            raise ValueError("Ablation requires all 17 candidate ROIs per frame")
        row = dict(video_id=video, frame_index=frame, split=recovery[key]["split"],
                   timestamp_sec=recovery[key]["timestamp_sec"],
                   sam2_status=sam2[key]["segmentation_status"],
                   sam2_mask_quality=_number(sam2[key].get("mask_quality")),
                   sam2_temporal_consistency=_number(sam2[key].get("temporal_consistency")),
                   sam2_detector_mask_consistency=_number(sam2[key].get("detector_mask_consistency")))
        for label, source in (("baseline", base), ("sam2", enhanced)):
            pose = source["pose"][key]
            row[f"{label}_pose_confidence"] = _number(pose.get("confidence"))
            row[f"{label}_pose_quality"] = _number(pose.get("pose_quality"))
            rois = grouped[label]["roi"][key]
            row[f"{label}_valid_roi_rate"] = sum(entry["roi_status"] == "AVAILABLE" for entry in rois)/17
            for table_name, column, metric in (("quality", "temporal_consistency", "roi_temporal_stability"),
                                                ("confidence", "q", "roi_confidence"),
                                                ("roi", "SAM2_mask_support", "roi_mask_support")):
                values = [_number(entry.get(column)) for entry in grouped[label][table_name][key]]
                values = [value for value in values if value is not None]
                row[f"{label}_{metric}"] = float(np.mean(values)) if values else None
        selected_a = {value["roi_name"] for value in grouped["baseline"]["confidence"][key]
                      if str(value.get("selected")).lower() == "true"}
        selected_b = {value["roi_name"] for value in grouped["sam2"]["confidence"][key]
                      if str(value.get("selected")).lower() == "true"}
        row["selected_roi_jaccard"] = (len(selected_a & selected_b)/len(selected_a | selected_b)
                                       if selected_a or selected_b else None)
        per_frame.append(row)
    frame_table = pd.DataFrame(per_frame)
    paired = []
    for video, group in [*frame_table.groupby("video_id", sort=False), ("ALL", frame_table)]:
        for metric in ("pose_confidence", "pose_quality", "valid_roi_rate",
                       "roi_temporal_stability", "roi_confidence"):
            a = pd.to_numeric(group[f"baseline_{metric}"], errors="coerce")
            b = pd.to_numeric(group[f"sam2_{metric}"], errors="coerce")
            valid = a.notna() & b.notna()
            paired.append(dict(video_id=video, metric=metric, n_paired=int(valid.sum()),
                               baseline_mean=float(a[valid].mean()) if valid.any() else None,
                               sam2_mean=float(b[valid].mean()) if valid.any() else None,
                               paired_difference_mean=float((b[valid]-a[valid]).mean()) if valid.any() else None,
                               status="DESCRIPTIVE" if valid.any() else "NOT_AVAILABLE",
                               ground_truth_status="NOT_AVAILABLE"))
        for metric in ("sam2_mask_quality", "sam2_temporal_consistency",
                       "sam2_detector_mask_consistency", "sam2_roi_mask_support",
                       "selected_roi_jaccard"):
            values = pd.to_numeric(group[metric], errors="coerce")
            paired.append(dict(video_id=video, metric=metric, n_paired=int(values.notna().sum()),
                               baseline_mean=None, sam2_mean=float(values.mean()) if values.notna().any() else None,
                               paired_difference_mean=None,
                               status="DESCRIPTIVE" if values.notna().any() else "NOT_AVAILABLE",
                               ground_truth_status="NOT_AVAILABLE"))
    return frame_table, pd.DataFrame(paired)
