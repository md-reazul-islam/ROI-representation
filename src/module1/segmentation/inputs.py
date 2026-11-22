"""Validated Phase 12 records and original-resolution mask loading."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .sam2 import RELIABLE_STATES, STATES


def load_sam2_table(sam2_dir):
    root = Path(sam2_dir)
    csv_path = root / "sam2_masks.csv"
    metadata_path = root / "metadata" / "experiment_metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("SAM 2 directory requires sam2_masks.csv and metadata/experiment_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    table = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    required = {"video_id", "split", "frame_index", "timestamp_sec", "track_id",
                "segmentation_status", "mask_path", "mask_quality", "original_height",
                "original_width", "final_infant_bbox"}
    if table.empty or not required.issubset(table.columns):
        raise ValueError("SAM 2 CSV is empty or missing required columns")
    records = {}
    for row in table.to_dict("records"):
        video = str(row["video_id"])
        index = int(float(row["frame_index"]))
        timestamp = float(row["timestamp_sec"])
        if (float(row["frame_index"]) != index or index < 0 or not math.isfinite(timestamp)
                or timestamp < 0 or row["segmentation_status"] not in STATES
                or row["split"] not in ("train", "validation", "test")):
            raise ValueError("Invalid SAM 2 frame metadata")
        key = video, index
        if key in records:
            raise ValueError("Duplicate SAM 2 frame")
        if row["mask_path"]:
            mask_path = root / str(row["mask_path"])
            if root.resolve() not in mask_path.resolve().parents or not mask_path.is_file():
                raise ValueError("SAM 2 mask path escapes output directory or is missing")
        elif row["segmentation_status"] in RELIABLE_STATES:
            raise ValueError("Reliable SAM 2 record has no mask")
        quality = row["mask_quality"]
        if row["segmentation_status"] in RELIABLE_STATES and quality == "":
            raise ValueError("Reliable SAM 2 record has no mask quality")
        if quality != "" and (not math.isfinite(float(quality)) or not 0 <= float(quality) <= 1):
            raise ValueError("Invalid SAM 2 mask quality")
        row["frame_index"] = index
        row["timestamp_sec"] = timestamp
        records[key] = row
    return records, metadata


def load_mask(sam2_dir, row, *, expected_shape=None, reliable_only=True):
    if reliable_only and row["segmentation_status"] not in RELIABLE_STATES:
        return None
    relative = row.get("mask_path")
    if not relative:
        return None
    root = Path(sam2_dir).resolve()
    path = (root / str(relative)).resolve()
    if root not in path.parents:
        raise ValueError("SAM 2 mask path escapes output directory")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or not np.isin(image, (0, 255)).all():
        raise ValueError("SAM 2 mask is missing or not binary PNG")
    shape = (int(row["original_height"]), int(row["original_width"]))
    if image.shape != shape or (expected_shape is not None and image.shape != tuple(expected_shape)):
        raise ValueError("SAM 2 mask is not in original-frame coordinates")
    return image > 0
