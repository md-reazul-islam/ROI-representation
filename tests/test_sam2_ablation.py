import json

import cv2
import numpy as np
import pandas as pd
import pytest

from module1.data.manifest import sha256_file
from module1.pose.inputs import load_recovery_frames
from module1.roi import ROI_NAMES
from module1.segmentation.ablation import compare_phase12
from module1.segmentation.sam2 import TemporalSAM2Segmenter


def ablation_inputs(tmp_path):
    recovery = tmp_path/"recovery.csv"
    pd.DataFrame([dict(video_id="v", split="test", frame_index=0, timestamp_sec=0.,
                       final_status="DETECTED", final_infant_bbox="[0,0,10,10]",
                       final_infant_confidence=.9, detector_status="DETECTED",
                       recovered_status="NOT_NEEDED", track_id=1)]).to_csv(recovery, index=False)
    digest = sha256_file(recovery)
    sam2_dir = tmp_path/"sam2"
    (sam2_dir/"metadata").mkdir(parents=True)
    (sam2_dir/"masks").mkdir()
    mask = np.ones((10, 10), np.uint8)*255
    cv2.imwrite(str(sam2_dir/"masks/m.png"), mask)
    row = TemporalSAM2Segmenter._base(load_recovery_frames(recovery)[("v", 0)])
    row.update(segmentation_status="INITIALIZED", mask_path="masks/m.png",
               mask_quality=.8, detector_mask_consistency=.9,
               temporal_consistency=None, original_height=10, original_width=10)
    pd.DataFrame([row]).to_csv(sam2_dir/"sam2_masks.csv", index=False)
    (sam2_dir/"metadata/experiment_metadata.json").write_text(json.dumps(dict(
        mode="SAM2_ENABLED", run_status="COMPLETE", source_hashes=dict(recovery=digest))))
    directories = {}
    for label, pose_conf, valid_count in (("baseline", .8, 10), ("enhanced", .9, 12)):
        parts = {}
        for name in ("pose", "roi", "quality", "confidence"):
            path = tmp_path/label/name
            path.mkdir(parents=True)
            parts[name] = path
        input_hashes = dict(recovery=digest, manifest="same-manifest")
        if label == "enhanced":
            input_hashes[str(sam2_dir/"sam2_masks.csv")] = sha256_file(sam2_dir/"sam2_masks.csv")
        (parts["pose"]/"experiment_metadata.json").write_text(json.dumps(dict(
            run_status="COMPLETE",
            sam2_mode="DIRECT_RECOVERY_BOX" if label == "baseline" else "MASK_CONSTRAINED",
            input_hashes=input_hashes)))
        pd.DataFrame([dict(video_id="v", split="test", frame_index=0, timestamp_sec=0.,
                           confidence=pose_conf, pose_quality=pose_conf)]).to_csv(
            parts["pose"]/"pose_results.csv", index=False)
        common = [dict(video_id="v", split="test", frame_index=0, timestamp_sec=0.,
                       roi_name=name) for name in ROI_NAMES]
        pd.DataFrame([dict(row, roi_status="AVAILABLE" if index < valid_count else "UNAVAILABLE",
                           SAM2_mask_support=.9 if label == "enhanced" else None)
                      for index, row in enumerate(common)]).to_csv(parts["roi"]/"candidate_rois.csv", index=False)
        pd.DataFrame([dict(row, temporal_consistency=.7 if label == "enhanced" else .5)
                      for row in common]).to_csv(parts["quality"]/"roi_quality_features.csv", index=False)
        pd.DataFrame([dict(row, q=.6 if label == "enhanced" else .4,
                           selected=index == (1 if label == "enhanced" else 0))
                      for index, row in enumerate(common)]).to_csv(
            parts["confidence"]/"roi_confidence.csv", index=False)
        for downstream, source_name, upstream, filename in (
                ("roi", "pose_results", "pose", "pose_results.csv"),
                ("quality", "roi_results", "roi", "candidate_rois.csv"),
                ("confidence", "quality_features", "quality", "roi_quality_features.csv")):
            (parts[downstream]/"experiment_metadata.json").write_text(json.dumps(dict(
                run_status="COMPLETE",
                source_hashes={source_name: dict(
                    sha256=sha256_file(parts[upstream]/filename))})))
        directories[label] = parts
    return recovery, sam2_dir, directories


def test_sam2_ablation_pairs_same_frames_and_reports_descriptive_differences(tmp_path):
    recovery, sam2_dir, directories = ablation_inputs(tmp_path)
    frames, summary = compare_phase12(recovery, sam2_dir, directories["baseline"], directories["enhanced"])
    assert len(frames) == 1
    assert frames.iloc[0].baseline_pose_confidence == .8
    assert frames.iloc[0].sam2_pose_confidence == .9
    assert frames.iloc[0].selected_roi_jaccard == 0.
    pose = summary[(summary.video_id == "ALL") & (summary.metric == "pose_confidence")].iloc[0]
    assert pose.n_paired == 1
    assert pose.paired_difference_mean == pytest.approx(.1)
    assert pose.ground_truth_status == "NOT_AVAILABLE"
    valid = summary[(summary.video_id == "ALL") & (summary.metric == "valid_roi_rate")].iloc[0]
    assert valid.paired_difference_mean == pytest.approx(2/17)


def test_sam2_ablation_rejects_different_upstream_inputs(tmp_path):
    recovery, sam2_dir, directories = ablation_inputs(tmp_path)
    meta_path = directories["enhanced"]["pose"]/"experiment_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["input_hashes"]["recovery"] = "different"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="upstream inputs"):
        compare_phase12(recovery, sam2_dir, directories["baseline"], directories["enhanced"])
