import json

import numpy as np
import pandas as pd
import pytest

from module1.ablation import AblationEvaluator, ABLATION_CONFIGURATIONS, METRICS, evaluate_paths, load_ground_truth


def bbox(values):
    return json.dumps(values)


def manifest(subject="S1"):
    return pd.DataFrame([
        dict(video_id="V1", split="test", infant_id=subject, frame_count=3),
    ])


def quality():
    rows = []
    for frame in range(3):
        for name, offset in (("FACE", 0.), ("CHEST", 5.)):
            rows.append(dict(video_id="V1", split="test", frame_index=frame, roi_name=name,
                             roi_status="AVAILABLE", bbox=bbox([10 + offset + frame, 10, 30 + offset + frame, 40]),
                             pose_confidence=.9 if name == "FACE" else .5,
                             anatomical_confidence=.9, occlusion_proxy=.1, motion_magnitude=.1,
                             temporal_consistency=.9, q=.8 if name == "FACE" else .3))
    return pd.DataFrame(rows)


def truth():
    return pd.DataFrame([
        dict(video_id="V1", frame_index=0, subject_id="S1", infant_present=1, bbox=bbox([10, 10, 30, 40]), quality_score=.9, signal_snr_proxy=4.),
        dict(video_id="V1", frame_index=1, subject_id="S1", infant_present=1, bbox=bbox([11, 10, 31, 40]), quality_score=.8, signal_snr_proxy=3.),
        dict(video_id="V1", frame_index=2, subject_id="S1", infant_present=0, bbox="", quality_score=.2, signal_snr_proxy=1.),
    ])


def dynamic():
    return pd.DataFrame([
        dict(video_id="V1", frame_index=i, selection_strategy="topk", selected_roi_names=json.dumps(["FACE"]), fusion_weights=json.dumps([1.]))
        for i in range(3)
    ])


def detections():
    return pd.DataFrame([
        dict(video_id="V1", frame_index=0, fusion_strategy="or", detection_status="DETECTED"),
        dict(video_id="V1", frame_index=1, fusion_strategy="or", detection_status="MISSING"),
        dict(video_id="V1", frame_index=2, fusion_strategy="or", detection_status="DETECTED"),
    ])


def recoveries():
    return pd.DataFrame([
        dict(video_id="V1", frame_index=0, recovered_status="NOT_NEEDED"),
        dict(video_id="V1", frame_index=1, recovered_status="ACCEPTED"),
        dict(video_id="V1", frame_index=2, recovered_status="NOT_NEEDED"),
    ])


def tracks():
    return pd.DataFrame([
        dict(video_id="V1", frame_index=i, observed=True, track_id=1) for i in range(3)
    ])


def test_ground_truth_loader_does_not_infer_missing_annotations(tmp_path):
    table, metadata = load_ground_truth(None)
    assert table.empty
    assert metadata["status"] == "NOT_AVAILABLE"
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"video_id": "V1"}]).to_csv(path, index=False)
    table, metadata = load_ground_truth(path)
    assert table.empty
    assert "missing columns" in metadata["reason"]


def test_evaluator_reports_available_truth_metrics_and_diagnostics():
    evaluator = AblationEvaluator(top_k=1)
    result = evaluator.evaluate(manifest(), quality(), dynamic(), truth=truth(), detections=detections(), recoveries=recoveries(), tracks=tracks())
    assert set(result.configuration) == set(ABLATION_CONFIGURATIONS)
    assert set(result.metric_name) == set(METRICS)
    b5_iou = result[(result.configuration == "B5_proposed_module1") & (result.metric_name == "roi_localization_iou")].iloc[0]
    assert b5_iou.status == "AVAILABLE"
    assert b5_iou.value == pytest.approx(1.)
    recovery = result[(result.configuration == "B5_proposed_module1") & (result.metric_name == "temporal_recovery_rate")].iloc[0]
    assert recovery.value == pytest.approx(1.)
    stability = result[(result.configuration == "B5_proposed_module1") & (result.metric_name == "temporal_stability")].iloc[0]
    assert stability.status == "AVAILABLE_DIAGNOSTIC"


def test_evaluator_marks_truth_metrics_unavailable_without_truth():
    evaluator = AblationEvaluator()
    result = evaluator.evaluate(manifest(subject="UNAVAILABLE"), quality())
    truth_rows = result[result.metric_name.isin({"detection_recall", "roi_localization_iou", "confidence_quality_correlation"})]
    assert set(truth_rows.status) == {"NOT_AVAILABLE"}
    assert truth_rows.value.isna().all()
    diagnostic = result[(result.metric_name == "temporal_stability") & result.configuration.isin(["B3_pose_anatomy_visibility", "B4_pose_anatomy_visibility_motion"])]
    assert set(diagnostic.status) == {"AVAILABLE_DIAGNOSTIC"}


def test_evaluate_paths_writes_required_reports(tmp_path):
    quality_path = tmp_path / "quality.csv"
    dynamic_path = tmp_path / "dynamic.csv"
    gt_path = tmp_path / "truth.csv"
    detection_path = tmp_path / "detections.csv"
    recovery_path = tmp_path / "recoveries.csv"
    tracking_path = tmp_path / "tracks.csv"
    quality().to_csv(quality_path, index=False)
    dynamic().to_csv(dynamic_path, index=False)
    truth().to_csv(gt_path, index=False)
    detections().to_csv(detection_path, index=False)
    recoveries().to_csv(recovery_path, index=False)
    tracks().to_csv(tracking_path, index=False)
    results, statistics, tests, metadata = evaluate_paths(manifest(), tmp_path / "reports", quality_path, dynamic_path, None, gt_path, detection_path, recovery_path, tracking_path)
    assert len(results) == len(ABLATION_CONFIGURATIONS) * len(METRICS)
    assert (tmp_path / "reports" / "ablation_results.csv").exists()
    assert (tmp_path / "reports" / "ablation_statistics.csv").exists()
    assert (tmp_path / "reports" / "statistical_tests.csv").exists()
    assert (tmp_path / "reports" / "figures" / "ablation_metric_overview.png").exists()
    assert metadata["no_fabrication_policy"]


def test_statistical_tests_require_subject_ids_when_manifest_has_none():
    result = AblationEvaluator().evaluate(manifest(subject="UNAVAILABLE"), quality())
    from module1.ablation.evaluation import statistical_tests
    tests = statistical_tests(result)
    assert set(tests.status) == {"NOT_AVAILABLE"}
    assert tests.p_value.isna().all()
