from module1.detection.metrics import calculate_detection_metrics, calculate_gap_metrics


def test_detection_metrics_do_not_use_ground_truth():
    rows = [
        {"frame_index": 0, "timestamp_sec": 0.0, "detection_status": "DETECTED", "selected_detection": True, "confidence": 0.8},
        {"frame_index": 1, "timestamp_sec": 0.05, "detection_status": "MISSED", "selected_detection": False, "confidence": None},
        {"frame_index": 2, "timestamp_sec": 0.10, "detection_status": "MISSED", "selected_detection": False, "confidence": None},
        {"frame_index": 3, "timestamp_sec": 0.15, "detection_status": "DETECTED", "selected_detection": True, "confidence": 0.6},
    ]
    m = calculate_detection_metrics(rows)
    assert m["total_processed_frames"] == 4
    assert m["detected_frames"] == 2
    assert m["detector_miss_frames"] == 2
    assert m["detection_rate"] == 0.5
    assert m["miss_rate"] == 0.5
    assert m["mean_confidence"] == 0.7
    assert m["number_of_detection_gaps"] == 1
    assert m["longest_consecutive_miss"] == 2
    assert m["ground_truth_status"] == "UNAVAILABLE"


def test_multiple_candidates_counted_once_for_frame_metrics():
    rows = [
        {"frame_index": 0, "timestamp_sec": 0.0, "detection_status": "DETECTED", "selected_detection": True, "confidence": 0.9},
        {"frame_index": 0, "timestamp_sec": 0.0, "detection_status": "DETECTED", "selected_detection": False, "confidence": 0.5},
        {"frame_index": 1, "timestamp_sec": 0.05, "detection_status": "MISSED", "selected_detection": False, "confidence": None},
    ]
    m = calculate_detection_metrics(rows)
    assert m["total_processed_frames"] == 2
    assert m["detected_frames"] == 1


def test_gap_metrics_trailing_gap():
    m = calculate_gap_metrics(["DETECTED", "MISSED", "MISSED"])
    assert m["number_of_detection_gaps"] == 1
    assert m["longest_consecutive_miss"] == 2
