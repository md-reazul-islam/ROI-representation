"""RT-DETR summaries with independent temporal sequences for each video."""
from .metrics import calculate_detection_metrics


def aggregate_metrics(records, group_key=None):
    groups = {}
    for row in records:
        groups.setdefault(row.get(group_key) if group_key else "all", []).append(row)
    summaries = []
    for key, rows in groups.items():
        result = calculate_detection_metrics(rows)
        videos = {}
        for row in rows:
            videos.setdefault(row["video_id"], []).append(row)
        per_video = [calculate_detection_metrics(video) for video in videos.values()]
        gaps = sum(m["number_of_detection_gaps"] for m in per_video)
        result.update(
            group=key,
            number_of_detection_gaps=gaps,
            longest_consecutive_miss=max(m["longest_consecutive_miss"] for m in per_video),
            average_gap_length=(sum(m["detector_miss_frames"] for m in per_video) / gaps if gaps else 0.0),
        )
        summaries.append(result)
    return summaries


def verify_frame_alignment(actual, expected):
    """Check unique frame identities and native timestamps, including misses."""
    def index(rows):
        return {(r["video_id"], int(r["frame_index"])): float(r["timestamp_sec"]) for r in rows}
    left, right = index(actual), index(expected)
    if left.keys() != right.keys() or any(abs(left[k] - right[k]) > 1e-8 for k in left):
        raise ValueError("RT-DETR frames do not match the YOLO baseline; check manifest and processing FPS")
