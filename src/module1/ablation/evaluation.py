from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


ABLATION_CONFIGURATIONS = (
    "B0_manual_fixed_roi",
    "B1_pose_fixed_roi",
    "B2_pose_anatomy_fixed_roi",
    "B3_pose_anatomy_visibility",
    "B4_pose_anatomy_visibility_motion",
    "B5_proposed_module1",
)

ROI_METRICS = (
    "roi_localization_iou",
    "roi_signal_quality_snr_proxy",
    "confidence_quality_correlation",
    "temporal_stability",
    "roi_selection_consistency",
)
DETECTION_METRICS = (
    "detection_recall",
    "detection_precision",
    "detection_f1",
    "miss_rate",
    "temporal_recovery_rate",
    "track_continuity",
    "track_fragmentation",
)
METRICS = ROI_METRICS + DETECTION_METRICS

TRUTH_METRICS = {
    "roi_localization_iou",
    "roi_signal_quality_snr_proxy",
    "confidence_quality_correlation",
    *DETECTION_METRICS,
}


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    status: str
    reason: str
    sample_count: int = 0
    diagnostic_value: float | None = None
    diagnostic_status: str = "NOT_AVAILABLE"
    interpretation: str = ""


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) or str(value).strip().lower() in {"", "nan", "none", "null", "unavailable"}
    except (TypeError, ValueError):
        return False


def _json(value, default=None):
    if _is_missing(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default
    return value


def _number(value, default=None):
    if _is_missing(value):
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _bbox(value):
    raw = _json(value)
    if raw is None or len(raw) != 4:
        return None
    values = [_number(item) for item in raw]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(first, second) -> float | None:
    first, second = _bbox(first), _bbox(second)
    if first is None or second is None:
        return None
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else None


def union_bbox(boxes: Iterable) -> list[float] | None:
    valid = [_bbox(box) for box in boxes]
    valid = [box for box in valid if box is not None]
    if not valid:
        return None
    return [min(box[0] for box in valid), min(box[1] for box in valid),
            max(box[2] for box in valid), max(box[3] for box in valid)]


def _truth_text(value) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "present", "infant"}:
        return True
    if text in {"0", "false", "no", "absent", "none"}:
        return False
    return None


def load_ground_truth(path: str | Path | None) -> tuple[pd.DataFrame, dict]:
    """Load optional frame-level truth without inferring annotations."""
    if path is None or str(path).strip() == "":
        return pd.DataFrame(), {"status": "NOT_AVAILABLE", "reason": "No ground-truth CSV configured"}
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), {"status": "NOT_AVAILABLE", "reason": f"Ground-truth file does not exist: {path}"}
    table = pd.read_csv(path)
    required = {"video_id", "frame_index"}
    missing = sorted(required - set(table.columns))
    if missing:
        return pd.DataFrame(), {"status": "NOT_AVAILABLE", "reason": f"Ground-truth CSV missing columns: {missing}"}
    table = table.copy()
    table["video_id"] = table["video_id"].astype(str)
    table["frame_index"] = pd.to_numeric(table["frame_index"], errors="coerce").astype("Int64")
    table = table[table["frame_index"].notna()].copy()
    if "bbox" in table.columns:
        table["_bbox"] = table["bbox"].map(_bbox)
    elif "roi_bbox" in table.columns:
        table["_bbox"] = table["roi_bbox"].map(_bbox)
    else:
        table["_bbox"] = None
    if "infant_present" in table.columns:
        table["_present"] = table["infant_present"].map(_truth_text)
    else:
        table["_present"] = None
    if "subject_id" not in table.columns:
        table["subject_id"] = None
    table["subject_id"] = table["subject_id"].where(table["subject_id"].notna(), None)
    if table.empty:
        return table, {"status": "NOT_AVAILABLE", "reason": "Ground-truth CSV has no valid frame rows"}
    return table, {"status": "AVAILABLE", "reason": "Frame-level ground truth loaded", "path": str(path)}


def _empty_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _available(value) -> bool:
    return value is not None and math.isfinite(float(value))


class AblationEvaluator:
    """Evaluate ROI ablations while keeping diagnostic and truth metrics separate."""

    def __init__(self, top_k: int = 3, selection_strategy: str = "topk"):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if selection_strategy not in {"top1", "top2", "topk", "confidence_weighted_fusion"}:
            raise ValueError("Unsupported Phase 10 selection strategy")
        self.top_k = top_k
        self.selection_strategy = selection_strategy

    @staticmethod
    def _frame_key(row):
        return str(row.get("video_id")), int(float(row.get("frame_index", 0)))

    def _quality_rows(self, quality: pd.DataFrame):
        grouped = {}
        if quality.empty:
            return grouped
        for _, source in quality.iterrows():
            row = source.to_dict()
            if str(row.get("roi_status", "AVAILABLE")) != "AVAILABLE":
                continue
            box = _bbox(row.get("bbox"))
            if box is None:
                continue
            row["_bbox"] = box
            row["_q"] = _number(row.get("q"))
            grouped.setdefault(self._frame_key(row), []).append(row)
        return grouped

    def _dynamic_rows(self, dynamic: pd.DataFrame):
        grouped = {}
        if dynamic.empty:
            return grouped
        selected = dynamic[dynamic.get("selection_strategy", "") == self.selection_strategy] if "selection_strategy" in dynamic else dynamic
        for _, source in selected.iterrows():
            names = _json(source.get("selected_roi_names"), []) or []
            weights = _json(source.get("fusion_weights"), []) or []
            grouped[self._frame_key(source)] = {
                "names": [str(name) for name in names],
                "weights": [float(weight) for weight in weights if _available(_number(weight))],
            }
        return grouped

    def _fixed_rows(self, fixed: pd.DataFrame):
        grouped = {}
        if fixed.empty:
            return grouped
        for _, source in fixed.iterrows():
            row = source.to_dict()
            raw_box = row.get("bbox", row.get("roi_bbox"))
            if _is_missing(raw_box) and all(name in row for name in ("x1", "y1", "x2", "y2")):
                raw_box = [row["x1"], row["y1"], row["x2"], row["y2"]]
            box = _bbox(raw_box)
            if box is None:
                continue
            grouped.setdefault(self._frame_key(row), []).append({
                "roi_name": str(row.get("roi_name", "FIXED_ROI")), "_bbox": box,
                "_q": _number(row.get("confidence"), 1.),
            })
        return grouped

    def _selected(self, config, key, quality, dynamic, fixed):
        if config in {"B0_manual_fixed_roi", "B1_pose_fixed_roi", "B2_pose_anatomy_fixed_roi"}:
            return fixed.get(key, [])
        if config == "B5_proposed_module1":
            selection = dynamic.get(key)
            if not selection:
                return []
            by_name = {row["roi_name"]: row for row in quality.get(key, [])}
            return [by_name[name] for name in selection["names"] if name in by_name]
        candidates = quality.get(key, [])
        scored = []
        for row in candidates:
            pose = _number(row.get("pose_confidence"), _number(row.get("anatomical_confidence"), 0.))
            visibility = 1. - _number(row.get("occlusion_proxy"), 1.)
            motion = _number(row.get("motion_magnitude"), 0.)
            if config == "B3_pose_anatomy_visibility":
                score = .5 * pose + .5 * visibility
            else:
                score = (pose + visibility + (1. - motion)) / 3.
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1].get("roi_name", "")))
        return [dict(row, _q=score) for score, row in scored[:self.top_k]]

    @staticmethod
    def _selection_stability(selected_by_frame):
        ordered = sorted(selected_by_frame.items(), key=lambda item: item[0][1])
        values = []
        previous = None
        for _, rows in ordered:
            current = {str(row.get("roi_name", "FIXED_ROI")) for row in rows}
            if previous is not None:
                union = len(previous | current)
                values.append(len(previous & current) / union if union else 1.)
            previous = current
        return float(np.mean(values)) if values else None, len(values)

    @staticmethod
    def _temporal_stability(selected_by_frame):
        ordered = sorted(selected_by_frame.items(), key=lambda item: item[0][1])
        values = []
        previous = None
        for _, rows in ordered:
            current = union_bbox(row.get("_bbox") for row in rows)
            if previous is not None and current is not None:
                value = bbox_iou(previous, current)
                if value is not None:
                    values.append(value)
            previous = current
        return float(np.mean(values)) if values else None, len(values)

    @staticmethod
    def _truth_for_video(truth, video):
        if truth.empty:
            return {}
        result = {}
        for _, row in truth[truth["video_id"].astype(str) == str(video)].iterrows():
            result[int(row["frame_index"])] = row.to_dict()
        return result

    @staticmethod
    def _truth_metric_reason(metric, truth_video):
        if not truth_video:
            return "No frame-level ground-truth rows for this video"
        if metric in {"detection_recall", "detection_precision", "detection_f1", "miss_rate"}:
            return "No frame-level infant presence ground truth"
        if metric == "roi_localization_iou":
            return "No frame-level infant bounding-box ground truth"
        if metric == "roi_signal_quality_snr_proxy":
            return "No signal-quality ground-truth or signal annotation"
        if metric == "confidence_quality_correlation":
            return "No quality score or signal annotation for correlation"
        if metric == "temporal_recovery_rate":
            return "No recovery input or frame-level infant presence ground truth"
        if metric in {"track_continuity", "track_fragmentation"}:
            return "No tracking input or frame-level infant presence ground truth"
        return "Metric unavailable"

    @staticmethod
    def _na(metric, reason, diagnostic=None, interpretation=""):
        return MetricValue(None, "NOT_AVAILABLE", reason, 0, diagnostic, "AVAILABLE_DIAGNOSTIC" if diagnostic is not None else "NOT_AVAILABLE", interpretation)

    def _roi_metrics(self, selected, truth_video, config):
        if not selected:
            return {
                "roi_localization_iou": self._na("roi_localization_iou", "No selected ROI records"),
                "roi_signal_quality_snr_proxy": self._na("roi_signal_quality_snr_proxy", "No selected ROI records"),
                "confidence_quality_correlation": self._na("confidence_quality_correlation", "No selected ROI records"),
            }
        localization, snr, confidence, quality = [], [], [], []
        for key, rows in selected.items():
            truth = truth_video.get(key[1], {})
            if truth.get("_bbox") is not None:
                box = union_bbox(row.get("_bbox") for row in rows)
                score = bbox_iou(box, truth.get("_bbox"))
                if score is not None:
                    localization.append(score)
            signal = _number(truth.get("signal_snr_proxy"))
            if signal is not None:
                snr.append(signal)
            target = _number(truth.get("quality_score"), signal)
            q = np.mean([_number(row.get("_q"), 0.) for row in rows]) if rows else None
            if target is not None and q is not None:
                confidence.append(q)
                quality.append(target)
        result = {}
        if localization:
            result["roi_localization_iou"] = MetricValue(float(np.mean(localization)), "AVAILABLE", "Frame-level bbox annotations loaded", len(localization), interpretation="ROI union-bbox IoU against frame-level ground truth")
        else:
            result["roi_localization_iou"] = self._na("roi_localization_iou", self._truth_metric_reason("roi_localization_iou", truth_video))
        if snr:
            result["roi_signal_quality_snr_proxy"] = MetricValue(float(np.mean(snr)), "AVAILABLE", "Signal SNR proxy annotation loaded", len(snr), interpretation="Annotated signal-quality SNR proxy")
        else:
            result["roi_signal_quality_snr_proxy"] = self._na("roi_signal_quality_snr_proxy", self._truth_metric_reason("roi_signal_quality_snr_proxy", truth_video))
        if len(confidence) >= 2 and np.std(confidence) > 0 and np.std(quality) > 0:
            result["confidence_quality_correlation"] = MetricValue(float(np.corrcoef(confidence, quality)[0, 1]), "AVAILABLE", "Quality annotations loaded", len(confidence), interpretation="Pearson correlation between ROI Q and annotated quality")
        else:
            result["confidence_quality_correlation"] = self._na("confidence_quality_correlation", self._truth_metric_reason("confidence_quality_correlation", truth_video))
        return result

    def _detection_metrics(self, video, truth_video, detections, recoveries, tracks, method):
        present = {frame: row.get("_present") for frame, row in truth_video.items() if row.get("_present") is not None}
        result = {}
        if not present:
            reason = "No frame-level infant presence ground truth"
            for metric in ("detection_recall", "detection_precision", "detection_f1", "miss_rate", "temporal_recovery_rate", "track_continuity", "track_fragmentation"):
                diagnostic = None
                result[metric] = self._na(metric, reason, diagnostic)
            return result
        det = detections[(detections.get("video_id", pd.Series(dtype=str)).astype(str) == str(video))] if not detections.empty else pd.DataFrame()
        if "fusion_strategy" in det.columns:
            preferred = det[det["fusion_strategy"].astype(str).str.lower() == method.lower()]
            if not preferred.empty:
                det = preferred
        det_map = {int(float(row.frame_index)): str(row.get("detection_status", "MISSING")).upper() == "DETECTED" for _, row in det.iterrows()}
        pairs = [(truth, det_map.get(frame, False)) for frame, truth in present.items()]
        tp = sum(actual and observed for actual, observed in pairs)
        fp = sum((not actual) and observed for actual, observed in pairs)
        fn = sum(actual and (not observed) for actual, observed in pairs)
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        miss = fn / (tp + fn) if tp + fn else None
        for metric, value, count in (("detection_recall", recall, tp + fn), ("detection_precision", precision, tp + fp), ("detection_f1", f1, len(pairs)), ("miss_rate", miss, tp + fn)):
            result[metric] = MetricValue(value, "AVAILABLE" if value is not None else "NOT_AVAILABLE", "Frame-level infant presence ground truth loaded" if value is not None else "No positive or negative truth cases", count, interpretation="Frame-level detection classification against ground truth")
        rec = recoveries[recoveries.get("video_id", pd.Series(dtype=str)).astype(str) == str(video)] if not recoveries.empty else pd.DataFrame()
        if rec.empty:
            result["temporal_recovery_rate"] = self._na("temporal_recovery_rate", "No recovery input")
        else:
            accepted = {int(float(row.frame_index)): str(row.get("recovered_status", "")).upper() in {"ACCEPTED", "RECOVERED"} for _, row in rec.iterrows()}
            missed_present = [frame for frame, actual in present.items() if actual and not det_map.get(frame, False)]
            recovered = sum(accepted.get(frame, False) for frame in missed_present)
            value = recovered / len(missed_present) if missed_present else None
            result["temporal_recovery_rate"] = MetricValue(value, "AVAILABLE" if value is not None else "NOT_AVAILABLE", "Recovery and presence annotations loaded" if value is not None else "No positive detector misses", len(missed_present), interpretation="Recovered positive detector misses divided by positive detector misses")
        track = tracks[tracks.get("video_id", pd.Series(dtype=str)).astype(str) == str(video)] if not tracks.empty else pd.DataFrame()
        if track.empty:
            result["track_continuity"] = self._na("track_continuity", "No tracking input")
            result["track_fragmentation"] = self._na("track_fragmentation", "No tracking input")
        else:
            observed = []
            ids = []
            for frame in sorted(present):
                rows = track[pd.to_numeric(track["frame_index"], errors="coerce") == frame]
                row = rows.iloc[0] if not rows.empty else None
                is_observed = bool(row is not None and _truth_text(row.get("observed")) is True and _number(row.get("track_id")) is not None)
                observed.append(is_observed)
                ids.append(str(row.get("track_id")) if is_observed else None)
            continuity = float(np.mean(observed)) if observed else None
            fragments = 0
            previous = None
            for current in ids:
                if current is not None and current != previous:
                    fragments += 1
                previous = current
            result["track_continuity"] = MetricValue(continuity, "AVAILABLE" if continuity is not None else "NOT_AVAILABLE", "Tracking and presence annotations loaded", len(observed), interpretation="Observed primary track coverage over annotated-present frames")
            result["track_fragmentation"] = MetricValue(float(fragments), "AVAILABLE", "Tracking and presence annotations loaded", len(ids), interpretation="Observed track ID runs over annotated-present frames")
        return result

    def evaluate(self, manifest: pd.DataFrame, quality: pd.DataFrame | None = None,
                 dynamic: pd.DataFrame | None = None, fixed: pd.DataFrame | None = None,
                 truth: pd.DataFrame | None = None, detections: pd.DataFrame | None = None,
                 recoveries: pd.DataFrame | None = None, tracks: pd.DataFrame | None = None,
                 detection_method: str = "or"):
        quality, dynamic, fixed = quality if quality is not None else pd.DataFrame(), dynamic if dynamic is not None else pd.DataFrame(), fixed if fixed is not None else pd.DataFrame()
        truth, detections = truth if truth is not None else pd.DataFrame(), detections if detections is not None else pd.DataFrame()
        recoveries, tracks = recoveries if recoveries is not None else pd.DataFrame(), tracks if tracks is not None else pd.DataFrame()
        if not truth.empty:
            truth = truth.copy()
            if "_bbox" not in truth.columns:
                truth["_bbox"] = truth.get("bbox", pd.Series(index=truth.index, dtype=object)).map(_bbox)
            if "_present" not in truth.columns:
                truth["_present"] = truth.get("infant_present", pd.Series(index=truth.index, dtype=object)).map(_truth_text)
        quality_rows, dynamic_rows, fixed_rows = self._quality_rows(quality), self._dynamic_rows(dynamic), self._fixed_rows(fixed)
        output = []
        for _, record in manifest.iterrows():
            video = str(record.get("video_id"))
            split = str(record.get("split", "UNASSIGNED"))
            subject = record.get("infant_id")
            subject = None if _is_missing(subject) or str(subject).upper() == "UNAVAILABLE" else str(subject)
            truth_video = self._truth_for_video(truth, video)
            for config in ABLATION_CONFIGURATIONS:
                frames = {key for key in quality_rows if key[0] == video}
                frames |= {key for key in dynamic_rows if key[0] == video}
                frames |= {key for key in fixed_rows if key[0] == video}
                selected = {key: self._selected(config, key, quality_rows, dynamic_rows, fixed_rows) for key in frames}
                selected = {key: rows for key, rows in selected.items() if rows}
                roi = self._roi_metrics(selected, truth_video, config)
                stability, stability_n = self._temporal_stability(selected)
                consistency, consistency_n = self._selection_stability(selected)
                roi["temporal_stability"] = MetricValue(stability, "AVAILABLE_DIAGNOSTIC" if stability is not None else "NOT_AVAILABLE", "Consecutive selected ROI boxes" if stability is not None else "No consecutive selected ROI boxes", stability_n, interpretation="Diagnostic consecutive selected-box IoU; not ground-truth accuracy")
                roi["roi_selection_consistency"] = MetricValue(consistency, "AVAILABLE_DIAGNOSTIC" if consistency is not None else "NOT_AVAILABLE", "Consecutive selected ROI sets" if consistency is not None else "No consecutive selected ROI sets", consistency_n, interpretation="Diagnostic adjacent-frame Jaccard consistency of selected ROI names")
                detection = self._detection_metrics(video, truth_video, detections, recoveries, tracks, detection_method)
                all_metrics = {**roi, **detection}
                for metric in METRICS:
                    value = all_metrics[metric]
                    output.append({
                        "scope": "video", "configuration": config, "video_id": video, "split": split,
                        "subject_id": subject, "metric_name": metric, "value": value.value,
                        "status": value.status, "reason": value.reason, "sample_count": value.sample_count,
                        "diagnostic_value": value.diagnostic_value, "diagnostic_status": value.diagnostic_status,
                        "metric_interpretation": value.interpretation,
                    })
        return pd.DataFrame(output)


def _ci95(values):
    if len(values) < 2:
        return None, None
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    try:
        from scipy.stats import t
        critical = float(t.ppf(.975, len(values) - 1))
    except Exception:
        critical = 1.96
    return mean - critical * standard_error, mean + critical * standard_error


def aggregate_statistics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, metric), group in results.groupby(["configuration", "metric_name"], dropna=False):
        valid = pd.to_numeric(group.loc[group["status"].isin(["AVAILABLE", "AVAILABLE_DIAGNOSTIC"]), "value"], errors="coerce"),
        values = valid[0].dropna().astype(float).tolist()
        diagnostic = pd.to_numeric(group["diagnostic_value"], errors="coerce").dropna().astype(float).tolist()
        source = values if values else diagnostic
        if not source:
            status = "NOT_AVAILABLE"
            reason = str(group.iloc[0]["reason"])
            rows.append({"scope": "overall", "configuration": config, "metric_name": metric, "value": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "status": status, "reason": reason, "unit": "count" if metric == "track_fragmentation" else "ratio", "metric_interpretation": str(group.iloc[0]["metric_interpretation"])})
            continue
        low, high = _ci95(source)
        rows.append({"scope": "overall", "configuration": config, "metric_name": metric, "value": float(np.mean(source)), "mean": float(np.mean(source)), "median": float(np.median(source)), "std": float(np.std(source, ddof=1)) if len(source) > 1 else 0., "ci_low": low if low is not None else np.nan, "ci_high": high if high is not None else np.nan, "n": len(source), "status": "AVAILABLE_DIAGNOSTIC" if not values else "AVAILABLE", "reason": "Aggregate across valid per-video values", "unit": "count" if metric == "track_fragmentation" else "ratio", "metric_interpretation": str(group.iloc[0]["metric_interpretation"])})
    return pd.DataFrame(rows)


def statistical_tests(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for baseline in ABLATION_CONFIGURATIONS[:-1]:
            subset = results[results["metric_name"] == metric]
            def subject_values(configuration):
                selected = subset[(subset["configuration"] == configuration) & subset["subject_id"].notna()].copy()
                selected = selected[selected["subject_id"].astype(str).str.upper() != "UNAVAILABLE"]
                selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
                selected = selected.dropna(subset=["value"])
                return selected.groupby("subject_id")["value"].mean() if not selected.empty else pd.Series(dtype=float)
            a = subject_values(baseline) if not subset.empty else pd.Series(dtype=float)
            b = subject_values("B5_proposed_module1") if not subset.empty else pd.Series(dtype=float)
            paired = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
            paired = paired[paired.index.notna() & (paired.index.astype(str).str.upper() != "UNAVAILABLE")]
            base = {"comparison": f"B5_proposed_module1_vs_{baseline}", "configuration_a": baseline, "configuration_b": "B5_proposed_module1", "metric_name": metric, "test": "paired_t_test;wilcoxon", "n_pairs": len(paired), "statistic": np.nan, "p_value": np.nan, "p_value_adjusted": np.nan, "significant": False, "correction": "holm", "subject_level": True}
            if len(paired) < 2:
                base.update(status="NOT_AVAILABLE", reason="No subject IDs with valid paired measurements")
            else:
                try:
                    from scipy.stats import ttest_rel, wilcoxon
                    t_stat, t_p = ttest_rel(paired["a"], paired["b"])
                    try:
                        w_stat, w_p = wilcoxon(paired["a"], paired["b"])
                    except ValueError:
                        w_stat, w_p = np.nan, np.nan
                    base.update(test="paired_t_test", statistic=float(t_stat), p_value=float(t_p), status="AVAILABLE", reason="Subject-level paired measurements")
                    base["wilcoxon_statistic"] = w_stat
                    base["wilcoxon_p_value"] = w_p
                except Exception as exc:
                    base.update(status="NOT_AVAILABLE", reason=f"Statistical test unavailable: {exc}")
            rows.append(base)
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    if valid.any():
        ordered = result.loc[valid].sort_values("p_value")
        m = len(ordered)
        adjusted = []
        for rank, value in enumerate(ordered["p_value"].astype(float), start=1):
            adjusted.append(min(1., value * (m - rank + 1)))
        adjusted = np.maximum.accumulate(adjusted)
        result.loc[ordered.index, "p_value_adjusted"] = adjusted
        result.loc[ordered.index, "significant"] = adjusted < .05
    result["hypotheses_corrected"] = int(valid.sum())
    return result


def write_figure(statistics: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    image = np.full((720, 1280, 3), 248, dtype=np.uint8)
    cv2.putText(image, "Phase 11 Ablation Overview", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2, cv2.LINE_AA)
    available = statistics[statistics["status"].isin(["AVAILABLE", "AVAILABLE_DIAGNOSTIC"])] if not statistics.empty else pd.DataFrame()
    metrics = list(METRICS)
    configs = list(ABLATION_CONFIGURATIONS)
    chart = available.pivot(index="metric_name", columns="configuration", values="mean") if not available.empty else pd.DataFrame()
    for index, metric in enumerate(metrics):
        y = 80 + index * 85
        cv2.putText(image, metric[:30], (35, y + 28), cv2.FONT_HERSHEY_SIMPLEX, .45, (45, 45, 45), 1, cv2.LINE_AA)
        for column, config in enumerate(configs):
            value = _number(chart.loc[metric, config]) if metric in chart.index and config in chart.columns else None
            if value is None:
                continue
            x = 330 + column * 145
            width = max(2, int(min(1., max(0., value)) * 110))
            cv2.rectangle(image, (x, y + 10), (x + width, y + 28), (70, 145, 215), -1)
            cv2.putText(image, f"{value:.2f}", (x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, .32, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "ablation_metric_overview.png"), image)


def evaluate_paths(manifest: pd.DataFrame, output_dir: str | Path, quality_path: str | Path | None = None,
                   dynamic_path: str | Path | None = None, fixed_path: str | Path | None = None,
                   ground_truth_path: str | Path | None = None, detection_path: str | Path | None = None,
                   recovery_path: str | Path | None = None, tracking_path: str | Path | None = None,
                   top_k: int = 3, selection_strategy: str = "topk", detection_method: str = "or",
                   save_visualizations: bool = True):
    truth, truth_metadata = load_ground_truth(ground_truth_path)
    evaluator = AblationEvaluator(top_k, selection_strategy)
    results = evaluator.evaluate(manifest, _empty_table(Path(quality_path) if quality_path else None), _empty_table(Path(dynamic_path) if dynamic_path else None), _empty_table(Path(fixed_path) if fixed_path else None), truth, _empty_table(Path(detection_path) if detection_path else None), _empty_table(Path(recovery_path) if recovery_path else None), _empty_table(Path(tracking_path) if tracking_path else None), detection_method)
    statistics = aggregate_statistics(results)
    tests = statistical_tests(results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    if save_visualizations:
        write_figure(statistics, figures)
    results.to_csv(output / "ablation_results.csv", index=False)
    statistics.to_csv(output / "ablation_statistics.csv", index=False)
    tests.to_csv(output / "statistical_tests.csv", index=False)
    metadata = {
        "phase": "PHASE_11_MODULE_1_ABLATION_AND_QUANTITATIVE_EVALUATION",
        "ground_truth": truth_metadata,
        "configurations": list(ABLATION_CONFIGURATIONS),
        "metrics": list(METRICS),
        "selection_strategy": selection_strategy,
        "top_k": top_k,
        "detection_method": detection_method,
        "no_fabrication_policy": "Truth metrics are NOT_AVAILABLE without valid annotations; diagnostic descriptors are labeled separately.",
        "multiple_comparison_correction": "Holm over available paired t-test p-values",
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results, statistics, tests, metadata
