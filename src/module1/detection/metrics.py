from __future__ import annotations
import math
from statistics import median

def _safe_rate(n,d): return float(n/d) if d else math.nan

def calculate_gap_metrics(statuses):
    gaps=0; longest=0; lengths=[]; current=0
    for status in statuses:
        if status=="MISSED": current+=1
        elif current:
            gaps+=1; lengths.append(current); longest=max(longest,current); current=0
    if current:
        gaps+=1; lengths.append(current); longest=max(longest,current)
    return {"number_of_detection_gaps":gaps,"longest_consecutive_miss":longest,
            "average_gap_length":float(sum(lengths)/len(lengths)) if lengths else 0.0}

def _frame_records(records):
    return sorted([r for r in records if r.get("selected_detection") or r.get("detection_status")=="MISSED"],
                  key=lambda r:(int(r["frame_index"]),float(r["timestamp_sec"])))

def calculate_detection_metrics(records):
    frames=_frame_records(records); total=len(frames); detected=sum(r["detection_status"]=="DETECTED" for r in frames); missed=total-detected
    confs=[float(r["confidence"]) for r in frames if r["detection_status"]=="DETECTED" and r.get("confidence") is not None]
    return {"total_processed_frames":total,"detected_frames":detected,"detector_miss_frames":missed,
            "detection_rate":_safe_rate(detected,total),"miss_rate":_safe_rate(missed,total),
            "mean_confidence":float(sum(confs)/len(confs)) if confs else math.nan,
            "median_confidence":float(median(confs)) if confs else math.nan,
            "detections_per_video":detected,**calculate_gap_metrics([r["detection_status"] for r in frames]),
            "ground_truth_status":"UNAVAILABLE","metric_interpretation":"detector_observation_not_ground_truth_accuracy"}

def aggregate_metrics(records, group_key=None):
    groups={}
    for r in records: groups.setdefault(r.get(group_key) if group_key else "all",[]).append(r)
    out=[]
    for key,group in groups.items():
        m=calculate_detection_metrics(group); m["group"]=key; out.append(m)
    return out
