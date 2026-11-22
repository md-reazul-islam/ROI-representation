
import pandas as pd
import pytest
from module1.data.split import SubjectLeakageChecker

def test_unavailable_subjects_are_reported():
    df=pd.DataFrame({"split":["train","test"],"infant_id":["UNAVAILABLE","UNAVAILABLE"]})
    assert SubjectLeakageChecker.check(df)["status"]=="UNAVAILABLE"

def test_real_subject_leakage_raises():
    df=pd.DataFrame({"split":["train","test"],"infant_id":["I01","I01"]})
    with pytest.raises(ValueError):
        SubjectLeakageChecker.check(df)

from pathlib import Path
from module1.data.split import VideoSplitPolicy, VideoLeakageChecker


def test_pilot_video_split_policy():
    policy = VideoSplitPolicy()
    assert policy.assign(1) == "train"
    assert policy.assign(18) == "train"
    assert policy.assign(19) == "validation"
    assert policy.assign(21) == "validation"
    assert policy.assign(22) == "test"
    assert policy.assign(24) == "test"


def test_video_leakage_checker_passes_unique_videos():
    df = pd.DataFrame({
        "video_id": ["V1", "V2", "V3"],
        "session_id": ["S1", "S2", "S3"],
        "split": ["train", "validation", "test"],
    })
    assert VideoLeakageChecker.check(df)["status"] == "PASS"
