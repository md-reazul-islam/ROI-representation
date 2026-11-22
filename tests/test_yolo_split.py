import pandas as pd
from module1.data.split import VideoSplitPolicy, VideoLeakageChecker


def test_pilot_split_policy():
    policy = VideoSplitPolicy()
    assert policy.assign(1) == "train"
    assert policy.assign(18) == "train"
    assert policy.assign(19) == "validation"
    assert policy.assign(21) == "validation"
    assert policy.assign(22) == "test"
    assert policy.assign(24) == "test"


def test_video_split_leakage_checker():
    df = pd.DataFrame({"video_id":["VID_1","VID_2","VID_3"],"session_id":["S1","S2","S3"],"split":["train","validation","test"]})
    r = VideoLeakageChecker.check(df)
    assert r["status"] == "PASS"
