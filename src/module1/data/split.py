from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


UNAVAILABLE = {"", "NAN", "NONE", "UNKNOWN", "UNAVAILABLE"}


class VideoSplitPolicy:
    """Deterministic pilot split that keeps complete videos in one partition.

    The pilot dataset contains 24 supplied video files named 1.mp4 ... 24.mp4.
    The policy assigns physical video order 1-18 to train 19-21 to validation
    and 22-24 to test. No frame-level randomization is used.
    """

    def __init__(self, train_end: int = 18, val_end: int = 21, total_videos: int = 24):
        if not (0 < train_end < val_end == total_videos - 3):
            raise ValueError("Pilot split must be 1-18 train 19-21 validation 22-24 test.")
        self.train_end = train_end
        self.val_end = val_end
        self.total_videos = total_videos

    @staticmethod
    def physical_video_number(path_or_name: str | Path) -> int:
        stem = Path(str(path_or_name)).stem
        match = re.fullmatch(r"(\d+)", stem)
        if not match:
            raise ValueError(
                f"Pilot video split requires numeric physical filenames such as 1.mp4. Got: {path_or_name}"
            )
        return int(match.group(1))

    def assign(self, physical_number: int) -> str:
        n = int(physical_number)
        if n < 1 or n > self.total_videos:
            raise ValueError(f"Physical video number {n} is outside 1..{self.total_videos}.")
        if n <= self.train_end:
            return "train"
        if n <= self.val_end:
            return "validation"
        return "test"

    def apply(self, df: pd.DataFrame, resolved_paths: dict[str, Path]) -> pd.DataFrame:
        out = df.copy()
        assignments = []
        for _, row in out.iterrows():
            vid = str(row["video_id"])
            path = resolved_paths.get(vid)
            if path is None:
                raise ValueError(f"No resolved physical path available for {vid}.")
            n = self.physical_video_number(path)
            assignments.append(self.assign(n))
        out["split"] = assignments
        out["split_unit"] = "video"
        # Subject independence is deliberately NOT inferred from video order.
        out["subject_independent_verified"] = False
        return out


class VideoLeakageChecker:
    """Audits that each video/session belongs to exactly one split."""

    @classmethod
    def check(cls, df: pd.DataFrame) -> dict:
        required = {"video_id", "split"}
        missing = required - set(df.columns)
        if missing:
            return {"status": "FAIL", "reason": f"Missing columns: {sorted(missing)}", "overlap": []}

        invalid_rows = []
        for _, row in df.iterrows():
            split = str(row["split"]).strip().lower()
            if split not in {"train", "validation", "val", "test"}:
                invalid_rows.append({"video_id": str(row["video_id"]), "split": split})
        if invalid_rows:
            return {"status": "FAIL", "reason": "Video has an invalid or unassigned split.", "invalid": invalid_rows}

        dup_videos = df[df["video_id"].astype(str).duplicated(keep=False)]["video_id"].astype(str).unique().tolist()
        if dup_videos:
            return {"status": "FAIL", "reason": "Duplicate video IDs can create leakage.", "overlap": dup_videos}

        result = {
            "status": "PASS",
            "reason": "Each video is assigned to exactly one split. Frames must inherit the video split.",
            "video_overlap": [],
            "session_overlap": [],
        }

        if "session_id" in df.columns:
            usable = ~df["session_id"].astype(str).str.strip().str.upper().isin(UNAVAILABLE)
            sessions = df.loc[usable, ["session_id", "split"]].copy()
            session_split_counts = sessions.groupby(sessions["session_id"].astype(str))["split"].nunique()
            result["session_overlap"] = session_split_counts[session_split_counts > 1].index.tolist()
            if result["session_overlap"]:
                result["status"] = "FAIL"
                result["reason"] = "A session occurs in more than one split."
        return result


class SubjectLeakageChecker:
    UNAVAILABLE = UNAVAILABLE

    @classmethod
    def check(cls, df, train_split="train", test_split="test"):
        if "infant_id" not in df.columns:
            return {"status":"UNAVAILABLE","reason":"infant_id column absent","overlap":[]}
        train = df[df["split"].astype(str).str.lower().eq(train_split)]
        test = df[df["split"].astype(str).str.lower().eq(test_split)]
        a = set(train["infant_id"].astype(str).str.strip())
        b = set(test["infant_id"].astype(str).str.strip())
        if a & cls.UNAVAILABLE or b & cls.UNAVAILABLE:
            return {"status":"UNAVAILABLE","reason":"infant_id unavailable in train or test","overlap":[]}
        overlap = sorted(a & b)
        if overlap:
            raise ValueError(f"Subject leakage detected. Shared infant IDs: {overlap}")
        return {"status":"PASS","reason":"No infant ID occurs in both train and test.","overlap":[]}
