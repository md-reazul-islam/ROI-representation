
from dataclasses import dataclass, asdict
from pathlib import Path
import json, pandas as pd
from .video import VideoReader
from .manifest import DatasetManifest

@dataclass
class ValidationReport:
    status: str
    video_count: int
    valid_video_count: int
    missing_files: list
    unreadable_videos: list
    duplicate_video_ids: list
    duplicate_session_ids: list
    invalid_splits: list
    missing_infant_ids: list
    missing_session_ids: list
    inconsistent_fps: list
    inconsistent_resolution: list
    metadata_mismatches: list
    subject_independent_evaluation: str
    warnings: list
    def to_dict(self): return asdict(self)

class DatasetValidator:
    def __init__(self, manifest, project_root):
        self.manifest = manifest
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def unavailable(v):
        return str(v).strip().upper() in {"","NAN","NONE","UNKNOWN","UNAVAILABLE"}

    def run(self):
        df = self.manifest.df
        missing, unreadable, mismatches = [], [], []
        invalid, missing_infant, missing_session = [], [], []
        actual_fps, actual_res = [], []
        warnings = []

        schema = self.manifest.validate_schema()
        if schema:
            warnings.append({"type":"missing_required_columns","columns":schema})

        path_col = self.manifest.relative_path_column()

        for _, r in df.iterrows():
            vid = str(r.get("video_id", r.get("filename","")))
            path = self.manifest.resolve_video_path(self.project_root, r)
            if not path.exists():
                missing.append({
                    "video_id": vid,
                    "manifest_path": str(r.get(path_col,"")),
                    "resolved_expected_path": str(path)
                })
                continue

            m = VideoReader(path).metadata()
            if not m.opened_successfully:
                unreadable.append({"video_id":vid,"path":str(path)})
                continue

            actual_fps.append(round(m.native_fps,6))
            actual_res.append((m.width,m.height))

            expected_fps = r.get("fps", r.get("input_fps", None))
            if pd.notna(expected_fps) if expected_fps is not None else False:
                if abs(float(m.native_fps)-float(expected_fps)) > 0.01:
                    mismatches.append({"video_id":vid,"field":"fps","manifest":float(expected_fps),"actual":m.native_fps})

            for field, actual in [("frame_count",m.frame_count),("width",m.width),("height",m.height)]:
                if field in r and pd.notna(r[field]):
                    expected = int(r[field])
                    if int(actual) != expected:
                        mismatches.append({"video_id":vid,"field":field,"manifest":expected,"actual":actual})

            if self.unavailable(r.get("infant_id","UNAVAILABLE")): missing_infant.append(vid)
            if self.unavailable(r.get("session_id","UNAVAILABLE")): missing_session.append(vid)

            split = str(r.get("split","UNASSIGNED")).strip().lower()
            if split not in DatasetManifest.ALLOWED_SPLITS:
                invalid.append({"video_id":vid,"split":split})

        if "video_id" in df.columns:
            dup_vid = df.loc[df["video_id"].duplicated(keep=False),"video_id"].astype(str).unique().tolist()
        else:
            dup_vid = []

        if "session_id" in df.columns:
            s = df["session_id"].astype(str)
            valid_s = ~s.str.strip().str.upper().isin({"","NAN","NONE","UNKNOWN","UNAVAILABLE"})
            dup_sess = s[valid_s & s.duplicated(keep=False)].astype(str).unique().tolist()
        else:
            dup_sess = []

        if len(set(actual_fps)) > 1:
            warnings.append({"type":"inconsistent_fps_across_dataset","values":sorted(set(actual_fps))})
        if len(set(actual_res)) > 1:
            warnings.append({"type":"inconsistent_resolution_across_dataset","values":[f"{w}x{h}" for w,h in sorted(set(actual_res))]})
        warnings.append({
            "type": "dataset_manifest_vs_physical_file_count",
            "manifest_records": int(len(df)),
            "physical_videos_resolved": int(len(actual_fps)),
            "unresolved_manifest_records": int(len(missing))
        })

        subject = ("UNAVAILABLE: infant_id is missing for one or more records"
                   if missing_infant else
                   "VERIFIABLE: infant_id is present; leakage checker required before experiments")

        # Heterogeneous FPS/resolution is not an integrity failure. Missing or
        # unreadable files and metadata mismatches are hard validation failures.
        hard = bool(missing or unreadable or dup_vid or dup_sess or invalid or mismatches or schema)
        return ValidationReport(
            "FAIL" if hard else "PASS",
            len(df),
            len(df)-len(missing)-len(unreadable),
            missing, unreadable, dup_vid, [str(x) for x in dup_sess], invalid,
            missing_infant, missing_session,
            sorted(set(actual_fps)) if len(set(actual_fps))>1 else [],
            sorted(set(actual_res)) if len(set(actual_res))>1 else [],
            mismatches, subject, warnings
        )

    def write_report(self, report, path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report.to_dict(),indent=2),encoding="utf-8")
