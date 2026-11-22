
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pandas as pd
from .video import VideoReader


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def discover_videos(video_dir: str | Path, extensions: list[str]) -> list[Path]:
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    ext = {e.lower() for e in extensions}
    return sorted(p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in ext)

def build_manifest(video_dir: str | Path, source_name: str, extensions: list[str]) -> pd.DataFrame:
    records = []
    video_dir = Path(video_dir).resolve()
    for i, path in enumerate(discover_videos(video_dir, extensions), start=1):
        meta = asdict(VideoReader(path).metadata())
        records.append({
            "video_id": path.stem,
            "record_id": i,
            "infant_id": "UNAVAILABLE",
            "session_id": "UNAVAILABLE",
            "dataset_source": source_name,
            "filename": path.name,
            "relative_video_path": path.relative_to(video_dir).as_posix(),
            "width": meta["width"],
            "height": meta["height"],
            "input_fps": meta["native_fps"],
            "frame_count": meta["frame_count"],
            "duration_sec": meta["duration_sec"],
            "codec": meta["codec"],
            "ground_truth_hr_available": "UNKNOWN",
            "ground_truth_rr_available": "UNKNOWN",
            "roi_annotation_available": "UNKNOWN",
            "split": "UNASSIGNED",
            "split_unit": "UNAVAILABLE",
            "subject_independent_verified": False,
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(records)

def validate_manifest(df: pd.DataFrame) -> list[str]:
    errors = []
    required = {"video_id", "filename", "relative_video_path", "input_fps", "frame_count"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return errors
    if df.empty:
        errors.append("Manifest contains no videos.")
        return errors
    if df["video_id"].duplicated().any():
        errors.append("Duplicate video_id values detected.")
    if (pd.to_numeric(df["input_fps"], errors="coerce") <= 0).any():
        errors.append("One or more videos have non-positive FPS.")
    if (pd.to_numeric(df["frame_count"], errors="coerce") <= 0).any():
        errors.append("One or more videos have non-positive frame counts.")
    return errors

@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    filename: str
    relative_video_path: str
    infant_id: str
    session_id: str
    dataset_source: str
    split: str
    split_unit: str
    camera_viewpoint: str
    input_fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    codec: str
    ground_truth_hr_available: str
    ground_truth_rr_available: str
    roi_annotation_available: str
    subject_independent_verified: bool
    sha256: str

class DatasetManifest:
    REQUIRED_COLUMNS = {
        "video_id", "filename", "infant_id", "session_id",
        "split", "split_unit"
    }
    ALLOWED_SPLITS = {"train", "val", "validation", "test", "unassigned"}

    def __init__(self, dataframe, manifest_path=None):
        self.df = dataframe.copy()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(pd.read_csv(path), path)

    def validate_schema(self):
        return sorted(self.REQUIRED_COLUMNS - set(self.df.columns))

    def relative_path_column(self):
        if "video_path" in self.df.columns:
            return "video_path"
        if "relative_video_path" in self.df.columns:
            return "relative_video_path"
        return "filename"

    def resolve_video_path(self, project_root, row):
        root = Path(project_root).resolve()
        raw = str(row.get(self.relative_path_column(), row.get("filename", "")))
        rel = Path(raw)

        # Supplied manifest uses Dataset/<filename>. Future manifests may use
        # just <filename>. Resolve both forms without absolute machine paths.
        candidates = [
            root / "data" / rel,
            root / "data" / "Dataset" / rel,
            root / "data" / "Dataset" / Path(rel.name),
            root / rel,
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()

        # Portable fallback: the supplied manifest contains SHA-256 hashes.
        # This allows a renamed/moved video to be resolved without depending
        # on an absolute path or the original filename.
        expected_sha = str(row.get("sha256", "")).strip().lower()
        if expected_sha and expected_sha not in {"nan", "unknown", "unavailable"}:
            dataset_root = root / "data" / "Dataset"
            if dataset_root.exists():
                import hashlib
                for candidate in dataset_root.rglob("*"):
                    if not candidate.is_file() or candidate.suffix.lower() not in {".mp4",".avi",".mov",".mkv"}:
                        continue
                    h = hashlib.sha256()
                    with candidate.open("rb") as f:
                        while chunk := f.read(1024 * 1024):
                            h.update(chunk)
                    if h.hexdigest().lower() == expected_sha:
                        return candidate.resolve()

        return candidates[0].resolve()

    def to_records(self):
        def get(r, *names, default="UNKNOWN"):
            for n in names:
                if n in r and pd.notna(r[n]):
                    return r[n]
            return default
        records = []
        for _, r in self.df.iterrows():
            records.append(VideoRecord(
                video_id=str(get(r,"video_id",default="")),
                filename=str(get(r,"filename",default="")),
                relative_video_path=str(get(r,"video_path","relative_video_path","filename",default="")),
                infant_id=str(get(r,"infant_id",default="UNAVAILABLE")),
                session_id=str(get(r,"session_id",default="UNAVAILABLE")),
                dataset_source=str(get(r,"source_dataset","dataset_source",default="UNKNOWN")),
                split=str(get(r,"split",default="UNASSIGNED")),
                split_unit=str(get(r,"split_unit",default="UNAVAILABLE")),
                camera_viewpoint=str(get(r,"camera_viewpoint",default="UNKNOWN")),
                input_fps=float(get(r,"fps","input_fps",default=0)),
                frame_count=int(get(r,"frame_count",default=0)),
                duration_sec=float(get(r,"duration_sec",default=0)),
                width=int(get(r,"width",default=0)),
                height=int(get(r,"height",default=0)),
                codec=str(get(r,"codec",default="")),
                ground_truth_hr_available=str(get(r,"ground_truth_hr_available",default="UNKNOWN")),
                ground_truth_rr_available=str(get(r,"ground_truth_rr_available",default="UNKNOWN")),
                roi_annotation_available=str(get(r,"roi_annotation_available",default="UNKNOWN")),
                subject_independent_verified=str(get(r,"subject_independent_verified",default=False)).lower()=="true",
                sha256=str(get(r,"sha256",default=""))
            ))
        return records
