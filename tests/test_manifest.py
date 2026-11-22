
from pathlib import Path
from module1.data.manifest import DatasetManifest

def test_authoritative_manifest_loads():
    root=Path(__file__).resolve().parents[1]
    m=DatasetManifest.load(root/"dataset_manifest.csv")
    assert len(m.df)>0
    assert {"video_id","infant_id","session_id","split"}.issubset(m.df.columns)

def test_phase0_manifest_builder_compatibility():
    root = Path(__file__).resolve().parents[1]
    from module1.data.manifest import build_manifest, validate_manifest
    df = build_manifest(root / "data" / "Dataset", "pilot", [".mp4"])
    assert len(df) == 24
    assert validate_manifest(df) == []
