
from pathlib import Path
from module1.data.manifest import DatasetManifest

def test_manifest_video_can_resolve_by_sha256_when_filename_differs():
    root=Path(__file__).resolve().parents[1]
    m=DatasetManifest.load(root/"dataset_manifest.csv")
    row=m.df.iloc[0]
    path=m.resolve_video_path(root,row)
    assert path.exists()
