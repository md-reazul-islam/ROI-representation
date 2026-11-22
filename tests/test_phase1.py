
from pathlib import Path
from module1.data.video import VideoReader
from module1.data.sampler import FrameSampler

def test_native_fps_and_deterministic_20fps_sampling():
    root=Path(__file__).resolve().parents[1]
    video=sorted((root/"data"/"Dataset").glob("*.mp4"))[0]
    reader=VideoReader(video)
    meta=reader.metadata()
    assert meta.opened_successfully
    assert meta.native_fps>0
    assert meta.frame_count>0
    assert meta.width>0 and meta.height>0
    a=FrameSampler(reader,20).selected_indices(meta)
    b=FrameSampler(reader,20).selected_indices(meta)
    assert a==b
    assert a==sorted(set(a))
    assert a[0]==0
    assert all(0 <= i < meta.frame_count for i in a)
