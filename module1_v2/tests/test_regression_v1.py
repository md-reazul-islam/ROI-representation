"""v2 code path with the v1_equivalent flags must reproduce v1 'Full' selection exactly on real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1 import paper_eval as pe
from module1_v2 import selection as sel

V2 = Path(__file__).resolve().parents[1]
RUN = V2.parent/'runs/pilot24_smoke/outputs/confidence_sam2/roi_confidence.csv'


@pytest.mark.skipif(not RUN.is_file(), reason='needs the v1 smoke run outputs')
def test_v1_flags_reproduce_full_selection():
    flags = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())['v1_equivalent']
    table = pd.read_csv(RUN, low_memory=False)
    for video, cand in table.groupby('video_id'):
        frames = sorted(cand.frame_index.unique())
        geometry, fusion, _, _ = sel.select_video(cand, frames, flags, 20.)
        v1 = pe.selections(cand, 'Full')
        assert set(v1) == set(geometry)
        for f in frames:
            assert [(r[0], round(r[1], 12)) for r in v1[f]] == [(r[0], round(r[1], 12)) for r in geometry[f]]
            assert [r[3] for r in v1[f]] == pytest.approx([r[3] for r in geometry[f]], nan_ok=True)
            assert sorted((r[0], round(r[1], 12)) for r in v1[f]) == sorted((r[0], round(r[1], 12)) for r in fusion[f])
