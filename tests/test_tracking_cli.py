import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata
from module1.tracking.inputs import load_fused_frames


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_tracking.py'
    spec = importlib.util.spec_from_file_location('tracking_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path / 'configs').mkdir()
    config = tmp_path / 'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'),
                                         tracking=dict(gmc_method='none', track_buffer=3, primary_hold_frames=2))))
    pd.DataFrame([dict(video_id='v', split='train', filename='v.mp4')]).to_csv(tmp_path / 'manifest.csv', index=False)
    source = tmp_path / 'outputs/fusion'
    source.mkdir(parents=True)
    rows = []
    for i in range(6):
        candidates = [] if i == 4 else [dict(accepted=True, selected=True,
                          fused=dict(bbox=[10, 10, 50, 60], confidence=.9, class_id=0, class_name='person'))]
        rows.append(dict(video_id='v', split='train', frame_index=i, timestamp_sec=i/20, fusion_strategy='or',
                         detection_status='DETECTED' if candidates else 'MISSED', candidates_json=json.dumps(candidates)))
    pd.DataFrame(rows).to_csv(source / 'detection_fusion.csv', index=False)
    meta = VideoMetadata('v.mp4', 'v.mp4', 100, 100, 20, 6, .3, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: meta)

    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, np.zeros((100, 100, 3), dtype=np.uint8)) if self.index <= 6 else (False, None)
        def release(self): pass

    monkeypatch.setattr(cli.cv2, 'VideoCapture', Capture)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv']


def test_full_cli_with_both_real_tracking_engines(experiment):
    cli, root, args = experiment
    source = root / 'outputs/fusion/detection_fusion.csv'
    before = source.read_bytes()
    assert cli.main(args) == 0
    for backend in ('bytetrack', 'botsort'):
        directory = root / 'outputs/tracking' / backend
        frames = pd.read_csv(directory / 'tracking_results.csv')
        assert len(frames) == 6
        assert frames.observed.sum() == 3
        assert frames.loc[4, 'track_state'] == 'LOST'
        assert frames.loc[5, 'track_id'] == frames.loc[3, 'track_id']
        assert frames.loc[4, 'confidence'] == 0
        tracks = pd.read_csv(directory / 'tracks.csv')
        assert {'track_id', 'bbox', 'confidence', 'frame_index', 'timestamp', 'velocity', 'age',
                'number_of_hits', 'number_of_misses', 'track_state'} <= set(tracks.columns)
        assert len(list((directory / 'visualizations').glob('*.jpg'))) == 6
        metrics = pd.read_csv(directory / 'tracking_metrics.csv')
        assert metrics.loc[metrics.scope == 'overall', 'track_coverage'].iloc[0] == .5
        assert metrics.id_switches.isna().all()
        assert json.loads((directory / 'experiment_metadata.json').read_text())['frame_alignment'] == 'PASS'
    assert source.read_bytes() == before


def test_missing_fusion_frame_rejected_before_tracking(experiment):
    cli, root, args = experiment
    path = root / 'outputs/fusion/detection_fusion.csv'
    table = pd.read_csv(path).iloc[:-1]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='sampling timeline'):
        cli.main(args)
    assert not (root / 'outputs/tracking').exists()


def test_loader_returns_unselected_accepted_candidates(experiment):
    _, root, _ = experiment
    path = root / 'outputs/fusion/detection_fusion.csv'
    table = pd.read_csv(path)
    candidates = json.loads(table.loc[0, 'candidates_json'])
    candidates.append(dict(accepted=True, selected=False,
                          fused=dict(bbox=[65, 10, 95, 60], confidence=.8, class_id=0, class_name='person')))
    table.loc[0, 'candidates_json'] = json.dumps(candidates)
    table.to_csv(path, index=False)
    assert len(load_fused_frames(path, 'or')[('v', 0)]['detections']) == 2


@pytest.mark.parametrize('mutation', ['duplicate', 'status', 'timestamp', 'flags'])
def test_fusion_input_corruption_rejected(experiment, mutation):
    _, root, _ = experiment
    path = root / 'outputs/fusion/detection_fusion.csv'
    table = pd.read_csv(path)
    if mutation == 'duplicate':
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif mutation == 'status':
        table.loc[0, 'detection_status'] = 'MISSED'
    elif mutation == 'timestamp':
        table.loc[0, 'timestamp_sec'] = -1
    else:
        candidates = json.loads(table.loc[0, 'candidates_json'])
        candidates[0]['accepted'] = False
        table.loc[0, 'candidates_json'] = json.dumps(candidates)
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_fused_frames(path, 'or')
