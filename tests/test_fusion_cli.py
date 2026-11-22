import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_detection_fusion.py'
    spec = importlib.util.spec_from_file_location('fusion_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path / 'configs').mkdir()
    config = tmp_path / 'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'))))
    pd.DataFrame([dict(video_id='v', split='train', filename='video.mp4')]).to_csv(tmp_path / 'manifest.csv', index=False)
    metadata = VideoMetadata('video.mp4', 'video.mp4', 64, 64, 20, 4, .2, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: metadata)

    class Capture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            self.index += 1
            return (True, np.zeros((64, 64, 3), dtype=np.uint8)) if self.index <= 4 else (False, None)
        def release(self):
            pass

    monkeypatch.setattr(cli.cv2, 'VideoCapture', Capture)
    for name, positives in [('yolo', {0, 1}), ('rtdetr', {0, 2})]:
        path = tmp_path / 'outputs' / name
        path.mkdir(parents=True)
        rows = []
        for index in range(4):
            positive = index in positives
            rows.append(dict(video_id='v', split='train', frame_index=index, timestamp_sec=index / 20,
                             detection_status='DETECTED' if positive else 'MISSED', class_id=0,
                             class_name='person', confidence=.8 if positive else None,
                             x1=10, y1=10, x2=50, y2=50))
        pd.DataFrame(rows).to_csv(path / f'{name}_detections.csv', index=False)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv']


def test_end_to_end_cli_artifacts_and_ablation(experiment):
    cli, root, args = experiment
    original = (root / 'outputs/yolo/yolo_detections.csv').read_bytes()
    assert cli.main(args) == 0
    output = root / 'outputs/fusion'
    rows = pd.read_csv(output / 'detection_fusion.csv')
    assert len(rows) == 16
    assert rows.groupby('fusion_strategy').size().to_dict() == dict.fromkeys(
        ['or', 'and', 'confidence_weighted', 'weighted_box_fusion'], 4)
    metrics = pd.read_csv(output / 'fusion_metrics.csv')
    overall = metrics[metrics.scope == 'overall'].set_index('method')
    assert overall.loc['YOLO', 'detection_rate'] == .5
    assert overall.loc['RT-DETR', 'detection_rate'] == .5
    assert overall.loc['YOLO+RT-DETR:or', 'detection_rate'] == .75
    assert overall.loc['YOLO+RT-DETR:and', 'detection_rate'] == .25
    assert len(list((output / 'visualizations').glob('*.jpg'))) == 4
    assert cli.frame_filename('v', 2) == 'v_frame_00000002.jpg'
    meta = json.loads((output / 'experiment_metadata.json').read_text())
    assert meta['total_frames'] == 4
    assert meta['input_frame_alignment'] == 'PASS'
    assert meta['source_csvs']['yolo']['sha256']
    assert (root / 'outputs/yolo/yolo_detections.csv').read_bytes() == original


def test_cli_rejects_alignment_errors_before_outputs(experiment):
    cli, root, args = experiment
    path = root / 'outputs/rtdetr/rtdetr_detections.csv'
    table = pd.read_csv(path)
    table.loc[0, 'timestamp_sec'] = 1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='mismatch'):
        cli.main(args)
    assert not (root / 'outputs/fusion/detection_fusion.csv').exists()


def test_cli_strategy_override_and_no_visualizations(experiment):
    cli, root, args = experiment
    assert cli.main([*args, '--strategy', 'and', '--no-visualizations']) == 0
    rows = pd.read_csv(root / 'outputs/fusion/detection_fusion.csv')
    assert set(rows.loc[rows.is_default_strategy, 'fusion_strategy']) == {'and'}
    assert not list((root / 'outputs/fusion/visualizations').glob('*.jpg'))
