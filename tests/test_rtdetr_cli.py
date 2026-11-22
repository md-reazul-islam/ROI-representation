import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from module1.detection.base import Detection


def test_cli_writes_detections_metrics_and_visualizations(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_rtdetr_baseline.py'
    spec = importlib.util.spec_from_file_location('rtdetr_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    video = tmp_path / 'video.mp4'
    video.touch()
    cfg = {'_project_root': str(tmp_path), 'logging': {'log_dir': 'logs', 'level': 'INFO'}}
    monkeypatch.setattr(cli, 'load_config', lambda _: cfg)
    manifest = SimpleNamespace(
        df=pd.DataFrame([{'video_id': 'v'}]),
        to_records=lambda: [SimpleNamespace(video_id='v', split='train')],
        resolve_video_path=lambda root, row: video,
    )
    monkeypatch.setattr(cli.DatasetManifest, 'load', lambda _: manifest)
    metadata = SimpleNamespace(native_fps=20, width=16, height=16, frame_count=2, duration_sec=0.1)
    monkeypatch.setattr(cli, 'VideoReader', lambda _: SimpleNamespace(metadata=lambda: metadata))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    samples = [SimpleNamespace(frame=frame, frame_index=i, timestamp_sec=i / 20) for i in range(2)]
    monkeypatch.setattr(cli, 'FrameSampler', lambda *args: SimpleNamespace(iter_sampled=lambda: iter(samples)))

    def detect(self, frame, frame_index=0, timestamp=0.0):
        if frame_index == 1:
            return []
        return [Detection((1, 1, 10, 10), 0.8, 0, 'person', frame_index, timestamp)]

    monkeypatch.setattr(cli.RTDETRDetector, 'detect', detect)
    monkeypatch.setattr(sys, 'argv', [str(script), '--manifest', 'dataset_manifest.csv', '--device', 'cpu'])
    yolo = tmp_path / 'outputs/yolo'
    yolo.mkdir(parents=True)
    pd.DataFrame([dict(video_id='v', frame_index=i, timestamp_sec=i / 20) for i in range(2)]).to_csv(
        yolo / 'yolo_detections.csv', index=False)
    assert cli.main() == 0
    output = tmp_path / 'outputs/rtdetr'
    detections = pd.read_csv(output / 'rtdetr_detections.csv')
    assert list(detections.detection_status) == ['DETECTED', 'MISSED']
    metrics = pd.read_csv(output / 'rtdetr_metrics.csv')
    assert metrics.iloc[0].detection_rate == 0.5
    assert metrics.iloc[0].longest_consecutive_miss == 1
    assert len(list((output / 'visualizations').glob('*.jpg'))) == 2
    assert (output / 'metadata/experiment_metadata.json').exists()
