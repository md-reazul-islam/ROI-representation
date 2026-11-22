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
    script = Path(__file__).resolve().parents[1] / 'scripts/run_temporal_recovery.py'
    spec = importlib.util.spec_from_file_location('recovery_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path / 'configs').mkdir()
    config = tmp_path / 'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'),
                                         recovery=dict(max_recovery_frames=2, min_recovery_confidence=.1))))
    pd.DataFrame([dict(video_id='v', split='train', filename='v.mp4')]).to_csv(tmp_path / 'manifest.csv', index=False)
    fusion = tmp_path / 'outputs/fusion'
    tracking = tmp_path / 'outputs/tracking/bytetrack'
    fusion.mkdir(parents=True)
    tracking.mkdir(parents=True)
    frames, tracks = [], []
    for i in range(6):
        positive = i in (0, 4)
        candidates = [dict(accepted=True, selected=True, fused=dict(
            bbox=[20, 20, 60, 60], confidence=.9, class_id=0, class_name='person'))] if positive else []
        frames.append(dict(video_id='v', split='train', frame_index=i, timestamp_sec=i/20, fusion_strategy='or',
                           detection_status='DETECTED' if positive else 'MISSED', candidates_json=json.dumps(candidates)))
        tracks.append(dict(video_id='v', split='train', frame_index=i, timestamp_sec=i/20, fusion_strategy='or',
                           tracker='bytetrack', track_id=1, bbox=json.dumps([20, 20, 60, 60]),
                           velocity=json.dumps([0, 0]), track_state='CONFIRMED' if positive else 'LOST',
                           observed=positive, is_primary=True, confidence=.9 if positive else 0., reliability=.9))
    pd.DataFrame(frames).to_csv(fusion / 'detection_fusion.csv', index=False)
    pd.DataFrame(tracks).to_csv(tracking / 'tracking_results.csv', index=False)
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


def test_full_cli_artifacts_and_bounded_recovery(experiment):
    cli, root, args = experiment
    source = root / 'outputs/fusion/detection_fusion.csv'
    before = source.read_bytes()
    assert cli.main(args) == 0
    output = root / 'outputs/recovery'
    rows = pd.read_csv(output / 'recovery_results.csv')
    assert list(rows.final_status) == ['DETECTED', 'RECOVERED', 'RECOVERED', 'MISSED', 'DETECTED', 'RECOVERED']
    assert rows.loc[3, 'rejection_reason'] == 'RECOVERY_LIMIT_EXCEEDED'
    assert rows.loc[3, 'final_infant_confidence'] == 0
    assert rows.loc[3, 'final_infant_bbox'] != rows.loc[3, 'final_infant_bbox']
    assert len(list((output / 'visualizations').glob('*.jpg'))) == 6
    m = pd.read_csv(output / 'recovery_metrics.csv')
    overall = m[m.scope == 'overall'].iloc[0]
    assert overall.detector_miss_rate == pytest.approx(4/6)
    assert overall.recovered_miss_rate == pytest.approx(1/6)
    assert overall.temporal_recovery_rate == .75
    assert overall.longest_recovered_sequence == 2
    assert m.false_recovery_rate.isna().all()
    metadata = json.loads((output / 'experiment_metadata.json').read_text())
    assert metadata['total_frames'] == 6 and metadata['input_validation'] == 'PASS'
    assert metadata['pose_validation_status'] == 'UNAVAILABLE'
    assert source.read_bytes() == before


@pytest.mark.parametrize('mutation', ['missing', 'timestamp', 'split', 'duplicate', 'strategy', 'observed', 'provenance'])
def test_bad_tracking_inputs_stop_before_outputs(experiment, mutation):
    cli, root, args = experiment
    path = root / 'outputs/tracking/bytetrack/tracking_results.csv'
    table = pd.read_csv(path)
    if mutation == 'missing': table = table.iloc[:-1]
    elif mutation == 'timestamp': table.loc[1, 'timestamp_sec'] = 1
    elif mutation == 'split': table.loc[1, 'split'] = 'test'
    elif mutation == 'duplicate': table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif mutation == 'strategy': table.loc[1, 'fusion_strategy'] = 'and'
    elif mutation == 'observed': table.loc[1, 'observed'] = True
    else:
        (path.parent / 'experiment_metadata.json').write_text(json.dumps(dict(fusion_sha256='wrong', fusion_strategy='or')))
    table.to_csv(path, index=False)
    with pytest.raises(ValueError): cli.main(args)
    assert not (root / 'outputs/recovery').exists()


def test_pose_quality_can_veto_recovery(experiment):
    cli, root, args = experiment
    pd.DataFrame([dict(video_id='v', frame_index=1, timestamp_sec=.05, pose_quality=.1)]).to_csv(root/'pose.csv', index=False)
    assert cli.main([*args, '--pose-quality', 'pose.csv', '--no-visualizations']) == 0
    rows = pd.read_csv(root / 'outputs/recovery/recovery_results.csv')
    assert rows.loc[1, 'rejection_reason'] == 'POSE_VALIDATION_FAILED'
    assert rows.loc[2, 'pose_validation_status'] == 'UNAVAILABLE'
    assert rows.loc[2, 'recovered_status'] == 'ACCEPTED'


def test_bad_pose_timestamps_rejected(experiment):
    cli, root, args = experiment
    pd.DataFrame([dict(video_id='v', frame_index=1, timestamp_sec=10, pose_quality=.8)]).to_csv(root/'pose.csv', index=False)
    with pytest.raises(ValueError, match='align'):
        cli.main([*args, '--pose-quality', 'pose.csv'])
