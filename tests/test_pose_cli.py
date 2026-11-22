import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata
from module1.pose import Sapiens2PoseEstimator
from test_pose import FakeBackend


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/run_sapiens2_pose.py'
    spec = importlib.util.spec_from_file_location('pose_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/'configs').mkdir()
    source = tmp_path/'external/sapiens2'
    (source/'sapiens/pose/configs/_base_').mkdir(parents=True)
    (source/'sapiens/pose/configs/_base_/keypoints308.py').write_text('# test-only metadata')
    (tmp_path/'model.py').write_text('# test-only config')
    (tmp_path/'pose.safetensors').write_bytes(b'test-only weights; never loaded by a real model')
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'),
                                         pose=dict(model_config='model.py', checkpoint='pose.safetensors'))))
    pd.DataFrame([dict(video_id='v', filename='v.mp4', split='train')]).to_csv(tmp_path/'manifest.csv', index=False)
    recovery = tmp_path/'outputs/recovery'
    recovery.mkdir(parents=True)
    rows = []
    for i, status in enumerate(['DETECTED', 'MISSED', 'RECOVERED']):
        rows.append(dict(video_id='v', split='train', frame_index=i, timestamp_sec=i/20,
                         final_status=status, final_infant_bbox=json.dumps([0, 0, 100, 100]) if i != 1 else '',
                         final_infant_confidence=.9 if i != 1 else 0, track_id=1,
                         detector_status='DETECTED' if i == 0 else 'MISSED',
                         recovered_status=['NOT_NEEDED', 'REJECTED', 'ACCEPTED'][i]))
    path = recovery/'recovery_results.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    meta = VideoMetadata('v.mp4', 'v.mp4', 100, 100, 20, 3, .15, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: meta)
    backend = FakeBackend()
    monkeypatch.setattr(cli, 'Sapiens2PoseEstimator', lambda *args, **kwargs: Sapiens2PoseEstimator(backend=backend))

    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, np.zeros((100, 100, 3), np.uint8)) if self.index <= 3 else (False, None)
        def release(self): pass

    monkeypatch.setattr(cli.cv2, 'VideoCapture', Capture)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv'], backend


def test_cli_artifacts_alignment_and_no_box_fallback(experiment):
    cli, root, args, backend = experiment
    before = (root/'outputs/recovery/recovery_results.csv').read_bytes()
    assert cli.main(args) == 0
    output = root/'outputs/pose'
    rows = pd.read_csv(output/'pose_results.csv')
    assert list(rows.pose_status) == ['ESTIMATED', 'NO_BBOX', 'ESTIMATED']
    assert list(rows.keypoint_count) == [308, 0, 308]
    assert list(rows.timestamp) == [0, .05, .1]
    assert len(backend.calls) == 2
    assert len(list((output/'visualizations').glob('*.jpg'))) == 3
    assert len(list((output/'keypoints').glob('*.json'))) == 3
    for i in (0, 2):
        record = json.loads((output/'keypoints'/f'v_frame_{i:08d}.json').read_text())
        assert record['bbox'] == [0, 0, 100, 100]
        assert len(record['keypoints']) == 308 and record['keypoints'][0]['visibility'] is None
    missing = json.loads((output/'keypoints/v_frame_00000001.json').read_text())
    assert missing['bbox'] is None and missing['keypoints'] == []
    assert missing['pose_quality'] is None and missing['confidence'] is None
    meta = json.loads((output/'experiment_metadata.json').read_text())
    assert meta['run_status'] == 'COMPLETE' and meta['processed_frames'] == 3
    assert meta['ground_truth_status'] == 'UNAVAILABLE'
    assert (root/'outputs/recovery/recovery_results.csv').read_bytes() == before
    with pytest.raises(FileExistsError): cli.main(args)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'timestamp', 'split', 'box', 'status', 'id'])
def test_bad_recovery_stops_before_inference(experiment, mutation):
    cli, root, args, backend = experiment
    path = root/'outputs/recovery/recovery_results.csv'
    rows = pd.read_csv(path, keep_default_na=False)
    if mutation == 'missing': rows = rows.iloc[:-1]
    elif mutation == 'duplicate': rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif mutation == 'timestamp': rows.loc[0, 'timestamp_sec'] = 1
    elif mutation == 'split': rows.loc[0, 'split'] = 'test'
    elif mutation == 'box': rows.loc[1, 'final_infant_bbox'] = '[0,0,10,10]'
    elif mutation == 'status': rows.loc[2, 'recovered_status'] = 'REJECTED'
    elif mutation == 'id': rows.loc[0, 'video_id'] = '../bad'
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError): cli.main(args)
    assert not backend.calls and not (root/'outputs/pose').exists()


def test_validation_only_never_loads_model_or_writes_pose(experiment, monkeypatch):
    cli, root, args, backend = experiment
    monkeypatch.setattr(cli, 'Sapiens2PoseEstimator', lambda *a: pytest.fail('Model should not load'))
    assert cli.main([*args, '--validate-only']) == 0
    assert not (root/'outputs/pose').exists()


def test_partial_run_is_labelled(experiment):
    cli, root, args, backend = experiment
    assert cli.main([*args, '--max-frames', '1', '--no-visualizations']) == 0
    meta = json.loads((root/'outputs/pose/experiment_metadata.json').read_text())
    assert meta['run_status'] == 'PARTIAL' and meta['processed_frames'] == 1
    assert meta['visualization_count'] == 0


def test_track_change_resets_temporal_reference(experiment):
    cli, root, args, backend = experiment
    path = root/'outputs/recovery/recovery_results.csv'
    rows = pd.read_csv(path)
    rows.loc[2, 'track_id'] = 2
    rows.to_csv(path, index=False)
    cli.main([*args, '--no-visualizations'])
    record = json.loads((root/'outputs/pose/keypoints/v_frame_00000002.json').read_text())
    assert record['checks']['temporal_reason'] == 'NO_PREVIOUS_VALID_POSE'


def test_video_boundary_resets_temporal_reference(experiment):
    cli, root, args, backend = experiment
    path = root/'outputs/recovery/recovery_results.csv'
    rows = pd.read_csv(path)
    second = rows.copy()
    second['video_id'] = 'v2'
    pd.concat([rows, second], ignore_index=True).to_csv(path, index=False)
    pd.DataFrame([dict(video_id=v, filename=f'{v}.mp4', split='train') for v in ['v', 'v2']]).to_csv(root/'manifest.csv', index=False)
    cli.main([*args, '--no-visualizations'])
    record = json.loads((root/'outputs/pose/keypoints/v2_frame_00000000.json').read_text())
    assert record['checks']['temporal_reason'] == 'NO_PREVIOUS_VALID_POSE'


def test_missing_checkpoint_no_fake_outputs(experiment, monkeypatch):
    cli, root, args, backend = experiment
    monkeypatch.setattr(cli, 'Sapiens2PoseEstimator', Sapiens2PoseEstimator)
    with pytest.raises(FileNotFoundError): cli.main([*args, '--checkpoint', 'absent.safetensors'])
    assert not (root/'outputs/pose').exists()


def test_failed_model_run_is_not_reported_complete(experiment):
    cli, root, args, backend = experiment
    backend.xy = backend.xy[:17]
    with pytest.raises(ValueError, match='308'): cli.main(args)
    meta = json.loads((root/'outputs/pose/experiment_metadata.json').read_text())
    assert meta['run_status'] == 'FAILED'
    assert not (root/'outputs/pose/pose_results.csv').exists()


def test_nonfinite_points_preserved_as_null_with_failure_flag(experiment):
    cli, root, args, backend = experiment
    backend.xy[0, 0] = np.nan
    assert cli.main([*args, '--no-visualizations']) == 0
    record = json.loads((root/'outputs/pose/keypoints/v_frame_00000000.json').read_text())
    assert record['keypoints'][0]['x'] is None
    assert record['checks']['coordinate_validity'] == 'FAIL' and record['status'] == 'FLAGGED'
