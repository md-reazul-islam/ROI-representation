import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata
from test_roi import keypoint_names, points


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/run_anatomical_rois.py'
    spec = importlib.util.spec_from_file_location('roi_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/'configs').mkdir()
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'),
                                         roi=dict(save_annotated_frames=False))))
    pd.DataFrame([dict(video_id='v', filename='v.mp4', split='train')]).to_csv(tmp_path/'manifest.csv', index=False)
    pose = tmp_path/'outputs/pose'
    (pose/'keypoints').mkdir(parents=True)
    names = keypoint_names()
    metadata = dict(phase='PHASE_7_SAPIENS2_POSE', run_status='PARTIAL', keypoint_count=308,
                    keypoint_names=list(names))
    (pose/'experiment_metadata.json').write_text(json.dumps(metadata))
    payload = dict(video_id='v', split='train', frame_index=0, timestamp=0., source_final_status='DETECTED',
                   bbox=[0., 0., 100., 320.], source_final_confidence=.9, status='ESTIMATED',
                   keypoints=points(names), confidence=.9, pose_quality=.8, checks={})
    (pose/'keypoints/v_frame_00000000.json').write_text(json.dumps(payload))
    pd.DataFrame([dict(video_id='v', split='train', frame_index=0, timestamp=0., timestamp_sec=0.,
                       final_status='DETECTED', final_infant_bbox=json.dumps(payload['bbox']), final_infant_confidence=.9,
                       track_id=1, pose_status='ESTIMATED', confidence=.9, pose_quality=.8, keypoint_count=308,
                       keypoints_path='keypoints/v_frame_00000000.json')]).to_csv(pose/'pose_results.csv', index=False)
    meta = VideoMetadata('v.mp4', 'v.mp4', 100, 320, 20, 1, .05, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: meta)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv', '--no-visualizations']


def test_cli_writes_all_candidate_roi_rows_and_metadata(experiment):
    cli, root, args = experiment
    assert cli.main(args) == 0
    output = root/'outputs/roi'
    rows = pd.read_csv(output/'candidate_rois.csv')
    assert len(rows) == 17
    assert set(rows.roi_name) == set(__import__('module1.roi', fromlist=['ROI_NAMES']).ROI_NAMES)
    assert rows.anatomical_confidence.notna().any()
    meta = json.loads((output/'experiment_metadata.json').read_text())
    assert meta['run_status'] == 'COMPLETE' and meta['confidence_scoring_implemented'] is False
    assert meta['roi_scoring_implemented'] is False
    with pytest.raises(FileExistsError): cli.main(args)
