import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata
from test_roi import keypoint_names, points
from module1.roi import ROI_NAMES


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/run_roi_quality.py'
    spec = importlib.util.spec_from_file_location('roi_quality_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/'configs').mkdir()
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'),
                                         roi_quality=dict(save_annotated_frames=False))))
    pd.DataFrame([dict(video_id='v', filename='v.mp4', split='train')]).to_csv(tmp_path/'manifest.csv', index=False)
    pose = tmp_path/'outputs/pose'
    (pose/'keypoints').mkdir(parents=True)
    names = keypoint_names()
    (pose/'experiment_metadata.json').write_text(json.dumps(dict(keypoint_count=308, keypoint_names=list(names), run_status='PARTIAL')))
    payload = dict(video_id='v', frame_index=0, timestamp=0., bbox=[0., 0., 100., 100.], status='ESTIMATED',
                   keypoints=points(names), confidence=.8, pose_quality=.7)
    (pose/'keypoints/v_frame_00000000.json').write_text(json.dumps(payload))
    roi = tmp_path/'outputs/roi'
    roi.mkdir(parents=True)
    roi_rows = []
    for name in ROI_NAMES:
        roi_rows.append(dict(video_id='v', split='train', frame_index=0, timestamp_sec=0., roi_name=name,
                             roi_status='AVAILABLE', geometry_type='POLYGON',
                             polygon=json.dumps([[10, 10], [60, 10], [60, 60], [10, 60]]),
                             bbox=json.dumps([10, 10, 60, 60]), centroid=json.dumps([35, 35]), width=50, height=50,
                             bbox_normalized=json.dumps([.1, .1, .6, .6]),
                             polygon_bbox_normalized=json.dumps([[.1, .1], [.6, .1], [.6, .6], [.1, .6]]),
                             centroid_bbox_normalized=json.dumps([.35, .35]), centroid_torso_normalized='',
                             support_keypoint_indices=json.dumps([0, 1, 2]), support_keypoint_names='[]',
                             anatomical_confidence=.8, unavailable_reason='', keypoints_path='keypoints/v_frame_00000000.json',
                             pose_confidence=.8))
    pd.DataFrame(roi_rows).to_csv(roi/'candidate_rois.csv', index=False)
    (roi/'experiment_metadata.json').write_text(json.dumps(dict(run_status='COMPLETE')))
    meta = VideoMetadata('v.mp4', 'v.mp4', 100, 100, 20, 1, .05, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: meta)

    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, np.full((100, 100, 3), 120, dtype=np.uint8)) if self.index == 1 else (False, None)
        def release(self): pass

    monkeypatch.setattr(cli.cv2, 'VideoCapture', Capture)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv', '--no-visualizations']


def test_quality_cli_outputs_all_regions_and_no_final_score(experiment):
    cli, root, args = experiment
    assert cli.main(args) == 0
    output = root/'outputs/roi_quality'
    rows = pd.read_csv(output/'roi_quality_features.csv')
    assert len(rows) == 17 and set(rows.roi_name) == set(ROI_NAMES)
    assert rows.pose_confidence.eq(.8).all()
    for field in ('visible_skin_proportion', 'occlusion_proxy', 'blur_score', 'illumination_score',
                  'saturation_score', 'swaddling_proxy'):
        assert rows[field].between(0, 1).all()
    assert rows.motion_magnitude.isna().all()
    assert 'final_roi_confidence' not in rows.columns
    metadata = json.loads((output/'experiment_metadata.json').read_text())
    assert metadata['confidence_scoring_implemented'] is False
    with pytest.raises(FileExistsError): cli.main(args)
