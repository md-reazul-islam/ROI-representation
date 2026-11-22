import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.video import VideoMetadata
from module1.roi import ROI_NAMES


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/run_roi_confidence.py'
    spec = importlib.util.spec_from_file_location('confidence_cli', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/'configs').mkdir()
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(logging=dict(log_dir='logs', level='INFO'), confidence=dict(save_annotated_frames=False))))
    pd.DataFrame([dict(video_id='v', filename='v.mp4', split='validation')]).to_csv(tmp_path/'manifest.csv', index=False)
    quality = tmp_path/'outputs/roi_quality'
    quality.mkdir(parents=True)
    quality_rows = []
    for index, name in enumerate(ROI_NAMES):
        q = .2 + index/20
        quality_rows.append(dict(video_id='v', split='validation', frame_index=0, timestamp_sec=0., roi_name=name,
                                 roi_status='AVAILABLE', pose_confidence=q, visible_skin_proportion=q,
                                 occlusion_proxy=1-q, blur_score=q, illumination_score=q, saturation_score=q,
                                 motion_magnitude=1-q, landmark_velocity=q, roi_centroid_displacement=q,
                                 roi_area_variation=1-q, temporal_consistency=q, swaddling_proxy=1-q,
                                 bbox_normalized=json.dumps([0., 0., q, q]), centroid_bbox_normalized=json.dumps([q/2, q/2]),
                                 width=10*q, height=10*q, anatomical_confidence=q,
                                 polygon=json.dumps([[0, 0], [10, 0], [10, 10], [0, 10]]),
                                 bbox=json.dumps([0, 0, 10, 10]), centroid=json.dumps([5, 5])))
    pd.DataFrame(quality_rows).to_csv(quality/'roi_quality_features.csv', index=False)
    (quality/'experiment_metadata.json').write_text(json.dumps(dict(run_status='COMPLETE')))
    meta = VideoMetadata('v.mp4', 'v.mp4', 20, 20, 20, 1, .05, 'mock', True)
    monkeypatch.setattr(cli.VideoReader, 'metadata', lambda self: meta)
    return cli, tmp_path, ['--config', str(config), '--manifest', 'manifest.csv', '--no-visualizations']


def test_cli_outputs_confidence_dynamic_selection_and_fusion(experiment):
    cli, root, args = experiment
    assert cli.main(args) == 0
    output = root/'outputs/confidence'
    confidence = pd.read_csv(output/'roi_confidence.csv')
    dynamic = pd.read_csv(output/'dynamic_roi.csv')
    fused = pd.read_csv(output/'fused_representation.csv')
    assert len(confidence) == 17 and confidence.alpha.sum() == pytest.approx(1.)
    assert confidence.selected.sum() == 3
    assert len(dynamic) == 4 and set(dynamic.selection_strategy) == {'top1', 'top2', 'topk', 'confidence_weighted_fusion'}
    assert dynamic[dynamic.selection_strategy == 'top1'].selection_count.iloc[0] == 1
    assert dynamic[dynamic.selection_strategy == 'top2'].selection_count.iloc[0] == 2
    assert dynamic[dynamic.selection_strategy == 'topk'].selection_count.iloc[0] == 3
    assert dynamic[dynamic.selection_strategy == 'confidence_weighted_fusion'].selection_count.iloc[0] == 17
    assert len(fused) == 4 and all(fused.representation_dimension == 19)
    assert 'final_confidence' not in confidence.columns
    metadata = json.loads((output/'experiment_metadata.json').read_text())
    assert metadata['selected_regime'] == 'equal_fixed'
    assert metadata['hr_calculated'] is False and metadata['rr_calculated'] is False
    with pytest.raises(FileExistsError): cli.main(args)


def test_cli_validation_tuned_and_separate_output(experiment):
    cli, root, args = experiment
    assert cli.main([*args, '--regime', 'validation_tuned', '--output-dir', 'outputs/confidence_validation']) == 0
    metadata = json.loads((root/'outputs/confidence_validation/experiment_metadata.json').read_text())
    assert metadata['selected_regime'] == 'validation_tuned'
    assert metadata['fit_metadata']['training_rows'] == 17
