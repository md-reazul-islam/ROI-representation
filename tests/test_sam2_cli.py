import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from module1.data.manifest import sha256_file
from module1.data.video import VideoMetadata
from module1.segmentation import SAM2VideoBackend
from module1.segmentation.inputs import load_mask, load_sam2_table
from module1.segmentation.sam2 import TemporalSAM2Segmenter
from test_pose_cli import experiment as pose_experiment
from test_sam2 import FakeVideoPredictor, square


@pytest.fixture
def sam2_experiment(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/"scripts/run_sam2_segmentation.py"
    spec = importlib.util.spec_from_file_location("sam2_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path/"configs").mkdir()
    (tmp_path/"data/Dataset").mkdir(parents=True)
    (tmp_path/"data/Dataset/v.mp4").write_bytes(b"fake-video-for-unit-test")
    config = tmp_path/"configs/module1.yaml"
    config.write_text(yaml.safe_dump(dict(
        processing=dict(target_fps=20),
        logging=dict(log_dir="logs", level="INFO"),
        sam2=dict(enabled=False, chunk_frames=3, save_annotated_frames=False))))
    pd.DataFrame([dict(video_id="v", filename="v.mp4", split="train")]).to_csv(
        tmp_path/"manifest.csv", index=False)
    recovery = tmp_path/"outputs/recovery"
    recovery.mkdir(parents=True)
    rows = []
    for index, status in enumerate(("DETECTED", "MISSED", "RECOVERED")):
        rows.append(dict(video_id="v", split="train", frame_index=index, timestamp_sec=index/20,
                         final_status=status,
                         final_infant_bbox=json.dumps([10, 10, 90, 90]) if index != 1 else "",
                         final_infant_confidence=.9 if index != 1 else 0.,
                         track_id=1, detector_status="DETECTED" if index == 0 else "MISSED",
                         recovered_status=("NOT_NEEDED", "REJECTED", "ACCEPTED")[index]))
    pd.DataFrame(rows).to_csv(recovery/"recovery_results.csv", index=False)
    meta = VideoMetadata("v.mp4", "v.mp4", 100, 100, 20., 3, .15, "mock", True)
    monkeypatch.setattr(cli.VideoReader, "metadata", lambda self: meta)

    class Capture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            self.index += 1
            return (True, np.zeros((100, 100, 3), np.uint8)) if self.index <= 3 else (False, None)
        def release(self):
            pass
    monkeypatch.setattr(cli.cv2, "VideoCapture", Capture)
    return cli, tmp_path, ["--config", str(config), "--manifest", "manifest.csv"]


def test_disabled_run_keeps_all_recovery_frames_and_skips_model(sam2_experiment):
    cli, root, args = sam2_experiment
    assert cli.main(args, backend_factory=lambda *_: pytest.fail("model loaded while disabled")) == 0
    output = root/"outputs/sam2"
    table = pd.read_csv(output/"sam2_masks.csv")
    assert list(table.segmentation_status) == ["DISABLED"]*3
    assert list(table.frame_index) == [0, 1, 2]
    assert not list((output/"masks").glob("*.png"))
    metadata = json.loads((output/"metadata/experiment_metadata.json").read_text())
    assert metadata["run_status"] == "COMPLETE"
    assert metadata["ground_truth_status"] == "UNAVAILABLE"


def test_enabled_run_uses_one_box_prompt_and_temporal_propagation(sam2_experiment):
    cli, root, args = sam2_experiment
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    predictor = FakeVideoPredictor([mask]*3)
    factory = lambda config, project_root: SAM2VideoBackend(config, project_root, predictor=predictor)
    assert cli.main([*args, "--enable"], backend_factory=factory) == 0
    output = root/"outputs/sam2"
    records, metadata = load_sam2_table(output)
    assert metadata["mode"] == "SAM2_ENABLED" and metadata["run_status"] == "COMPLETE"
    assert [records[("v", i)]["segmentation_status"] for i in range(3)] == [
        "INITIALIZED", "PROPAGATED", "PROPAGATED"]
    assert predictor.prompts == [0] and predictor.propagations == [1]
    assert load_mask(output, records[("v", 1)]).shape == (100, 100)
    assert records[("v", 1)]["final_infant_bbox"] == ""
    assert records[("v", 2)]["frame_index"] == 2
    metrics = pd.read_csv(output/"sam2_metrics.csv")
    assert metrics.iloc[-1].reliable_mask_frames == 3
    assert metrics.iloc[-1].ground_truth_segmentation_accuracy == "NOT_AVAILABLE"


def test_model_load_failure_writes_failed_records_and_bbox_fallback(sam2_experiment):
    cli, root, args = sam2_experiment
    def fail_model(*_):
        raise FileNotFoundError("checkpoint unavailable")
    assert cli.main([*args, "--enable"], backend_factory=fail_model) == 0
    output = root/"outputs/sam2"
    records, metadata = load_sam2_table(output)
    assert metadata["run_status"] == "DEGRADED"
    assert all(row["segmentation_status"] == "FAILED" for row in records.values())
    assert records[("v", 0)]["final_infant_bbox"] == "[10, 10, 90, 90]"
    assert all(row["mask_path"] == "" for row in records.values())


def test_required_model_preflight_checks_model_without_creating_outputs(sam2_experiment):
    cli, root, args = sam2_experiment
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    calls = []

    def factory(config, project_root):
        calls.append((config.device, project_root))
        return SAM2VideoBackend(config, project_root, predictor=FakeVideoPredictor([mask]*3))

    assert cli.main([*args, "--enable", "--device", "cuda:0", "--validate-only",
                     "--require-model"], backend_factory=factory) == 0
    assert calls == [("cuda:0", root)]
    assert not (root/"outputs/sam2").exists()


def test_required_model_fails_fast_and_preserves_default_fallback(sam2_experiment):
    cli, root, args = sam2_experiment

    def fail_model(*_):
        raise FileNotFoundError("checkpoint unavailable")

    with pytest.raises(ValueError, match="needs SAM 2 enabled"):
        cli.main([*args, "--require-model"], backend_factory=fail_model)
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        cli.main([*args, "--enable", "--require-model"], backend_factory=fail_model)
    assert not (root/"outputs/sam2").exists()


def test_sam2_mask_loader_rejects_escape_and_wrong_original_resolution(sam2_experiment):
    cli, root, args = sam2_experiment
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    cli.main([*args, "--enable"], backend_factory=lambda cfg, project:
             SAM2VideoBackend(cfg, project, predictor=FakeVideoPredictor([mask]*3)))
    output = root/"outputs/sam2"
    records, _ = load_sam2_table(output)
    with pytest.raises(ValueError, match="original-frame"):
        load_mask(output, records[("v", 0)], expected_shape=(20, 20))
    csv_path = output/"sam2_masks.csv"
    table = pd.read_csv(csv_path, keep_default_na=False)
    table.loc[0, "mask_path"] = "../outside.png"
    table.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="escapes"):
        load_sam2_table(output)


def test_pose_cli_direct_and_sam2_masked_modes_with_fallback(pose_experiment):
    cli, root, args, backend = pose_experiment
    recovery_path = root/"outputs/recovery/recovery_results.csv"
    from module1.pose.inputs import load_recovery_frames
    recovery = load_recovery_frames(recovery_path)
    sam2 = root/"outputs/sam2"
    (sam2/"masks").mkdir(parents=True)
    (sam2/"metadata").mkdir()
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    cv2.imwrite(str(sam2/"masks/v_frame_00000000.png"), mask)
    cv2.imwrite(str(sam2/"masks/v_frame_00000001.png"), mask)
    output_rows = []
    for key in sorted(recovery):
        record = TemporalSAM2Segmenter._base(recovery[key])
        record["original_height"] = record["original_width"] = 100
        record["segmentation_status"] = ("INITIALIZED", "PROPAGATED", "FAILED")[key[1]]
        record["mask_path"] = f"masks/v_frame_{key[1]:08d}.png" if key[1] < 2 else None
        record["mask_quality"] = .9 if key[1] < 2 else None
        output_rows.append(record)
    pd.DataFrame(output_rows).to_csv(sam2/"sam2_masks.csv", index=False)
    (sam2/"metadata/experiment_metadata.json").write_text(json.dumps(dict(
        phase="PHASE_12_SAM2_TEMPORAL_SEGMENTATION", mode="SAM2_ENABLED",
        run_status="DEGRADED", source_hashes=dict(recovery=sha256_file(recovery_path)))))
    assert cli.main([*args, "--without-sam2", "--no-visualizations",
                     "--output-dir", "outputs/pose_direct"]) == 0
    assert cli.main([*args, "--sam2-dir", "outputs/sam2", "--no-visualizations",
                     "--output-dir", "outputs/pose_sam2"]) == 0
    direct = pd.read_csv(root/"outputs/pose_direct/pose_results.csv")
    enhanced = pd.read_csv(root/"outputs/pose_sam2/pose_results.csv")
    assert list(direct.frame_index) == list(enhanced.frame_index) == [0, 1, 2]
    assert list(enhanced.sam2_mask_used) == [True, False, False]
    assert enhanced.loc[1, "pose_status"] == "NO_BBOX"
    assert enhanced.loc[2, "final_infant_bbox"] == direct.loc[2, "final_infant_bbox"]
    assert np.array_equal(backend.calls[2][0][0, 0], [104, 116, 124])
    assert np.array_equal(backend.calls[2][0][30, 30], [0, 0, 0])
    assert np.all(backend.calls[3][0] == 0)
    metadata = json.loads((root/"outputs/pose_sam2/experiment_metadata.json").read_text())
    assert metadata["sam2_mode"] == "MASK_CONSTRAINED"
