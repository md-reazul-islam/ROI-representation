from pathlib import Path

from scripts.run_ablation_evaluation import main


def test_ablation_cli_creates_reports_for_current_annotation_free_project(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "reports"
    assert main(["--manifest", str(root / "dataset_manifest.csv"), "--output-dir", str(output), "--no-visualizations"]) == 0
    assert (output / "ablation_results.csv").exists()
    assert (output / "ablation_statistics.csv").exists()
    assert (output / "statistical_tests.csv").exists()
