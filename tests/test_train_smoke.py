# tests/test_train_smoke.py
from pathlib import Path
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]

def test_train_smoke(tmp_path):
    # Copy sample data into a temp folder (to prove we can read from anywhere)
    data_dir = tmp_path / "data" / "sample"
    data_dir.mkdir(parents=True)
    src_csv = ROOT / "data/sample/customers.csv"
    (data_dir / "customers.csv").write_text(src_csv.read_text())

    # Ensure child process can import src-layout packages
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    # Run trainer from repo root; artifacts must land in ROOT/outputs/
    cmd = [
        sys.executable, "-m", "training.train",
        "--data", str(data_dir / "customers.csv"),
        "--target", "churn",
        "--model", "rf",
        # optional: you can pass just a filename; it will be placed in ./outputs/
        "--out", "model.joblib",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    # Assertions inside repo outputs
    model_path = ROOT / "outputs" / "model.joblib"
    metrics_path = ROOT / "outputs" / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics
