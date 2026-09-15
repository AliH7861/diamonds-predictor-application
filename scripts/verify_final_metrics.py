"""Re-evaluate saved full models on the held-out test rows without training."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_classification() -> dict:
    """Evaluate the saved primary ANN and benchmark classifier."""
    from src.classification.evaluation import evaluate_model
    from src.classification.preprocessing import prepare_classification_data
    from src.classification.prediction import load_best_model

    prepared = prepare_classification_data(PROJECT_ROOT / "data" / "raw" / "diamonds.csv")
    test_rows = prepared["diamonds"].iloc[prepared["test_indices"]]
    results = {}

    for role, directory in (
        ("primary", PROJECT_ROOT / "models" / "classification"),
        ("benchmark_winner", PROJECT_ROOT / "models" / "classification" / "benchmark_winner"),
    ):
        bundle = load_best_model(directory)
        features = bundle["metadata"]["features"]
        matrix = bundle["preprocessor"].transform(test_rows[features]).astype(np.float32)
        if bundle["metadata"]["model_type"] == "ANN":
            predicted = bundle["model"].predict(matrix, verbose=0).argmax(axis=1)
        else:
            predicted = bundle["model"].predict(matrix)
        metrics = evaluate_model(
            prepared["y_test"],
            predicted,
            f"Experiment {bundle['metadata']['experiment']}",
            bundle["metadata"]["model_name"].split(" - ")[-1],
        )
        results[role] = metrics
        name = "best_test_metrics.json" if role == "primary" else "benchmark_winner_test_metrics.json"
        output = PROJECT_ROOT / "outputs" / "classification" / "metrics" / name
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"Classification {role}: Accuracy={metrics['Accuracy']:.2f}%  "
            f"Macro F1={metrics['Macro_F1']:.2f}%"
        )

    return results


def verify_regression() -> dict:
    """Evaluate the saved primary ANN and benchmark price regressor."""
    from src.regression.prediction import load_best_model
    from src.regression.preprocessing import prepare_regression_data
    from src.regression.training import evaluate_on_test

    prepared = prepare_regression_data(PROJECT_ROOT / "data" / "raw" / "diamonds.csv")
    results = {}

    for role, directory in (
        ("primary", PROJECT_ROOT / "models" / "regression"),
        ("benchmark_winner", PROJECT_ROOT / "models" / "regression" / "benchmark_winner"),
    ):
        run = load_best_model(directory)
        _, metrics = evaluate_on_test(run, prepared)
        results[role] = metrics
        name = "best_test_metrics.json" if role == "primary" else "benchmark_winner_test_metrics.json"
        output = PROJECT_ROOT / "outputs" / "regression" / "metrics" / name
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"Regression {role}: MAE=${metrics['MAE']:,.2f}  "
            f"R2={metrics['R2']:.4f}"
        )

    return results


def run_inside_docker() -> None:
    """Write fresh test reports from every saved full supervised model."""
    classification = verify_classification()
    regression = verify_regression()
    combined = {"classification": classification, "regression": regression}
    output = PROJECT_ROOT / "outputs" / "final_model_test_metrics.json"
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nFINAL MODEL METRICS: VERIFIED\nCombined report: {output}")


def run_with_docker() -> None:
    """Use the training image to load TensorFlow and XGBoost consistently."""
    if subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    ).returncode != 0:
        raise SystemExit("Docker Desktop is not running. Start it, then retry.")
    if subprocess.run(
        ["docker", "image", "inspect", "diamond-training"],
        capture_output=True,
        check=False,
    ).returncode != 0:
        raise SystemExit(
            "The training image is missing. Run "
            "'python scripts/train_all_models.py --smoke' once, then retry."
        )
    subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "python",
            "-e", "CUDA_VISIBLE_DEVICES=-1",
            "-v", f"{PROJECT_ROOT}:/workspace", "-w", "/workspace",
            "diamond-training", "scripts/verify_final_metrics.py", "--inside-docker",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-docker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_inside_docker() if args.inside_docker else run_with_docker()


if __name__ == "__main__":
    main()
