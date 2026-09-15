"""Train XGBoost, ANN, and Random Forest for both maintained ML tasks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Force TensorFlow to remain on CPU before it is imported. The training container
# does not need CUDA, and probing an unavailable GPU caused the previous crashes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "void")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALGORITHMS = ("XGBoost", "ANN", "Random Forest")
REGRESSION_ALGORITHMS = ("XGBoost", "ANN", "RandomForest")
REGRESSION_FEATURE_SETS = ("CURRENT_BASELINE", "HUMAN_ONLY", "HUMAN_PLUS_RAW")


def _print_classification_score(metrics: dict) -> None:
    """Print the complete validation scorecard after one clarity candidate."""
    print(
        "    Validation: "
        f"Accuracy={metrics['Accuracy']:.2f}%  "
        f"Macro Precision={metrics['Macro_Precision']:.2f}%  "
        f"Macro Recall={metrics['Macro_Recall']:.2f}%  "
        f"Macro F1={metrics['Macro_F1']:.2f}%  "
        f"Balanced Accuracy={metrics['Balanced_Accuracy']:.2f}%",
        flush=True,
    )


def _print_regression_score(metrics: dict) -> None:
    """Print the main dollar and goodness-of-fit validation metrics."""
    print(
        "    Validation: "
        f"MAE=${metrics['MAE']:,.2f}  RMSE=${metrics['RMSE']:,.2f}  "
        f"R2={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.2f}%  "
        f"Within 10%={metrics['Within_10']:.2f}%",
        flush=True,
    )


def _initialize_cpu_tensorflow() -> None:
    """Initialize TensorFlow once without printing irrelevant CUDA probe messages."""
    saved_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        import tensorflow as tf
    finally:
        os.dup2(saved_stderr, 2)
        os.close(null_stderr)
        os.close(saved_stderr)
    tf.config.set_visible_devices([], "GPU")
    tf.get_logger().setLevel("ERROR")


def _training_data_path(smoke: bool) -> Path:
    """Use the real CSV for full training and a representative sample for smoke tests."""
    source = PROJECT_ROOT / "data" / "raw" / "diamonds.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if not smoke:
        return source

    import pandas as pd

    destination = Path("/tmp/diamonds_smoke.csv")
    frame = pd.read_csv(source)
    frame.sample(min(1500, len(frame)), random_state=42).to_csv(destination, index=False)
    return destination


def train_classification(data_path: Path, smoke: bool) -> dict:
    """Train nine clarity candidates, select by validation Macro F1, and save winners."""
    import pandas as pd

    from src.classification.preprocessing import prepare_classification_data
    from src.classification.prediction import load_best_model, predict_diamonds, save_best_model
    from src.classification.training import evaluate_on_test, train_candidate

    print("\nCLASSIFICATION: XGBoost, ANN, Random Forest", flush=True)
    prepared = prepare_classification_data(data_path)
    runs = []
    for experiment in (1, 2, 3):
        for algorithm in ALGORITHMS:
            print(f"  Training Experiment {experiment} / {algorithm}...", flush=True)
            run = train_candidate(prepared, experiment, algorithm, smoke)
            runs.append(run)
            _print_classification_score(run["metrics"])

    best = max(runs, key=lambda run: run["metrics"]["Macro_F1"])
    best_ann = max(
        (run for run in runs if run["algorithm"] == "ANN"),
        key=lambda run: run["metrics"]["Macro_F1"],
    )
    output_dir = PROJECT_ROOT / "models" / "classification"
    report_dir = PROJECT_ROOT / "outputs" / "classification" / "metrics"
    if smoke:
        output_dir /= "smoke_test"
        report_dir /= "smoke_test"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([run["metrics"] for run in runs]).to_csv(
        report_dir / "model_comparison.csv", index=False
    )

    # The assignment requires an ANN result, so the strongest ANN is the primary
    # deployable artifact. The strongest model of any family remains available as
    # a transparent benchmark comparison.
    for run, directory in (
        (best_ann, output_dir),
        (best_ann, output_dir / "ann"),
        (best, output_dir / "benchmark_winner"),
    ):
        name = f"Experiment {run['experiment']} - {run['algorithm']}"
        save_best_model(
            run["model"], run["model_type"], run["preprocessor"], run["experiment"],
            name, run["metrics"], directory,
        )
    test_metrics = evaluate_on_test(best_ann, prepared)
    (report_dir / "best_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    benchmark_test_metrics = evaluate_on_test(best, prepared)
    (report_dir / "benchmark_winner_test_metrics.json").write_text(
        json.dumps(benchmark_test_metrics, indent=2), encoding="utf-8"
    )

    sample = {
        "carat": 1.0, "cut": "Ideal", "color": "G", "depth": 61.5,
        "table": 57.0, "x": 6.45, "y": 6.43, "z": 3.96, "price": 6000.0,
    }
    prediction = predict_diamonds(load_best_model(output_dir), sample)[0]
    print(
        f"  Saved course model: Experiment {best_ann['experiment']} / ANN",
        flush=True,
    )
    print(f"  Benchmark winner: Experiment {best['experiment']} / {best['algorithm']}", flush=True)
    print(f"  Reloaded ANN prediction: {prediction['clarity_family']}", flush=True)
    return {
        "best": best,
        "best_ann": best_ann,
        "test_metrics": test_metrics,
        "benchmark_test_metrics": benchmark_test_metrics,
    }


def train_regression(data_path: Path, smoke: bool) -> dict:
    """Train nine price candidates, select by validation MAE, and save winners."""
    import pandas as pd

    from src.regression.prediction import load_best_model, predict_prices, save_best_model
    from src.regression.preprocessing import prepare_regression_data
    from src.regression.training import evaluate_on_test, train_model

    print("\nREGRESSION: XGBoost, ANN, Random Forest", flush=True)
    prepared = prepare_regression_data(data_path)
    runs = []
    for feature_set in REGRESSION_FEATURE_SETS:
        for algorithm in REGRESSION_ALGORITHMS:
            display_name = "Random Forest" if algorithm == "RandomForest" else algorithm
            print(f"  Training {feature_set} / {display_name}...", flush=True)
            run = train_model(prepared, feature_set, algorithm, smoke)
            runs.append(run)
            _print_regression_score(run["metrics"])

    best = min(runs, key=lambda run: run["metrics"]["MAE"])
    best_ann = min(
        (run for run in runs if run["algorithm"] == "ANN"),
        key=lambda run: run["metrics"]["MAE"],
    )
    output_dir = PROJECT_ROOT / "models" / "regression"
    report_dir = PROJECT_ROOT / "outputs" / "regression" / "metrics"
    if smoke:
        output_dir /= "smoke_test"
        report_dir /= "smoke_test"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([run["metrics"] for run in runs]).to_csv(
        report_dir / "model_comparison.csv", index=False
    )
    save_best_model(
        best_ann,
        output_dir,
        provenance={"role": "course-required ANN", "selected_by": "validation MAE among ANNs"},
    )
    save_best_model(best_ann, output_dir / "ann", provenance={"selected_by": "validation MAE"})
    save_best_model(
        best,
        output_dir / "benchmark_winner",
        provenance={"role": "algorithm benchmark", "selected_by": "validation MAE"},
    )
    _, test_metrics = evaluate_on_test(best_ann, prepared)
    (report_dir / "best_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    _, benchmark_test_metrics = evaluate_on_test(best, prepared)
    (report_dir / "benchmark_winner_test_metrics.json").write_text(
        json.dumps(benchmark_test_metrics, indent=2), encoding="utf-8"
    )

    sample = {
        "carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS1",
        "depth": 61.5, "table": 57.0, "x": 6.45, "y": 6.43, "z": 3.96,
    }
    prediction = predict_prices(load_best_model(output_dir), sample)[0]
    print(f"  Saved course model: {best_ann['feature_set']} / ANN", flush=True)
    print(f"  Benchmark winner: {best['feature_set']} / {best['algorithm']}", flush=True)
    print(f"  Reloaded ANN prediction: ${prediction['predicted_price']:,.2f}", flush=True)
    return {
        "best": best,
        "best_ann": best_ann,
        "test_metrics": test_metrics,
        "benchmark_test_metrics": benchmark_test_metrics,
    }


def train_inside_docker(smoke: bool) -> None:
    """Run both complete workflows inside the CPU-only training container."""
    _initialize_cpu_tensorflow()
    data_path = _training_data_path(smoke)
    train_classification(data_path, smoke)
    train_regression(data_path, smoke)
    from src.clustering.pipeline import run_buyer_segmentation

    print("\nCLUSTERING: K-Means buyer segmentation", flush=True)
    run_buyer_segmentation(data_path=data_path, smoke=smoke)
    print("\nALL MODEL TRAINING AND RELOAD CHECKS COMPLETE", flush=True)


def run_with_docker(smoke: bool) -> None:
    """Build the reproducible CPU image and mount the project for saved outputs."""
    try:
        check = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise SystemExit("Docker is not installed or is not available on PATH.") from error
    if check.returncode != 0:
        raise SystemExit("Docker Desktop is not running. Start it, then run this command again.")

    subprocess.run(["docker", "build", "-t", "diamond-project", "."], cwd=PROJECT_ROOT, check=True)
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.train", "-t", "diamond-training", "."],
        cwd=PROJECT_ROOT,
        check=True,
    )
    command = [
        "docker", "run", "--rm", "--cpus", "4", "--memory", "8g",
        "-e", "CUDA_VISIBLE_DEVICES=-1", "-e", "NVIDIA_VISIBLE_DEVICES=void",
        "-v", f"{PROJECT_ROOT}:/workspace", "diamond-training",
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-docker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help="Train tiny CPU models on 1,500 rows")
    args = parser.parse_args()
    if args.inside_docker:
        train_inside_docker(args.smoke)
    else:
        run_with_docker(args.smoke)


if __name__ == "__main__":
    main()
