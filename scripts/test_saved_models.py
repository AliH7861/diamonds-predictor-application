"""Reload saved production models and make one prediction without training anything."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.prediction import load_best_model as load_classifier  # noqa: E402
from src.classification.prediction import predict_diamonds  # noqa: E402
from src.regression.prediction import load_best_model as load_regressor  # noqa: E402
from src.regression.prediction import predict_prices  # noqa: E402

SAMPLE = {
    "carat": 1.0,
    "cut": "Ideal",
    "color": "G",
    "clarity": "VS1",
    "depth": 61.5,
    "table": 57.0,
    "x": 6.45,
    "y": 6.43,
    "z": 3.96,
}


def test_classification(directory: Path) -> dict:
    bundle = load_classifier(directory)
    record = {key: value for key, value in SAMPLE.items() if key != "clarity"}
    if "price" in bundle["metadata"]["required_inputs"]:
        record["price"] = 6000.0
    return predict_diamonds(bundle, record)[0]


def test_regression(directory: Path) -> dict:
    return predict_prices(load_regressor(directory), SAMPLE)[0]


def test_regression_ann(directory: Path) -> dict:
    """Load the separately saved ANN through the same stable regression interface."""
    return test_regression(directory)


def run_checks(models_dir: Path, skip_benchmark: bool, smoke: bool) -> None:
    """Load default artifacts and separately saved algorithm artifacts."""
    classification_dir = models_dir / "classification"
    regression_dir = models_dir / "regression"
    production_ready = (classification_dir / "metadata.json").is_file() and (
        regression_dir / "pipeline.joblib"
    ).is_file()
    if smoke or not production_ready:
        classification_dir /= "smoke_test"
        regression_dir /= "smoke_test"
        print("Using smoke-test artifacts because full production models are not available.")

    required = [classification_dir / "metadata.json", regression_dir / "pipeline.joblib"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Saved models are missing. First run: python scripts/train_all_models.py --smoke\n"
            + "\n".join(missing)
        )

    if smoke or not production_ready:
        checks = [
            ("Classification smoke winner", lambda: test_classification(classification_dir)),
            ("Regression smoke winner", lambda: test_regression(regression_dir)),
            ("Classification smoke ANN", lambda: test_classification(classification_dir / "ann")),
            ("Regression smoke ANN", lambda: test_regression_ann(regression_dir / "ann")),
        ]
    else:
        checks = [
            ("Classification default", lambda: test_classification(classification_dir)),
            ("Regression selected model", lambda: test_regression(regression_dir)),
        ]
        if not skip_benchmark:
            checks.extend(
                [
                    (
                        "Classification ANN",
                        lambda: test_classification(classification_dir / "ann"),
                    ),
                    (
                        "Regression benchmark",
                        lambda: test_regression(regression_dir / "benchmark_winner"),
                    ),
                ]
            )
    for label, check in checks:
        result = check()
        print(f"{label:.<28} PASS  {json.dumps(result)}")
    print("\nSAVED MODEL STATUS: SUCCESS")


def run_in_docker(skip_benchmark: bool, smoke: bool) -> None:
    """Use the same Linux environment that created the serialized artifacts."""
    if (
        subprocess.run(["docker", "info"], capture_output=True, text=True, check=False).returncode
        != 0
    ):
        raise SystemExit("Docker Desktop is not running. Start it, then retry.")
    if (
        subprocess.run(
            ["docker", "image", "inspect", "diamond-training"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "train_all_models.py"), "--smoke"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "-e",
        "CUDA_VISIBLE_DEVICES=-1",
        "-v",
        f"{PROJECT_ROOT}:/workspace",
        "diamond-training",
        "scripts/test_saved_models.py",
        "--inside-docker",
        "--models-dir",
        "/workspace/models",
    ]
    if skip_benchmark:
        command.append("--skip-benchmark")
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Check only the two primary course ANN models",
    )
    parser.add_argument("--smoke", action="store_true", help="Use saved smoke-test artifacts")
    parser.add_argument("--inside-docker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_docker:
        run_checks(args.models_dir, args.skip_benchmark, args.smoke)
    else:
        run_in_docker(args.skip_benchmark, args.smoke)


if __name__ == "__main__":
    main()
