"""Run a fast local validation of both supervised ANN pipelines."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.preprocessing import prepare_classification_data  # noqa: E402
from src.classification.training import train_ann_smoke as train_classification_ann  # noqa: E402
from src.regression.preprocessing import prepare_regression_data  # noqa: E402
from src.regression.training import train_ann_smoke as train_regression_ann  # noqa: E402
from tests.helpers import make_diamonds  # noqa: E402


def run_validation() -> bool:
    """Execute ten named checks and raise at the exact failing step."""
    print("=" * 48)
    print("PHASE 1 END-TO-END VALIDATION")
    print("=" * 48)

    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        data_path = work / "diamonds.csv"

        def step(number, label, action):
            try:
                result = action()
            except Exception:
                print(f"[{number}/10] {label:.<34} FAIL")
                raise
            print(f"[{number}/10] {label:.<34} PASS")
            return result

        step(1, "Dataset loading", lambda: make_diamonds().to_csv(data_path, index=False))
        classification = step(
            2, "Classification preprocessing", lambda: prepare_classification_data(data_path)
        )
        classification_data = classification["experiments"][1]
        step(
            3,
            "Classification model build",
            lambda: __import__("src.classification.model", fromlist=["build_ann"]).build_ann(
                classification_data["X_train"].shape[1]
            ),
        )
        classification_run = step(
            4,
            "Classification training",
            lambda: train_classification_ann(
                classification_data["X_train"],
                classification["y_train"],
                classification_data["X_valid"],
                classification["y_valid"],
                1,
            ),
        )

        def classification_reload():
            import tensorflow as tf

            path = work / "clarity_ann.keras"
            classification_run[0].save(path)
            loaded = tf.keras.models.load_model(path)
            before = classification_run[0].predict(classification_data["X_valid"][:2], verbose=0)
            after = loaded.predict(classification_data["X_valid"][:2], verbose=0)
            np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)

        step(5, "Classification save/reload", classification_reload)
        regression = step(6, "Regression preprocessing", lambda: prepare_regression_data(data_path))
        regression_data = regression["experiments"]["CURRENT_BASELINE"]["matrices"]
        step(
            7,
            "Regression model build",
            lambda: __import__("src.regression.model", fromlist=["build_ann"]).build_ann(
                regression_data["train"].shape[1]
            ),
        )
        regression_run = step(
            8,
            "Regression training",
            lambda: train_regression_ann(
                regression_data["train"],
                regression["targets"]["train"],
                regression_data["valid"],
                regression["targets"]["valid"],
                1,
            ),
        )

        def regression_reload():
            import tensorflow as tf

            path = work / "price_ann.keras"
            regression_run[0].save(path)
            loaded = tf.keras.models.load_model(path)
            before = regression_run[0].predict(regression_data["valid"][:2], verbose=0)
            after = loaded.predict(regression_data["valid"][:2], verbose=0)
            np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)

        step(9, "Regression save/reload", regression_reload)
        step(
            10,
            "Metrics generation",
            lambda: (classification_run[2]["Macro_F1"], regression_run[2]["MAE"]),
        )

    print("\nPHASE 1 STATUS: SUCCESS")
    return True


def run_in_docker() -> None:
    """Run this validator inside the project's Linux image."""
    image_name = "diamond-project"
    print("Windows blocked TensorFlow. Running the ANN validation in Docker...", flush=True)
    print("Updating the Docker image from the current source...", flush=True)
    subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd=PROJECT_ROOT,
        check=True,
    )

    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            image_name,
            "scripts/phase1_end_to_end.py",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    """Run natively, or use Docker for the specific Windows policy block."""
    try:
        import tensorflow  # noqa: F401
    except (ImportError, OSError) as error:
        policy_blocked = os.name == "nt" and "Application Control policy" in str(error)
        if policy_blocked:
            run_in_docker()
            return
        raise

    run_validation()


if __name__ == "__main__":
    main()
