"""Run buyer segmentation for K = 3, 5, 7, and 10 and save the selected model."""

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_inside_docker(smoke: bool) -> None:
    """Build the reproducible Linux image and save outputs through a workspace mount."""
    check = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if check.returncode != 0:
        raise SystemExit("Docker Desktop is not running. Start it and retry this command.")
    subprocess.run(
        ["docker", "build", "-t", "diamond-project", "."],
        cwd=PROJECT_ROOT,
        check=True,
    )
    command = [
        "docker", "run", "--rm", "--cpus", "4", "--memory", "8g",
        "-v", f"{PROJECT_ROOT}:/workspace", "-w", "/workspace",
        "diamond-project", "python", "scripts/run_clustering.py", "--inside-docker",
    ]
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use 2,500 rows for a quick check")
    parser.add_argument("--inside-docker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_docker:
        from src.clustering.pipeline import run_buyer_segmentation

        run_buyer_segmentation(smoke=args.smoke)
    else:
        run_inside_docker(args.smoke)


if __name__ == "__main__":
    main()
