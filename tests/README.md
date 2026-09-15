# Tests

`unit/` checks features, targets, preprocessing, and model shapes independently.
`integration/` briefly trains and persists each ANN pipeline. `e2e/` runs both
pipelines through the terminal validator. Tests use generated fixtures and never
retrain production models or write into the project model folders.
Exploratory analyses and production training runs do not belong under `tests/`.

If Windows Application Control blocks TensorFlow's native extension, pytest marks
the five ANN-dependent host checks as skipped. Docker and GitHub Actions run those
checks on Linux; skips there indicate a failed environment setup and should be
investigated.
