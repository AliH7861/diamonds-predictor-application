# Saved models

- `classification/` stores the primary course ANN clarity model and its fitted preprocessor.
- `classification/benchmark_winner/` stores the strongest classification comparison model.
- `regression/` stores the primary course ANN price model and its fitted pipeline.
- `regression/benchmark_winner/` stores the strongest regression comparison model.
- `clustering/` stores the fitted preprocessing and selected K-Means buyer-segmentation model.

Every deployable artifact must include the fitted preprocessing state required to transform raw diamond inputs.
Generated model files are ignored by Git. Use Git LFS or external artifact storage if
trained models need to be published with the repository.
Datasets, figures, and experiment prose do not belong in this folder.
