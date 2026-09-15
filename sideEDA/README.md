# Diamond sideEDA

Earlier EDA outputs and notebook snapshots are organized here. The maintained
classification and regression notebooks are in `../notebooks/`. Production
modules and newly generated outputs do not belong in this folder.

## Start here

Open [notebooks/EDAOverview.ipynb](notebooks/EDAOverview.ipynb) and run all cells.
It uses paths relative to the Diamond project, displays CSV versions and statistics,
reads earlier diagnostics and reports, and plots the current clarity distribution.
This is a newly assembled overview, not a recovered historical EDA notebook.

## Layout

- `notebooks/EDAOverview.ipynb`: runnable inventory and data review.
- `notebooks/history/`: classification snapshots with saved historical outputs.
  Paths are adapted, but original code issues remain. Use the maintained notebook
  for current training; these snapshots are for reference.
- `data/cleaned/`: earlier `diamonds_cleaned.csv` and `diamonds_final_cleaned.csv`.
- `data/diagnostics/`: extreme dimensions, removed outliers, and size/carat mismatches.
- `outputs/reports/`: four earlier Experiment M reports.
- `manifest.json`: locations, original locations, row counts, columns, and SHA256 hashes.

## Which diamonds CSV is used?

| Version | Rows | Location relative to Diamond |
|---|---:|---|
| Raw input used by current classification | 53,940 | `data/raw/diamonds.csv` |
| Earlier EDA cleaned | 53,920 | `sideEDA/data/cleaned/diamonds_cleaned.csv` |
| Earlier EDA final cleaned | 53,899 | `sideEDA/data/cleaned/diamonds_final_cleaned.csv` |
| Current classification cleaned export | 53,572 | `data/processed/classification/cleaned/dataset.csv` |

The classifier reads the raw input and applies its current cleaning rules.
The earlier final cleaned CSV is retained as an EDA artifact, not substituted
as the classifier input. Current exports include split and source-row metadata.
Counts/hashes in this document and manifest describe the organization-time snapshot.

## Recovered original notebooks and regression artifacts

The original notebooks have been recovered and organized here:

- [DiamondCaptsone.ipynb](DiamondCaptsone.ipynb)
- [DiamondCaptsoneANN.ipynb](DiamondCaptsoneANN.ipynb)
- [DiamondCaptsonePrice.ipynb](DiamondCaptsonePrice.ipynb)

Their embedded historical outputs are preserved. A project-relative setup cell
and file-path updates route known CSVs, reports, and model artifacts into sideEDA.
Run the setup cell before individual experiments. Training has not been rerun.
The Price notebook also contains earlier classification exploration.

Historical regression reports are under `sideEDA/outputs/reports/regression/{final,r10,r10_5,r12}/`.
Definitions and parameters are under `configuration/{regression,classification}/`.
The full price snapshot is under `references/regression/`.
The recovered historical model bundle is under `models/regression/final_price_model/`;
its identity should be verified against its metadata before calling it the final winner.
`recovered_files.json` records original locations and hashes. Notebook records also
retain their original pre-edit hashes. Unrelated Leetcode and empty Untitled notebooks
were left in their original locations.

## Active project paths

- Training: `notebooks/02_classification.ipynb`
- Implementation: `src/classification/`
- Raw input: `data/raw/diamonds.csv`
- Classification exports: `data/processed/classification/`
- Winning model after training: `models/classification/`
- Local API: `GET /health`, `GET /model`, `POST /predict` at `http://127.0.0.1:8765`

Historical EDA CSVs were moved from `data/` and reports from `outputs/reports/`.
The current classification code does not reference those old locations. If an older
notebook is recovered later, update its paths using this manifest before running it.

The three original project notebooks are directly in sideEDA: DiamondCaptsone.ipynb, DiamondCaptsonePrice.ipynb, and DiamondCaptsoneANN.ipynb. The generated overview and additional snapshots remain under notebooks/.
