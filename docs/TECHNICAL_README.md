# Diamonds Predictor: Technical Guide

This document explains installation, repository structure, pipelines, commands, APIs, testing, Docker, and frontend/backend operation. The [main README](../README.md) contains the project narrative, experiment reasoning, findings, and results.

## Requirements and installation

- Python 3.11
- Minimum 4 GB RAM; 8 GB recommended
- Git and a Python IDE
- Docker Desktop recommended for TensorFlow on managed Windows
- Ollama for the live local assistant

```powershell
git clone https://github.com/AliH7861/diamonds-predictor-application.git
cd diamonds-predictor-application
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name diamond-ml --display-name "Diamond ML (.venv)"
```

Download the Kaggle Diamonds dataset to `data/raw/diamonds.csv`. Expected columns are `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `price`, `x`, `y`, and `z`. An optional `Unnamed: 0` export index is removed.

## Repository structure

```text
app.py                         Streamlit frontend
config.py                      Shared paths and random seed
data/                          Raw input location and generated datasets
knowledge/                     Trusted RAG Markdown
models/                        Generated model bundles
notebooks/                     EDA, classification, regression, segmentation
outputs/                       Generated metrics, figures, and reports
scripts/                       Training, tracing, and verification commands
sideEDA/                       Historical notebooks and recovered evidence
src/
  assistant/                   Routing, retrieval, API, client, UI
  classification/              Clarity pipeline
  regression/                  Price pipeline
  clustering/                  Buyer-segmentation pipeline
tests/                         Unit, integration, RAG, HTTP, and E2E tests
.github/workflows/             GitHub Actions
```

## Pipeline map

```text
diamonds.csv → cleaning → feature engineering
                         ├─ classification → ANN/XGBoost/Random Forest
                         ├─ regression     → ANN/XGBoost/Random Forest
                         └─ clustering     → K-Means K=3/5/7/10

knowledge/*.md → chunks → embeddings → Chroma

dataset + saved ANNs + Chroma + memory → router → compact evidence → Qwen
```

## Data preparation

- `RANDOM_STATE = 42` controls reproducibility.
- Classification uses one stratified 70/15/15 split for all nine runs.
- Regression uses one 70/15/15 split for all nine runs.
- Each preprocessor is fitted with training rows and reused for validation, test, and inference.
- Clustering creates one scaled/encoded matrix reused for K = 3, 5, 7, and 10.

## Classification files

| File | Responsibility |
| --- | --- |
| `src/classification/cleaning.py` | Clean rows and create the five-family target |
| `feature_engineering.py` | Physical and price-relative features; experiment lists |
| `preprocessing.py` | Shared stratified split, imputation, scaling, encoding |
| `model.py` | ANN architectures |
| `training.py` | ANN, XGBoost, and Random Forest training |
| `evaluation.py` | Accuracy, precision, recall, F1, balance, ordinal errors |
| `diagnostics.py` | Distributions, correlations, top-eight importance/relations |
| `datasets.py` | DataFrame, encoded matrix, and statistics exports |
| `prediction.py` | Save/load fitted artifacts and predict from raw input |
| `api.py` | Local clarity HTTP API |

Target mapping: `I1 → I`, `SI1/SI2 → SI`, `VS1/VS2 → VS`, `VVS1/VVS2 → VVS`, `IF → IF`.

## Regression files

| File | Responsibility |
| --- | --- |
| `src/regression/cleaning.py` | Reproduce the maintained regression population |
| `feature_engineering.py` | Baseline, human, threshold, and peer features |
| `preprocessing.py` | Shared split and three experiment matrices |
| `model.py` | Price ANN |
| `training.py` | Train on `log1p(price)`, validate, select, and test |
| `evaluation.py` | Dollar, percentage, bias, percentile, and band metrics |
| `visualizations.py` | EDA, top features, predictions, and errors |
| `datasets.py` | Original and processed dataset/statistics exports |
| `prediction.py` | Save/load model, preprocessing, and human context |
| `api.py` | Local price HTTP API |

Every prediction path applies `expm1` before returning dollar values.

## Clustering files

| File | Responsibility |
| --- | --- |
| `src/clustering/feature_engineering.py` | Purchase-oriented geometry and value features |
| `preprocessing.py` | Clean, scale numeric values, encode categories |
| `evaluation.py` | Silhouette, inertia, balance, separation, selection score |
| `profiling.py` | Original-unit statistics and buyer interpretations |
| `visualizations.py` | K comparison, sizes, and PCA view |
| `pipeline.py` | Fit, compare, select, export, and persist |
| `prediction.py` | Assign a future profile with the saved pipeline |

## Assistant files

| File | Responsibility |
| --- | --- |
| `src/assistant/routing.py` | Choose intent and required evidence |
| `clarification.py` | Parse natural language and request missing fields |
| `dataset_search.py` | Exact Pandas constraints |
| `similarity_search.py` | Scaled/encoded structured neighbours |
| `vector_store.py` | Separate Chroma knowledge and memory collections |
| `retrieval.py` | Semantic search and optional focused second pass |
| `model_evidence.py` | Load primary ANNs once and infer |
| `memory.py` | Retrieve/save explicit preferences |
| `prompt_builder.py` | Bound and assemble compact evidence |
| `generation.py` | Ollama embeddings, structured output, and response |
| `service.py` | Question-to-answer orchestration |
| `api.py` | Local `/health` and `/chat` backend |
| `transport.py` | JSON-safe DataFrame and NumPy conversion |
| `client.py` | Streamlit-compatible HTTP client |
| `runtime.py` | In-process production assembly |
| `ui.py` | Normal and developer Streamlit views |
| `ui_content.py` | Editable page wording |

## Streamlit modes

### One-process local mode

With `DIAMOND_ASSISTANT_API_URL` unset, Streamlit loads the dataset, vector index, models, and Ollama client in its own process.

```powershell
ollama serve
python -m streamlit run app.py
```

### Separate frontend and local backend

Terminal 1:

```powershell
$env:DIAMOND_ASSISTANT_API_TOKEN="choose-a-long-random-token"
python -m src.assistant.api --host 127.0.0.1 --port 8770
```

Terminal 2:

```powershell
$env:DIAMOND_ASSISTANT_API_URL="http://127.0.0.1:8770"
$env:DIAMOND_ASSISTANT_API_TOKEN="choose-a-long-random-token"
python -m streamlit run app.py
```

The frontend sends natural-language turns to `POST /chat`. The local API retains the dataset, saved models, vector data, and Ollama access.

### Hosted frontend, local backend

A cloud server cannot reach your computer at `127.0.0.1`. Create an authenticated HTTPS tunnel to port 8770 and set the frontend host’s secrets:

```text
DIAMOND_ASSISTANT_API_URL=https://your-tunnel-host.example
DIAMOND_ASSISTANT_API_TOKEN=the-same-local-secret
```

The computer, API, Ollama, and tunnel must stay running. Streamlit Community Cloud can host this Python frontend. Vercel requires a separate supported web frontend.

`requirements-frontend.txt` contains only the packages needed by the separated Streamlit frontend. The client imports the larger local ML/RAG runtime only when `DIAMOND_ASSISTANT_API_URL` is unset.

## APIs

### Assistant

```powershell
python -m src.assistant.api --port 8770
```

- `GET /health`
- `POST /chat`

```json
{
  "question": "I have a $6,000 budget and clarity matters most.",
  "conversation": []
}
```

If `DIAMOND_ASSISTANT_API_TOKEN` is set, send `Authorization: Bearer <token>`.

### Saved predictions

```powershell
python -m src.classification.api --port 8765
python -m src.regression.api --port 8766
```

Both provide `GET /health`, `GET /model`, and `POST /predict`. Callers provide raw diamond attributes; each model bundle contains its fitted preprocessing.

## Training

```powershell
# All 3 × 3 supervised comparisons plus segmentation.
python scripts/train_all_models.py

# Quick rehearsal with separate smoke artifacts.
python scripts/train_all_models.py --smoke

# Clustering only.
python scripts/run_clustering.py
python scripts/run_clustering.py --smoke
```

The full command saves the strongest validation ANN for each course task and the strongest overall comparison under `benchmark_winner/`.

## Notebooks

Run with **Restart Kernel and Run All**:

1. `notebooks/01_old_eda.ipynb`
2. `notebooks/02_classification.ipynb`
3. `notebooks/03_regression.ipynb`
4. `notebooks/04_buyer_segmentation.ipynb`

Notebooks display explanations, DataFrames, statistics, calls, comparisons, and plots. Reusable implementation lives in `src/`.

## Verification

| Command | Trains? | Purpose |
| --- | ---: | --- |
| `python scripts/test_saved_models.py` | No | Reload saved supervised artifacts and predict |
| `python scripts/verify_final_metrics.py` | No | Recalculate held-out metrics |
| `python scripts/test_rag.py --ci` | No | Deterministic embedding/Chroma/conversation test |
| `python scripts/test_rag.py` | No | Live Ollama RAG test |
| `python scripts/trace_rag.py` | No | Print retrieval and evidence stages |
| `python scripts/phase1_end_to_end.py` | Brief test fits | Validate supervised structure |
| `pytest -q` | Brief test fits | Complete automated suite |
| `ruff check src tests scripts config.py app.py` | No | Static quality checks |

The HTTP integration test authenticates, calls `/chat`, serializes model/data evidence, and restores result rows for Streamlit.

## CI/CD and Docker

`.github/workflows/phase1-ci.yml` runs on pushes and pull requests:

1. Ruff lint
2. Unit and integration tests
3. Assistant, UI, and HTTP component tests
4. Real Chroma RAG test with deterministic embeddings
5. K=3/5/7/10 clustering test
6. End-to-end supervised validation
7. Clean Linux Docker build and validation

```powershell
docker build -t diamond-project .
docker run --rm diamond-project
```

[View GitHub Actions](https://github.com/AliH7861/diamonds-predictor-application/actions).

## Generated artifacts

| Path | Contents |
| --- | --- |
| `data/processed/classification/` | Feature, split, encoded, and statistics exports |
| `data/processed/regression/` | Three price representations and statistics |
| `models/classification/` | Primary ANN and preprocessing |
| `models/classification/benchmark_winner/` | Best classification comparison |
| `models/regression/` | Primary ANN, preprocessing, human context |
| `models/regression/benchmark_winner/` | Best regression comparison |
| `models/clustering/` | Selected K-Means pipeline |
| `outputs/` | Metrics, profiles, figures, and reports |
| `vector_db/chroma_v2/` | Local knowledge and preference vectors |

Large generated datasets, model files, vector databases, and reports are ignored by Git and recreated with the scripts.

## Troubleshooting

### TensorFlow DLL blocked

Use Docker-backed training/validation. Linux avoids the managed Windows native DLL restriction.

### Ollama unavailable

Run `ollama serve` and install `qwen3.5:4b` plus `nomic-embed-text`. Override with `DIAMOND_CHAT_MODEL`, `DIAMOND_EMBEDDING_MODEL`, or `OLLAMA_BASE_URL`.

### Frontend cannot reach backend

Open `http://127.0.0.1:8770/health`, check both processes use the same token, and ensure `DIAMOND_ASSISTANT_API_URL` contains only the backend base URL.

### Hosted frontend is offline

Confirm the local computer, assistant API, Ollama, and HTTPS tunnel are running. A hosted frontend cannot start a private local backend.
