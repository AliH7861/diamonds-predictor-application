# Diamonds Predictor Application — Capstone Presentation Guide

This guide explains the project in presentation order and in plain language. It is designed to
help explain what the system does, why each decision was made, how the code flows from one file
to another, what the results mean, and what should be improved next.

## Presentation map

1. [The idea](#1-the-idea)
2. [Setup and demonstration](#2-setup-and-demonstration)
3. [Project design](#3-project-design)
4. [Clarity classification](#4-clarity-classification)
5. [Price regression](#5-price-regression)
6. [Buyer segmentation](#6-buyer-segmentation)
7. [Diamond assistant and RAG](#7-diamond-assistant-and-rag)
8. [End-to-end use cases](#8-end-to-end-use-cases)
9. [Questions to prepare for](#9-questions-to-prepare-for)
10. [Main conclusions](#10-main-conclusions)

---

## 1. The idea

Buying a diamond involves several connected decisions. A buyer may want to know its likely
clarity family, estimate its price, understand what type of buyer usually prefers similar
diamonds, and ask follow-up questions without reading technical tables.

This project turns one structured diamond dataset into four connected capabilities:

| Capability | Question answered | Output |
| --- | --- | --- |
| Classification | What clarity family is most likely from physical attributes? | I, SI, VS, VVS, or IF |
| Regression | What price is reasonable for these attributes? | Predicted price in dollars |
| Clustering | Which product-derived buyer profile fits this diamond? | One of five buyer archetypes |
| Assistant | How can a user search, compare, analyze, predict, and learn naturally? | Grounded conversational answer |

### Why this project matters

- A buyer can compare real examples within a budget.
- A seller can use consistent price and product-profile evidence.
- A student can compare ANN, XGBoost, and Random Forest on the same data.
- The assistant makes the dataset, models, and project knowledge accessible through natural
  language.

### Important limitation

The dataset describes diamonds, not identified customers. The clustering output therefore
represents **buyer-preference archetypes inferred from product characteristics**. It does not
claim to measure real customer psychology or demographics.

---

## 2. Setup and demonstration

### Requirements

- Python 3.11
- Node.js and npm for the React interface
- Docker Desktop for reproducible full model training
- Ollama for local chat generation and embeddings

### Install the project

```powershell
git clone <repository-url>
cd diamonds-predictor-application

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

cd frontend
npm install
cd ..

ollama pull qwen3.5:0.8b
ollama pull nomic-embed-text
```

The project dataset is stored at `data/raw/diamonds.csv`. Saved model artifacts are stored under
`models/`, so the application can make predictions without retraining every time it starts.

### Start the complete application

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_dev.ps1
```

Then open `http://localhost:5173`. The launcher starts:

- React frontend: `http://localhost:5173`
- Python backend: `http://localhost:8770`
- Health check: `http://localhost:8770/health`

### Train and verify models

```powershell
# Quick structural rehearsal
python scripts\train_all_models.py --smoke

# Full CPU-only training in Docker
python scripts\train_all_models.py

# Reload saved models and make predictions without retraining
python scripts\test_saved_models.py

# Run clustering separately
python scripts\run_clustering.py

# Run automated tests
python -m pytest -q
```

---

## 3. Project design

The project separates research, reusable machine-learning code, saved artifacts, results, and
the user interface.

```text
data/raw/                 Original dataset
data/processed/           Exported cleaned, engineered, split, and experiment datasets
sideEDA/                  Research notebooks and earlier analysis artifacts
src/classification/       Clarity classification pipeline
src/regression/           Price regression pipeline
src/clustering/           Buyer segmentation pipeline
src/assistant/            Routing, evidence tools, RAG, models, and HTTP API
models/                   Reloadable trained artifacts and metadata
outputs/                  Metrics, reports, figures, and evaluation results
knowledge/                Curated Markdown documents used by RAG
frontend/                 React chat interface
scripts/                  Training, testing, tracing, and startup entry points
tests/                    Unit, integration, and end-to-end checks
```

### Core design principle

Not every question should be sent to an LLM. The system chooses the most reliable source for the
job:

| User need | Evidence source |
| --- | --- |
| Exact count, average, range, or distribution | Pandas calculation |
| Budget and attribute search | Filtered real dataset rows |
| Price, clarity, or buyer-profile prediction | Saved ML artifact |
| Definition or project explanation | Fixed domain answer or RAG |
| Greeting | Direct response |

This keeps calculations deterministic and uses the language model mainly to explain evidence.

---

## 4. Clarity classification

### 4.1 Objective

Predict a diamond's clarity family from its physical measurements, cut, and color. Price is
excluded because the goal is to learn physical clarity evidence rather than market value.

The original eight clarity grades are grouped into five ordered families:

```text
I1 → I
SI1, SI2 → SI
VS1, VS2 → VS
VVS1, VVS2 → VVS
IF → IF
```

Professional clarity grading depends on microscopic inclusions, but the dataset contains no
microscope or inclusion fields. Combining close grades makes the target more realistic for the
available evidence.

### 4.2 Data preprocessing

1. Load the CSV and remove the extra index column if present.
2. Validate the required physical and categorical columns.
3. Remove missing or impossible physical measurements.
4. Map the original clarity labels into five ordered families.
5. Create physical Y23 engineered features.
6. Make one stratified 70/15/15 train, validation, and test split.
7. Rank numeric candidates with mutual information using training rows only.
8. Remove strongly redundant numeric features with correlation pruning at `0.75`.
9. Median-impute and standardize numeric features.
10. One-hot encode `cut` and `color`.

The same split and prepared representation are used for all three algorithms, making the
comparison fair. A guard verifies that no input feature contains `price`.

### 4.3 Feature-engineering rationale

The raw measurements `x`, `y`, `z`, `depth`, `table`, and `carat` describe individual values.
The Y23 features express physical relationships that may carry more clarity signal:

| Feature idea | Why it was created |
| --- | --- |
| Face area, diagonal, and volume | Represent visible and three-dimensional size |
| Dimension ratios and differences | Represent shape and asymmetry |
| Carat relative to area or volume | Represent how weight is distributed |
| Calculated depth and depth error | Compare reported depth with geometry |
| Depth/table and depth/face ratios | Represent proportions rather than isolated values |
| Size interactions | Let the model observe how proportions behave at different carat weights |

More columns are not automatically better. Mutual-information ranking and correlation pruning
retain useful signal while removing repeated versions of the same geometry.

### 4.4 Models and why they were chosen

| Model | Reason for inclusion |
| --- | --- |
| XGBoost | Course-required model and a strong choice for structured tabular data |
| ANN | Nonlinear comparison model with a different learning approach |
| Random Forest | Stable tree ensemble and an interpretable baseline against boosting and ANN |

Because the five classes have a natural order, each algorithm predicts four cumulative
boundaries: above I, above SI, above VS, and above VVS. The boundary probabilities are converted
into probabilities for the five final families. This penalizes distant errors more meaningfully
than treating all labels as unrelated.

### 4.5 Recorded classification scores

Validation results for the maintained Y23 physical-only representation:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | **57.39%** | **50.21%** | **49.37%** | **49.07%** | **49.37%** |
| Random Forest | 55.91% | 48.45% | 43.12% | 44.00% | 43.12% |
| ANN | 50.20% | 42.07% | 26.88% | 26.14% | 26.88% |

XGBoost was selected as the default classifier using an equal-weighted balanced score based on
macro precision, macro recall, macro F1, and balanced accuracy. On the untouched test set it
reached **56.49% accuracy**, **47.67% macro F1**, and placed **91.31% of predictions within one
neighbouring clarity family**.

### 4.6 Findings and improvements

**Findings**

- Physical features contain some clarity signal, but the classes overlap strongly.
- Most errors occur between neighbouring clarity families.
- Removing price-derived hints makes the result scientifically cleaner.
- XGBoost handled this tabular feature space better than the maintained ANN.

**What could be better**

- Add microscope images, inclusion type, location, size, and certification details.
- Research larger or richer grading datasets.
- Test more sophisticated ordinal neural networks and calibrated probabilities.
- Improve rare-family balance through better data collection or careful class weighting.

### 4.7 Classification code flow

```text
scripts/train_all_models.py: train_classification()
  → cleaning.py: load_and_clean_data() + create_target()
  → feature_engineering.py: engineer_features()
  → preprocessing.py: prepare_classification_data()
       → select_numeric_features()
       → create_preprocessor()
  → training.py: train_candidate() for ANN, XGBoost, and Random Forest
  → evaluation.py: classification_metrics()
  → prediction.py: save_best_model()

New prediction
  → prediction.py: load_best_model()
  → prediction.py: predict_diamonds()
  → saved feature engineering + preprocessing + model
  → clarity family and probabilities
```

Start debugging in `preprocessing.py` when input columns or splits are wrong, `training.py` when
fitting is wrong, and `prediction.py` when a saved model behaves differently after reload.

---

## 5. Price regression

### 5.1 Objective

Predict `price` from carat, quality grades, physical geometry, and engineered market/peer
features. Clarity is an input in this workflow because it contributes to price.

### 5.2 Data preprocessing

1. Load the dataset and remove duplicates, missing values, and impossible dimensions.
2. Create baseline physical features.
3. Use one shared 70/15/15 split: 37,729 train, 8,085 validation, and 8,085 test rows.
4. Fit human-context calculations using training rows only.
5. Apply that saved context to validation and test rows.
6. Pass numeric inputs through and one-hot encode categorical inputs.
7. Keep price separate as the target.
8. Fit every model on the same rows for a fair comparison.

### 5.3 Three feature experiments

| Experiment | Question | Encoded inputs |
| --- | --- | ---: |
| CURRENT_BASELINE | How well do raw attributes and basic geometry work? | 34 |
| HUMAN_ONLY | Can understandable quality, size, rarity, and peer features replace raw geometry? | 62 |
| HUMAN_PLUS_RAW | Do human features and exact raw geometry complement each other? | 67 |

### 5.4 Feature-engineering rationale

| Feature idea | Meaning |
| --- | --- |
| Face area, diameter, diagonal, and volume | Approximate visible and physical size |
| Aspect ratio and asymmetry | Describe shape balance |
| Weight per face/volume | Describe how carat is distributed |
| Ordered cut, color, and clarity scores | Represent the natural direction of quality grades |
| Quality bundle | Combine the three certificate-quality dimensions |
| Carat bands and milestone distance | Represent market thresholds such as 0.50, 0.75, 1.00, 1.50, and 2.00 ct |
| Expected face area and volume | Compare a diamond with training-derived size expectations |
| Depth/table deviations | Compare proportions with similar-size and similar-cut peers |
| Quality at size | Model interactions between size and quality |
| Peer rarity | Measure how unusual a training peer group is |

Price-derived inputs are not used. The model must predict price rather than receive a disguised
version of its target.

### 5.5 Models and why they were chosen

The same ANN, XGBoost, and Random Forest families are used to compare a neural network, gradient
boosting, and bagged trees. Every algorithm is trained on all three feature representations,
producing nine comparable candidates.

### 5.6 Recorded regression scores

Validation results:

| Feature set | XGBoost MAE | ANN MAE | Random Forest MAE |
| --- | ---: | ---: | ---: |
| CURRENT_BASELINE | $257.99 | $336.90 | $268.41 |
| HUMAN_ONLY | $266.78 | $290.43 | $272.94 |
| **HUMAN_PLUS_RAW** | **$253.80** | **$279.49** | **$261.69** |

The **course-required and selected model** is HUMAN_PLUS_RAW XGBoost. Its untouched-test result
was:

| MAE | Median AE | RMSE | R² | MAPE | Within 10% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **$245.44** | **$87.20** | **$505.07** | **0.9834** | **5.89%** | **82.46%** |

The HUMAN_PLUS_RAW ANN remains a comparison model. Its untouched-test result was **$271.15 MAE**,
**$557.31 RMSE**, **0.9798 R²**, and **77.61% within 10% error**.

### 5.7 Findings and improvements

**Findings**

- Carat was the strongest basic price driver.
- Human-only features were understandable but lost exact geometric information.
- Raw geometry alone was strong, while human context explained useful market relationships.
- Combining both representations produced the best results.
- XGBoost was strongest overall; the ANN still produced a strong capstone result.

**What could be better**

- Add dated market, retailer, certification, shape, and location data.
- Research industry pricing practices and how they change over time.
- Add prediction intervals so users see uncertainty, not only one dollar value.
- Test feature stability across periods and external datasets.
- Explore a smaller hybrid representation that retains accuracy with lower complexity.

### 5.8 Regression code flow

```text
scripts/train_all_models.py: train_regression()
  → cleaning.py: load_and_clean_data()
  → feature_engineering.py: engineer_base_features()
  → preprocessing.py: split_data()
  → feature_engineering.py: fit_human_context() on train only
  → feature_engineering.py: transform_human_features()
  → preprocessing.py: prepare_experiments()
  → training.py: train_model() for 3 feature sets × 3 algorithms
  → evaluation.py: full_regression_metrics()
  → prediction.py: save_best_model()

New prediction
  → prediction.py: load_best_model()
  → prediction.py: predict_prices()
  → saved context + preprocessor + XGBoost
  → predicted dollar price
```

The course-required XGBoost artifact is stored under `models/regression/benchmark_winner/`. The
ANN artifact remains saved for model comparison.

---

## 6. Buyer segmentation

### 6.1 Objective

Group diamonds into understandable purchase profiles, then infer the buyer preference that each
profile may represent. Each diamond row acts as one anonymous purchase profile because the
dataset has no customer IDs.

### 6.2 How the groups were created

1. Validate and clean the required diamond columns.
2. Create physical, quality, value, rarity, and milestone features.
3. Hold out 203 rows with several extreme geometry signals so anomalies do not become fake
   segments.
4. Combine related variables into eight equally weighted customer pillars.
5. Median-fill any remaining numeric gaps and standardize the pillars.
6. Test K-Means, Gaussian Mixture Models, and Ward Agglomerative clustering.
7. Test `K = 3, 5, 7, 10`.
8. Evaluate separation, balance, stability, coverage, and interpretability.
9. Select K-Means with `K = 5`.
10. Profile each cluster in original human-readable diamond units.

### 6.3 The eight segmentation pillars

```text
Visual presence      Certificate quality
Value                Price level
Quality balance      Proportion consistency
Rarity               Carat-milestone positioning
```

Pillars prevent multiple correlated geometry columns from overpowering business meaning simply
because geometry happens to have more raw columns.

### 6.4 Why K-Means with five clusters was selected

The system did not select a solution using silhouette alone. It considered silhouette,
Davies-Bouldin, Calinski-Harabasz, segment sizes, balance, centroid separation, within-cluster
spread, stability, coverage, and interpretability.

| Selected solution | Silhouette | Stability ARI | Davies-Bouldin | Smallest segment | Composite score |
| --- | ---: | ---: | ---: | ---: | ---: |
| **K-Means, K = 5** | **0.171** | **0.982** | **1.730** | **10.60%** | **0.732** |

K = 3 had a slightly higher silhouette, but five groups gave a more useful level of buyer-facing
detail while remaining stable and reasonably balanced. K = 10 fragmented the market too much.

### 6.5 Final buyer-preference archetypes

| Archetype | Share | Typical diamond | Main priorities |
| --- | ---: | --- | --- |
| Quality-Conscious Budget Buyers | 29.24% | 0.33 ct, $772, Ideal, E, VS | certificate quality, proportions, value |
| Milestone-Conscious Value Buyers | 25.32% | 0.76 ct, $3,239, Ideal, E, SI | milestone size, value, proportions |
| Balanced Mid-Market Pragmatists | 20.30% | 0.72 ct, $2,639, Very Good, G, SI | balanced quality, rarity, milestone |
| Premium Visual-Impact Seekers | 14.54% | 1.23 ct, $7,363, Ideal, H, SI | visible size, market tier, rarity |
| Rarity / Distinctiveness Seekers | 10.60% | 1.00 ct, $3,853, Good, F, SI | rarity, visual size, milestone |

### 6.6 What the clustering tells us

- The market is a continuum, so some overlap is expected.
- Budget-oriented profiles can still prioritize certificate quality.
- Larger visual impact does not always mean the highest clarity.
- Carat milestones influence some profiles independently of pure size.
- A mathematically tighter solution is not always the clearest business segmentation.

**What could be better**

- Validate the profiles with real customer IDs, transactions, surveys, and repeat purchases.
- Compare additional methods such as HDBSCAN with careful tuning and soft membership.
- Test the stability of profiles on another retailer or time period.
- Separate value-driven and quality-driven research questions when richer data becomes available.

### 6.7 Clustering code flow

```text
scripts/run_clustering.py
  → pipeline.py: run_buyer_segmentation()
  → feature_engineering.py: build_segmentation_features()
  → feature_engineering.py: split_geometry_anomalies()
  → feature_engineering.py: build_customer_pillars()
  → StandardScaler
  → clustering.py: run_candidate_clustering()
  → evaluation.py: evaluate_candidates() + select_best_solution()
  → profiling.py: build_profiles() + attach_profile_names()
  → rag.py: build_rag_records()
  → visualizations.py: generate_readme_visualizations()
  → pipeline.py: persist_segmentation_result()
```

The reusable model, scaler, pillar list, and profiles are stored in
`models/clustering/buyer_segmentation.joblib` and `profile_registry.json`.

---

## 7. Diamond assistant and RAG

### 7.1 Design rationale

The assistant is an evidence router around the project. It should understand the user's intent,
choose the correct tool, gather compact evidence, and return an understandable response.

It supports seven main intents:

```text
search                 compare displayed diamonds
dataset analysis       diamond/project knowledge
model prediction       small talk
out of scope
```

### 7.2 Why this RAG approach was used

The complete CSV is not embedded into the vector database. Exact data questions are better
answered with Pandas, searches are better answered by filters, and predictions are better
answered by saved models.

RAG is reserved for explanations such as:

- What does VS clarity mean?
- Why does carat affect price?
- How were the buyer profiles created?
- What features does the price model use?

This hybrid approach reduces hallucination because each question uses the most appropriate
evidence source.

### 7.3 RAG data preprocessing

1. Store project and diamond knowledge as focused Markdown documents in `knowledge/`.
2. Split documents by headings and compact text chunks.
3. Embed chunks locally with `nomic-embed-text`.
4. Index embeddings and metadata in Chroma.
5. Normalize a user's question and route its intent.
6. Build up to two focused retrieval queries.
7. Retrieve semantically similar chunks and apply relevance filtering/reranking.
8. Compact the retrieved evidence into a bounded prompt.
9. Generate and stream the explanation with local `qwen3.5:0.8b` through Ollama.

Greetings, fixed category questions, exact dataset calculations, and model predictions bypass
RAG when it is unnecessary.

### 7.4 Live request flow

```text
React App
  → frontend/src/api.js: streamChat()
  → assistant/api.py: AssistantRequestHandler
  → assistant/runtime.py: create_assistant()
  → assistant/service.py: DiamondAssistant.ask()
  → assistant/routing.py: route_question()
       ├─ search → search_planning.py → dataset_search.py
       ├─ compare → comparison.py
       ├─ analysis → data_analysis.py
       ├─ fixed knowledge → domain_answers.py
       ├─ RAG → retrieval.py → vector_store.py → prompt_builder.py → generation.py
       └─ prediction → clarification.py → tool_executor.py → model_evidence.py
  → assistant/transport.py: encode_result()
  → streamed React response
```

### 7.5 Where to debug

| Symptom | Start in | What to inspect |
| --- | --- | --- |
| Everything receives a generic/default answer | `routing.py` | selected intent, action, and reason |
| Budget, carat, cut, or clarity is misunderstood | `search_planning.py` | extracted and validated plan |
| A follow-up reuses the wrong preference | `service.py` | incoming state and recovered search state |
| Correct plan returns strange diamonds | `dataset_search.py` | filters, qualifying rows, and ranking |
| Wrong knowledge appears | `retrieval.py` / `vector_store.py` | retrieval queries, chunks, and scores |
| Right evidence becomes a poor answer | `prompt_builder.py` / `generation.py` | final evidence payload and prompt |
| Backend is correct but UI is wrong | `transport.py` / `frontend/src/App.jsx` | streamed final event and rendering |

Use this command to see the production checkpoints for one question:

```powershell
python scripts\trace_assistant_question.py "Find me a diamond around $2500"
```

### 7.6 Findings and improvements

**Findings**

- Deterministic tools are better than an LLM for exact calculations and filters.
- Compact, labeled evidence makes generated answers easier to ground.
- Loading models once at backend startup makes inference faster.
- Streaming improves perceived response time even when local generation is still running.
- Small local language models reduce cost and keep data local, but they require strong routing and
  prompts.

**What could be better**

- Replace brittle keyword coverage with a tested hierarchical or hybrid intent classifier.
- Improve query rewriting and retrieval reranking while keeping prompts compact.
- Add broader held-out conversational tests that match real frontend conversations.
- Measure retrieval relevance, answer faithfulness, latency, and token use separately.
- Improve state boundaries so a new chat never inherits another chat's search criteria.
- Use a stronger local model when hardware permits.

---

## 8. End-to-end use cases

### Use case A: Find a diamond within a budget

```text
User: "Find me a balanced diamond below $4,000 around 0.75 carat."
→ router chooses search
→ search planner extracts budget, size, and priorities
→ dataset search filters and ranks real rows
→ response explains the closest options and trade-offs
```

### Use case B: Predict price and clarity

```text
User supplies raw diamond measurements and quality grades
→ router chooses model prediction
→ parser validates required inputs
→ saved preprocessing and feature engineering run
→ saved XGBoost model returns predictions
→ assistant explains the result and evidence source
```

### Use case C: Understand the market groups

```text
User: "What types of buyers did the project find?"
→ router chooses fixed project knowledge
→ profile registry supplies the five saved archetypes
→ assistant explains priorities, typical diamonds, and limitations
```

### Use case D: Ask an educational question

```text
User: "Why does a one-carat milestone affect price?"
→ router chooses knowledge/RAG
→ question is embedded
→ relevant knowledge chunks are retrieved
→ local Qwen explains the grounded evidence
```

---

## 9. Questions to prepare for

### Why did classification perform worse than regression?

Price is strongly represented by carat, dimensions, and quality grades. Clarity grading depends
on microscopic inclusions that are missing from the dataset. The classification ceiling is
therefore largely a data limitation, not simply a model limitation.

### Why remove price from classification?

Price is market evidence affected by clarity and many other variables. Using it to predict clarity
would let the classifier learn a market shortcut rather than the physical grading relationship the
experiment is meant to test.

### Why use five clarity families?

The input data cannot reliably distinguish every fine clarity grade. The five ordered families
reduce unsupported precision while preserving the direction from included to flawless.

### Why compare ANN and Random Forest if XGBoost is required?

XGBoost is the course-required and selected model. ANN and Random Forest provide meaningful
comparison points, showing whether boosting actually performs better than neural-network and
bagged-tree alternatives on the same rows and features.

### Why did HUMAN_PLUS_RAW win regression?

Human features added meaningful concepts such as peer rarity and milestone distance, while raw
geometry preserved exact measurements. The combined representation kept both forms of signal.

### Why was K = 5 selected when K = 3 had a higher silhouette?

Silhouette measures separation, but the project also needs stable, balanced, and understandable
buyer profiles. K = 5 achieved the best composite score and added useful market detail without
the fragmentation seen at K = 10.

### Is the clustering really customer segmentation?

It is product-based buyer-preference segmentation. Without customer IDs or behaviour, the project
cannot validate psychological customer types. The names describe the purchase patterns represented
by each diamond cluster.

### Why not use RAG for every question?

Embeddings retrieve similar text; they do not reliably calculate exact counts, apply numeric
filters, or run trained models. Pandas and saved artifacts are more trustworthy for those tasks.

### How is leakage prevented?

Splits are created before learned preprocessing or contextual features. Feature selection,
imputation, scaling, encoding, peer profiles, rarity counts, and expected-size models are fitted
using training rows only, then applied to validation and test data.

### Does the application retrain when it starts?

No. The backend loads saved artifacts once and uses them for inference. Training is a separate,
explicit workflow.

---

## 10. Main conclusions

1. Diamond price is highly predictable from size, geometry, and quality, especially when raw and
   human-interpretable features are combined.
2. Clarity is harder because the dataset lacks the microscopic evidence used by professional
   graders.
3. Five stable product-derived profiles provide a useful view of buyer preferences without
   overstating what anonymous diamond rows can prove.
4. The assistant works best as an orchestrator: exact data tools perform calculations, saved
   models perform inference, and RAG explains curated knowledge.
5. The largest opportunity is better source data: inclusion evidence for clarity, market history
   for price, real customer behaviour for clustering, and broader conversational evaluation for
   the assistant.

### One-sentence presentation summary

> This capstone turns a raw diamond dataset into clarity classification, price prediction,
> buyer-preference segmentation, and a local evidence-grounded assistant, while keeping the
> limitations of each data source visible.
