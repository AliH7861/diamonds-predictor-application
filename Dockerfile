FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements-ci.txt requirements-rag-ci.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-ci.txt \
    && pip install --no-cache-dir -r requirements-rag-ci.txt

COPY config.py pytest.ini pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

# The evolving assistant routing benchmark runs separately so its failures stay visible.
CMD ["sh", "-c", "pytest -q tests/unit --ignore=tests/unit/assistant && pytest -q tests/unit/assistant/test_dataset.py tests/unit/assistant/test_retrieval.py && pytest -q tests/integration/test_classification_pipeline.py tests/integration/test_classification_pipeline_existing.py tests/integration/test_regression_pipeline.py tests/integration/test_clustering_pipeline.py tests/integration/test_assistant_http.py tests/integration/test_assistant_ui.py tests/e2e/test_phase1_end_to_end.py && python scripts/phase1_end_to_end.py"]
