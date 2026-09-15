FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements-ci.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-ci.txt

COPY config.py pytest.ini pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

CMD ["sh", "-c", "pytest -q tests && python scripts/phase1_end_to_end.py"]
