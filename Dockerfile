FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.2.1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
COPY pyproject.toml poetry.lock README.md ./
RUN poetry sync --only main --no-root
COPY src/ ./src/
RUN poetry install --only-root

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app /app
COPY keis7-main/train.csv keis7-main/test.csv ./keis7-main/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ML_DATA_DIR=/app/keis7-main \
    ML_ARTIFACTS_DIR=/app/artifacts \
    MLFLOW_TRACKING_URI=sqlite:////app/artifacts/mlflow.db
# MLflow 2.x initializes ./mlruns even when an experiment has an explicit artifact URI.
RUN mkdir -p /app/artifacts /app/mlruns && chown app:app /app/artifacts /app/mlruns
USER app
ENTRYPOINT ["python", "-m"]
CMD ["src.train", "--smoke"]
