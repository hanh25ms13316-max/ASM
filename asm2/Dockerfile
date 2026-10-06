# Reproducible training/scoring image for the dropout early-warning pipeline.
FROM python:3.11.10-slim

# Deterministic Python behaviour and no .pyc clutter
ENV PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install pinned dependencies first to maximise layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and configuration (data is mounted or pulled with DVC at runtime)
COPY src/ src/
COPY params.yaml .

# Run as non-root
RUN useradd --create-home appuser
USER appuser

# Default: run the experiment matrix; Airflow overrides the command per task
CMD ["python", "-m", "dropout.experiments"]
