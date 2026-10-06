# Dropout Early-Warning Pipeline (DDM501 Assignment 2 prototype)

Prototype of the ML pipeline designed in Assignment 1. It uses the public UCI dataset
*Predict Students' Dropout and Academic Success* (4,424 students) as a proxy for
college data; the structure of the pipeline is what would be reused in production.

## Layout
```
params.yaml              all tunable values (seed, splits, feature groups, gates)
.env.example             infrastructure settings (MLflow URI, data path)
src/dropout/
  config.py              config loading, environment variables, global seed
  data.py                ingestion, validation (pandera), stratified split
  features.py            shared feature engineering + preprocessing
  evaluate.py            PR-AUC, Recall@k, Precision@k, Brier, ECE, fairness gap
  train.py               one experiment = one MLflow run
  experiments.py         experiment matrix E01-E12, selection, registration
  registry.py            quality gate, challenger/champion aliases
  pipeline_tasks.py      functions called by the Airflow DAGs
dags/dropout_training_dag.py   weekly scoring DAG + training DAG
tests/                   unit tests (pytest)
Dockerfile               reproducible runtime image
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # adjust if needed
# place the dataset at data/raw/students.csv (semicolon separated)
PYTHONPATH=src python -m dropout.experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q && ruff check .
```

Results are written to `reports/experiment_results.csv` and
`reports/selection_summary.json`; every run is also in MLflow.
