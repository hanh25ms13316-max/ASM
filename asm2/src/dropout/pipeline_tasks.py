"""Task functions called by the Airflow DAGs.

Each function takes and returns file paths, so tasks stay independent,
restartable and testable without Airflow.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from dropout.config import PROJECT_ROOT, load_params, load_settings
from dropout.data import DataValidationError, ingest, validate
from dropout.features import EngineeredFeatures, select_columns
from dropout.train import rule_score

logger = logging.getLogger(__name__)
WORK_DIR = PROJECT_ROOT / "data" / "work"


def extract(run_date: str, source: str | Path | None = None) -> str:
    """Copy the nightly export into an immutable, dated snapshot."""
    params = load_params()
    src = Path(source or load_settings().data_path or PROJECT_ROOT / params["data"]["raw_path"])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dest = WORK_DIR / f"snapshot_{run_date}.csv"
    shutil.copyfile(src, dest)
    return str(dest)


def validate_snapshot(snapshot: str) -> bool:
    """True if the snapshot passes the data quality gate."""
    params = load_params()
    try:
        validate(ingest(snapshot, params["data"]["separator"]), params)
    except DataValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return False
    return True


def build_weekly_features(snapshot: str) -> str:
    """Select model columns and add engineered features; save as Parquet/CSV."""
    params = load_params()
    df = ingest(snapshot, params["data"]["separator"])
    cols = select_columns(params, ["enrollment", "financial", "semester1", "semester2"])
    feats = df[cols].copy()
    feats.insert(0, "student_key", np.arange(len(df)))  # pseudonymised key
    out = Path(snapshot).with_name(Path(snapshot).stem + "_features.csv")
    feats.to_csv(out, index=False)
    return str(out)


def score_with_champion(features_path: str) -> str:
    """Score with the registry model currently holding alias ``champion``."""
    params = load_params()
    mlflow.set_tracking_uri(load_settings().tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{params['mlflow']['registered_model']}@champion")
    feats = pd.read_csv(features_path)
    out = feats[["student_key"]].copy()
    out["risk_score"] = model.predict_proba(feats.drop(columns=["student_key"]))[:, 1]
    out["source"] = "model"
    path = WORK_DIR / "scores.csv"
    out.to_csv(path, index=False)
    return str(path)


def score_with_rule(snapshot: str) -> str:
    """Fallback scoring with rule B2 when data validation fails."""
    params = load_params()
    df = ingest(snapshot, params["data"]["separator"])
    out = pd.DataFrame(
        {"student_key": np.arange(len(df)), "risk_score": rule_score(df), "source": "fallback_rule"}
    )
    path = WORK_DIR / "scores.csv"
    out.to_csv(path, index=False)
    return str(path)


def assign_and_publish(run_date: str) -> str:
    """Rank students and assign Tier A (advisor) / Tier B (care team)."""
    ev = load_params()["evaluation"]
    scores = pd.read_csv(WORK_DIR / "scores.csv").sort_values("risk_score", ascending=False)
    n = len(scores)
    rank = np.arange(1, n + 1) / n
    scores["tier"] = np.select(
        [rank <= ev["tier_a_fraction"], rank <= ev["tier_b_fraction"]], ["A", "B"], default="none"
    )
    scores["contact_owner"] = scores["tier"].map({"A": "advisor", "B": "care_team"}).fillna("")
    path = WORK_DIR / f"tiers_{run_date}.csv"
    scores.to_csv(path, index=False)
    logger.info("Tier A=%d, Tier B=%d", (scores.tier == "A").sum(), (scores.tier == "B").sum())
    return str(path)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def max_feature_psi(features_path: str, reference: str | Path | None = None) -> float:
    """Largest PSI across numeric features versus the training reference."""
    params = load_params()
    ref_path = reference or PROJECT_ROOT / params["data"]["raw_path"]
    ref = EngineeredFeatures().transform(ingest(ref_path, params["data"]["separator"]))
    cur = EngineeredFeatures().transform(pd.read_csv(features_path))
    shared = [c for c in cur.columns if c in ref.columns and c != "student_key"]
    return max(psi(ref[c], cur[c]) for c in shared)
