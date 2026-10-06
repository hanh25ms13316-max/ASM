"""Training stage: build, fit and log one experiment configuration to MLflow."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from dropout.data import DatasetSplits
from dropout.evaluate import evaluate
from dropout.features import (
    EngineeredFeatures,
    build_preprocessor,
    engineered_columns,
    select_columns,
)

logger = logging.getLogger(__name__)

XGB_SEARCH_SPACE: dict[str, list[Any]] = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.9],
    "model__min_child_weight": [1, 3, 5],
}


@dataclass
class ExperimentConfig:
    """One row of the experiment matrix."""

    exp_id: str
    description: str
    algorithm: str  # rule | logreg | rf | xgb
    feature_groups: list[str] = field(
        default_factory=lambda: ["enrollment", "financial", "semester1", "semester2"]
    )
    hyperparams: dict[str, Any] = field(default_factory=dict)
    balance: str = "none"  # none | class_weight | smote
    tune: bool = False
    calibrate: bool = False


@dataclass
class ExperimentResult:
    """Metrics and fitted model of a finished run."""

    config: ExperimentConfig
    run_id: str
    val_metrics: dict[str, float]
    model: Any | None


def rule_score(X: pd.DataFrame) -> np.ndarray:
    """Baseline B2: the extended rule currently used as fallback.

    One point each for debt, unpaid tuition and passing no first-semester
    module; ties are broken by first-semester grade (lower = riskier).
    """
    points = (
        X["Debtor"]
        + (1 - X["Tuition fees up to date"])
        + (X["Curricular units 1st sem (approved)"] == 0).astype(int)
    )
    return (points - X["Curricular units 1st sem (grade)"] / 100).to_numpy(dtype=float)


def build_estimator(cfg: ExperimentConfig, pos_weight: float, seed: int) -> Any:
    """Create the classifier described by ``cfg`` with a fixed seed."""
    hp = dict(cfg.hyperparams)
    if cfg.algorithm == "logreg":
        weight = "balanced" if cfg.balance == "class_weight" else None
        return LogisticRegression(max_iter=2000, class_weight=weight, random_state=seed, **hp)
    if cfg.algorithm == "rf":
        weight = "balanced" if cfg.balance == "class_weight" else None
        return RandomForestClassifier(
            n_estimators=400, class_weight=weight, n_jobs=-1, random_state=seed, **hp
        )
    if cfg.algorithm == "xgb":
        if cfg.balance == "class_weight":
            hp.setdefault("scale_pos_weight", pos_weight)
        return XGBClassifier(eval_metric="logloss", n_jobs=4, random_state=seed, **hp)
    raise ValueError(f"Unknown algorithm: {cfg.algorithm}")


def build_pipeline(cfg: ExperimentConfig, columns: list[str], pos_weight: float, seed: int):
    """Assemble feature engineering -> preprocessing -> [SMOTE] -> model."""
    steps: list[tuple[str, Any]] = [
        ("engineer", EngineeredFeatures()),
        ("preprocess", build_preprocessor(engineered_columns(columns), cfg.algorithm == "logreg")),
    ]
    if cfg.balance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", build_estimator(cfg, pos_weight, seed)))
    return ImbPipeline(steps)


def run_experiment(
    cfg: ExperimentConfig, splits: DatasetSplits, params: dict[str, Any]
) -> ExperimentResult:
    """Train one configuration, evaluate on validation data and log to MLflow.

    Args:
        cfg: experiment definition (one row of the matrix).
        splits: validated train/validation/test data.
        params: configuration loaded from ``params.yaml``.

    Returns:
        ExperimentResult: run id, validation metrics and fitted pipeline.
    """
    seed = params["project"]["seed"]
    ev = params["evaluation"]
    fair_attr = params["features"]["fairness_attribute"]
    columns = select_columns(params, cfg.feature_groups)
    X_tr, X_val = splits.X_train[columns], splits.X_val[columns]
    y_tr, y_val = splits.y_train, splits.y_val
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    with mlflow.start_run(run_name=f"{cfg.exp_id}-{cfg.algorithm}") as run:
        mlflow.set_tags(
            {"exp_id": cfg.exp_id, "description": cfg.description, "data_sha256": splits.data_hash}
        )
        mlflow.log_params(
            {
                "algorithm": cfg.algorithm,
                "feature_groups": ",".join(cfg.feature_groups),
                "n_features_raw": len(columns),
                "balance": cfg.balance,
                "tune": cfg.tune,
                "calibrate": cfg.calibrate,
                "seed": seed,
                **cfg.hyperparams,
            }
        )
        start = time.perf_counter()
        model: Any | None = None
        if cfg.algorithm == "rule":
            scores = rule_score(X_val)
        else:
            model = build_pipeline(cfg, columns, pos_weight, seed)
            if cfg.tune:
                search = RandomizedSearchCV(
                    model,
                    XGB_SEARCH_SPACE,
                    n_iter=25,
                    scoring="average_precision",
                    cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                    random_state=seed,
                    n_jobs=1,
                )
                search.fit(X_tr, y_tr)
                model = search.best_estimator_
                best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                mlflow.log_params({f"best_{k}": v for k, v in best.items()})
                mlflow.log_metric("cv_pr_auc", round(search.best_score_, 4))
                cfg.hyperparams = best  # reused by dependent experiments
            if cfg.calibrate:
                model = CalibratedClassifierCV(model, method="isotonic", cv=5)
            # Cross-validated PR-AUC on the training data: tells us whether
            # differences between configurations exceed run-to-run noise.
            cv_scores = cross_val_score(
                clone(model),
                X_tr,
                y_tr,
                scoring="average_precision",
                cv=StratifiedKFold(ev["cv_folds"], shuffle=True, random_state=seed),
            )
            mlflow.log_metrics(
                {
                    "cv_pr_auc_mean": round(float(cv_scores.mean()), 4),
                    "cv_pr_auc_std": round(float(cv_scores.std()), 4),
                }
            )
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_val)[:, 1]
        train_seconds = time.perf_counter() - start

        metrics = evaluate(
            y_val,
            scores,
            splits.X_val[fair_attr],
            ev["tier_a_fraction"],
            ev["tier_b_fraction"],
            is_probability=model is not None,
        )
        metrics["train_seconds"] = round(train_seconds, 2)
        if model is not None:
            metrics["cv_pr_auc_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_pr_auc_std"] = round(float(cv_scores.std()), 4)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"columns": columns}, "feature_columns.json")
        if model is not None:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=infer_signature(X_val.head(5), scores[:5]),
                input_example=X_val.head(3),
            )
        logger.info("%s %s", cfg.exp_id, json.dumps(metrics))
        return ExperimentResult(cfg, run.info.run_id, metrics, model)
