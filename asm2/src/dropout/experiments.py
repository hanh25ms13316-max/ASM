"""Run the full experiment matrix, select a model and register it.

Usage:
    python -m dropout.experiments
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from dropout.config import PROJECT_ROOT, load_params, load_settings, set_global_seed
from dropout.data import load_splits
from dropout.evaluate import evaluate
from dropout.features import select_columns
from dropout.registry import champion_metric, passes_quality_gate, promote, register_candidate
from dropout.train import ExperimentConfig, ExperimentResult, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dropout.experiments")

FULL = ["enrollment", "financial", "semester1", "semester2"]

# Experiment matrix. E07 tunes XGBoost; E08-E12 reuse its best hyperparameters.
MATRIX: list[ExperimentConfig] = [
    ExperimentConfig("E01", "Baseline B2: extended rule", "rule"),
    ExperimentConfig("E02", "Baseline B3: logistic regression", "logreg"),
    ExperimentConfig("E03", "Logistic regression, class_weight", "logreg", balance="class_weight"),
    ExperimentConfig(
        "E04",
        "Random forest, class_weight",
        "rf",
        balance="class_weight",
        hyperparams={"min_samples_leaf": 3},
    ),
    ExperimentConfig("E05", "XGBoost default", "xgb"),
    ExperimentConfig("E06", "XGBoost, scale_pos_weight", "xgb", balance="class_weight"),
    ExperimentConfig("E07", "XGBoost tuned (random search, 5-fold CV)", "xgb", tune=True),
    ExperimentConfig("E08", "XGBoost tuned + SMOTE", "xgb", balance="smote"),
    ExperimentConfig("E09", "XGBoost tuned + isotonic calibration", "xgb", calibrate=True),
    ExperimentConfig(
        "E10",
        "XGBoost tuned, enrollment data only (earliest)",
        "xgb",
        feature_groups=["enrollment", "financial"],
    ),
    ExperimentConfig(
        "E11",
        "XGBoost tuned, up to semester 1 (early)",
        "xgb",
        feature_groups=["enrollment", "financial", "semester1"],
    ),
    ExperimentConfig(
        "E12",
        "XGBoost tuned, no financial features (privacy)",
        "xgb",
        feature_groups=["enrollment", "semester1", "semester2"],
    ),
]
DEPENDS_ON_TUNED = {"E08", "E09", "E10", "E11", "E12"}


def git_commit() -> str:
    """Current git commit, recorded on every registered model."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def select_model(results: list[ExperimentResult], params: dict) -> ExperimentResult:
    """Pick the production candidate.

    Rules: (1) only ML models using the full feature set that pass the
    fairness limit are eligible; (2) the top PR-AUC model is chosen only if
    it beats the simple baseline by ``min_gain_over_baseline``; otherwise the
    simpler, explainable baseline is kept.
    """
    sel = params["selection"]
    max_gap = params["quality_gate"]["max_fairness_gap"]
    eligible = [
        r
        for r in results
        if r.model is not None
        and r.config.feature_groups == FULL
        and r.val_metrics["fairness_gap_at_25"] <= max_gap
    ]
    top = max(eligible, key=lambda r: r.val_metrics["pr_auc"])
    simple = next(r for r in results if r.config.exp_id == sel["simple_baseline"])
    gain = top.val_metrics["pr_auc"] - simple.val_metrics["pr_auc"]
    logger.info("Top model %s, gain over %s = %.4f", top.config.exp_id, simple.config.exp_id, gain)
    return top if gain >= sel["min_gain_over_baseline"] else simple


def main() -> pd.DataFrame:
    """Run every experiment, then gate, register and promote the best model."""
    params = load_params()
    set_global_seed(params["project"]["seed"])
    mlflow.set_tracking_uri(load_settings().tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    splits = load_splits(params, PROJECT_ROOT / params["data"]["raw_path"])
    results: list[ExperimentResult] = []
    tuned: dict = {}
    for cfg in MATRIX:
        if cfg.exp_id in DEPENDS_ON_TUNED:
            cfg.hyperparams = dict(tuned)
        res = run_experiment(cfg, splits, params)
        if cfg.exp_id == "E07":
            tuned = dict(res.config.hyperparams)
        results.append(res)

    table = pd.DataFrame(
        [
            {
                "exp_id": r.config.exp_id,
                "description": r.config.description,
                **r.val_metrics,
                "run_id": r.run_id,
            }
            for r in results
        ]
    )
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "experiment_results.csv", index=False)

    best = select_model(results, params)
    client = MlflowClient()
    name = params["mlflow"]["registered_model"]
    passed, reasons = passes_quality_gate(best.val_metrics, params, champion_metric(client, name))
    summary: dict = {
        "selected": best.config.exp_id,
        "top_by_pr_auc": max(
            (r for r in results if r.model is not None), key=lambda r: r.val_metrics["pr_auc"]
        ).config.exp_id,
        "gate_passed": passed,
        "reasons": reasons,
        "tuned_hyperparams": tuned,
    }
    if passed:
        version = register_candidate(
            best.run_id,
            params,
            {
                "exp_id": best.config.exp_id,
                "git_commit": git_commit(),
                "data_sha256": splits.data_hash[:12],
            },
        )
        promote(params, version, approved_by="demo-reviewer")
        # Final, one-time check on the untouched test set.
        cols = select_columns(params, best.config.feature_groups)
        test_scores = best.model.predict_proba(splits.X_test[cols])[:, 1]
        ev = params["evaluation"]
        summary["test_metrics"] = evaluate(
            splits.y_test,
            test_scores,
            splits.X_test[params["features"]["fairness_attribute"]],
            ev["tier_a_fraction"],
            ev["tier_b_fraction"],
        )
        summary["model_version"] = version
    Path(out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Selection: %s", json.dumps(summary))
    return table


if __name__ == "__main__":
    main()
