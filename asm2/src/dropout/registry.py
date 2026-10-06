"""Model registry helpers: quality gate, registration and alias promotion."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def passes_quality_gate(
    metrics: dict[str, float], params: dict[str, Any], champion_pr_auc: float | None
) -> tuple[bool, list[str]]:
    """Check a candidate against the rules defined in ``params.yaml``.

    Returns:
        tuple: (passed, list of human-readable reasons for failure).
    """
    gate = params["quality_gate"]
    reasons: list[str] = []
    if metrics["pr_auc"] < gate["min_pr_auc"]:
        reasons.append(f"pr_auc {metrics['pr_auc']} < {gate['min_pr_auc']}")
    if metrics["fairness_gap_at_25"] > gate["max_fairness_gap"]:
        reasons.append(f"fairness gap {metrics['fairness_gap_at_25']} > {gate['max_fairness_gap']}")
    if champion_pr_auc is not None and (
        metrics["pr_auc"] < champion_pr_auc - gate["max_regression_vs_champion"]
    ):
        reasons.append(f"worse than champion ({champion_pr_auc})")
    return not reasons, reasons


def champion_metric(client: MlflowClient, name: str, metric: str = "pr_auc") -> float | None:
    """Return the champion's metric, or None if no champion exists yet."""
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get(metric)


def register_candidate(run_id: str, params: dict[str, Any], tags: dict[str, str]) -> str:
    """Register the run's model as a new version with alias ``challenger``.

    Returns:
        str: the new model version number.
    """
    name = params["mlflow"]["registered_model"]
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.set_registered_model_alias(name, "challenger", version.version)
    logger.info("Registered %s v%s as challenger", name, version.version)
    return str(version.version)


def promote(params: dict[str, Any], version: str, approved_by: str) -> None:
    """Move the ``champion`` alias after human approval (keeps history for rollback)."""
    name = params["mlflow"]["registered_model"]
    client = MlflowClient()
    try:
        previous = client.get_model_version_by_alias(name, "champion").version
        client.set_registered_model_alias(name, "previous-champion", previous)
    except MlflowException:
        pass  # first deployment: nothing to keep
    client.set_registered_model_alias(name, "champion", version)
    client.set_model_version_tag(name, version, "approved_by", approved_by)
    logger.info("Promoted %s v%s to champion (approved by %s)", name, version, approved_by)
