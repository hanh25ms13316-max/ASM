"""Evaluation metrics aligned with the operating constraints of the system.

Lists are cut by contact capacity (top-k), not by a probability threshold,
so ranking metrics at k are the operational metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def top_k_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask selecting the ``fraction`` highest-risk students."""
    k = max(1, int(round(len(scores) * fraction)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def recall_at(y: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Share of all dropouts that fall inside the top ``fraction`` list."""
    return float(y[top_k_mask(scores, fraction)].sum() / max(1, y.sum()))


def precision_at(y: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Share of the top ``fraction`` list who actually drop out."""
    return float(y[top_k_mask(scores, fraction)].mean())


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Weighted gap between predicted probability and observed rate per bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(prob[sel].mean() - y[sel].mean())
    return float(ece)


def fairness_gap(y: np.ndarray, scores: np.ndarray, groups: pd.Series, fraction: float) -> float:
    """Largest difference in recall@fraction between groups (equal opportunity)."""
    mask = top_k_mask(scores, fraction)
    recalls = []
    for g in groups.unique():
        sel = (groups.values == g) & (y == 1)
        if sel.sum() > 0:
            recalls.append(mask[sel].mean())
    return float(max(recalls) - min(recalls)) if len(recalls) > 1 else 0.0


def evaluate(
    y: pd.Series,
    scores: np.ndarray,
    groups: pd.Series,
    tier_a: float,
    tier_b: float,
    is_probability: bool = True,
) -> dict[str, float]:
    """Compute the full metric set for one model on one data split.

    Args:
        y: true binary labels.
        scores: risk scores (probabilities for ML models, rule points for B2).
        groups: protected attribute used only for the fairness audit.
        tier_a: fraction contacted by advisors (e.g. 0.10).
        tier_b: cumulative fraction covered with the care team (e.g. 0.25).
        is_probability: False for rule scores, which have no calibration.
    """
    y_arr = y.to_numpy()
    metrics = {
        "pr_auc": average_precision_score(y_arr, scores),
        "roc_auc": roc_auc_score(y_arr, scores),
        "recall_at_10": recall_at(y_arr, scores, tier_a),
        "precision_at_10": precision_at(y_arr, scores, tier_a),
        "recall_at_25": recall_at(y_arr, scores, tier_b),
        "fairness_gap_at_25": fairness_gap(y_arr, scores, groups, tier_b),
    }
    if is_probability:
        metrics["brier"] = brier_score_loss(y_arr, scores)
        metrics["ece"] = expected_calibration_error(y_arr, scores)
    return {k: round(float(v), 4) for k, v in metrics.items()}
