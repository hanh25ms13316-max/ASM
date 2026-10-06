"""Preprocessing and feature engineering shared by training and scoring.

Keeping one implementation for both paths is the main defence against
training-serving skew.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Application mode", "Course", "Previous qualification"]


def select_columns(params: dict[str, Any], groups: list[str]) -> list[str]:
    """Return the raw columns belonging to the requested feature groups.

    Sensitive attributes are removed even if a group lists them.
    """
    sensitive = set(params["features"]["sensitive"])
    cols: list[str] = []
    for group in groups:
        cols.extend(c for c in params["features"]["groups"][group] if c not in sensitive)
    return cols


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """Add ratio and trend features derived from semester results.

    Only features whose source columns are present are created, so the same
    transformer works for "enrollment only" and "full" feature sets.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> EngineeredFeatures:
        """Stateless transformer; nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``X`` with engineered columns appended."""
        out = X.copy()
        for sem, tag in (("1st", "s1"), ("2nd", "s2")):
            enrolled = f"Curricular units {sem} sem (enrolled)"
            approved = f"Curricular units {sem} sem (approved)"
            if enrolled in out and approved in out:
                # Share of enrolled modules passed; 0 when nothing was enrolled.
                out[f"approval_rate_{tag}"] = np.where(
                    out[enrolled] > 0, out[approved] / out[enrolled].clip(lower=1), 0.0
                )
        grade1, grade2 = "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"
        if grade1 in out and grade2 in out:
            out["grade_trend"] = out[grade2] - out[grade1]
        if "Debtor" in out and "Tuition fees up to date" in out:
            out["financial_risk"] = out["Debtor"] + (1 - out["Tuition fees up to date"])
        return out


def build_preprocessor(columns: list[str], scale: bool) -> ColumnTransformer:
    """One-hot encode coded categoricals; optionally scale numeric columns.

    Args:
        columns: columns present after feature engineering.
        scale: True for linear models, False for tree models.
    """
    cats = [c for c in CATEGORICAL if c in columns]
    nums = [c for c in columns if c not in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            ("num", StandardScaler() if scale else "passthrough", nums),
        ],
        verbose_feature_names_out=False,
    )


def engineered_columns(raw_columns: list[str]) -> list[str]:
    """Column names after ``EngineeredFeatures`` for a given raw selection."""
    sample = pd.DataFrame(np.zeros((1, len(raw_columns))), columns=raw_columns)
    return list(EngineeredFeatures().transform(sample).columns)
