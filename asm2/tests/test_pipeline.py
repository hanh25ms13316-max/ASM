"""Unit tests for data validation, features, metrics and orchestration tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout.config import PROJECT_ROOT, load_params
from dropout.data import DataValidationError, ingest, load_splits, validate
from dropout.evaluate import precision_at, recall_at, top_k_mask
from dropout.features import EngineeredFeatures, select_columns
from dropout.pipeline_tasks import psi

PARAMS = load_params(PROJECT_ROOT / "params.yaml")
RAW = PROJECT_ROOT / PARAMS["data"]["raw_path"]


def test_sensitive_attributes_never_selected() -> None:
    cols = select_columns(PARAMS, list(PARAMS["features"]["groups"]))
    assert not set(cols) & set(PARAMS["features"]["sensitive"])


def test_validation_rejects_out_of_range_values() -> None:
    df = ingest(RAW)
    df.loc[0, "Age at enrollment"] = 7
    with pytest.raises(DataValidationError):
        validate(df, PARAMS)


def test_split_is_reproducible_and_disjoint() -> None:
    a, b = load_splits(PARAMS, RAW), load_splits(PARAMS, RAW)
    assert a.X_test.index.equals(b.X_test.index)
    assert not set(a.X_train.index) & set(a.X_test.index)
    assert abs(a.y_train.mean() - a.y_test.mean()) < 0.01  # stratified


def test_engineered_features_handle_zero_enrolment() -> None:
    df = pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [0, 5],
            "Curricular units 1st sem (approved)": [0, 4],
        }
    )
    out = EngineeredFeatures().transform(df)
    assert out["approval_rate_s1"].tolist() == [0.0, 0.8]


def test_top_k_metrics() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1, 0.2, 0.1, 0.05])
    assert top_k_mask(s, 0.2).sum() == 2
    assert recall_at(y, s, 0.2) == pytest.approx(2 / 3)
    assert precision_at(y, s, 0.2) == 1.0


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.2
