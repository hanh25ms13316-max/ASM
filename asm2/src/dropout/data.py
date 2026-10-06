"""Data ingestion, validation and splitting stages."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Raised when a data quality gate fails; the pipeline must stop."""


@dataclass
class DatasetSplits:
    """Train / validation / test partitions plus the data fingerprint."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    data_hash: str


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file so every run records which data it used."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest(path: str | Path, separator: str = ";") -> pd.DataFrame:
    """Stage 1 - read the raw extract and normalise column names.

    Args:
        path: CSV exported from the source systems.
        separator: field separator of the export.

    Returns:
        pd.DataFrame: raw records with stripped column names.
    """
    df = pd.read_csv(path, sep=separator)
    df.columns = [c.strip() for c in df.columns]  # source export has a trailing tab
    logger.info("Ingested %d rows x %d columns from %s", *df.shape, path)
    return df


def build_schema(params: dict[str, Any]) -> pa.DataFrameSchema:
    """Declare the expected structure of the raw data as code."""
    target = params["data"]["target_column"]
    return pa.DataFrameSchema(
        {
            target: pa.Column(str, pa.Check.isin(["Dropout", "Enrolled", "Graduate"])),
            "Age at enrollment": pa.Column(int, pa.Check.in_range(15, 80)),
            "Admission grade": pa.Column(float, pa.Check.in_range(0, 200)),
            "Curricular units 1st sem (approved)": pa.Column(int, pa.Check.ge(0)),
            "Curricular units 1st sem (grade)": pa.Column(float, pa.Check.in_range(0, 20)),
            "Tuition fees up to date": pa.Column(int, pa.Check.isin([0, 1])),
            "Debtor": pa.Column(int, pa.Check.isin([0, 1])),
        },
        strict=False,
        coerce=True,
    )


def validate(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Stage 2 - quality gate: schema, volume and completeness checks.

    Raises:
        DataValidationError: if any check fails, so no model is trained or
            scored on corrupted data.
    """
    cfg = params["data"]
    if len(df) < cfg["min_rows"]:
        raise DataValidationError(f"Only {len(df)} rows, expected >= {cfg['min_rows']}")
    missing = float(df.isna().mean().max())
    if missing > cfg["max_missing_ratio"]:
        raise DataValidationError(f"Missing ratio {missing:.2%} above threshold")
    try:
        df = build_schema(params).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(str(exc.failure_cases.head())) from exc
    logger.info("Validation passed (max missing ratio %.2f%%)", missing * 100)
    return df


def make_label(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Binary label: 1 if the student dropped out, 0 if enrolled or graduated."""
    cfg = params["data"]
    return (df[cfg["target_column"]] == cfg["positive_label"]).astype(int)


def split(df: pd.DataFrame, params: dict[str, Any], data_hash: str) -> DatasetSplits:
    """Stratified train / validation / test split with a fixed seed.

    The public proxy dataset has no enrolment date, so the split is stratified
    rather than time-based; production data would use a semester-based split.
    """
    seed = params["project"]["seed"]
    y = make_label(df, params)
    X = df.drop(columns=[params["data"]["target_column"]])
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], stratify=y, random_state=seed
    )
    val_share = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_share, stratify=y_tmp, random_state=seed
    )
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test, data_hash)


def load_splits(params: dict[str, Any], path: str | Path | None = None) -> DatasetSplits:
    """Run ingestion -> validation -> splitting in one call."""
    raw_path = Path(path or params["data"]["raw_path"])
    df = validate(ingest(raw_path, params["data"]["separator"]), params)
    return split(df, params, file_sha256(raw_path))
