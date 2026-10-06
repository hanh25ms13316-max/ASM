"""Configuration loading and global reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Infrastructure settings that come from environment variables."""

    tracking_uri: str
    params_path: Path
    data_path: Path | None


def load_settings() -> Settings:
    """Read environment variables, falling back to safe local defaults.

    Returns:
        Settings: resolved infrastructure settings.
    """
    data_env = os.getenv("DROPOUT_DATA_PATH")
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"),
        params_path=Path(os.getenv("DROPOUT_PARAMS_PATH", PROJECT_ROOT / "params.yaml")),
        data_path=Path(data_env) if data_env else None,
    )


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML parameter file.

    Args:
        path: Optional explicit path; defaults to ``DROPOUT_PARAMS_PATH`` or
            ``params.yaml`` at the project root.

    Returns:
        dict: nested configuration dictionary.
    """
    params_path = Path(path) if path else load_settings().params_path
    with open(params_path, encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params


def set_global_seed(seed: int) -> None:
    """Fix every source of randomness used by the pipeline.

    Args:
        seed: integer seed taken from ``params.yaml``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
