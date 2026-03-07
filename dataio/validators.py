from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("ts", "symbol", "horizon", "close_t")
KNOWN_QUANTILES = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_required_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLUMNS) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]
    return []


def validate_unique_key(df: pd.DataFrame, key_cols: Sequence[str] = ("ts", "symbol", "horizon")) -> list[str]:
    if not set(key_cols).issubset(df.columns):
        return []
    dup = df.duplicated(list(key_cols), keep=False)
    if dup.any():
        n = int(dup.sum())
        return [f"Duplicate key rows by {key_cols}: {n}"]
    return []


def validate_quantile_monotonicity(df: pd.DataFrame, quantile_cols: Iterable[str] = KNOWN_QUANTILES) -> list[str]:
    qcols = [c for c in quantile_cols if c in df.columns]
    if len(qcols) < 2:
        return []

    arr = df[qcols].to_numpy(dtype=float)
    bad = np.any(np.diff(arr, axis=1) < -1e-12, axis=1)
    if bad.any():
        return [f"Quantile monotonicity violated on {int(bad.sum())} rows for columns {qcols}"]
    return []


def validate_ts_dtype(df: pd.DataFrame) -> list[str]:
    if "ts" not in df.columns:
        return []
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        return ["Column 'ts' is not datetime type after parsing."]
    return []


def run_all_validations(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(validate_required_columns(df))
    errors.extend(validate_ts_dtype(df))
    errors.extend(validate_unique_key(df))
    errors.extend(validate_quantile_monotonicity(df))

    # Soft warning: no predictive columns
    if "mu_pred" not in df.columns and not any(c in df.columns for c in KNOWN_QUANTILES):
        warnings.append("No predictive columns found (mu_pred or quantile columns).")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)
