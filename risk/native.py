from __future__ import annotations

import numpy as np
import pandas as pd


def infer_expected_return(df: pd.DataFrame) -> pd.Series:
    if "mu_pred" in df.columns:
        return df["mu_pred"].astype(float)
    if "q50" in df.columns:
        return df["q50"].astype(float)
    raise ValueError("Cannot infer expected return: missing mu_pred and q50.")


def infer_native_risk(df: pd.DataFrame, low_q: str = "q10", high_q: str = "q90") -> pd.Series:
    if "sigma_pred" in df.columns:
        return df["sigma_pred"].astype(float)
    if low_q in df.columns and high_q in df.columns:
        band = (df[high_q].astype(float) - df[low_q].astype(float)).clip(lower=0.0)
        return (band / 2.0).astype(float)
    if "q25" in df.columns and "q75" in df.columns:
        band = (df["q75"].astype(float) - df["q25"].astype(float)).clip(lower=0.0)
        return (band / 2.0).astype(float)
    raise ValueError("Cannot infer native risk: missing sigma_pred or quantile bands.")


def attach_native_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expected_return"] = infer_expected_return(out)
    out["native_risk"] = infer_native_risk(out)
    out["native_risk"] = out["native_risk"].replace(0.0, np.nan)
    return out
