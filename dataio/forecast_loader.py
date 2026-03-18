from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from dataio.validators import REQUIRED_COLUMNS, run_all_validations


@dataclass
class ForecastLoadConfig:
    path: str
    column_map: Mapping[str, str] | None = None
    symbol: str | None = None
    horizon: int = 1


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported forecast file format: {suffix}")


def standardize_forecast_df(
    raw_df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
    symbol: str | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    df = raw_df.copy()

    if column_map:
        reverse = {v: k for k, v in column_map.items()}
        df = df.rename(columns=reverse)

    if "ts" not in df.columns:
        # fallback for common names
        for alt in ("ts_decision", "datetime", "timestamp", "date", "time"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ts"})
                break

    # zeroshot black-box conventions
    alias_pairs = (
        ("pred_ret_t+1_mean", "mu_pred"),
        ("pred_ret_t+1_q0.1", "q10"),
        ("pred_ret_t+1_q0.2", "q20"),
        ("pred_ret_t+1_q0.5", "q50"),
        ("pred_ret_t+1_q0.8", "q80"),
        ("pred_ret_t+1_q0.9", "q90"),
        ("y_true_ret_t+1", "y_true_next"),
    )
    for src, dst in alias_pairs:
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # "pred_ret_t+1_mean" in current zeroshot files is actually median (q50).
    if "q50" not in df.columns and "mu_pred" in df.columns:
        df["q50"] = df["mu_pred"]

    if "close_t" not in df.columns:
        for alt in ("close", "close_price", "price"):
            if alt in df.columns:
                df = df.rename(columns={alt: "close_t"})
                break

    if "symbol" not in df.columns:
        df["symbol"] = symbol if symbol is not None else "UNKNOWN"

    if "horizon" not in df.columns:
        df["horizon"] = int(horizon)

    if "ts" not in df.columns:
        raise ValueError("Cannot infer 'ts' column. Please provide column_map.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # enforce numeric conversion for expected numeric columns if present
    numeric_candidates = [
        "close_t",
        "mu_pred",
        "sigma_pred",
        "q05",
        "q10",
        "q20",
        "q25",
        "q50",
        "q75",
        "q80",
        "q90",
        "q95",
        "y_true_next",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c in df.columns])
    return df


def load_external_forecast(cfg: ForecastLoadConfig) -> pd.DataFrame:
    raw = _read_table(cfg.path)
    df = standardize_forecast_df(
        raw_df=raw,
        column_map=cfg.column_map,
        symbol=cfg.symbol,
        horizon=cfg.horizon,
    )

    result = run_all_validations(df)
    if not result.ok:
        raise ValueError("; ".join(result.errors))

    for w in result.warnings:
        print(f"[forecast_loader][WARN] {w}")

    return df


def align_forecast_with_market(forecast_df: pd.DataFrame, close: pd.Series, symbol: str) -> pd.DataFrame:
    df = forecast_df.copy()
    idx = close.index
    df = df[df["symbol"] == symbol].copy()
    df = df.set_index("ts").sort_index()
    df = df.loc[df.index.intersection(idx)].copy()

    if "close_t" not in df.columns:
        df["close_t"] = close.reindex(df.index).astype(float)
    else:
        # trust market data as the source of truth for close_t
        df["close_t"] = close.reindex(df.index).astype(float)

    return df.reset_index().rename(columns={"index": "ts"})
