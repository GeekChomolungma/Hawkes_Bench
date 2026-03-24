from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_parent(path)
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=index)
    elif p.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(p, index=index)
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}")


def save_metrics(metrics: dict, path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _to_npy_value(v):
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, pd.DatetimeIndex):
        idx = v
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx.asi8
    if isinstance(v, pd.Index):
        return v.to_numpy()
    if isinstance(v, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(v.dtype):
            s = pd.to_datetime(v, utc=True, errors="coerce")
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            return s.to_numpy(dtype="datetime64[ns]").astype("int64")
        return v.to_numpy()
    if isinstance(v, (list, tuple)):
        return np.asarray(v)
    if isinstance(v, (str, bytes)):
        return np.asarray(v, dtype=object)
    if np.isscalar(v):
        return np.asarray(v)
    return np.asarray(v, dtype=object)


def save_npy_payload(payload: dict, path: str) -> None:
    """
    Save figure-aligned metadata payload as .npy (pickled dict of numpy arrays/scalars).
    Datetime indices are normalized to int64 nanoseconds (UTC-compatible).
    """
    ensure_parent(path)
    out = {k: _to_npy_value(v) for k, v in payload.items()}
    np.save(path, out, allow_pickle=True)
