from __future__ import annotations

import json
from pathlib import Path

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
