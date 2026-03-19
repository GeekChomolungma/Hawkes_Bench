from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_NAME_RE = re.compile(
    r"(?P<symbol>[A-Za-z0-9]+)_(?P<interval>[0-9]+[A-Za-z]+)_Binance\.csv$",
    flags=re.IGNORECASE,
)


def _interval_to_pandas_freq(interval: str) -> str:
    s = interval.strip().lower()
    m = re.match(r"^(\d+)([a-z]+)$", s)
    if not m:
        raise ValueError(f"Unsupported interval format: {interval}")
    n = int(m.group(1))
    u = m.group(2)
    if u == "d":
        return f"{n}D"
    if u == "h":
        return f"{n}h"
    if u == "m":
        return f"{n}min"
    raise ValueError(f"Unsupported interval unit: {interval}")


def _freq_from_filename(csv_path: Path) -> str | None:
    m = _NAME_RE.match(csv_path.name)
    if not m:
        return None
    return _interval_to_pandas_freq(m.group("interval"))


def _pick_ts_col(df: pd.DataFrame) -> str:
    if "starttime" in df.columns:
        return "starttime"
    if "eventtime" in df.columns:
        return "eventtime"
    raise ValueError("CSV must contain either 'starttime' or 'eventtime'.")


def _infer_epoch_unit(raw: pd.Series) -> str:
    v = pd.to_numeric(raw, errors="coerce").dropna().abs()
    if v.empty:
        return "ms"
    med = float(v.median())
    # Typical Unix epoch:
    # - seconds: ~1e9
    # - milliseconds: ~1e12
    if med >= 1e11:
        return "ms"
    return "s"


def _infer_freq_from_data(ts: pd.Series) -> str:
    dt = ts.sort_values().diff().dropna()
    if dt.empty:
        raise ValueError("Cannot infer frequency from <= 1 timestamp.")
    # Use Timedelta arithmetic to avoid depending on datetime storage unit (ms/ns).
    ms = int(dt.median() / pd.Timedelta(milliseconds=1))
    if ms <= 0:
        raise ValueError("Invalid timestamp sequence for frequency inference.")
    return f"{ms}ms"


@dataclass
class CleanResult:
    file_name: str
    freq_used: str
    rows_before: int
    rows_after: int
    inserted_rows: int
    out_path: str


def clean_one_csv(csv_path: Path, out_dir: Path) -> CleanResult:
    df = pd.read_csv(csv_path)
    ts_col = _pick_ts_col(df)

    ts_unit = _infer_epoch_unit(df[ts_col])
    ts = pd.to_datetime(pd.to_numeric(df[ts_col], errors="coerce"), unit=ts_unit, utc=True, errors="coerce")
    bad_ts = ts.isna().sum()
    if bad_ts > 0:
        df = df.loc[~ts.isna()].copy()
        ts = ts.loc[~ts.isna()]

    df["_ts"] = ts
    df = df.sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last").set_index("_ts")
    # Snapshot original (post-dedup) rows for strict preservation checks.
    raw_aligned = df.copy()

    freq = _freq_from_filename(csv_path)
    if freq is None:
        freq = _infer_freq_from_data(df.index.to_series())

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz="UTC")
    before = len(df)
    df = df.reindex(full_idx)
    inserted = len(df) - before
    inserted_mask = ~df.index.isin(raw_aligned.index)

    # Interpolate numeric columns in time, then ffill/bfill as fallback
    ts_cols = {"starttime", "eventtime"}
    numeric_cols = [c for c in df.columns if c not in ts_cols and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        num_filled = df[numeric_cols].interpolate(method="time").ffill().bfill()
        # Only fill inserted rows; never modify existing raw-timestamp rows.
        df.loc[inserted_mask, numeric_cols] = num_filled.loc[inserted_mask, numeric_cols]

    # Fill non-numeric columns by nearest known values
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in ts_cols]
    for c in other_cols:
        col_filled = df[c].ffill().bfill()
        # Only fill inserted rows; preserve raw values exactly on overlap timestamps.
        df.loc[inserted_mask, c] = col_filled.loc[inserted_mask]

    # Rebuild ms timestamp columns from repaired index.
    # Force conversion to ns first, then convert ns -> ms so this works for
    # both datetime64[ms, UTC] and datetime64[ns, UTC] backends.
    ts_ns = (
        df.index.tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    ts_ms = (ts_ns // 10**6).astype("int64")
    if "starttime" in df.columns:
        start_col = df["starttime"].copy()
        start_col.loc[inserted_mask] = ts_ms[inserted_mask]
        df["starttime"] = pd.to_numeric(start_col, errors="coerce").astype("int64")
    if "eventtime" in df.columns:
        event_col = df["eventtime"].copy()
        event_col.loc[inserted_mask] = ts_ms[inserted_mask]
        df["eventtime"] = pd.to_numeric(event_col, errors="coerce").astype("int64")

    # Strict guarantee: on raw timestamps, every shared column value must match.
    overlap_idx = raw_aligned.index.intersection(df.index)
    shared_cols = [c for c in raw_aligned.columns if c in df.columns]
    mismatch_count = 0
    for c in shared_cols:
        a = raw_aligned.loc[overlap_idx, c]
        b = df.loc[overlap_idx, c]
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        both_num = (a_num.notna() | b_num.notna())
        if bool(both_num.any()):
            equal = np.isclose(a_num.to_numpy(dtype=float), b_num.to_numpy(dtype=float), equal_nan=True, rtol=0.0, atol=0.0)
            mismatch_count += int((~equal).sum())
        else:
            aa = a.astype("string").fillna("<NA>")
            bb = b.astype("string").fillna("<NA>")
            mismatch_count += int((aa != bb).sum())
    if mismatch_count > 0:
        raise ValueError(
            f"Strict clean check failed for {csv_path.name}: "
            f"{mismatch_count} mismatched cell(s) on raw timestamps."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_cleaned.csv"
    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False)

    return CleanResult(
        file_name=csv_path.name,
        freq_used=freq,
        rows_before=before,
        rows_after=len(df),
        inserted_rows=inserted,
        out_path=str(out_path),
    )


def clean_market_info_dir(
    in_dir: str = "market_info",
    out_dir: str = "market_info/cleaned",
) -> pd.DataFrame:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    files = sorted(in_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {in_path}")

    results: list[CleanResult] = []
    for f in files:
        # skip already cleaned files if they are in the same folder by accident
        if f.stem.endswith("_cleaned"):
            continue
        res = clean_one_csv(f, out_path)
        results.append(res)
        print(
            f"[cleaned] {res.file_name} | freq={res.freq_used} | "
            f"before={res.rows_before}, after={res.rows_after}, inserted={res.inserted_rows}"
        )

    summary = pd.DataFrame([r.__dict__ for r in results])
    summary_path = out_path / "clean_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[summary] saved: {summary_path}")
    return summary


if __name__ == "__main__":
    clean_market_info_dir()
