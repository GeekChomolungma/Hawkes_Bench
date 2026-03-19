from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


_NAME_RE = re.compile(
    r"(?P<symbol>[A-Za-z0-9]+)_(?P<interval>[0-9]+[A-Za-z]+)_Binance(?:_cleaned)?\.csv$",
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


def _pick_ts_col(df: pd.DataFrame) -> str:
    if "starttime" in df.columns:
        return "starttime"
    if "eventtime" in df.columns:
        return "eventtime"
    raise ValueError("CSV must contain either 'starttime' or 'eventtime'.")


@dataclass
class VerifyResult:
    file_name: str
    status: str
    freq_expected: str
    rows: int
    ts_missing: int
    ts_duplicate: int
    ts_not_monotonic: int
    gap_count: int
    missing_close: int
    note: str


def verify_one_file(csv_path: Path) -> VerifyResult:
    df = pd.read_csv(csv_path)
    ts_col = _pick_ts_col(df)
    ts = pd.to_datetime(df[ts_col], unit="ms", utc=True, errors="coerce")

    m = _NAME_RE.match(csv_path.name)
    if not m:
        return VerifyResult(
            file_name=csv_path.name,
            status="FAIL",
            freq_expected="unknown",
            rows=len(df),
            ts_missing=int(ts.isna().sum()),
            ts_duplicate=-1,
            ts_not_monotonic=-1,
            gap_count=-1,
            missing_close=int(df["close"].isna().sum()) if "close" in df.columns else -1,
            note="Filename does not match {SYMBOL}_{INTERVAL}_Binance[_cleaned].csv",
        )

    freq = _interval_to_pandas_freq(m.group("interval"))
    ts_missing = int(ts.isna().sum())
    if ts_missing > 0:
        return VerifyResult(
            file_name=csv_path.name,
            status="FAIL",
            freq_expected=freq,
            rows=len(df),
            ts_missing=ts_missing,
            ts_duplicate=-1,
            ts_not_monotonic=-1,
            gap_count=-1,
            missing_close=int(df["close"].isna().sum()) if "close" in df.columns else -1,
            note="Timestamp parsing failed for some rows",
        )

    idx = pd.DatetimeIndex(ts).sort_values()
    ts_duplicate = int(idx.duplicated().sum())
    ts_not_monotonic = int(not idx.is_monotonic_increasing)
    year_min = int(idx.min().year)
    year_max = int(idx.max().year)

    full_idx = pd.date_range(start=idx.min(), end=idx.max(), freq=freq, tz="UTC")
    gap_count = int(len(full_idx) - len(idx.unique()))
    missing_close = int(df["close"].isna().sum()) if "close" in df.columns else -1

    fail_reasons: list[str] = []
    if ts_duplicate > 0:
        fail_reasons.append(f"duplicate_ts={ts_duplicate}")
    if ts_not_monotonic > 0:
        fail_reasons.append("timestamp not monotonic")
    # Guardrail for common seconds-vs-milliseconds scaling bugs.
    if year_min < 2000 or year_max > 2100:
        fail_reasons.append(f"timestamp year out of expected range: [{year_min}, {year_max}]")
    if gap_count > 0:
        fail_reasons.append(f"gaps={gap_count}")
    if "close" not in df.columns:
        fail_reasons.append("missing close column")
    elif missing_close > 0:
        fail_reasons.append(f"missing_close={missing_close}")

    status = "PASS" if not fail_reasons else "FAIL"
    note = "OK" if status == "PASS" else "; ".join(fail_reasons)

    return VerifyResult(
        file_name=csv_path.name,
        status=status,
        freq_expected=freq,
        rows=len(df),
        ts_missing=ts_missing,
        ts_duplicate=ts_duplicate,
        ts_not_monotonic=ts_not_monotonic,
        gap_count=gap_count,
        missing_close=missing_close,
        note=note,
    )


def verify_cleaned_dir(cleaned_dir: str = "market_info/cleaned") -> pd.DataFrame:
    base = Path(cleaned_dir)
    files = sorted(base.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found in {base}")

    results: list[VerifyResult] = []
    for f in files:
        if f.name in {"clean_summary.csv", "verify_summary.csv", "clean_trace_audit.csv"}:
            continue
        try:
            res = verify_one_file(f)
        except Exception as e:
            res = VerifyResult(
                file_name=f.name,
                status="FAIL",
                freq_expected="unknown",
                rows=len(pd.read_csv(f)) if f.exists() else -1,
                ts_missing=-1,
                ts_duplicate=-1,
                ts_not_monotonic=-1,
                gap_count=-1,
                missing_close=-1,
                note=f"verification exception: {e}",
            )
        results.append(res)
        print(f"[verify] {res.file_name} | {res.status} | {res.note}")

    summary = pd.DataFrame([asdict(r) for r in results])
    out = base / "verify_summary.csv"
    summary.to_csv(out, index=False)
    print(f"[verify-summary] saved: {out}")
    return summary


if __name__ == "__main__":
    verify_cleaned_dir()
