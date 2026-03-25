from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _decode_ts_ns(v: object) -> pd.DatetimeIndex:
    arr = np.asarray(v)
    if arr.size == 0:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.to_datetime(arr.astype("int64"), utc=True)


def _as_float(v: object) -> np.ndarray:
    return np.asarray(v, dtype=float)


def _collect_npy(path_like: str) -> list[Path]:
    p = Path(path_like)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.npy"))
    return []


def convert_one(npy_path: Path) -> Path | None:
    meta = np.load(str(npy_path), allow_pickle=True).item()
    if str(meta.get("kind", "")) != "forecast_layer":
        return None

    x = _decode_ts_ns(meta.get("target_ts_ns", []))
    close_gt = _as_float(meta.get("close_gt_t1", []))
    pred = _as_float(meta.get("pred_price_t1", []))
    band_lo = _as_float(meta.get("band_lo_t1", []))
    band_hi = _as_float(meta.get("band_hi_t1", []))
    title = str(meta.get("title", npy_path.stem))

    n = min(len(x), len(close_gt), len(pred))
    if n == 0:
        return None
    x = x[:n]
    close_gt = close_gt[:n]
    pred = pred[:n]

    has_band = len(band_lo) >= n and len(band_hi) >= n
    if has_band:
        band_lo = band_lo[:n]
        band_hi = band_hi[:n]

    fig = plt.figure(figsize=(14, 6))
    plt.plot(x, close_gt, label="Close (GT, t+1)", alpha=0.8, color="tab:orange")
    plt.plot(x, pred, label="Pred Price (t+1)", linewidth=2, color="tab:blue")
    if has_band:
        plt.fill_between(x, band_lo, band_hi, alpha=0.25, label="Pred Band", color="lightblue")

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = npy_path.with_suffix(".svg")
    plt.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render forecast_layer .npy into SVG in the same folder."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="reports/exp_results_meta",
        help="A .npy file or a directory (recursive search).",
    )
    args = parser.parse_args()

    npy_files = _collect_npy(args.input)
    if not npy_files:
        print(f"[WARN] no input files found: {args.input}")
        return

    converted = 0
    for npy_path in npy_files:
        out = convert_one(npy_path)
        if out is not None:
            converted += 1
            print(f"[OK] {npy_path} -> {out}")
    print(f"[DONE] converted {converted} forecast_layer file(s).")


if __name__ == "__main__":
    main()

