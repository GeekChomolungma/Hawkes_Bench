from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


# -------------------------------------------------------------------
# 1) Hardcoded inputs
# -------------------------------------------------------------------

MARKET_CSV_PATHS: Dict[str, str] = {
    "BTCUSDT": "market_info/cleaned/BTCUSDT_1d_Binance_cleaned.csv",
    "DOGEUSDT": "market_info/cleaned/DOGEUSDT_1d_Binance_cleaned.csv",
}

# Shared windows for both datasets (editable)
TRAIN_START = "2021-01-01"
TRAIN_END = "2025-10-31"
VAL_START = "2025-11-01"
VAL_END = "2025-11-30"
TEST_START = "2025-12-10"
TEST_END = "2026-01-25"

# Segment colors (editable)
TRAIN_BG_COLOR = "#4E677D"  # light blue
VAL_BG_COLOR = "#5F3D4E"    # light pink
TEST_BG_COLOR = "#BC9B6D"   # light orange

# Plot styles
KLINE_UP_COLOR = "#2ca02c"
KLINE_DOWN_COLOR = "#d62728"
# Candlestick body width ratio relative to median bar spacing.
# Smaller value -> larger visual gap between candles.
KLINE_BODY_WIDTH_RATIO = 0.38
FIGSIZE = (16, 9)
# Extra x-axis padding on both sides (ratio of [train_start, test_end] span).
# Increase right pad to separate labels in late-period segments.
X_LEFT_PAD_RATIO = 0.01
X_RIGHT_PAD_RATIO = 0.08
# Typography (hardcoded, editable)
TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
AXIS_TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 16

OUTPUT_DIR = Path("reports/figures/market_kline")


# -------------------------------------------------------------------
# 2) Helpers
# -------------------------------------------------------------------

def _to_utc_naive(ts_like: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None)


def _decode_epoch_auto(arr_like: object) -> pd.DatetimeIndex:
    arr = np.asarray(arr_like)
    if arr.size == 0:
        return pd.DatetimeIndex([])

    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, utc=True).tz_convert("UTC").tz_localize(None)

    raw = arr.astype("int64")
    m = int(np.nanmax(np.abs(raw)))
    if m >= 10**17:
        unit = "ns"
    elif m >= 10**14:
        unit = "us"
    elif m >= 10**11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(raw, unit=unit, utc=True).tz_convert("UTC").tz_localize(None)


def _load_market_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    required = {"starttime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {p}: {sorted(missing)}")

    ts = _decode_epoch_auto(df["starttime"].to_numpy())
    out = pd.DataFrame(
        {
            "ts": ts,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    ).dropna()

    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return out


def _shade_windows(ax: plt.Axes) -> None:
    t0 = _to_utc_naive(TRAIN_START)
    t1 = _to_utc_naive(TRAIN_END)
    v0 = _to_utc_naive(VAL_START)
    v1 = _to_utc_naive(VAL_END)
    e0 = _to_utc_naive(TEST_START)
    e1 = _to_utc_naive(TEST_END)

    ax.axvspan(t0, t1, color=TRAIN_BG_COLOR, alpha=0.35, zorder=0)
    ax.axvspan(v0, v1, color=VAL_BG_COLOR, alpha=0.35, zorder=0)
    ax.axvspan(e0, e1, color=TEST_BG_COLOR, alpha=0.35, zorder=0)


def _draw_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = mdates.date2num(df["ts"].to_numpy())
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    price_span = float(np.nanmax(h) - np.nanmin(l)) if len(h) > 0 else 0.0
    min_body = price_span * 0.0008 if price_span > 0 else 1e-6

    step = np.diff(x)
    step = step[np.isfinite(step) & (step > 0)]
    median_step = float(np.median(step)) if len(step) > 0 else 1.0
    body_width = max(0.01, median_step * max(0.05, float(KLINE_BODY_WIDTH_RATIO)))

    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        up = ci >= oi
        color = KLINE_UP_COLOR if up else KLINE_DOWN_COLOR

        ax.vlines(xi, li, hi, color=color, linewidth=0.8, alpha=0.9, zorder=2)

        body_bottom = min(oi, ci)
        body_height = max(abs(ci - oi), min_body)
        rect = Rectangle(
            (xi - body_width / 2.0, body_bottom),
            body_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)


def _add_segment_labels_on_timeline(ax: plt.Axes) -> None:
    t0 = _to_utc_naive(TRAIN_START)
    t1 = _to_utc_naive(TRAIN_END)
    v0 = _to_utc_naive(VAL_START)
    v1 = _to_utc_naive(VAL_END)
    e0 = _to_utc_naive(TEST_START)
    e1 = _to_utc_naive(TEST_END)

    segments = [
        ("Train Data", t0, t1, "#2F6FAA"),
        ("Val Data", v0, v1, "#B24E8E"),
        ("Test Data", e0, e1, "#C47A1F"),
    ]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for name, a, b, color in segments:
        if b <= a:
            continue
        mid = a + (b - a) / 2
        ax.text(
            mid,
            -0.18,
            name,
            transform=trans,
            ha="center",
            va="top",
            fontsize=10,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _compute_logreturn_ylim(csv_path: str) -> Tuple[float, float]:
    df = _load_market_csv(csv_path)
    start = _to_utc_naive(TRAIN_START)
    end = _to_utc_naive(TEST_END)
    view = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if view.empty:
        raise ValueError(f"No rows in [{TRAIN_START}, {TEST_END}] for DOGEUSDT: {csv_path}")

    view["log_return"] = np.log(view["close"]).diff().fillna(0.0)
    y_min = float(view["log_return"].min())
    y_max = float(view["log_return"].max())

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError(f"Invalid DOGE logreturn range from: {csv_path}")
    if y_min == y_max:
        pad = 1e-6
        return y_min - pad, y_max + pad
    return y_min, y_max


def _plot_one_symbol(
    symbol: str,
    csv_path: str,
    output_dir: Path,
    logreturn_ylim: Tuple[float, float] | None = None,
) -> Path:
    df = _load_market_csv(csv_path)

    start = _to_utc_naive(TRAIN_START)
    end = _to_utc_naive(TEST_END)
    view = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if view.empty:
        raise ValueError(f"No rows in [{TRAIN_START}, {TEST_END}] for {symbol}: {csv_path}")

    view["log_return"] = np.log(view["close"]).diff().fillna(0.0)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )

    _shade_windows(ax1)
    _shade_windows(ax2)

    _draw_candles(ax1, view)
    ax1.set_ylabel("Price", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title(f"{symbol} 1d Candlestick with Train/Val/Test Segments", fontsize=TITLE_FONTSIZE)
    ax1.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax1.grid(True, linestyle="--", alpha=0.25)

    ax2.plot(view["ts"], view["log_return"], color="#4C4C4C", linewidth=1.1, alpha=0.9)
    ax2.axhline(0.0, color="#303030", linewidth=0.8, alpha=0.9)
    ax2.set_ylabel("LogReturn", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax2.grid(True, linestyle="--", alpha=0.25)
    if logreturn_ylim is not None:
        ax2.set_ylim(*logreturn_ylim)

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _add_segment_labels_on_timeline(ax2)

    # Expand horizontal viewport so right-tail segment labels have more room.
    base_left = _to_utc_naive(TRAIN_START)
    base_right = _to_utc_naive(TEST_END)
    span = base_right - base_left
    if span > pd.Timedelta(0):
        x_left = base_left - span * float(max(0.0, X_LEFT_PAD_RATIO))
        x_right = base_right + span * float(max(0.0, X_RIGHT_PAD_RATIO))
        ax1.set_xlim(x_left, x_right)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=TRAIN_BG_COLOR, edgecolor="none", alpha=0.35, label="Train"),
        Rectangle((0, 0), 1, 1, facecolor=VAL_BG_COLOR, edgecolor="none", alpha=0.35, label="Val"),
        Rectangle((0, 0), 1, 1, facecolor=TEST_BG_COLOR, edgecolor="none", alpha=0.35, label="Test"),
    ]
    ax1.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=LEGEND_FONTSIZE)

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_kline_logreturn_train_val_test.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------
# 3) Entry
# -------------------------------------------------------------------

def main() -> None:
    doge_ylim = _compute_logreturn_ylim(MARKET_CSV_PATHS["DOGEUSDT"])

    generated: list[Path] = []
    for symbol, csv_path in MARKET_CSV_PATHS.items():
        generated.append(
            _plot_one_symbol(
                symbol=symbol,
                csv_path=csv_path,
                output_dir=OUTPUT_DIR,
                logreturn_ylim=doge_ylim,
            )
        )

    print(f"[DONE] generated {len(generated)} SVG files:")
    for p in generated:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
