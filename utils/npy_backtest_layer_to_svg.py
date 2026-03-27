from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# 1. Manually fill in the 4 models path
# -------------------------------------------------------------------

@dataclass
class BacktestPairPath:
    native_no_hawkes: str
    hawkes_scaled: str


# IMPORTANT:
# 1) Please hardcode your own .npy paths here.
# 2) Each path must point to a backtest_layer .npy file.
# 3) native_no_hawkes -> dashed line
# 4) hawkes_scaled    -> solid line
MODEL_BACKTEST_PATHS: Dict[str, BacktestPairPath] = {
    "ARIMA+GARCH": BacktestPairPath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/zeroshot/btcusdt/exp2_white_no_hawkes_BTCUSDT_1d.npy",
        hawkes_scaled="reports/exp_results_meta/ft/1d/zeroshot/btcusdt/exp2_white_hawkes_q70_BTCUSDT_1d.npy",
    ),
    "Chronos2 Zeroshot": BacktestPairPath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/zeroshot/btcusdt/exp2_black_no_hawkes_BTCUSDT_1d.npy",
        hawkes_scaled="reports/exp_results_meta/ft/1d/zeroshot/btcusdt/exp2_black_hawkes_q70_BTCUSDT_1d.npy",
    ),
    "Chronos2 Native FT": BacktestPairPath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/pretrained_native_all/btcusdt/exp2_black_no_hawkes_BTCUSDT_1d.npy",
        hawkes_scaled="reports/exp_results_meta/ft/1d/pretrained_native_all/btcusdt/exp2_black_hawkes_q70_BTCUSDT_1d.npy",
    ),
    "Chronos2 Proposed FT": BacktestPairPath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/btcusdt/exp2_black_no_hawkes_BTCUSDT_1d.npy",
        hawkes_scaled="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/btcusdt/exp2_black_hawkes_q70_BTCUSDT_1d.npy",
    ),
}


# -------------------------------------------------------------------
# 2. Color mapping (same as bar_plot_for_metrics_btc)
# -------------------------------------------------------------------

MODEL_COLORS = {
    "ARIMA+GARCH": "#4C78A8",
    "Chronos2 Zeroshot": "#72B7B2",
    "Chronos2 Native FT": "#F58518",
    "Chronos2 Proposed FT": "#E45756",
}

BUY_HOLD_COLOR = "#7A7A7A"


# -------------------------------------------------------------------
# 3. Plot/output settings
# -------------------------------------------------------------------

FIGSIZE = (15, 7.5)
TITLE = "BTCUSDT Equity Curves: Native Signal (Dashed) vs Hawkes-Scaled Signal (Solid)"
X_LABEL = "Time"
Y_LABEL = "Equity"
OUTPUT_SVG = "reports/figures/manual/exp2_equity_multi_model.svg"
SHOW_LEGEND = False
END_LABEL_FONT_SIZE = 9
END_LABEL_X_OFFSET_PTS = 7
END_LABEL_MIN_GAP_RATIO = 0.018
# Right-side x-axis padding ratio for tail labels.
# Final xlim right = x_right + x_span * X_RIGHT_PAD_RATIO
X_RIGHT_PAD_RATIO = 0.18


# -------------------------------------------------------------------
# 4. Data loading helpers
# -------------------------------------------------------------------

def _decode_epoch_auto(v: object) -> pd.DatetimeIndex:
    arr = np.asarray(v)
    if arr.size == 0:
        return pd.DatetimeIndex([], tz="UTC")
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, utc=True)

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
    return pd.to_datetime(raw, unit=unit, utc=True)


def _as_float(v: object) -> np.ndarray:
    return np.asarray(v, dtype=float)


def _load_backtest_meta(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    obj = np.load(str(p), allow_pickle=True).item()
    if str(obj.get("kind", "")) != "backtest_layer":
        raise ValueError(f"Not a backtest_layer npy: {p}")
    return obj


def _load_equity_triplet(path: str | Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Returns:
    - equity_ts_ns decoded to DatetimeIndex
    - equity_line as float array
    - buy_hold_line as float array
    """
    meta = _load_backtest_meta(path)
    ts = _decode_epoch_auto(meta.get("equity_ts_ns", []))
    eq = _as_float(meta.get("equity_line", []))
    bh = _as_float(meta.get("buy_hold_line", []))

    n = min(len(ts), len(eq), len(bh))
    if n == 0:
        raise ValueError(f"Empty equity series in {path}")
    return ts[:n], eq[:n], bh[:n]


def _validate_manual_paths() -> None:
    missing = []
    for model_name, pair in MODEL_BACKTEST_PATHS.items():
        if not str(pair.native_no_hawkes).strip():
            missing.append(f"{model_name} native_no_hawkes")
        if not str(pair.hawkes_scaled).strip():
            missing.append(f"{model_name} hawkes_scaled")
    if missing:
        joined = "\n - ".join(missing)
        raise ValueError(
            "Please fill all manual npy paths in MODEL_BACKTEST_PATHS before running:\n - " + joined
        )


# -------------------------------------------------------------------
# 5. Tail labeling helper
# -------------------------------------------------------------------

def _add_tail_labels(
    ax: plt.Axes,
    endpoints: list[dict],
    min_gap_ratio: float = END_LABEL_MIN_GAP_RATIO,
    fontsize: float = END_LABEL_FONT_SIZE,
    x_offset_pts: float = END_LABEL_X_OFFSET_PTS,
) -> None:
    if not endpoints:
        return

    ys = np.array([float(p["y"]) for p in endpoints], dtype=float)
    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    span = y_max - y_min
    if span <= 0:
        span = 1.0

    min_gap = span * max(0.0, float(min_gap_ratio))

    # Small anti-overlap pass on y labels.
    sorted_idx = np.argsort(ys)
    adjusted = ys.copy()
    prev = -np.inf
    for i in sorted_idx:
        yv = adjusted[i]
        if yv < prev + min_gap:
            yv = prev + min_gap
        adjusted[i] = yv
        prev = yv

    # Clamp to keep labels inside visible y-range.
    lower, upper = ax.get_ylim()
    adjusted = np.clip(adjusted, lower, upper)

    for idx, p in enumerate(endpoints):
        is_hawkes = bool(p.get("is_hawkes", False))
        is_bold = bool(p.get("bold", False))
        bbox_style = {
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": p["color"],
            "linewidth": 0.9,
            "linestyle": "-" if is_hawkes else "--",
            "alpha": 0.92,
        }
        ax.annotate(
            p["label"],
            xy=(p["x"], adjusted[idx]),
            xytext=(x_offset_pts, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if is_bold else "normal",
            color=p["color"],
            bbox=bbox_style,
            clip_on=False,
            zorder=4,
        )


# -------------------------------------------------------------------
# 6. Main plotting logic (single large panel, no subplots)
# -------------------------------------------------------------------

def plot_multi_model_equity(output_svg: str = OUTPUT_SVG) -> Path:
    _validate_manual_paths()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 5.1 Draw the 8 strategy curves (4 models * 2 lines each)
    first_buy_hold: tuple[pd.DatetimeIndex, np.ndarray] | None = None
    line_endpoints: list[dict] = []

    for model_name, pair in MODEL_BACKTEST_PATHS.items():
        if model_name not in MODEL_COLORS:
            raise KeyError(f"Missing color mapping for model: {model_name}")
        color = MODEL_COLORS[model_name]

        ts_native, eq_native, bh_native = _load_equity_triplet(pair.native_no_hawkes)
        ts_hawkes, eq_hawkes, _ = _load_equity_triplet(pair.hawkes_scaled)

        # Native risk (no Hawkes): dashed
        ax.plot(
            ts_native,
            eq_native,
            linestyle="--",
            linewidth=1.9,
            color=color,
            alpha=0.95,
            label=f"{model_name} | Native",
        )
        line_endpoints.append(
            {
                "x": ts_native[-1],
                "y": float(eq_native[-1]),
                "label": f"{model_name} | Native",
                "color": color,
                "is_hawkes": False,
                "bold": False,
            }
        )

        # Hawkes-scaled risk: solid
        ax.plot(
            ts_hawkes,
            eq_hawkes,
            linestyle="-",
            linewidth=2.2,
            color=color,
            alpha=0.95,
            label=f"{model_name} | Hawkes",
        )
        line_endpoints.append(
            {
                "x": ts_hawkes[-1],
                "y": float(eq_hawkes[-1]),
                "label": f"{model_name} | Hawkes",
                "color": color,
                "is_hawkes": True,
                # Minimal-intrusion hardcode:
                # only Proposed FT + Hawkes tail label is bold.
                "bold": (model_name == "Chronos2 Proposed FT"),
            }
        )

        # Keep buy&hold from the first model only (per requirement).
        if first_buy_hold is None:
            first_buy_hold = (ts_native, bh_native)

    # 5.2 Draw buy&hold line from first model
    if first_buy_hold is not None:
        bh_ts, bh_line = first_buy_hold
        ax.plot(
            bh_ts,
            bh_line,
            linestyle="--",
            linewidth=2.0,
            color=BUY_HOLD_COLOR,
            alpha=0.95,
            label="Buy & Hold",
        )
        line_endpoints.append(
            {
                "x": bh_ts[-1],
                "y": float(bh_line[-1]),
                "label": "Buy & Hold",
                "color": BUY_HOLD_COLOR,
                "is_hawkes": False,
                "bold": False,
            }
        )

    # 5.3 Styling
    ax.set_title(TITLE, fontsize=14)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    if SHOW_LEGEND:
        ax.legend(frameon=False, ncol=3, fontsize=10)

    # Reserve right-side room for tail labels.
    x_left, x_right = ax.get_xlim()
    x_span = x_right - x_left
    ax.set_xlim(x_left, x_right + x_span * X_RIGHT_PAD_RATIO)
    _add_tail_labels(ax, line_endpoints)
    plt.tight_layout()

    out_path = Path(output_svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------
# 7. Entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    out = plot_multi_model_equity(OUTPUT_SVG)
    print(f"[DONE] saved svg: {out}")
