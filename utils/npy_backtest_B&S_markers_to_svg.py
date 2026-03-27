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
# 1. Manually fill in the 4 models path (3 strategies each)
# -------------------------------------------------------------------

@dataclass
class BacktestTriplePath:
    native_no_hawkes: str
    hawkes_scaled_q70: str
    hawkes_scaled_q90: str


# IMPORTANT:
# 1) Please hardcode your own .npy paths here.
# 2) Each path must point to a backtest_layer .npy file.
MODEL_BACKTEST_PATHS: Dict[str, BacktestTriplePath] = {
    "ARIMA+GARCH": BacktestTriplePath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_white_no_hawkes_DOGEUSDT_1d.npy",
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_white_hawkes_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_white_hawkes_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Zeroshot": BacktestTriplePath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_black_no_hawkes_DOGEUSDT_1d.npy",
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_black_hawkes_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_black_hawkes_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Native FT": BacktestTriplePath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/pretrained_native_all/dogeusdt/exp2_black_no_hawkes_DOGEUSDT_1d.npy",
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/pretrained_native_all/dogeusdt/exp2_black_hawkes_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/pretrained_native_all/dogeusdt/exp2_black_hawkes_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Proposed FT": BacktestTriplePath(
        native_no_hawkes="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/dogeusdt/exp2_black_no_hawkes_DOGEUSDT_1d.npy",
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/dogeusdt/exp2_black_hawkes_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/dogeusdt/exp2_black_hawkes_q90_DOGEUSDT_1d.npy",
    ),
}


# -------------------------------------------------------------------
# 2. Output settings
# -------------------------------------------------------------------

OUTPUT_ROOT = Path("reports/figures/backtest")
OUTPUT_SUBDIR = "dogeusdt_1d"

FIGSIZE = (15, 7)
LINE_COLOR = "#1f77b4"

MARKER_STYLES = {
    "open_long_ts_ns": {
        "label": "Open Long",
        "marker": "^",
        "color": "#2ca02c",
        "side": "below",
        "kind": "triangle",
    },
    "close_long_ts_ns": {
        "label": "Close Long",
        "marker": "o",
        "color": "#d62728",
        "side": "above",
        "kind": "circle",
    },
    "open_short_ts_ns": {
        "label": "Open Short",
        "marker": "v",
        "color": "#d62728",
        "side": "above",
        "kind": "triangle",
    },
    "close_short_ts_ns": {
        "label": "Close Short",
        "marker": "o",
        "color": "#2ca02c",
        "side": "below",
        "kind": "circle",
    },
}


# -------------------------------------------------------------------
# 3. Data loading helpers
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


def _validate_manual_paths() -> None:
    missing = []
    for model_name, triple in MODEL_BACKTEST_PATHS.items():
        if not str(triple.native_no_hawkes).strip():
            missing.append(f"{model_name} native_no_hawkes")
        if not str(triple.hawkes_scaled_q70).strip():
            missing.append(f"{model_name} hawkes_scaled_q70")
        if not str(triple.hawkes_scaled_q90).strip():
            missing.append(f"{model_name} hawkes_scaled_q90")
    if missing:
        joined = "\n - ".join(missing)
        raise ValueError(
            "Please fill all manual npy paths in MODEL_BACKTEST_PATHS before running:\n - " + joined
        )


def _nearest_index(base_ns: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(base_ns, query_ns)
    pos = np.clip(pos, 0, len(base_ns) - 1)
    left = np.maximum(pos - 1, 0)
    right = pos
    choose_left = np.abs(query_ns - base_ns[left]) <= np.abs(base_ns[right] - query_ns)
    return np.where(choose_left, left, right)


def _extract_marker_points(
    meta: dict,
    close_ts: pd.DatetimeIndex,
    close: np.ndarray,
    marker_key: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    close_ns = close_ts.view("int64")
    marker_ts_raw = np.asarray(meta.get(marker_key, []), dtype="int64")
    marker_ts_raw = np.unique(marker_ts_raw)
    marker_ts = _decode_epoch_auto(marker_ts_raw)

    if len(close_ts) == 0:
        return marker_ts, np.array([], dtype=int), np.array([], dtype=float)
    if len(marker_ts) == 0:
        return marker_ts, np.array([], dtype=int), np.array([], dtype=float)

    marker_idx = _nearest_index(close_ns, marker_ts.view("int64"))
    marker_vals = close[marker_idx]
    return marker_ts, marker_idx.astype(int), marker_vals


def _compute_marker_y_positions(
    marker_data: dict[str, dict[str, np.ndarray | pd.DatetimeIndex]],
    close: np.ndarray,
) -> dict[str, np.ndarray]:
    y_min = float(np.nanmin(close))
    y_max = float(np.nanmax(close))
    span = y_max - y_min
    base_offset = span * 0.01 if span > 0 else 1e-6
    near_offset = base_offset
    far_offset = base_offset * 1.9

    out_y: dict[str, np.ndarray] = {}
    for marker_key, payload in marker_data.items():
        base_vals = np.asarray(payload["vals"], dtype=float)
        out_y[marker_key] = base_vals.copy()

    side_pairs = [
        ("above", "close_long_ts_ns", "open_short_ts_ns", 1.0),
        ("below", "close_short_ts_ns", "open_long_ts_ns", -1.0),
    ]

    for _, circle_key, triangle_key, sign in side_pairs:
        circle_idx_arr = np.asarray(marker_data[circle_key]["idx"], dtype=int)
        triangle_idx_arr = np.asarray(marker_data[triangle_key]["idx"], dtype=int)
        circle_idx_set = set(circle_idx_arr.tolist())
        triangle_idx_set = set(triangle_idx_arr.tolist())
        overlap_idx = circle_idx_set.intersection(triangle_idx_set)

        circle_vals = np.asarray(marker_data[circle_key]["vals"], dtype=float)
        circle_y = out_y[circle_key]
        for i, idx in enumerate(circle_idx_arr):
            if int(idx) in overlap_idx:
                circle_y[i] = circle_vals[i] + sign * near_offset
            else:
                circle_y[i] = circle_vals[i] + sign * near_offset

        triangle_vals = np.asarray(marker_data[triangle_key]["vals"], dtype=float)
        triangle_y = out_y[triangle_key]
        for i, idx in enumerate(triangle_idx_arr):
            if int(idx) in overlap_idx:
                triangle_y[i] = triangle_vals[i] + sign * far_offset
            else:
                triangle_y[i] = triangle_vals[i] + sign * near_offset

    return out_y


# -------------------------------------------------------------------
# 4. Plot helpers
# -------------------------------------------------------------------

def _sanitize_token(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "+", "/"):
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"


def _plot_close_with_markers(
    model_name: str,
    strategy_name: str,
    npy_path: str,
    output_dir: Path,
) -> Path:
    meta = _load_backtest_meta(npy_path)
    close_ts = _decode_epoch_auto(meta.get("close_ts_ns", []))
    close = _as_float(meta.get("close", []))

    n = min(len(close_ts), len(close))
    if n == 0:
        raise ValueError(f"Empty close series in {npy_path}")
    close_ts = close_ts[:n]
    close = close[:n]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(close_ts, close, color=LINE_COLOR, linewidth=1.6, alpha=0.92, label="Close")
    marker_data: dict[str, dict[str, np.ndarray | pd.DatetimeIndex]] = {}
    for marker_key, style in MARKER_STYLES.items():
        marker_ts, marker_idx, marker_vals = _extract_marker_points(
            meta=meta,
            close_ts=close_ts,
            close=close,
            marker_key=marker_key,
        )
        marker_data[marker_key] = {
            "ts": marker_ts,
            "idx": marker_idx,
            "vals": marker_vals,
        }

    marker_y_map = _compute_marker_y_positions(marker_data=marker_data, close=close)

    for marker_key, style in MARKER_STYLES.items():
        marker_ts = np.asarray(marker_data[marker_key]["ts"])
        marker_y = np.asarray(marker_y_map[marker_key], dtype=float)
        if len(marker_ts) == 0:
            continue
        is_circle = str(style.get("kind", "")) == "circle"
        ax.scatter(
            marker_ts,
            marker_y,
            marker=style["marker"],
            s=54,
            facecolors="none" if is_circle else style["color"],
            edgecolors=style["color"],
            linewidths=1.5 if is_circle else 0.8,
            alpha=0.92,
            label=style["label"],
            zorder=3,
        )

    ax.set_title(f"{model_name} | {strategy_name} | Close with 4 Backtest Markers", fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Close", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = _sanitize_token(model_name)
    strategy_tag = _sanitize_token(strategy_name)
    out_path = output_dir / f"{model_tag}__{strategy_tag}__close_bs_markers.svg"

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------
# 5. Main entry
# -------------------------------------------------------------------

def main() -> None:
    _validate_manual_paths()

    output_dir = OUTPUT_ROOT / OUTPUT_SUBDIR
    generated: list[Path] = []

    for model_name, triple in MODEL_BACKTEST_PATHS.items():
        generated.append(
            _plot_close_with_markers(
                model_name=model_name,
                strategy_name="native_no_hawkes",
                npy_path=triple.native_no_hawkes,
                output_dir=output_dir,
            )
        )
        generated.append(
            _plot_close_with_markers(
                model_name=model_name,
                strategy_name="hawkes_scaled_q70",
                npy_path=triple.hawkes_scaled_q70,
                output_dir=output_dir,
            )
        )
        generated.append(
            _plot_close_with_markers(
                model_name=model_name,
                strategy_name="hawkes_scaled_q90",
                npy_path=triple.hawkes_scaled_q90,
                output_dir=output_dir,
            )
        )

    print(f"[DONE] generated {len(generated)} svg files in: {output_dir}")
    for p in generated:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
