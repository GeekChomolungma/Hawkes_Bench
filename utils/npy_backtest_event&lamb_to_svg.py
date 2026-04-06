from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class EventLambdaPairPath:
    hawkes_scaled_q70: str
    hawkes_scaled_q90: str


# -------------------------------------------------------------------
# 1. Manually fill in the 4 models path (event+lambda recorder npy)
# -------------------------------------------------------------------

MODEL_EVENT_LAMBDA_PATHS: Dict[str, EventLambdaPairPath] = {
    "ARIMA+GARCH": EventLambdaPairPath(
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_hawkes_event_lambda_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_hawkes_event_lambda_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Zeroshot": EventLambdaPairPath(
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_hawkes_event_lambda_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/zeroshot/dogeusdt/exp2_hawkes_event_lambda_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Native FT": EventLambdaPairPath(
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/pretrained_native_all/dogeusdt/exp2_hawkes_event_lambda_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/pretrained_native_all/dogeusdt/exp2_hawkes_event_lambda_q90_DOGEUSDT_1d.npy",
    ),
    "Chronos2 Proposed FT": EventLambdaPairPath(
        hawkes_scaled_q70="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/dogeusdt/exp2_hawkes_event_lambda_q70_DOGEUSDT_1d.npy",
        hawkes_scaled_q90="reports/exp_results_meta/ft/1d/pretrained_QuEXTime_all/dogeusdt/exp2_hawkes_event_lambda_q90_DOGEUSDT_1d.npy",
    ),
}


# -------------------------------------------------------------------
# 2. Plot/output settings
# -------------------------------------------------------------------

OUTPUT_ROOT = Path("reports/figures/backtest")
OUTPUT_SUBDIR = "dogeusdt_1d"
FIGSIZE = (15, 8.5)
TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
THETA_TEXT_FONT_SIZE = 9


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


def _as_int(v: object) -> np.ndarray:
    return np.asarray(v, dtype=int)


def _as_bool_scalar(v: object, default: bool = False) -> bool:
    arr = np.asarray(v)
    if arr.size == 0:
        return bool(default)
    return bool(arr.reshape(-1)[0])


def _as_float_scalar(v: object, default: float = float("nan")) -> float:
    arr = np.asarray(v)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _load_event_lambda_meta(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    obj = np.load(str(p), allow_pickle=True).item()
    if str(obj.get("kind", "")) != "event_lambda_recorder":
        raise ValueError(f"Not an event_lambda_recorder npy: {p}")
    return obj


def _sanitize_token(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "+", "/"):
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "unknown"


def _validate_manual_paths() -> None:
    missing = []
    for model_name, pair in MODEL_EVENT_LAMBDA_PATHS.items():
        if not str(pair.hawkes_scaled_q70).strip():
            missing.append(f"{model_name} hawkes_scaled_q70")
        if not str(pair.hawkes_scaled_q90).strip():
            missing.append(f"{model_name} hawkes_scaled_q90")
    if missing:
        joined = "\n - ".join(missing)
        raise ValueError(
            "Please fill all manual npy paths in MODEL_EVENT_LAMBDA_PATHS before running:\n - " + joined
        )


# -------------------------------------------------------------------
# 4. Plot one recorder file
# -------------------------------------------------------------------

def _plot_one_event_lambda(
    model_name: str,
    strategy_name: str,
    npy_path: str,
    output_dir: Path,
) -> Path:
    meta = _load_event_lambda_meta(npy_path)

    ts = _decode_epoch_auto(meta.get("test_ts_ns", []))
    lam_total = _as_float(meta.get("lambda_total", meta.get("lambda_line", [])))
    lam_pos = _as_float(meta.get("lambda_pos", []))
    lam_neg = _as_float(meta.get("lambda_neg", []))
    lam_abs = _as_float(meta.get("lambda_abs", []))
    log_return = _as_float(meta.get("log_return", []))
    event_pos = _as_int(meta.get("event_pos", []))
    event_neg = _as_int(meta.get("event_neg", []))
    event_abs = _as_int(meta.get("event_abs", []))
    signed_events = _as_bool_scalar(meta.get("signed_events", False), default=False)

    n = min(len(ts), len(lam_total), len(log_return), len(event_pos), len(event_neg), len(event_abs))
    if n == 0:
        raise ValueError(f"Empty event/lambda series in {npy_path}")

    ts = ts[:n]
    lam_total = lam_total[:n]
    lam_pos = lam_pos[:n] if len(lam_pos) >= n else np.zeros(n, dtype=float)
    lam_neg = lam_neg[:n] if len(lam_neg) >= n else np.zeros(n, dtype=float)
    lam_abs = lam_abs[:n] if len(lam_abs) >= n else np.zeros(n, dtype=float)
    log_return = log_return[:n]
    event_pos = event_pos[:n]
    event_neg = event_neg[:n]
    event_abs = event_abs[:n]

    theta_pos_enabled = _as_bool_scalar(meta.get("theta_pos_enabled", False))
    theta_neg_enabled = _as_bool_scalar(meta.get("theta_neg_enabled", False))
    theta_abs_enabled = _as_bool_scalar(meta.get("theta_abs_enabled", False))
    theta_pos_mu = _as_float_scalar(meta.get("theta_pos_mu", np.nan))
    theta_pos_alpha = _as_float_scalar(meta.get("theta_pos_alpha", np.nan))
    theta_pos_beta = _as_float_scalar(meta.get("theta_pos_beta", np.nan))
    theta_neg_mu = _as_float_scalar(meta.get("theta_neg_mu", np.nan))
    theta_neg_alpha = _as_float_scalar(meta.get("theta_neg_alpha", np.nan))
    theta_neg_beta = _as_float_scalar(meta.get("theta_neg_beta", np.nan))
    theta_abs_mu = _as_float_scalar(meta.get("theta_abs_mu", np.nan))
    theta_abs_alpha = _as_float_scalar(meta.get("theta_abs_alpha", np.nan))
    theta_abs_beta = _as_float_scalar(meta.get("theta_abs_beta", np.nan))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    if signed_events:
        ax1.step(ts, event_pos.astype(float), where="post", color="tab:green", linewidth=1.2, label="event_pos")
        ax1.step(ts, -event_neg.astype(float), where="post", color="tab:red", linewidth=1.2, label="event_neg")
        ax1.set_ylim(-1.25, 1.25)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(["-1", "0", "1"])
    else:
        ax1.step(ts, event_abs.astype(float), where="post", color="tab:purple", linewidth=1.2, label="event_abs")
        ax1.set_ylim(-0.1, 1.25)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["0", "1"])

    ax1.set_ylabel("Event", fontsize=AXIS_LABEL_FONT_SIZE)
    ax1.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1_r = ax1.twinx()
    ax1_r.plot(ts, log_return, color="0.35", linewidth=1.0, alpha=0.55, label="log_return")
    ax1_r.set_ylabel("LogReturn", color="0.35", fontsize=AXIS_LABEL_FONT_SIZE)
    ax1_r.tick_params(axis="y", colors="0.35", labelsize=TICK_LABEL_FONT_SIZE)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=LEGEND_FONT_SIZE)

    ax2.plot(ts, lam_total, color="tab:blue", linewidth=1.7, alpha=0.95, label="lambda_total")
    if signed_events:
        ax2.plot(ts, lam_pos, color="tab:green", linewidth=1.3, alpha=0.9, linestyle="--", label="lambda_pos")
        ax2.plot(ts, lam_neg, color="tab:red", linewidth=1.3, alpha=0.9, linestyle="--", label="lambda_neg")
        theta_txt = (
            f"theta_pos({'on' if theta_pos_enabled else 'off'}): "
            f"mu={theta_pos_mu:.4g}, a={theta_pos_alpha:.4g}, b={theta_pos_beta:.4g}\n"
            f"theta_neg({'on' if theta_neg_enabled else 'off'}): "
            f"mu={theta_neg_mu:.4g}, a={theta_neg_alpha:.4g}, b={theta_neg_beta:.4g}"
        )
    else:
        ax2.plot(ts, lam_abs, color="tab:purple", linewidth=1.3, alpha=0.9, linestyle="--", label="lambda_abs")
        theta_txt = (
            f"theta_abs({'on' if theta_abs_enabled else 'off'}): "
            f"mu={theta_abs_mu:.4g}, a={theta_abs_alpha:.4g}, b={theta_abs_beta:.4g}"
        )
    ax2.set_ylabel("Lambda", fontsize=AXIS_LABEL_FONT_SIZE)
    ax2.set_xlabel("Time", fontsize=AXIS_LABEL_FONT_SIZE)
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, loc="upper right", fontsize=LEGEND_FONT_SIZE)
    ax2.text(
        0.01,
        0.98,
        theta_txt,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=THETA_TEXT_FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    fig.suptitle(f"{strategy_name} | Event and Lambda", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = _sanitize_token(model_name)
    strategy_tag = _sanitize_token(strategy_name)
    out_path = output_dir / f"{model_tag}__{strategy_tag}__event_lambda.svg"

    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------

def main() -> None:
    _validate_manual_paths()
    output_dir = OUTPUT_ROOT / OUTPUT_SUBDIR
    generated: list[Path] = []

    for model_name, pair in MODEL_EVENT_LAMBDA_PATHS.items():
        generated.append(
            _plot_one_event_lambda(
                model_name=model_name,
                strategy_name="hawkes_scaled_q70",
                npy_path=pair.hawkes_scaled_q70,
                output_dir=output_dir,
            )
        )
        generated.append(
            _plot_one_event_lambda(
                model_name=model_name,
                strategy_name="hawkes_scaled_q90",
                npy_path=pair.hawkes_scaled_q90,
                output_dir=output_dir,
            )
        )

    print(f"[DONE] generated {len(generated)} svg files in: {output_dir}")
    for p in generated:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
