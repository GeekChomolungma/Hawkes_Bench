from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModelMetrics:
    name: str
    metrics: Dict[str, float]


# -------------------------------------------------------------------
# 1. Manually fill in the metrics for the 4 models here
# -------------------------------------------------------------------

whitebox = ModelMetrics(
    name="ARIMA+GARCH",
    # Doge metrics
    metrics={
        "mae": 0.026029564413503282,
        "rmse": 0.0357428128596628,
        "pinball_q50": None,  # white-box currently unavailable / optional
        "pinball_q10": 0.004741493041721666,
        "pinball_q90": 0.008679979536122135,
        "sign_accuracy": 0.4906382978723404,
        "rank_ic_spearman": -0.17934782608695657,
    },
)

zeroshot = ModelMetrics(
    name="Chronos2 Zeroshot",
    # Doge metrics
    metrics={
        "mae": 0.02551316594610521,
        "rmse": 0.03514076519552154,
        "pinball_q50": 0.012756582973052604,
        "pinball_q10": 0.005035683910400414,
        "pinball_q90": 0.00851738324388475,
        "sign_accuracy": 0.4870212765957447,
        "rank_ic_spearman": -0.1344858489764373,
    },
)

native_ft = ModelMetrics(
    name="Chronos2 Native FT",
    # Doge metrics
    metrics={
        "mae": 0.030250445508426545,
        "rmse": 0.03744389767732508,
        "pinball_q50": 0.015125222754213272,
        "pinball_q10": 0.010793771679417235,
        "pinball_q90": 0.007496124922390927,
        "sign_accuracy": 0.48680851063829785,
        "rank_ic_spearman": 0.2840705410935244,
    },
)

proposed_ft = ModelMetrics(
    name="Chronos2 Proposed FT",
    # Doge metrics
    metrics={
        "mae": 0.03221851408135618,
        "rmse": 0.03958706454582015,
        "pinball_q50": 0.01610925704067809,
        "pinball_q10": 0.01178801186730175,
        "pinball_q90": 0.007489048245740977,
        "sign_accuracy": 0.467021276595745,
        "rank_ic_spearman": 0.22606382978723405,
    },
)

MODELS: List[ModelMetrics] = [whitebox, zeroshot, native_ft, proposed_ft]


# -------------------------------------------------------------------
# 2. Color design:
#    same model keeps similar color family across all figures
# -------------------------------------------------------------------

MODEL_COLORS = {
    "ARIMA+GARCH": "#4C78A8",
    "Chronos2 Zeroshot": "#72B7B2",
    "Chronos2 Native FT": "#F58518",
    "Chronos2 Proposed FT": "#E45756",
}

# -------------------------------------------------------------------
# 2.1 Plot style knobs (manual tuning)
# -------------------------------------------------------------------

# Make y=0 baseline clearly visible.
ZERO_AXIS_LINEWIDTH = 0.5
ZERO_AXIS_COLOR = "#222222"

# Extra white space on top/bottom of y-axis range.
# Final ylim is:
#   [data_min - span * y_pad_lower, data_max + span * y_pad_upper]
#
# Decoupled per figure type:
# - Traditional
# - Risk
# - Trend
Y_PAD_UPPER_TRADITIONAL = 0.35
Y_PAD_LOWER_TRADITIONAL = 0

Y_PAD_UPPER_RISK = 0.35
Y_PAD_LOWER_RISK = 0

Y_PAD_UPPER_TREND = 0.40
Y_PAD_LOWER_TREND = 0.20

# Value label style on bar ends.
VALUE_TEXT_OFFSET_RATIO = 0.018
VALUE_TEXT_FONTSIZE = 9
DEFAULT_VALUE_TEXT_COLOR = "black"

# Bar thickness (horizontal width). Increase this to make bars wider.
BAR_WIDTH = 0.32

# Optional visibility overrides (same key style as color overrides).
# True  -> force show value label
# False -> force hide value label
BAR_VALUE_TEXT_VISIBLE_OVERRIDES: Dict[tuple[str, str], bool] = {}
BAR_VALUE_TEXT_VISIBLE_OVERRIDES = {
    # Risk metrics
    # ("ARIMA+GARCH", "Pinball q90"): True,
    # ("Chronos2 Zeroshot", "Pinball q90"): True,
    # ("Chronos2 Native FT", "Pinball q90"): True,
    # ("Chronos2 Proposed FT", "Pinball q90"): True,
    
    ("ARIMA+GARCH", "Tail Avg"): True,
    ("Chronos2 Zeroshot", "Tail Avg"): True,
    ("Chronos2 Native FT", "Tail Avg"): True,
    ("Chronos2 Proposed FT", "Tail Avg"): True,

    # Trend metrics
    ("ARIMA+GARCH", "Sign Accuracy"): True,
    ("Chronos2 Zeroshot", "Sign Accuracy"): True,
    ("Chronos2 Native FT", "Sign Accuracy"): True,
    ("Chronos2 Proposed FT", "Sign Accuracy"): True,

    ("ARIMA+GARCH", "Rank IC (Spearman)"): True,
    ("Chronos2 Zeroshot", "Rank IC (Spearman)"): True,
    ("Chronos2 Native FT", "Rank IC (Spearman)"): True,
    ("Chronos2 Proposed FT", "Rank IC (Spearman)"): True,
}

# Default visibility for labels not in BAR_VALUE_TEXT_VISIBLE_OVERRIDES.
DEFAULT_SHOW_VALUE_TEXT = False

# Optional hard-coded color overrides per (model_name, metric_label).
# Example:
# ("Chronos2 Proposed FT", "Rank IC (Spearman)"): "red",
BAR_VALUE_TEXT_COLOR_OVERRIDES: Dict[tuple[str, str], str] = {}
BAR_VALUE_TEXT_COLOR_OVERRIDES = {
    # Risk metrics
    # ("ARIMA+GARCH", "Pinball q90"): "red",
    # ("Chronos2 Zeroshot", "Pinball q90"): "green",
    # ("Chronos2 Native FT", "Pinball q90"): "red",
    # ("Chronos2 Proposed FT", "Pinball q90"): "red",

    # ("ARIMA+GARCH", "Tail Avg"): "red",
    # ("Chronos2 Zeroshot", "Tail Avg"): "green",
    ("Chronos2 Native FT", "Tail Avg"): "green",
    ("Chronos2 Proposed FT", "Tail Avg"): "green",

    # Trend metrics
    # ("ARIMA+GARCH", "Sign Accuracy"): "red",
    # ("Chronos2 Zeroshot", "Sign Accuracy"): "green",
    # ("Chronos2 Native FT", "Sign Accuracy"): "green",
    # ("Chronos2 Proposed FT", "Sign Accuracy"): "green",

    ("ARIMA+GARCH", "Rank IC (Spearman)"): "gray",
    ("Chronos2 Zeroshot", "Rank IC (Spearman)"): "gray",
    ("Chronos2 Native FT", "Rank IC (Spearman)"): "green",
    ("Chronos2 Proposed FT", "Rank IC (Spearman)"): "green",
}


# Optional arrow control (same key style as color overrides).
# If a key is present here, this value has highest priority:
#   "up" / "down" / "none"
BAR_VALUE_TEXT_ARROW_OVERRIDES: Dict[tuple[str, str], str] = {}
BAR_VALUE_TEXT_ARROW_OVERRIDES = {
    # Risk metrics
    # ("ARIMA+GARCH", "Pinball q90"): "up",
    # ("Chronos2 Zeroshot", "Pinball q90"): "up",
    # ("Chronos2 Native FT", "Pinball q90"): "down",
    # ("Chronos2 Proposed FT", "Pinball q90"): "down",

    # ("ARIMA+GARCH", "Tail Avg"): "up",
    # ("Chronos2 Zeroshot", "Tail Avg"): "up",
    # ("Chronos2 Native FT", "Tail Avg"): "up",
    ("Chronos2 Proposed FT", "Tail Avg"): "up",

    # Trend metrics
    # ("ARIMA+GARCH", "Sign Accuracy"): "red",
    # ("Chronos2 Zeroshot", "Sign Accuracy"): "green",
    # ("Chronos2 Native FT", "Sign Accuracy"): "up",
    # ("Chronos2 Proposed FT", "Sign Accuracy"): "up",

    # ("ARIMA+GARCH", "Rank IC (Spearman)"): "up",
    # ("Chronos2 Zeroshot", "Rank IC (Spearman)"): "up",
    ("Chronos2 Native FT", "Rank IC (Spearman)"): "up",
    # ("Chronos2 Proposed FT", "Rank IC (Spearman)"): "up",
}
ARROW_UP_COLOR = "green"
ARROW_DOWN_COLOR = "red"
ARROW_FONTSIZE = VALUE_TEXT_FONTSIZE

# Arrow horizontal spacing (in points) when number + arrow are shown together.
# Final offset = ARROW_BASE_OFFSET_PTS + max(0, len(value_text)-4) * ARROW_PER_CHAR_OFFSET_PTS
ARROW_BASE_OFFSET_PTS = 6
ARROW_PER_CHAR_OFFSET_PTS = 6

def _resolve_arrow_symbol_and_color(model_name: str, block_label: str) -> tuple[str, str | None]:
    """
    Resolve arrow symbol/color for a bar label.
    Arrow is controlled ONLY by BAR_VALUE_TEXT_ARROW_OVERRIDES.
    No coupling with color.
    """
    key = (model_name, block_label)
    if key in BAR_VALUE_TEXT_ARROW_OVERRIDES:
        mode = str(BAR_VALUE_TEXT_ARROW_OVERRIDES[key]).strip().lower()
        if mode == "up":
            return "\u2191", ARROW_UP_COLOR
        if mode == "down":
            return "\u2193", ARROW_DOWN_COLOR
        return "", None
    return "", None


def _resolve_value_label_visible(model_name: str, block_label: str) -> bool:
    """Resolve whether a bar value label should be shown."""
    key = (model_name, block_label)
    if key in BAR_VALUE_TEXT_VISIBLE_OVERRIDES:
        return bool(BAR_VALUE_TEXT_VISIBLE_OVERRIDES[key])
    return bool(DEFAULT_SHOW_VALUE_TEXT)


# -------------------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------------------

def tail_pinball_avg(model: ModelMetrics) -> float:
    """Average tail pinball loss using q10 and q90."""
    q10 = model.metrics["pinball_q10"]
    q90 = model.metrics["pinball_q90"]
    return (q10 + q90) / 2.0


def format_metric_value(v: float) -> str:
    """
    Format metric values with adaptive precision for cleaner bar labels.
    """
    av = abs(v)
    if av >= 1:
        return f"{v:.3f}"
    if av >= 0.1:
        return f"{v:.3f}"
    if av >= 0.01:
        return f"{v:.4f}"
    return f"{v:.5f}"


def plot_grouped_metric_blocks(
    metric_blocks: List[Dict],
    title: str,
    ylabel: str,
    out_path: str,
    figsize=(12, 6),
    y_pad_upper: float = 0.40,
    y_pad_lower: float = 0.20,
    value_text_offset_ratio: float = VALUE_TEXT_OFFSET_RATIO,
    value_text_fontsize: int = VALUE_TEXT_FONTSIZE,
    bar_width: float = BAR_WIDTH,
) -> None:
    """
    Draw grouped bars where each x-block is one metric, and each model contributes one bar.

    metric_blocks examples:
    [{"label": "MAE", "key": "mae"}, {"label": "RMSE", "key": "rmse"}]
    - label: displayed x-axis text
    - key: metric key in model.metrics
    - func: optional callable(model) -> float, used for derived metrics
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(MODELS)
    n_blocks = len(metric_blocks)

    # X center positions of metric blocks. Larger step widens spacing between blocks.
    block_centers = np.arange(n_blocks) * 1.8
    bar_width = float(bar_width)
    if bar_width <= 0:
        raise ValueError("bar_width must be > 0")

    # Offsets place each model's bar around the block center.
    # This is symmetric for any model count, e.g. 4 -> [-1.5w, -0.5w, 0.5w, 1.5w].
    half = (n_models - 1) / 2.0
    offsets = np.linspace(-half * bar_width, half * bar_width, n_models)

    # bar_records keeps references for post-processing (value text placement/coloring).
    # tuple: (model_obj, bar_container, y_values_array)
    bar_records = []
    # all_vals is used to compute global y-range and padding.
    all_vals = []

    for model_idx, model in enumerate(MODELS):
        # Final x positions for this model across all blocks.
        xs = block_centers + offsets[model_idx]
        ys = []
        for block in metric_blocks:
            if "func" in block:
                val = block["func"](model)
            else:
                val = model.metrics.get(block["key"], np.nan)
            if val is None:
                val = np.nan
            ys.append(val)

        # Convert to numeric array so None -> nan handling and finite checks are uniform.
        ys_arr = np.asarray(ys, dtype=float)
        bars = ax.bar(
            xs,
            ys_arr,
            width=bar_width,
            label=model.name,
            color=MODEL_COLORS[model.name],
            alpha=0.9,
        )
        bar_records.append((model, bars, ys_arr))
        all_vals.extend([v for v in ys_arr if np.isfinite(v)])

    # Include 0.0 in range so the zero baseline is always visible.
    if all_vals:
        data_min = min(min(all_vals), 0.0)
        data_max = max(max(all_vals), 0.0)
    else:
        data_min, data_max = -1.0, 1.0
    # span is the full data range used by both ylim padding and text offsets.
    span = data_max - data_min
    if span <= 0:
        span = max(abs(data_max), 1.0)

    # Asymmetric y padding allows manual control of extra white space above/below bars.
    pad_up = span * max(0.0, float(y_pad_upper))
    pad_dn = span * max(0.0, float(y_pad_lower))
    ax.set_ylim(data_min - pad_dn, data_max + pad_up)

    # Clear zero baseline
    ax.axhline(0.0, color=ZERO_AXIS_COLOR, linewidth=ZERO_AXIS_LINEWIDTH, alpha=0.95, zorder=1.1)

    # Annotate each bar value on the outer side of the bar:
    # positive bar -> top outside, negative bar -> bottom outside.
    text_offset = span * max(0.0, float(value_text_offset_ratio))
    if text_offset == 0:
        text_offset = 1e-6
    for model, bars, ys_arr in bar_records:
        for i, (bar, yv) in enumerate(zip(bars, ys_arr)):
            if not np.isfinite(yv):
                continue
            # block_label + model.name is the key for optional hard-coded text coloring.
            block_label = metric_blocks[i]["label"]
            txt_color = BAR_VALUE_TEXT_COLOR_OVERRIDES.get(
                (model.name, block_label),
                DEFAULT_VALUE_TEXT_COLOR,
            )
            arrow_symbol, arrow_color = _resolve_arrow_symbol_and_color(model.name, block_label)
            show_value = _resolve_value_label_visible(model.name, block_label)
            value_text = format_metric_value(float(yv)) if show_value else ""
            if value_text == "" and arrow_symbol == "":
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            if yv >= 0:
                y_txt = yv + text_offset
                va = "bottom"
            else:
                y_txt = yv - text_offset
                va = "top"
            if value_text != "":
                ax.text(
                    x,
                    y_txt,
                    value_text,
                    ha="center",
                    va=va,
                    fontsize=value_text_fontsize,
                    color=txt_color,
                    rotation=0,
                    clip_on=False,
                    zorder=3,
                )

            if arrow_symbol != "":
                if value_text != "":
                    # Place arrow to the right of numeric value with dynamic offset
                    # based on text length, to avoid overlap.
                    arrow_dx_pts = (
                        ARROW_BASE_OFFSET_PTS
                        + max(0, len(value_text) - 4) * ARROW_PER_CHAR_OFFSET_PTS
                    )
                    ax.annotate(
                        arrow_symbol,
                        xy=(x, y_txt),
                        xytext=(arrow_dx_pts, 0),
                        textcoords="offset points",
                        ha="left",
                        va=va,
                        fontsize=ARROW_FONTSIZE,
                        color=arrow_color if arrow_color is not None else "black",
                        clip_on=False,
                        zorder=3,
                    )
                else:
                    # If number is hidden but arrow is enabled, place arrow at the label anchor.
                    ax.text(
                        x,
                        y_txt,
                        arrow_symbol,
                        ha="center",
                        va=va,
                        fontsize=ARROW_FONTSIZE,
                        color=arrow_color if arrow_color is not None else "black",
                        rotation=0,
                        clip_on=False,
                        zorder=3,
                    )

    ax.set_xticks(block_centers)
    ax.set_xticklabels([b["label"] for b in metric_blocks], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# 4. Figure 1: Traditional error metrics
# -------------------------------------------------------------------

def plot_traditional_metrics(out_path: str = "doge_traditional_metrics.svg") -> None:
    metric_blocks = [
        {"label": "MAE", "key": "mae"},
        {"label": "RMSE", "key": "rmse"},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="DOGE Traditional Error Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(9, 5.5),
        y_pad_upper=Y_PAD_UPPER_TRADITIONAL,
        y_pad_lower=Y_PAD_LOWER_TRADITIONAL,
    )


# -------------------------------------------------------------------
# 5. Figure 2: Risk characterization metrics
# -------------------------------------------------------------------

def plot_risk_metrics(out_path: str = "doge_risk_metrics.svg") -> None:
    metric_blocks = [
        {"label": "Pinball q10", "key": "pinball_q10"},
        {"label": "Pinball q90", "key": "pinball_q90"},
        {"label": "Tail Avg", "func": tail_pinball_avg},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="DOGE Risk Characterization Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(12, 5.5),
        y_pad_upper=Y_PAD_UPPER_RISK,
        y_pad_lower=Y_PAD_LOWER_RISK,
    )


# -------------------------------------------------------------------
# 6. Figure 3: Trend-following metrics
# -------------------------------------------------------------------

def plot_trend_metrics(out_path: str = "doge_trend_metrics.svg") -> None:
    metric_blocks = [
        {"label": "Sign Accuracy", "key": "sign_accuracy"},
        {"label": "Rank IC (Spearman)", "key": "rank_ic_spearman"},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="DOGE Trend-Following Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(10, 5.5),
        y_pad_upper=Y_PAD_UPPER_TREND,
        y_pad_lower=Y_PAD_LOWER_TREND,
    )


# -------------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    plot_traditional_metrics()
    plot_risk_metrics()
    plot_trend_metrics()
    print("Done. Saved:")
    print(" - doge_traditional_metrics.svg")
    print(" - doge_risk_metrics.svg")
    print(" - doge_trend_metrics.svg")

