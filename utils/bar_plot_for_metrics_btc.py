from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


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
    # metrics={
    #     "mae": 0.026029564413503282,
    #     "rmse": 0.0357428128596628,
    #     "pinball_q50": None,  # white-box currently unavailable / optional
    #     "pinball_q10": 0.004741493041721666,
    #     "pinball_q90": 0.008679979536122135,
    #     "sign_accuracy": 0.4906382978723404,
    #     "rank_ic_spearman": -0.17934782608695657,
    # },

    # BTC metrics
    metrics={
        "mae": 0.012048966209237429,
        "rmse": 0.016174684996494618,
        "pinball_q50": None,  # white-box currently unavailable / optional
        "pinball_q10": 0.0033625345749007216,
        "pinball_q90": 0.0031699228735634,
        "sign_accuracy": 0.3404255319148936 ,
        "rank_ic_spearman": -0.16709065679925994,
    },
)

zeroshot = ModelMetrics(
    name="Chronos2 Zeroshot",
    # Doge metrics
    # metrics={
    #     "mae": 0.02551316594610521,
    #     "rmse": 0.03514076519552154,
    #     "pinball_q50": 0.012756582973052604,
    #     "pinball_q10": 0.005035683910400414,
    #     "pinball_q90": 0.00851738324388475,
    #     "sign_accuracy": 0.4870212765957447,
    #     "rank_ic_spearman": -0.1344858489764373,
    # },
    
    # BTC metrics
    metrics={
        "mae": 0.01170353549489713,
        "rmse": 0.01599709227898297,
        "pinball_q10": 0.0029783736351348313,
        "pinball_q90": 0.003045043702925137,
        "sign_accuracy": 0.425531914893617,
        "rank_ic_spearman": -0.17496386268044065,
    },
)

native_ft = ModelMetrics(
    name="Chronos2 Native FT",
    # Doge metrics
    # metrics={
    #     "mae": 0.030250445508426545,
    #     "rmse": 0.03744389767732508,
    #     "pinball_q50": 0.015125222754213272,
    #     "pinball_q10": 0.010793771679417235,
    #     "pinball_q90": 0.007496124922390927,
    #     "sign_accuracy": 0.48680851063829785,
    #     "rank_ic_spearman": 0.2840705410935244,
    # },

    # BTC metrics
    metrics={
        "mae": 0.011688595781245224,
        "rmse": 0.01611459185722997,
        "pinball_q10": 0.0031033900473689795,
        "pinball_q90": 0.002794409976196414,
        "sign_accuracy": 0.48936170212765956,
        "rank_ic_spearman": 0.06696351225497632,
    },
)

proposed_ft = ModelMetrics(
    name="Chronos2 Proposed FT",
    # Doge metrics
    # metrics={
    #     "mae": 0.03221851408135618,
    #     "rmse": 0.03958706454582015,
    #     "pinball_q50": 0.01610925704067809,
    #     "pinball_q10": 0.01178801186730175,
    #     "pinball_q90": 0.007489048245740977,
    #     "sign_accuracy": 0.467021276595745,
    #     "rank_ic_spearman": 0.22606382978723405,
    # },

    # BTC metrics
        metrics={
        "mae": 0.01181861137187433,
        "rmse": 0.01636262073066979,
        "pinball_q10": 0.003175837566021858,
        "pinball_q90": 0.0025718311936331114,
        "sign_accuracy": 0.6170212765957447,
        "rank_ic_spearman": 0.2792553191489362,
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

# Typography (hardcoded, editable)
TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 18
X_TICK_FONTSIZE = 20
Y_TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

# Shared figure size for side-by-side placement in Overleaf.
FIGSIZE_TRADITIONAL = (9.5, 5.5)
FIGSIZE_TREND = FIGSIZE_TRADITIONAL
FIGSIZE_RISK = (FIGSIZE_TRADITIONAL[0] * 2.0, FIGSIZE_TRADITIONAL[1])

# Bar thickness (horizontal width). Increase this to make bars wider.
BAR_WIDTH = 0.32

# Traditional-metrics split style knobs:
# MAE (left block) -> hatched / hollow
# RMSE (right block) -> solid fill
TRADITIONAL_MAE_HATCH = "///"
TRADITIONAL_MAE_FILL = False
TRADITIONAL_MAE_ALPHA = 1.0
TRADITIONAL_MAE_EDGEWIDTH = 1.2

TRADITIONAL_RMSE_HATCH = None
TRADITIONAL_RMSE_FILL = True
TRADITIONAL_RMSE_ALPHA = 0.9
TRADITIONAL_RMSE_EDGEWIDTH = 1.0

# Risk-metrics style knobs (3 blocks, 3 distinct visual encodings)
# q10 -> rectangular frame + wave fill
RISK_Q10_SHAPE = "rect"
RISK_Q10_FILL = False
RISK_Q10_HATCH = None
RISK_Q10_ALPHA = 1.0
RISK_Q10_EDGEWIDTH = 1.8
RISK_Q10_WAVE_FILL = True
RISK_Q10_WAVE_ALPHA = 0.38
RISK_Q10_WAVE_LINEWIDTH = 0.95
RISK_Q10_WAVE_SPACING_RATIO = 0.20
RISK_Q10_WAVE_AMPLITUDE_RATIO = 0.08
RISK_Q10_WAVE_WAVELENGTH_RATIO = 0.55

# q90 -> hatched hollow (thick border)
RISK_Q90_SHAPE = "rect"
RISK_Q90_FILL = False
RISK_Q90_HATCH = "///"
RISK_Q90_ALPHA = 1.0
RISK_Q90_EDGEWIDTH = 1.7
RISK_Q90_ROUNDING_SIZE = 0.06
RISK_Q90_WAVE_FILL = False

# Tail Avg -> rounded solid
RISK_TAIL_SHAPE = "rounded"
RISK_TAIL_FILL = True
RISK_TAIL_HATCH = None
RISK_TAIL_ALPHA = 0.88
RISK_TAIL_EDGEWIDTH = 1.0
RISK_TAIL_ROUNDING_SIZE = 0.06
RISK_TAIL_WAVE_FILL = False

# Trend-metrics split style knobs:
# Sign Accuracy -> rounded solid bars
# Rank IC      -> rectangular hollow bars (thick border)
TREND_SIGN_SHAPE = "rect"
TREND_SIGN_FILL = False
TREND_SIGN_HATCH = None
TREND_SIGN_ALPHA = 1.0
TREND_SIGN_EDGEWIDTH = 2.0
# Rounded corner radius in data-x units (should be small, near bar width scale).
TREND_SIGN_ROUNDING_SIZE = 0.06
TREND_SIGN_WAVE_FILL = True
TREND_SIGN_WAVE_ALPHA = 0.40
TREND_SIGN_WAVE_LINEWIDTH = 1.0
TREND_SIGN_WAVE_SPACING_RATIO = 0.20
TREND_SIGN_WAVE_AMPLITUDE_RATIO = 0.08
TREND_SIGN_WAVE_WAVELENGTH_RATIO = 0.55

TREND_RANKIC_SHAPE = "rounded"
TREND_RANKIC_FILL = True
TREND_RANKIC_HATCH = None
TREND_RANKIC_ALPHA = 0.9
TREND_RANKIC_EDGEWIDTH = 1.0
TREND_RANKIC_ROUNDING_SIZE = 0.06
TREND_RANKIC_WAVE_FILL = False
TREND_RANKIC_WAVE_ALPHA = 0.40
TREND_RANKIC_WAVE_LINEWIDTH = 1.0
TREND_RANKIC_WAVE_SPACING_RATIO = 0.20
TREND_RANKIC_WAVE_AMPLITUDE_RATIO = 0.08
TREND_RANKIC_WAVE_WAVELENGTH_RATIO = 0.55

# Optional visibility overrides (same key style as color overrides).
# True  -> force show value label
# False -> force hide value label
BAR_VALUE_TEXT_VISIBLE_OVERRIDES: Dict[tuple[str, str], bool] = {}
BAR_VALUE_TEXT_VISIBLE_OVERRIDES = {
    # Risk metrics
    ("ARIMA+GARCH", "Pinball q90"): True,
    ("Chronos2 Zeroshot", "Pinball q90"): True,
    ("Chronos2 Native FT", "Pinball q90"): True,
    ("Chronos2 Proposed FT", "Pinball q90"): True,
    
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
    # ("ARIMA+GARCH", "pinball_q90"): "red",
    # ("Chronos2 Zeroshot", "pinball_q90"): "green",
    ("Chronos2 Native FT", "pinball_q90"): "red",
    ("Chronos2 Proposed FT", "pinball_q90"): "red",

    # ("ARIMA+GARCH", "Tail Avg"): "red",
    # ("Chronos2 Zeroshot", "Tail Avg"): "green",
    ("Chronos2 Native FT", "Tail Avg"): "red",
    ("Chronos2 Proposed FT", "Tail Avg"): "red",

    # Trend metrics
    # ("ARIMA+GARCH", "Sign Accuracy"): "red",
    # ("Chronos2 Zeroshot", "Sign Accuracy"): "green",
    ("Chronos2 Native FT", "Sign Accuracy"): "green",
    ("Chronos2 Proposed FT", "Sign Accuracy"): "green",

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
    # ("ARIMA+GARCH", "pinball_q90"): "up",
    # ("Chronos2 Zeroshot", "pinball_q90"): "up",
    # ("Chronos2 Native FT", "pinball_q90"): "down",
    ("Chronos2 Proposed FT", "pinball_q90"): "down",

    # ("ARIMA+GARCH", "Tail Avg"): "up",
    # ("Chronos2 Zeroshot", "Tail Avg"): "up",
    # ("Chronos2 Native FT", "Tail Avg"): "up",
    ("Chronos2 Proposed FT", "Tail Avg"): "down",

    # Trend metrics
    # ("ARIMA+GARCH", "Sign Accuracy"): "red",
    # ("Chronos2 Zeroshot", "Sign Accuracy"): "green",
    # ("Chronos2 Native FT", "Sign Accuracy"): "up",
    ("Chronos2 Proposed FT", "Sign Accuracy"): "up",

    # ("ARIMA+GARCH", "Rank IC (Spearman)"): "up",
    # ("Chronos2 Zeroshot", "Rank IC (Spearman)"): "up",
    # ("Chronos2 Native FT", "Rank IC (Spearman)"): "up",
    ("Chronos2 Proposed FT", "Rank IC (Spearman)"): "up",
}
ARROW_UP_COLOR = "green"
ARROW_DOWN_COLOR = "red"
ARROW_FONTSIZE = VALUE_TEXT_FONTSIZE

# Arrow horizontal spacing (in points) when number + arrow are shown together.
# Final offset = ARROW_BASE_OFFSET_PTS + max(0, len(value_text)-4) * ARROW_PER_CHAR_OFFSET_PTS
ARROW_BASE_OFFSET_PTS = 10
ARROW_PER_CHAR_OFFSET_PTS = 5

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


class _HalfSplitLegendHandle:
    """Legend proxy: left half MAE(hatched), right half RMSE(solid)."""

    def __init__(self, color: str, mae_hatch: str = TRADITIONAL_MAE_HATCH) -> None:
        self.color = color
        self.mae_hatch = mae_hatch


class _HalfSplitLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        left = Rectangle(
            (xdescent, ydescent),
            width * 0.5,
            height,
            facecolor="none",
            edgecolor=orig_handle.color,
            hatch=orig_handle.mae_hatch,
            linewidth=1.0,
            transform=trans,
        )
        right = Rectangle(
            (xdescent + width * 0.5, ydescent),
            width * 0.5,
            height,
            facecolor=orig_handle.color,
            edgecolor=orig_handle.color,
            linewidth=1.0,
            transform=trans,
            alpha=TRADITIONAL_RMSE_ALPHA,
        )
        return [left, right]


class _TrendSplitLegendHandle:
    """Legend proxy: left Sign Accuracy(solid rounded), right Rank IC(hollow + waves)."""

    def __init__(self, color: str) -> None:
        self.color = color


class _TrendSplitLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        if TREND_SIGN_SHAPE == "rounded":
            left = FancyBboxPatch(
                (xdescent, ydescent),
                width * 0.5,
                height,
                boxstyle="round,pad=0,rounding_size=1.8",
                facecolor=orig_handle.color if TREND_SIGN_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=TREND_SIGN_EDGEWIDTH,
                alpha=TREND_SIGN_ALPHA,
                transform=trans,
            )
        else:
            left = Rectangle(
                (xdescent, ydescent),
                width * 0.5,
                height,
                facecolor=orig_handle.color if TREND_SIGN_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=TREND_SIGN_EDGEWIDTH,
                alpha=TREND_SIGN_ALPHA,
                transform=trans,
            )

        if TREND_RANKIC_SHAPE == "rounded":
            right = FancyBboxPatch(
                (xdescent + width * 0.5, ydescent),
                width * 0.5,
                height,
                boxstyle="round,pad=0,rounding_size=1.8",
                facecolor=orig_handle.color if TREND_RANKIC_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=TREND_RANKIC_EDGEWIDTH,
                alpha=TREND_RANKIC_ALPHA,
                transform=trans,
            )
        else:
            right = Rectangle(
                (xdescent + width * 0.5, ydescent),
                width * 0.5,
                height,
                facecolor=orig_handle.color if TREND_RANKIC_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=TREND_RANKIC_EDGEWIDTH,
                alpha=TREND_RANKIC_ALPHA,
                transform=trans,
            )

        artists = [left, right]
        if TREND_SIGN_WAVE_FILL:
            xl0 = xdescent
            xl1 = xdescent + width * 0.5
            yl0 = ydescent
            yl1 = ydescent + height
            ys = np.linspace(yl0 + height * 0.18, yl1 - height * 0.18, 3)
            x = np.linspace(xl0, xl1, 40)
            for i, y0 in enumerate(ys):
                amp = height * 0.08
                wl = max(1e-6, (xl1 - xl0) * 0.7)
                y = y0 + amp * np.sin((2 * np.pi * (x - xl0) / wl) + i * np.pi / 3.0)
                line = Line2D(
                    x,
                    y,
                    color=orig_handle.color,
                    alpha=TREND_SIGN_WAVE_ALPHA,
                    linewidth=1.0,
                    transform=trans,
                )
                line.set_clip_path(left)
                artists.append(line)
        if TREND_RANKIC_WAVE_FILL:
            xr0 = xdescent + width * 0.5
            xr1 = xdescent + width
            yr0 = ydescent
            yr1 = ydescent + height
            ys = np.linspace(yr0 + height * 0.18, yr1 - height * 0.18, 3)
            x = np.linspace(xr0, xr1, 40)
            for i, y0 in enumerate(ys):
                amp = height * 0.08
                wl = max(1e-6, (xr1 - xr0) * 0.7)
                y = y0 + amp * np.sin((2 * np.pi * (x - xr0) / wl) + i * np.pi / 3.0)
                line = Line2D(
                    x,
                    y,
                    color=orig_handle.color,
                    alpha=TREND_RANKIC_WAVE_ALPHA,
                    linewidth=1.0,
                    transform=trans,
                )
                line.set_clip_path(right)
                artists.append(line)
        return artists


class _RiskTripleLegendHandle:
    """Legend proxy: q10(left wave-frame), q90(mid rounded-solid), tail(right hatched-frame)."""

    def __init__(self, color: str) -> None:
        self.color = color


class _RiskTripleLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        w = width / 3.0
        left = Rectangle(
            (xdescent, ydescent),
            w,
            height,
            facecolor=orig_handle.color if RISK_Q10_FILL else "none",
            edgecolor=orig_handle.color,
            linewidth=RISK_Q10_EDGEWIDTH,
            alpha=RISK_Q10_ALPHA,
            transform=trans,
        )
        if RISK_Q90_SHAPE == "rounded":
            mid = FancyBboxPatch(
                (xdescent + w, ydescent),
                w,
                height,
                boxstyle="round,pad=0,rounding_size=1.8",
                facecolor=orig_handle.color if RISK_Q90_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=RISK_Q90_EDGEWIDTH,
                hatch=RISK_Q90_HATCH,
                alpha=RISK_Q90_ALPHA,
                transform=trans,
            )
        else:
            mid = Rectangle(
                (xdescent + w, ydescent),
                w,
                height,
                facecolor=orig_handle.color if RISK_Q90_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=RISK_Q90_EDGEWIDTH,
                hatch=RISK_Q90_HATCH,
                alpha=RISK_Q90_ALPHA,
                transform=trans,
            )
        if RISK_TAIL_SHAPE == "rounded":
            right = FancyBboxPatch(
                (xdescent + 2 * w, ydescent),
                w,
                height,
                boxstyle="round,pad=0,rounding_size=1.8",
                facecolor=orig_handle.color if RISK_TAIL_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=RISK_TAIL_EDGEWIDTH,
                hatch=RISK_TAIL_HATCH,
                alpha=RISK_TAIL_ALPHA,
                transform=trans,
            )
        else:
            right = Rectangle(
                (xdescent + 2 * w, ydescent),
                w,
                height,
                facecolor=orig_handle.color if RISK_TAIL_FILL else "none",
                edgecolor=orig_handle.color,
                linewidth=RISK_TAIL_EDGEWIDTH,
                hatch=RISK_TAIL_HATCH,
                alpha=RISK_TAIL_ALPHA,
                transform=trans,
            )
        artists = [left, mid, right]
        if RISK_Q10_WAVE_FILL:
            ys = np.linspace(ydescent + height * 0.18, ydescent + height * 0.82, 3)
            x = np.linspace(xdescent, xdescent + w, 40)
            for i, y0 in enumerate(ys):
                amp = height * 0.08
                wl = max(1e-6, w * 0.7)
                y = y0 + amp * np.sin((2 * np.pi * (x - xdescent) / wl) + i * np.pi / 3.0)
                line = Line2D(
                    x,
                    y,
                    color=orig_handle.color,
                    alpha=RISK_Q10_WAVE_ALPHA,
                    linewidth=RISK_Q10_WAVE_LINEWIDTH,
                    transform=trans,
                )
                line.set_clip_path(left)
                artists.append(line)
        return artists


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


def _draw_wave_fill_in_bar(
    ax: plt.Axes,
    clip_patch,
    x_center: float,
    bar_width: float,
    y_value: float,
    color: str,
    alpha: float,
    linewidth: float,
    spacing_ratio: float,
    amplitude_ratio: float,
    wavelength_ratio: float,
) -> None:
    """Draw sine-wave strokes clipped inside one bar."""
    if not np.isfinite(y_value) or y_value == 0:
        return
    x_left = x_center - bar_width / 2.0
    x_right = x_center + bar_width / 2.0
    y_bottom = min(0.0, float(y_value))
    y_top = max(0.0, float(y_value))
    h = y_top - y_bottom
    if h <= 0:
        return

    spacing = max(1e-6, h * max(0.02, float(spacing_ratio)))
    amplitude = h * max(0.01, float(amplitude_ratio))
    wavelength = bar_width * max(0.15, float(wavelength_ratio))

    x_vals = np.linspace(x_left, x_right, 100)
    y_base = y_bottom + spacing * 0.5
    i = 0
    while y_base < y_top:
        phase = i * (np.pi / 3.0)
        y_vals = y_base + amplitude * np.sin((2.0 * np.pi * (x_vals - x_left) / wavelength) + phase)
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            alpha=float(alpha),
            linewidth=float(linewidth),
            clip_path=clip_patch,
            clip_on=True,
            zorder=2.3,
        )
        y_base += spacing
        i += 1


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
    block_spacing: float = 1.8,
    legend_handles=None,
    legend_labels=None,
    legend_handler_map=None,
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
    block_centers = np.arange(n_blocks) * float(block_spacing)
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
        ys = []
        model_bar_records = []
        for block_idx, block in enumerate(metric_blocks):
            if "func" in block:
                val = block["func"](model)
            else:
                val = model.metrics.get(block["key"], np.nan)
            if val is None:
                val = np.nan
            ys.append(val)

            x = float(block_centers[block_idx] + offsets[model_idx])
            y = float(val) if np.isfinite(val) else np.nan

            fill = bool(block.get("fill", True))
            hatch = block.get("hatch", None)
            alpha = float(block.get("alpha", 0.9))
            edge_lw = float(block.get("edge_linewidth", 1.0))
            base_color = MODEL_COLORS[model.name]
            face_color = base_color if fill else "none"

            shape = str(block.get("shape", "rect")).strip().lower()
            if np.isfinite(y) and shape == "rounded":
                y0 = 0.0 if y >= 0 else y
                h = abs(y)
                rounding_size = float(block.get("rounding_size", 0.06))
                # Prevent over-rounded geometry on tiny bars (can create SVG artifacts).
                max_rounding = max(1e-6, min(bar_width * 0.45, h * 0.45))
                rounding_size = min(rounding_size, max_rounding)
                rounded = FancyBboxPatch(
                    (x - bar_width / 2.0, y0),
                    bar_width,
                    h,
                    boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                    facecolor=face_color,
                    edgecolor=base_color,
                    linewidth=edge_lw,
                    hatch=hatch,
                    alpha=alpha,
                    zorder=2,
                )
                ax.add_patch(rounded)
                if block_idx == 0:
                    rounded.set_label(model.name)
                patch_obj = rounded
            else:
                bars = ax.bar(
                    [x],
                    [y],
                    width=bar_width,
                    label=model.name if block_idx == 0 else None,
                    color=face_color,
                    edgecolor=base_color,
                    linewidth=edge_lw,
                    hatch=hatch,
                    alpha=alpha,
                )
                patch_obj = bars.patches[0]

            if bool(block.get("wave_fill", False)) and np.isfinite(y):
                _draw_wave_fill_in_bar(
                    ax=ax,
                    clip_patch=patch_obj,
                    x_center=x,
                    bar_width=bar_width,
                    y_value=y,
                    color=base_color,
                    alpha=float(block.get("wave_alpha", 0.40)),
                    linewidth=float(block.get("wave_linewidth", 1.0)),
                    spacing_ratio=float(block.get("wave_spacing_ratio", 0.20)),
                    amplitude_ratio=float(block.get("wave_amplitude_ratio", 0.08)),
                    wavelength_ratio=float(block.get("wave_wavelength_ratio", 0.55)),
                )

            model_bar_records.append((x, y))

        ys_arr = np.asarray(ys, dtype=float)
        bar_records.append((model, model_bar_records, ys_arr))
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
    for model, model_bar_records, ys_arr in bar_records:
        for i, ((x, yv), _) in enumerate(zip(model_bar_records, ys_arr)):
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
    ax.set_xticklabels([b["label"] for b in metric_blocks], fontsize=X_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if legend_handles is not None:
        labels = legend_labels if legend_labels is not None else []
        ax.legend(
            legend_handles,
            labels,
            frameon=False,
            ncol=2,
            fontsize=LEGEND_FONTSIZE,
            handler_map=legend_handler_map if legend_handler_map is not None else {},
        )
    else:
        ax.legend(frameon=False, ncol=2, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# 4. Figure 1: Traditional error metrics
# -------------------------------------------------------------------

def plot_traditional_metrics(out_path: str = "btc_traditional_metrics.svg") -> None:
    metric_blocks = [
        {
            "label": "MAE",
            "key": "mae",
            "hatch": TRADITIONAL_MAE_HATCH,
            "fill": TRADITIONAL_MAE_FILL,
            "alpha": TRADITIONAL_MAE_ALPHA,
            "edge_linewidth": TRADITIONAL_MAE_EDGEWIDTH,
        },
        {
            "label": "RMSE",
            "key": "rmse",
            "hatch": TRADITIONAL_RMSE_HATCH,
            "fill": TRADITIONAL_RMSE_FILL,
            "alpha": TRADITIONAL_RMSE_ALPHA,
            "edge_linewidth": TRADITIONAL_RMSE_EDGEWIDTH,
        },
    ]
    legend_handles = [
        _HalfSplitLegendHandle(MODEL_COLORS[m.name], mae_hatch=TRADITIONAL_MAE_HATCH)
        for m in MODELS
    ]
    legend_labels = [m.name for m in MODELS]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Traditional Error Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=FIGSIZE_TRADITIONAL,
        y_pad_upper=Y_PAD_UPPER_TRADITIONAL,
        y_pad_lower=Y_PAD_LOWER_TRADITIONAL,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        legend_handler_map={_HalfSplitLegendHandle: _HalfSplitLegendHandler()},
    )


# -------------------------------------------------------------------
# 5. Figure 2: Risk characterization metrics
# -------------------------------------------------------------------

def plot_risk_metrics(out_path: str = "btc_risk_metrics.svg") -> None:
    metric_blocks = [
        {
            "label": "Pinball q10",
            "key": "pinball_q10",
            "shape": RISK_Q10_SHAPE,
            "fill": RISK_Q10_FILL,
            "hatch": RISK_Q10_HATCH,
            "alpha": RISK_Q10_ALPHA,
            "edge_linewidth": RISK_Q10_EDGEWIDTH,
            "wave_fill": RISK_Q10_WAVE_FILL,
            "wave_alpha": RISK_Q10_WAVE_ALPHA,
            "wave_linewidth": RISK_Q10_WAVE_LINEWIDTH,
            "wave_spacing_ratio": RISK_Q10_WAVE_SPACING_RATIO,
            "wave_amplitude_ratio": RISK_Q10_WAVE_AMPLITUDE_RATIO,
            "wave_wavelength_ratio": RISK_Q10_WAVE_WAVELENGTH_RATIO,
        },
        {
            "label": "Pinball q90",
            "key": "pinball_q90",
            "shape": RISK_Q90_SHAPE,
            "fill": RISK_Q90_FILL,
            "hatch": RISK_Q90_HATCH,
            "alpha": RISK_Q90_ALPHA,
            "edge_linewidth": RISK_Q90_EDGEWIDTH,
            "rounding_size": RISK_Q90_ROUNDING_SIZE,
            "wave_fill": RISK_Q90_WAVE_FILL,
        },
        {
            "label": "Tail Avg",
            "func": tail_pinball_avg,
            "shape": RISK_TAIL_SHAPE,
            "fill": RISK_TAIL_FILL,
            "hatch": RISK_TAIL_HATCH,
            "alpha": RISK_TAIL_ALPHA,
            "edge_linewidth": RISK_TAIL_EDGEWIDTH,
            "rounding_size": RISK_TAIL_ROUNDING_SIZE,
            "wave_fill": RISK_TAIL_WAVE_FILL,
        },
    ]
    legend_handles = [_RiskTripleLegendHandle(MODEL_COLORS[m.name]) for m in MODELS]
    legend_labels = [m.name for m in MODELS]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Risk Characterization Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=FIGSIZE_RISK,
        y_pad_upper=Y_PAD_UPPER_RISK,
        y_pad_lower=Y_PAD_LOWER_RISK,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        legend_handler_map={_RiskTripleLegendHandle: _RiskTripleLegendHandler()},
    )


# -------------------------------------------------------------------
# 6. Figure 3: Trend-following metrics
# -------------------------------------------------------------------

def plot_trend_metrics(out_path: str = "btc_trend_metrics.svg") -> None:
    metric_blocks = [
        {
            "label": "Sign Accuracy",
            "key": "sign_accuracy",
            "shape": TREND_SIGN_SHAPE,
            "fill": TREND_SIGN_FILL,
            "hatch": TREND_SIGN_HATCH,
            "alpha": TREND_SIGN_ALPHA,
            "edge_linewidth": TREND_SIGN_EDGEWIDTH,
            "rounding_size": TREND_SIGN_ROUNDING_SIZE,
            "wave_fill": TREND_SIGN_WAVE_FILL,
            "wave_alpha": TREND_SIGN_WAVE_ALPHA,
            "wave_linewidth": TREND_SIGN_WAVE_LINEWIDTH,
            "wave_spacing_ratio": TREND_SIGN_WAVE_SPACING_RATIO,
            "wave_amplitude_ratio": TREND_SIGN_WAVE_AMPLITUDE_RATIO,
            "wave_wavelength_ratio": TREND_SIGN_WAVE_WAVELENGTH_RATIO,
        },
        {
            "label": "Rank IC (Spearman)",
            "key": "rank_ic_spearman",
            "shape": TREND_RANKIC_SHAPE,
            "fill": TREND_RANKIC_FILL,
            "hatch": TREND_RANKIC_HATCH,
            "alpha": TREND_RANKIC_ALPHA,
            "edge_linewidth": TREND_RANKIC_EDGEWIDTH,
            "rounding_size": TREND_RANKIC_ROUNDING_SIZE,
            "wave_fill": TREND_RANKIC_WAVE_FILL,
            "wave_alpha": TREND_RANKIC_WAVE_ALPHA,
            "wave_linewidth": TREND_RANKIC_WAVE_LINEWIDTH,
            "wave_spacing_ratio": TREND_RANKIC_WAVE_SPACING_RATIO,
            "wave_amplitude_ratio": TREND_RANKIC_WAVE_AMPLITUDE_RATIO,
            "wave_wavelength_ratio": TREND_RANKIC_WAVE_WAVELENGTH_RATIO,
        },
    ]
    legend_handles = [_TrendSplitLegendHandle(MODEL_COLORS[m.name]) for m in MODELS]
    legend_labels = [m.name for m in MODELS]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Trend-Following Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=FIGSIZE_TREND,
        block_spacing=2.4,
        y_pad_upper=Y_PAD_UPPER_TREND,
        y_pad_lower=Y_PAD_LOWER_TREND,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        legend_handler_map={_TrendSplitLegendHandle: _TrendSplitLegendHandler()},
    )


# -------------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    plot_traditional_metrics()
    plot_risk_metrics()
    plot_trend_metrics()
    print("Done. Saved:")
    print(" - btc_traditional_metrics.svg")
    print(" - btc_risk_metrics.svg")
    print(" - btc_trend_metrics.svg")
