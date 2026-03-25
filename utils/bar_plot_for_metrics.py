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
    metrics={
        "mae": 0.012048966209237429,
        "rmse": 0.016174684996494618,
        "pinball_q50": None,  # white-box currently unavailable / optional
        "pinball_q10": 0.0033625345749007216,
        "pinball_q90": 0.0031699228735634,
        "sign_accuracy": 0.3404255319148936,
        "rank_ic_spearman": -0.16709065679925994,
    },
)

zeroshot = ModelMetrics(
    name="Chronos2 Zeroshot",
    metrics={
        "mae": 0.01170353549489713,
        "rmse": 0.01599709227898297,
        "pinball_q50": 0.005851767747448565,
        "pinball_q10": 0.0029783736351348313,
        "pinball_q90": 0.003045043702925137,
        "sign_accuracy": 0.425531914893617,
        "rank_ic_spearman": -0.17496386268044065,
    },
)

native_ft = ModelMetrics(
    name="Chronos2 Native FT",
    metrics={
        "mae": 0.011688595781245224,
        "rmse": 0.01611459185722997,
        "pinball_q50": 0.005844297890622612,
        "pinball_q10": 0.0031033900473689795,
        "pinball_q90": 0.002794409976196414,
        "sign_accuracy": 0.48936170212765956,
        "rank_ic_spearman": 0.06696351225497632,
    },
)

proposed_ft = ModelMetrics(
    name="Chronos2 Proposed FT",
    metrics={
        "mae": 0.01181861137187433,
        "rmse": 0.01636262073066979,
        "pinball_q50": 0.005909305685937165,
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
# 3. Helper functions
# -------------------------------------------------------------------

def tail_pinball_avg(model: ModelMetrics) -> float:
    q10 = model.metrics["pinball_q10"]
    q90 = model.metrics["pinball_q90"]
    return (q10 + q90) / 2.0


def plot_grouped_metric_blocks(
    metric_blocks: List[Dict],
    title: str,
    ylabel: str,
    out_path: str,
    figsize=(12, 6),
) -> None:
    """
    Each block is a metric region, e.g.
    [{"label": "MAE", "key": "mae"}, {"label": "RMSE", "key": "rmse"}]
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(MODELS)
    n_blocks = len(metric_blocks)

    block_centers = np.arange(n_blocks) * 1.8
    bar_width = 0.18

    offsets = np.linspace(
        -1.5 * bar_width, 1.5 * bar_width, n_models
    )

    for model_idx, model in enumerate(MODELS):
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

        ax.bar(
            xs,
            ys,
            width=bar_width,
            label=model.name,
            color=MODEL_COLORS[model.name],
            alpha=0.9,
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

def plot_traditional_metrics(out_path: str = "btc_traditional_metrics.svg") -> None:
    metric_blocks = [
        {"label": "MAE", "key": "mae"},
        {"label": "RMSE", "key": "rmse"},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Traditional Error Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(9, 5.5),
    )


# -------------------------------------------------------------------
# 5. Figure 2: Risk characterization metrics
# -------------------------------------------------------------------

def plot_risk_metrics(out_path: str = "btc_risk_metrics.svg") -> None:
    metric_blocks = [
        {"label": "Pinball q10", "key": "pinball_q10"},
        {"label": "Pinball q90", "key": "pinball_q90"},
        {"label": "Tail Avg", "func": tail_pinball_avg},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Risk Characterization Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(12, 5.5),
    )


# -------------------------------------------------------------------
# 6. Figure 3: Trend-following metrics
# -------------------------------------------------------------------

def plot_trend_metrics(out_path: str = "btc_trend_metrics.svg") -> None:
    metric_blocks = [
        {"label": "Sign Accuracy", "key": "sign_accuracy"},
        {"label": "Rank IC (Spearman)", "key": "rank_ic_spearman"},
    ]
    plot_grouped_metric_blocks(
        metric_blocks=metric_blocks,
        title="BTC Trend-Following Metrics",
        ylabel="Metric Value",
        out_path=out_path,
        figsize=(10, 5.5),
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
