import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())

def compute_metrics(
    bt: pd.DataFrame,
    bars_per_year: int = 252,   # 1d bar by default, change to 252*6.5 for 1m bars, etc.
) -> dict:
    """
    Compute thesis-friendly metrics from backtest output.
    bt must contain ['pnl','equity','dpos'] at least.
    pnl is per-bar log-return-like (because you used log returns).
    """

    pnl = bt["pnl"].astype(float)
    equity = bt["equity"].astype(float)

    n = len(bt)
    if n < 5:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "turnover": np.nan,
            "hit_rate": np.nan,
        }

    total_return = float(equity.iloc[-1] - 1.0)

    # Approx CAGR from equity curve length in years
    years = n / float(bars_per_year)
    cagr = float(equity.iloc[-1] ** (1.0 / (years + 1e-12)) - 1.0)

    ann_vol = float(pnl.std(ddof=1) * np.sqrt(bars_per_year))
    ann_mean = float(pnl.mean() * bars_per_year)

    sharpe = float(ann_mean / (ann_vol + 1e-12))

    # Sortino (downside deviation)
    downside = pnl[pnl < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(bars_per_year)) if len(downside) > 1 else 0.0
    sortino = float(ann_mean / (downside_vol + 1e-12))

    mdd = max_drawdown(equity)
    calmar = float(cagr / (abs(mdd) + 1e-12))

    turnover = float(bt["dpos"].mean())  # average abs position change per bar

    # hit-rate: proportion of pnl > 0 (excluding zeros)
    pnl_nonzero = pnl[pnl != 0]
    hit_rate = float((pnl_nonzero > 0).mean()) if len(pnl_nonzero) > 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "turnover": turnover,
        "hit_rate": hit_rate,
    }