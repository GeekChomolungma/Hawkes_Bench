from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def compute_backtest_metrics(
    bt: pd.DataFrame,
    bars_per_year: int = 252,
) -> dict:
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
    years = n / float(bars_per_year)
    cagr = float(equity.iloc[-1] ** (1.0 / (years + 1e-12)) - 1.0)

    ann_vol = float(pnl.std(ddof=1) * np.sqrt(bars_per_year))
    ann_mean = float(pnl.mean() * bars_per_year)
    sharpe = float(ann_mean / (ann_vol + 1e-12))

    downside = pnl[pnl < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(bars_per_year)) if len(downside) > 1 else 0.0
    sortino = float(ann_mean / (downside_vol + 1e-12))

    mdd = max_drawdown(equity)
    calmar = float(cagr / (abs(mdd) + 1e-12))
    turnover = float(bt["dpos"].mean())

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


def compute_pinball_loss(y_true: pd.Series, y_pred_q: pd.Series, q: float) -> float:
    y = y_true.astype(float).to_numpy()
    yq = y_pred_q.astype(float).to_numpy()
    err = y - yq
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.nanmean(loss))


def compute_forecast_metrics(
    y_true: pd.Series,
    mu_pred: pd.Series | None = None,
    quantile_preds: dict[float, pd.Series] | None = None,
) -> dict:
    y = y_true.astype(float) # returns
    out: dict[str, float] = {}

    if mu_pred is not None:
        yp = mu_pred.reindex(y.index).astype(float)
        err = y - yp
        out["mse"] = float(np.nanmean(err**2))
        out["mae"] = float(np.nanmean(np.abs(err)))
        out["rmse"] = float(np.sqrt(out["mse"]))

    if quantile_preds:
        q_losses = []
        for q, s in quantile_preds.items():
            aligned = s.reindex(y.index)
            pl = compute_pinball_loss(y, aligned, q=q)
            out[f"pinball_q{int(q*100):02d}"] = pl
            q_losses.append(pl)
        if q_losses:
            out["pinball_mean"] = float(np.nanmean(q_losses))

    return out


# backward compatibility

def compute_metrics(bt: pd.DataFrame, bars_per_year: int = 252) -> dict:
    return compute_backtest_metrics(bt=bt, bars_per_year=bars_per_year)
