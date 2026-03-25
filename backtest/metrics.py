from __future__ import annotations

import math

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
    turnover = float(bt["dpos"].abs().mean())

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


def _compute_picp(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    y = y_true.astype(float)
    lo = lower.reindex(y.index).astype(float)
    hi = upper.reindex(y.index).astype(float)
    hit = (y >= lo) & (y <= hi)
    return float(hit.mean())


def _compute_crps_gaussian(y_true: pd.Series, mu_pred: pd.Series, sigma_pred: pd.Series) -> float:
    y = y_true.astype(float)
    mu = mu_pred.reindex(y.index).astype(float)
    sig = sigma_pred.reindex(y.index).astype(float).clip(lower=1e-12)

    z = ((y - mu) / sig).to_numpy(dtype=float)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    crps = sig.to_numpy(dtype=float) * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.nanmean(crps))


def _compute_gaussian_interval(
    y_index: pd.Index,
    mu_pred: pd.Series,
    sigma_pred: pd.Series,
    z: float,
) -> tuple[pd.Series, pd.Series]:
    mu = mu_pred.reindex(y_index).astype(float)
    sig = sigma_pred.reindex(y_index).astype(float).clip(lower=1e-12)
    lower = mu - z * sig
    upper = mu + z * sig
    return lower, upper


def _append_interval_metrics(
    out: dict[str, float],
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    picp_key: str,
    q_lower: float,
    q_upper: float,
) -> None:
    out[picp_key] = _compute_picp(y_true=y_true, lower=lower, upper=upper)
    out[f"pinball_q{int(q_lower*100):02d}"] = compute_pinball_loss(y_true=y_true, y_pred_q=lower, q=q_lower)
    out[f"pinball_q{int(q_upper*100):02d}"] = compute_pinball_loss(y_true=y_true, y_pred_q=upper, q=q_upper)


def compute_forecast_metrics(
    y_true: pd.Series,
    mu_pred: pd.Series | None = None,
    sigma_pred: pd.Series | None = None,
    quantile_preds: dict[float, pd.Series] | None = None,
) -> dict:
    y = y_true.astype(float)
    out: dict[str, float] = {}
    point_pred: pd.Series | None = None

    if mu_pred is not None:
        yp = mu_pred.reindex(y.index).astype(float)
        point_pred = yp
        err = y - yp
        out["mse"] = float(np.nanmean(err**2))
        out["mae"] = float(np.nanmean(np.abs(err)))
        out["rmse"] = float(np.sqrt(out["mse"]))
    elif quantile_preds and 0.50 in quantile_preds:
        # Fallback point prediction for direction/ranking metrics when mu_pred is unavailable.
        point_pred = quantile_preds[0.50].reindex(y.index).astype(float)

    q_losses = []
    if quantile_preds:
        for q, s in quantile_preds.items():
            aligned = s.reindex(y.index)
            pl = compute_pinball_loss(y, aligned, q=q)
            out[f"pinball_q{int(q*100):02d}"] = pl
            q_losses.append(pl)
        if q_losses:
            out["pinball_mean"] = float(np.nanmean(q_losses))
            # CRPS approximation from quantile scores.
            out["crps_quantile_approx"] = float(2.0 * np.nanmean(q_losses))

    if sigma_pred is not None and mu_pred is not None:
        out["crps_gaussian"] = _compute_crps_gaussian(y_true=y, mu_pred=mu_pred, sigma_pred=sigma_pred)

    if quantile_preds and 0.10 in quantile_preds and 0.90 in quantile_preds:
        out["picp_80"] = _compute_picp(y_true=y, lower=quantile_preds[0.10], upper=quantile_preds[0.90])
    elif sigma_pred is not None and mu_pred is not None:
        # 80% interval for Gaussian: z ~= 1.28155
        z80 = 1.2815515655446004
        lower, upper = _compute_gaussian_interval(y_index=y.index, mu_pred=mu_pred, sigma_pred=sigma_pred, z=z80)
        _append_interval_metrics(
            out=out, y_true=y, lower=lower, upper=upper, picp_key="picp_80", q_lower=0.10, q_upper=0.90
        )

    # Additional PICP 20% ~ 80% if quantiles are available.
    if quantile_preds and 0.20 in quantile_preds and 0.80 in quantile_preds:
        out["picp_60"] = _compute_picp(y_true=y, lower=quantile_preds[0.20], upper=quantile_preds[0.80])
    elif sigma_pred is not None and mu_pred is not None:
        # 60% interval for Gaussian: z ~= 0.84162
        z60 = 0.8416212335729143
        lower, upper = _compute_gaussian_interval(y_index=y.index, mu_pred=mu_pred, sigma_pred=sigma_pred, z=z60)
        _append_interval_metrics(
            out=out, y_true=y, lower=lower, upper=upper, picp_key="picp_60", q_lower=0.20, q_upper=0.80
        )

    # Direction/ranking metrics:
    # - sign_accuracy: proportion of timestamps where sign(pred) == sign(y_true)
    # - ic_pearson: linear correlation between point forecast and y_true
    # - rank_ic_spearman: rank correlation (Information Coefficient)
    if point_pred is not None:
        yp = point_pred.reindex(y.index).astype(float)
        mask = y.notna() & yp.notna()
        yv = y[mask]
        pv = yp[mask]
        if len(yv) > 0:
            out["sign_accuracy"] = float((np.sign(pv.to_numpy()) == np.sign(yv.to_numpy())).mean())
            out["ic_pearson"] = float(yv.corr(pv, method="pearson"))
            out["rank_ic_spearman"] = float(yv.corr(pv, method="spearman"))

    return out


# backward compatibility

def compute_metrics(bt: pd.DataFrame, bars_per_year: int = 252) -> dict:
    return compute_backtest_metrics(bt=bt, bars_per_year=bars_per_year)
