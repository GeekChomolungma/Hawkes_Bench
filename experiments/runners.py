from __future__ import annotations

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_backtest_metrics, compute_forecast_metrics
from risk.hawkes_scaler import apply_hawkes_risk_scaling
from risk.native import attach_native_fields
from strategy_signal.unified_signal import build_position, build_stateful_all_in_position
from utils.persist import save_dataframe, save_metrics


def make_next_return_target(returns: pd.Series, decision_index: pd.DatetimeIndex) -> pd.Series:
    r_next = returns.shift(-1)
    y = r_next.reindex(decision_index).dropna()
    return y


def evaluate_forecast_frame(
    forecast_df: pd.DataFrame,
    returns: pd.Series,
    metrics_out_path: str,
    rows_out_path: str | None = None,
) -> dict:
    df = forecast_df.copy() # pred_for_ts
    if "ts" in df.columns:
        df = df.set_index("ts").sort_index()

    y_true = make_next_return_target(returns=returns, decision_index=df.index) # next-bar return at decision times
    df = df.reindex(y_true.index)

    quantile_preds = {}
    for q, col in ((0.05, "q05"), (0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90"), (0.95, "q95")):
        if col in df.columns:
            quantile_preds[q] = df[col]

    mu = df["mu_pred"] if "mu_pred" in df.columns else None
    metrics = compute_forecast_metrics(y_true=y_true, mu_pred=mu, quantile_preds=quantile_preds or None) # y_true -- r_next vs.  "pred_for_ts" mu_pred
    save_metrics(metrics, metrics_out_path)

    if rows_out_path:
        out = pd.DataFrame({"y_true": y_true})
        if mu is not None:
            out["mu_pred"] = mu.reindex(y_true.index)
        for q, s in quantile_preds.items():
            out[f"q{int(q*100):02d}"] = s.reindex(y_true.index)
        save_dataframe(out.reset_index(), rows_out_path, index=False)

    return metrics


def run_strategy_backtest(
    forecast_df: pd.DataFrame,
    close: pd.Series,
    lam: pd.Series,
    alpha_risk: float,
    fee_bps: float,
    slippage_bps: float,
    bars_per_year: int,
    position_cap: float,
    use_hawkes: bool,
    execution_mode: str = "stateful_all_in",
    entry_threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    df = forecast_df.copy()
    if "ts" in df.columns:
        df = df.set_index("ts").sort_index()

    df = attach_native_fields(df) # [ts, pred_for_ts, mu_pred, sigma_pred, native_risk...], the "native_risk" are binding to "pred_for_ts", default to "sigma_pred"
    idx = df.index.intersection(close.index)
    df = df.reindex(idx)

    native_risk = df["native_risk"].ffill().bfill() # index is decision tims -- ts.
    if use_hawkes:
        risk = apply_hawkes_risk_scaling(native_risk=native_risk, lam=lam.reindex(idx), alpha_risk=alpha_risk)
    else:
        risk = native_risk

    if execution_mode == "target_continuous":
        pos = build_position(expected_return=df["expected_return"], adjusted_risk=risk, cap=position_cap)
    elif execution_mode == "stateful_all_in":
        pos = build_stateful_all_in_position(
            expected_return=df["expected_return"],
            adjusted_risk=risk,
            cap=position_cap,
            entry_threshold=entry_threshold,
        )
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode}")
    bt = run_backtest(
        close=close.reindex(idx),
        position=pos,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    metrics = compute_backtest_metrics(bt=bt, bars_per_year=bars_per_year)
    return bt, metrics


def save_backtest_bundle(
    bt: pd.DataFrame,
    metrics: dict,
    bt_out_path: str,
    metrics_out_path: str,
) -> None:
    save_dataframe(bt.reset_index(), bt_out_path, index=False)
    save_metrics(metrics, metrics_out_path)
