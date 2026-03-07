from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import BacktestConfig, DataConfig, ExternalForecastConfig, HawkesConfig, OutputConfig, SignalConfig, WhiteBoxConfig
from data.loader import load_kline_csv, time_split_df
from data.preprocess import align_features, compute_log_return
from dataio.forecast_loader import ForecastLoadConfig, align_forecast_with_market, load_external_forecast
from experiments.runners import run_strategy_backtest, save_backtest_bundle
from hawkes.lambda_online import fit_hawkes_theta_from_train, hawkes_lambda_online
from models.whitebox.arima_garch_adapter import WhiteBoxForecaster
from utils.market_meta import parse_market_from_csv_path
from utils.persist import save_metrics
from utils.visual import plot_backtest_layer, plot_hawkes_lambda_splits


def _build_hawkes_lambda(
    returns: pd.Series,
    idx_train: pd.DatetimeIndex,
    idx_all: pd.DatetimeIndex,
    hawkes_cfg: HawkesConfig,
) -> pd.Series:
    r_train = returns.reindex(idx_train).dropna()
    unit = hawkes_cfg.time_unit if hawkes_cfg.time_unit in {"D", "s"} else "D"
    tau, theta = fit_hawkes_theta_from_train(
        returns_train=r_train,
        quantile=hawkes_cfg.quantile,
        signed=hawkes_cfg.signed_events,
        unit=unit,
    )
    origin = idx_all[0]
    lam = hawkes_lambda_online(
        returns=returns,
        index=idx_all,
        origin=origin,
        tau=tau,
        theta_by_key=theta,
        signed=hawkes_cfg.signed_events,
        unit=unit,
    )
    return lam


def run_exp2_hawkes_ablation(
    data_cfg: DataConfig,
    wb_cfg: WhiteBoxConfig,
    hawkes_cfg: HawkesConfig,
    sig_cfg: SignalConfig,
    bt_cfg: BacktestConfig,
    out_cfg: OutputConfig,
    ext_cfg: ExternalForecastConfig | None = None,
) -> dict:
    """
    Experiment 2: trading-layer Hawkes ablation.

    Workflow:
    1) Build white-box forecast frame on full timeline.
    2) Fit Hawkes parameters on train split only, then run online lambda on all splits.
    3) Backtest native-risk vs Hawkes-enhanced risk for white-box branch.
    4) Optionally repeat native vs Hawkes comparison for external black-box branch.

    Returns a dict of metric summaries and writes detailed CSV/JSON/figures to reports/.
    """
    Path(out_cfg.table_dir).mkdir(parents=True, exist_ok=True)
    Path(out_cfg.figure_dir).mkdir(parents=True, exist_ok=True)

    df = load_kline_csv(data_cfg.csv_path)
    df = align_features(df)
    close = df["close"].astype(float)
    returns = compute_log_return(close)
    meta = parse_market_from_csv_path(
        csv_path=data_cfg.csv_path,
        fallback_symbol=data_cfg.symbol,
        fallback_interval=data_cfg.interval,
    )
    mk = meta["key"]
    mt = meta["title_label"]

    df_train, df_val, df_test = time_split_df(df, ratios=(0.7, 0.1, 0.2))

    white = WhiteBoxForecaster(cfg=wb_cfg).forecast_frame(close=close, returns=returns, symbol=data_cfg.symbol)
    white = white.set_index("ts").sort_index()

    idx_all = white.index
    idx_train = idx_all[idx_all <= df_train.index[-1]]
    idx_val = idx_all[(idx_all > df_train.index[-1]) & (idx_all <= df_val.index[-1])]
    idx_test = idx_all[idx_all > df_val.index[-1]]

    lam = _build_hawkes_lambda(returns=returns, idx_train=idx_train, idx_all=idx_all, hawkes_cfg=hawkes_cfg)

    plot_hawkes_lambda_splits(
        close=close,
        lam=lam,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        title=f"{mt} | Hawkes Lambda Across Splits",
    )

    results = {}

    # whitebox native
    bt_w_native, m_w_native = run_strategy_backtest(
        forecast_df=white.reset_index(),
        close=close,
        lam=lam,
        alpha_risk=hawkes_cfg.alpha_risk,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
        bars_per_year=bt_cfg.bars_per_year,
        position_cap=sig_cfg.position_cap,
        use_hawkes=False,
        execution_mode=sig_cfg.execution_mode,
        entry_threshold=sig_cfg.entry_threshold,
    )
    save_backtest_bundle(
        bt=bt_w_native,
        metrics=m_w_native,
        bt_out_path=f"{out_cfg.table_dir}/exp2_white_native_bt_{mk}.csv",
        metrics_out_path=f"{out_cfg.table_dir}/exp2_white_native_metrics_{mk}.json",
    )
    plot_backtest_layer(
        close=close,
        bt=bt_w_native,
        title=f"{mt} | White-box Native Risk",
        out_path=f"{out_cfg.figure_dir}/exp2_white_native_{mk}.png",
    )
    results["white_native"] = m_w_native

    # whitebox hawkes-enhanced
    bt_w_hawkes, m_w_hawkes = run_strategy_backtest(
        forecast_df=white.reset_index(),
        close=close,
        lam=lam,
        alpha_risk=hawkes_cfg.alpha_risk,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
        bars_per_year=bt_cfg.bars_per_year,
        position_cap=sig_cfg.position_cap,
        use_hawkes=True,
        execution_mode=sig_cfg.execution_mode,
        entry_threshold=sig_cfg.entry_threshold,
    )
    save_backtest_bundle(
        bt=bt_w_hawkes,
        metrics=m_w_hawkes,
        bt_out_path=f"{out_cfg.table_dir}/exp2_white_hawkes_bt_{mk}.csv",
        metrics_out_path=f"{out_cfg.table_dir}/exp2_white_hawkes_metrics_{mk}.json",
    )
    plot_backtest_layer(
        close=close,
        bt=bt_w_hawkes,
        title=f"{mt} | White-box Hawkes Enhanced",
        out_path=f"{out_cfg.figure_dir}/exp2_white_hawkes_{mk}.png",
    )
    results["white_hawkes"] = m_w_hawkes

    if ext_cfg is not None and ext_cfg.enabled:
        black_raw = load_external_forecast(
            ForecastLoadConfig(path=ext_cfg.path, column_map=ext_cfg.column_map or None, symbol=data_cfg.symbol, horizon=1)
        )
        black = align_forecast_with_market(black_raw, close=close, symbol=data_cfg.symbol).set_index("ts").sort_index()

        bt_b_native, m_b_native = run_strategy_backtest(
            forecast_df=black.reset_index(),
            close=close,
            lam=lam.reindex(black.index).fillna(0.0),
            alpha_risk=hawkes_cfg.alpha_risk,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
            bars_per_year=bt_cfg.bars_per_year,
            position_cap=sig_cfg.position_cap,
            use_hawkes=False,
            execution_mode=sig_cfg.execution_mode,
            entry_threshold=sig_cfg.entry_threshold,
        )
        save_backtest_bundle(
            bt=bt_b_native,
            metrics=m_b_native,
            bt_out_path=f"{out_cfg.table_dir}/exp2_black_native_bt_{mk}.csv",
            metrics_out_path=f"{out_cfg.table_dir}/exp2_black_native_metrics_{mk}.json",
        )
        plot_backtest_layer(
            close=close,
            bt=bt_b_native,
            title=f"{mt} | Black-box Native Risk",
            out_path=f"{out_cfg.figure_dir}/exp2_black_native_{mk}.png",
        )
        results["black_native"] = m_b_native

        bt_b_hawkes, m_b_hawkes = run_strategy_backtest(
            forecast_df=black.reset_index(),
            close=close,
            lam=lam.reindex(black.index).fillna(0.0),
            alpha_risk=hawkes_cfg.alpha_risk,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
            bars_per_year=bt_cfg.bars_per_year,
            position_cap=sig_cfg.position_cap,
            use_hawkes=True,
            execution_mode=sig_cfg.execution_mode,
            entry_threshold=sig_cfg.entry_threshold,
        )
        save_backtest_bundle(
            bt=bt_b_hawkes,
            metrics=m_b_hawkes,
            bt_out_path=f"{out_cfg.table_dir}/exp2_black_hawkes_bt_{mk}.csv",
            metrics_out_path=f"{out_cfg.table_dir}/exp2_black_hawkes_metrics_{mk}.json",
        )
        plot_backtest_layer(
            close=close,
            bt=bt_b_hawkes,
            title=f"{mt} | Black-box Hawkes Enhanced",
            out_path=f"{out_cfg.figure_dir}/exp2_black_hawkes_{mk}.png",
        )
        results["black_hawkes"] = m_b_hawkes

    save_metrics(results, f"{out_cfg.table_dir}/exp2_summary_metrics_{mk}.json")
    return results
