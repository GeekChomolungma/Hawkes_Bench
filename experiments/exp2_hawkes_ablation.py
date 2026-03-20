from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import compute_backtest_metrics
from config import BacktestConfig, DataConfig, ExternalForecastConfig, HawkesConfig, OutputConfig, SignalConfig, WhiteBoxConfig
from data.loader import load_kline_csv, time_split_df
from data.preprocess import align_features, compute_log_return
from dataio.forecast_loader import ForecastLoadConfig, align_forecast_with_market, load_external_forecast
from experiments.runners import run_strategy_backtest
from hawkes.lambda_online import fit_hawkes_theta_from_train, hawkes_lambda_online
from models.whitebox.arima_garch_adapter import WhiteBoxForecaster
from utils.market_meta import parse_market_from_csv_path
from utils.persist import save_dataframe, save_metrics
from utils.visual import plot_backtest_layer


def _to_utc_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _build_exp2_split_indices(
    idx_all: pd.DatetimeIndex,
    df: pd.DataFrame,
    data_cfg: DataConfig,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, dict]:
    split = getattr(data_cfg, "split", None)
    if split is not None and split.train_end:
        train_end = _to_utc_ts(split.train_end)
        val_end = _to_utc_ts(split.val_end) if split.val_end else None

        idx_train = idx_all[idx_all <= train_end]
        if val_end is not None:
            idx_val = idx_all[(idx_all > train_end) & (idx_all <= val_end)]
            idx_test = idx_all[idx_all > val_end]
        else:
            idx_val = idx_all[:0]
            idx_test = idx_all[idx_all > train_end]

        info = {
            "mode": "explicit_dates",
            "train_end": str(train_end),
            "val_end": str(val_end) if val_end is not None else None,
            "train_rows": int(len(idx_train)),
            "val_rows": int(len(idx_val)),
            "test_rows": int(len(idx_test)),
        }
        return idx_train, idx_val, idx_test, info

    df_train, df_val, _ = time_split_df(df, ratios=(0.7, 0.1, 0.2))
    idx_train = idx_all[idx_all <= df_train.index[-1]]
    idx_val = idx_all[(idx_all > df_train.index[-1]) & (idx_all <= df_val.index[-1])]
    idx_test = idx_all[idx_all > df_val.index[-1]]

    info = {
        "mode": "ratio_70_10_20",
        "train_end": str(df_train.index[-1]),
        "val_end": str(df_val.index[-1]),
        "train_rows": int(len(idx_train)),
        "val_rows": int(len(idx_val)),
        "test_rows": int(len(idx_test)),
    }
    return idx_train, idx_val, idx_test, info


def _compact_bt_metrics(metrics: dict) -> dict[str, float]:
    return {
        "CumRet": float(metrics.get("total_return", float("nan"))),
        "Sharpe": float(metrics.get("sharpe", float("nan"))),
        "MaxDD": float(metrics.get("max_drawdown", float("nan"))),
        "Calmar": float(metrics.get("calmar", float("nan"))),
        "WinRate": float(metrics.get("hit_rate", float("nan"))),
    }


def _build_hawkes_lambda_for_quantile(
    returns: pd.Series,
    idx_fit: pd.DatetimeIndex,
    idx_all: pd.DatetimeIndex,
    quantile: float,
    signed_events: bool,
    unit: str,
) -> tuple[pd.Series, float]:
    r_fit = returns.reindex(idx_fit).dropna()
    tau, theta = fit_hawkes_theta_from_train(
        returns_train=r_fit,
        quantile=quantile,
        signed=signed_events,
        unit=unit,
    )
    lam = hawkes_lambda_online(
        returns=returns,
        index=idx_all,
        origin=idx_all[0],
        tau=tau,
        theta_by_key=theta,
        signed=signed_events,
        unit=unit,
    )
    return lam, float(tau)


def run_exp2_hawkes_ablation(
    data_cfg: DataConfig,
    wb_cfg: WhiteBoxConfig,
    hawkes_cfg: HawkesConfig,
    sig_cfg: SignalConfig,
    bt_cfg: BacktestConfig,
    out_cfg: OutputConfig,
    ext_cfg: ExternalForecastConfig | None = None,
    enable_whitebox: bool = False,
) -> dict:
    """
    Experiment 2 (test-focused):
    1) Fit Hawkes on train+val by default (or train only if online_update_enabled=True).
    2) Backtest on test split only.
    3) Cases:
       - no_hawkes: white / black
       - hawkes_qXX: white / black for each Qx
       - buy_and_hold

    Core artifacts (always):
    - case-level metrics json files
    - case-level backtest figures
    - exp2_summary_metrics_*.json

    Optional debug artifacts are controlled by OutputConfig.exp2_save_debug_tables.
    """
    Path(out_cfg.table_dir).mkdir(parents=True, exist_ok=True)
    Path(out_cfg.figure_dir).mkdir(parents=True, exist_ok=True)
    debug_tables = bool(getattr(out_cfg, "exp2_save_debug_tables", False))

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

    idx_all = close.index
    idx_train, idx_val, idx_test_base, split_info = _build_exp2_split_indices(idx_all=idx_all, df=df, data_cfg=data_cfg)

    white = None
    if enable_whitebox:
        white = WhiteBoxForecaster(cfg=wb_cfg).forecast_frame(close=close, returns=returns, symbol=data_cfg.symbol)
        white = white.set_index("ts").sort_index()

    black = None
    if ext_cfg is not None and ext_cfg.enabled:
        black_raw = load_external_forecast(
            ForecastLoadConfig(
                path=ext_cfg.path,
                column_map=ext_cfg.column_map or None,
                symbol=data_cfg.symbol,
                horizon=1,
            )
        )
        black = align_forecast_with_market(black_raw, close=close, symbol=data_cfg.symbol).set_index("ts").sort_index()

    if white is None and black is None:
        raise ValueError("Both white-box and black-box are disabled for Exp2. Nothing to backtest.")

    idx_test = idx_test_base
    if black is not None:
        idx_test = idx_test.intersection(black.index)

    if len(idx_test) == 0:
        raise ValueError("Exp2 test index is empty after alignment. Check split dates and external forecast coverage.")

    close_test = close.reindex(idx_test).dropna()
    white_test = white.reindex(idx_test).dropna() if white is not None else None
    black_test = black.reindex(idx_test).dropna() if black is not None else None

    unit = hawkes_cfg.time_unit if hawkes_cfg.time_unit in {"D", "s"} else "D"
    online_update_enabled = bool(getattr(hawkes_cfg, "online_update_enabled", False))
    # Current implementation does not retrain online; this flag currently controls fit window only.
    idx_hawkes_fit = idx_train if online_update_enabled else idx_train.union(idx_val)

    q_list = list(hawkes_cfg.quantile_grid) if getattr(hawkes_cfg, "quantile_grid", ()) else [hawkes_cfg.quantile]

    out: dict = {
        "split": split_info,
        "test_rows": int(len(idx_test)),
        "whitebox_enabled": bool(white is not None),
        "blackbox_enabled": bool(black is not None),
        "hawkes_fit_policy": {
            "online_update_enabled": online_update_enabled,
            "fit_window": "train" if online_update_enabled else "train+val",
            "fit_rows": int(len(idx_hawkes_fit)),
        },
        "no_hawkes": {},
        "hawkes": {},
        "buy_and_hold": {},
    }

    def _run_case(
        case_tag: str,
        branch_tag: str,
        forecast_df: pd.DataFrame,
        lam_case: pd.Series,
        use_hawkes: bool,
    ) -> dict:
        bt, m_full = run_strategy_backtest(
            forecast_df=forecast_df.reset_index(),
            close=close_test,
            lam=lam_case.reindex(idx_test).fillna(0.0),
            alpha_risk=hawkes_cfg.alpha_risk,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
            bars_per_year=bt_cfg.bars_per_year,
            position_cap=sig_cfg.position_cap,
            use_hawkes=use_hawkes,
            execution_mode=sig_cfg.execution_mode,
            entry_threshold=sig_cfg.entry_threshold,
        )

        m = _compact_bt_metrics(m_full)
        metrics_path = f"{out_cfg.table_dir}/exp2_{branch_tag}_{case_tag}_metrics_{mk}.json"
        save_metrics(m, metrics_path)

        if debug_tables:
            bt_path = f"{out_cfg.table_dir}/exp2_{branch_tag}_{case_tag}_bt_{mk}.csv"
            save_dataframe(bt.reset_index(), bt_path, index=False)

        fig_path = f"{out_cfg.figure_dir}/exp2_{branch_tag}_{case_tag}_{mk}.png"
        plot_backtest_layer(
            close=close_test,
            bt=bt,
            title=f"{mt} | {branch_tag} | {case_tag} | Test",
            out_path=fig_path,
        )
        return m

    # case 1: no Hawkes risk adjustment, i.e. lambda=0, for both white-box and black-box (if available)
    lam_zero = pd.Series(0.0, index=idx_test)
    if white_test is not None:
        out["no_hawkes"]["white"] = _run_case(
            case_tag="no_hawkes",
            branch_tag="white",
            forecast_df=white_test,
            lam_case=lam_zero,
            use_hawkes=False,
        )

    if black_test is not None:
        out["no_hawkes"]["black"] = _run_case(
            case_tag="no_hawkes",
            branch_tag="black",
            forecast_df=black_test,
            lam_case=lam_zero,
            use_hawkes=False,
        )

    # case 2: Hawkes risk adjustment with various quantile thresholds
    for q in q_list:
        lam_full, tau = _build_hawkes_lambda_for_quantile(
            returns=returns,
            idx_fit=idx_hawkes_fit,
            idx_all=idx_all,
            quantile=float(q),
            signed_events=hawkes_cfg.signed_events,
            unit=unit,
        )
        lam_test = lam_full.reindex(idx_test).fillna(0.0)

        q_tag = f"q{int(round(float(q) * 100)):02d}"
        out["hawkes"][q_tag] = {
            "event_threshold_quantile": float(q),
            "event_threshold_tau": float(tau),
        }

        if white_test is not None:
            out["hawkes"][q_tag]["white"] = _run_case(
                case_tag=f"hawkes_{q_tag}",
                branch_tag="white",
                forecast_df=white_test,
                lam_case=lam_test,
                use_hawkes=True,
            )

        if black_test is not None:
            out["hawkes"][q_tag]["black"] = _run_case(
                case_tag=f"hawkes_{q_tag}",
                branch_tag="black",
                forecast_df=black_test,
                lam_case=lam_test,
                use_hawkes=True,
            )

        if debug_tables:
            lam_path = f"{out_cfg.table_dir}/exp2_hawkes_lambda_{q_tag}_{mk}.csv"
            save_dataframe(lam_test.rename("lambda").reset_index(), lam_path, index=False)

    # case 3: buy and hold benchmark
    pos_bh = pd.Series(1.0, index=close_test.index)
    bt_bh = run_backtest(
        close=close_test,
        position=pos_bh,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
    )
    bh_metrics = _compact_bt_metrics(compute_backtest_metrics(bt=bt_bh, bars_per_year=bt_cfg.bars_per_year))
    save_metrics(bh_metrics, f"{out_cfg.table_dir}/exp2_buy_and_hold_metrics_{mk}.json")

    if debug_tables:
        save_dataframe(bt_bh.reset_index(), f"{out_cfg.table_dir}/exp2_buy_and_hold_bt_{mk}.csv", index=False)

    plot_backtest_layer(
        close=close_test,
        bt=bt_bh,
        title=f"{mt} | Buy & Hold | Test",
        out_path=f"{out_cfg.figure_dir}/exp2_buy_and_hold_{mk}.png",
    )
    out["buy_and_hold"] = bh_metrics

    save_metrics(out, f"{out_cfg.table_dir}/exp2_summary_metrics_{mk}.json")
    return out
