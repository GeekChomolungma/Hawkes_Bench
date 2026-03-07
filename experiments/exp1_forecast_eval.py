from __future__ import annotations

from pathlib import Path

from config import DataConfig, ExternalForecastConfig, OutputConfig, WhiteBoxConfig
from data.loader import load_kline_csv, time_split_df
from data.preprocess import align_features, compute_log_return
from dataio.forecast_loader import ForecastLoadConfig, align_forecast_with_market, load_external_forecast
from experiments.runners import evaluate_forecast_frame
from models.whitebox.arima_garch_adapter import WhiteBoxForecaster
from utils.market_meta import parse_market_from_csv_path
from utils.persist import save_dataframe, save_metrics
from utils.visual import plot_forecast_layer, plot_return_target_layer


def run_exp1_forecast_eval(
    data_cfg: DataConfig,
    wb_cfg: WhiteBoxConfig,
    out_cfg: OutputConfig,
    ext_cfg: ExternalForecastConfig | None = None,
) -> dict:
    """
    Experiment 1: forecast-layer evaluation.

    Workflow:
    1) Load and preprocess market data.
    2) Generate white-box forecasts (ARIMA+GARCH adapter).
    3) Evaluate metrics by split (train/val/test), with test as primary report.
    4) Evaluate a naive baseline on test for sanity check.
    5) Optionally load external black-box forecasts and evaluate on test.

    Outputs are persisted under reports/tables and reports/figures.
    """
    Path(out_cfg.table_dir).mkdir(parents=True, exist_ok=True)
    Path(out_cfg.figure_dir).mkdir(parents=True, exist_ok=True)

    df = load_kline_csv(data_cfg.csv_path)
    df = align_features(df)
    close = df["close"].astype(float)
    returns = compute_log_return(close)
    df_train, df_val, df_test = time_split_df(df, ratios=(0.7, 0.1, 0.2))
    meta = parse_market_from_csv_path(
        csv_path=data_cfg.csv_path,
        fallback_symbol=data_cfg.symbol,
        fallback_interval=data_cfg.interval,
    )
    mk = meta["key"]
    mt = meta["title_label"]

    white = WhiteBoxForecaster(cfg=wb_cfg).forecast_frame(close=close, returns=returns, symbol=data_cfg.symbol)
    white_idx = white.set_index("ts").index
    idx_train = white_idx[white_idx <= df_train.index[-1]]
    idx_val = white_idx[(white_idx > df_train.index[-1]) & (white_idx <= df_val.index[-1])]
    idx_test = white_idx[white_idx > df_val.index[-1]]

    white_metrics_all = evaluate_forecast_frame(
        forecast_df=white,
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_all_{mk}.json",
        rows_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_rows_all_{mk}.csv",
    )
    white_metrics_train = evaluate_forecast_frame(
        forecast_df=white[white["ts"].isin(idx_train)],
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_train_{mk}.json",
    )
    white_metrics_val = evaluate_forecast_frame(
        forecast_df=white[white["ts"].isin(idx_val)],
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_val_{mk}.json",
    )
    white_metrics_test = evaluate_forecast_frame(
        forecast_df=white[white["ts"].isin(idx_test)],
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_test_{mk}.json",
        rows_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_rows_test_{mk}.csv",
    )
    save_dataframe(white, f"{out_cfg.table_dir}/exp1_whitebox_forecast_frame_{mk}.csv", index=False)

    # Visualize forecast and return target layers for test split
    plot_forecast_layer(
        close=close,
        forecast_df=white,
        title=f"{mt} | White-box Forecast (ARIMA+GARCH)",
        out_path=f"{out_cfg.figure_dir}/exp1_whitebox_forecast_{mk}.png",
    )
    plot_return_target_layer(
        returns=returns,
        forecast_df=white[white["ts"].isin(idx_test)],
        title=f"{mt} | White-box Return Target (Test)",
        out_path=f"{out_cfg.figure_dir}/exp1_whitebox_return_target_test_{mk}.png",
    )

    # Naive baseline for leakage sanity check: predict r_{t+1} with r_t
    naive_df = white[["ts", "symbol", "horizon", "close_t"]].copy()
    naive_df["mu_pred"] = returns.reindex(white["ts"]).to_numpy()
    naive_test = evaluate_forecast_frame(
        forecast_df=naive_df[naive_df["ts"].isin(idx_test)],
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_naive_forecast_metrics_test_{mk}.json",
    )

    out = {
        "whitebox": {
            "all": white_metrics_all,
            "train": white_metrics_train,
            "val": white_metrics_val,
            "test": white_metrics_test,
        },
        "naive_baseline_test": naive_test,
    }
    save_metrics(out, f"{out_cfg.table_dir}/exp1_summary_metrics_{mk}.json")

    if ext_cfg is not None and ext_cfg.enabled:
        black_raw = load_external_forecast(
            ForecastLoadConfig(
                path=ext_cfg.path,
                column_map=ext_cfg.column_map or None,
                symbol=data_cfg.symbol,
                horizon=1,
            )
        )
        black = align_forecast_with_market(black_raw, close=close, symbol=data_cfg.symbol)
        black_idx = black.set_index("ts").index
        black_test = black[black["ts"].isin(black_idx[black_idx > df_val.index[-1]])]
        black_metrics = evaluate_forecast_frame(
            forecast_df=black_test,
            returns=returns,
            metrics_out_path=f"{out_cfg.table_dir}/exp1_blackbox_forecast_metrics_test_{mk}.json",
            rows_out_path=f"{out_cfg.table_dir}/exp1_blackbox_forecast_rows_test_{mk}.csv",
        )
        save_dataframe(black, f"{out_cfg.table_dir}/exp1_blackbox_forecast_frame_{mk}.csv", index=False)
        plot_forecast_layer(
            close=close,
            forecast_df=black,
            title=f"{mt} | Black-box Forecast (External Input)",
            out_path=f"{out_cfg.figure_dir}/exp1_blackbox_forecast_{mk}.png",
        )
        plot_return_target_layer(
            returns=returns,
            forecast_df=black_test,
            title=f"{mt} | Black-box Return Target (Test)",
            out_path=f"{out_cfg.figure_dir}/exp1_blackbox_return_target_test_{mk}.png",
        )
        out["blackbox_test"] = black_metrics
        save_metrics(out, f"{out_cfg.table_dir}/exp1_summary_metrics_{mk}.json")

    return out
