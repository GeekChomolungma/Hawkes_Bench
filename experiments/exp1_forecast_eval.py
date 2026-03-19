from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DataConfig, ExternalForecastConfig, OutputConfig, WhiteBoxConfig
from data.loader import load_kline_csv, time_split_df
from data.preprocess import align_features, compute_log_return
from dataio.forecast_loader import ForecastLoadConfig, align_forecast_with_market, load_external_forecast
from experiments.runners import evaluate_forecast_frame
from models.whitebox.arima_garch_adapter import WhiteBoxForecaster
from utils.market_meta import parse_market_from_csv_path
from utils.persist import save_dataframe, save_metrics
from utils.visual import plot_forecast_layer, plot_return_target_layer


def _to_utc_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _build_exp1_split_indices(
    white_idx: pd.DatetimeIndex,
    df: pd.DataFrame,
    data_cfg: DataConfig,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, dict]:
    split = getattr(data_cfg, "split", None)
    if split is not None and split.train_end:
        train_end = _to_utc_ts(split.train_end)
        val_end = _to_utc_ts(split.val_end) if split.val_end else None

        idx_train = white_idx[white_idx <= train_end]
        if val_end is not None:
            idx_val = white_idx[(white_idx > train_end) & (white_idx <= val_end)]
            idx_test = white_idx[white_idx > val_end]
        else:
            idx_val = white_idx[:0]
            idx_test = white_idx[white_idx > train_end]

        info = {
            "mode": "explicit_dates",
            "train_end": str(train_end),
            "val_end": str(val_end) if val_end is not None else None,
            "train_rows": int(len(idx_train)),
            "val_rows": int(len(idx_val)),
            "test_rows": int(len(idx_test)),
        }
        return idx_train, idx_val, idx_test, info

    # fallback: chronological ratio split
    df_train, df_val, _ = time_split_df(df, ratios=(0.7, 0.1, 0.2))
    idx_train = white_idx[white_idx <= df_train.index[-1]]
    idx_val = white_idx[(white_idx > df_train.index[-1]) & (white_idx <= df_val.index[-1])]
    idx_test = white_idx[white_idx > df_val.index[-1]]
    info = {
        "mode": "ratio_70_10_20",
        "train_end": str(df_train.index[-1]),
        "val_end": str(df_val.index[-1]),
        "train_rows": int(len(idx_train)),
        "val_rows": int(len(idx_val)),
        "test_rows": int(len(idx_test)),
    }
    return idx_train, idx_val, idx_test, info


def run_exp1_forecast_eval(
    data_cfg: DataConfig,
    wb_cfg: WhiteBoxConfig,
    out_cfg: OutputConfig,
    ext_cfg: ExternalForecastConfig | None = None,
) -> dict:
    """
    Experiment 1: forecast-layer evaluation.

    Core artifacts (always):
    - exp1_summary_metrics_*.json
    - exp1_whitebox_forecast_metrics_test_*.json
    - exp1_naive_forecast_metrics_test_*.json
    - exp1_blackbox_forecast_metrics_test_*.json (if enabled)

    Optional debug artifacts are controlled by OutputConfig.exp1_save_debug_tables.
    """
    Path(out_cfg.table_dir).mkdir(parents=True, exist_ok=True)
    Path(out_cfg.figure_dir).mkdir(parents=True, exist_ok=True)
    debug_tables = bool(getattr(out_cfg, "exp1_save_debug_tables", False))

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

    white = WhiteBoxForecaster(cfg=wb_cfg).forecast_frame(close=close, returns=returns, symbol=data_cfg.symbol)
    white_idx = white.set_index("ts").index
    idx_train, idx_val, idx_test, split_info = _build_exp1_split_indices(white_idx=white_idx, df=df, data_cfg=data_cfg)

    white_metrics_all = evaluate_forecast_frame(
        forecast_df=white,
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_all_{mk}.json" if debug_tables else None,
        rows_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_rows_all_{mk}.csv" if debug_tables else None,
    )
    white_metrics_train = evaluate_forecast_frame(
        forecast_df=white[white["ts"].isin(idx_train)],
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_train_{mk}.json" if debug_tables else None,
    )

    if len(idx_val) > 0:
        white_metrics_val = evaluate_forecast_frame(
            forecast_df=white[white["ts"].isin(idx_val)],
            returns=returns,
            metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_val_{mk}.json" if debug_tables else None,
        )
    else:
        white_metrics_val = {}

    white_test = white[white["ts"].isin(idx_test)].copy()
    white_metrics_test = evaluate_forecast_frame(
        forecast_df=white_test,
        returns=returns,
        metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_test_{mk}.json",
        rows_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_rows_test_{mk}.csv" if debug_tables else None,
    )
    if debug_tables:
        save_dataframe(white, f"{out_cfg.table_dir}/exp1_whitebox_forecast_frame_{mk}.csv", index=False)

    # Visualize forecast and return target layers for test split only
    close_test = close[close.index >= idx_test.min()] if len(idx_test) > 0 else close.iloc[0:0]

    plot_forecast_layer(
        close=close_test,
        forecast_df=white_test,
        title=f"{mt} | White-box Forecast (Test)",
        out_path=f"{out_cfg.figure_dir}/exp1_whitebox_forecast_{mk}.png",
    )
    plot_return_target_layer(
        returns=returns,
        forecast_df=white_test,
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
        "split": split_info,
        "whitebox": {
            "all": white_metrics_all,
            "train": white_metrics_train,
            "val": white_metrics_val,
            "test": white_metrics_test,
        },
        "naive_baseline_test": naive_test,
    }
    save_metrics(out, f"{out_cfg.table_dir}/exp1_summary_metrics_{mk}.json")

    # Black box evaluation
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

        # Fair comparison: evaluate white and black on the same test decision timestamps.
        idx_test_aligned = idx_test.intersection(black_idx)
        black_test = black[black["ts"].isin(idx_test_aligned)].copy()
        white_test_aligned = white[white["ts"].isin(idx_test_aligned)]

        black_metrics = evaluate_forecast_frame(
            forecast_df=black_test,
            returns=returns,
            metrics_out_path=f"{out_cfg.table_dir}/exp1_blackbox_forecast_metrics_test_{mk}.json",
            rows_out_path=f"{out_cfg.table_dir}/exp1_blackbox_forecast_rows_test_{mk}.csv" if debug_tables else None,
        )
        white_metrics_test_aligned = evaluate_forecast_frame(
            forecast_df=white_test_aligned,
            returns=returns,
            metrics_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_metrics_test_aligned_blackbox_{mk}.json" if debug_tables else None,
            rows_out_path=f"{out_cfg.table_dir}/exp1_whitebox_forecast_rows_test_aligned_blackbox_{mk}.csv" if debug_tables else None,
        )
        naive_test_aligned = evaluate_forecast_frame(
            forecast_df=naive_df[naive_df["ts"].isin(idx_test_aligned)],
            returns=returns,
            metrics_out_path=f"{out_cfg.table_dir}/exp1_naive_forecast_metrics_test_aligned_blackbox_{mk}.json" if debug_tables else None,
        )

        if debug_tables:
            save_dataframe(black, f"{out_cfg.table_dir}/exp1_blackbox_forecast_frame_{mk}.csv", index=False)

        close_test_aligned = close[close.index >= idx_test_aligned.min()] if len(idx_test_aligned) > 0 else close.iloc[0:0]

        plot_forecast_layer(
            close=close_test_aligned,
            forecast_df=black_test,
            title=f"{mt} | Black-box Forecast (Test)",
            out_path=f"{out_cfg.figure_dir}/exp1_blackbox_forecast_{mk}.png",
            band_low_col="q10",
            band_high_col="q90",
            band_label="Pred Band (q10-q90, t+1)",
        )
        plot_return_target_layer(
            returns=returns,
            forecast_df=black_test,
            title=f"{mt} | Black-box Return Target (Test)",
            out_path=f"{out_cfg.figure_dir}/exp1_blackbox_return_target_test_{mk}.png",
            band_low_col="q10",
            band_high_col="q90",
            band_label="Pred Band (q10-q90)",
        )

        out["blackbox_test"] = black_metrics
        out["aligned_test_window"] = {
            "rows": int(len(idx_test_aligned)),
            "whitebox": white_metrics_test_aligned,
            "blackbox": black_metrics,
            "naive": naive_test_aligned,
        }
        save_metrics(out, f"{out_cfg.table_dir}/exp1_summary_metrics_{mk}.json")

    return out
