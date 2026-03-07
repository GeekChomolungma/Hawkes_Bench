from __future__ import annotations

from pathlib import Path

from config import (
    BacktestConfig,
    DataConfig,
    ExternalForecastConfig,
    HawkesConfig,
    OutputConfig,
    SignalConfig,
    WhiteBoxConfig,
)
from experiments.exp1_forecast_eval import run_exp1_forecast_eval
from experiments.exp2_hawkes_ablation import run_exp2_hawkes_ablation
from utils.interval_policy import apply_interval_policy
from utils.market_meta import parse_market_from_csv_path


def main() -> None:
    """
    Entry point for the thesis experiment pipeline.

    Runs:
    - Experiment 1: forecast-layer evaluation
    - Experiment 2: Hawkes ablation on trading layer
    """
    data_cfg = DataConfig(csv_path="market_info/BTCUSDT_1d_Binance.csv", symbol="BTCUSDT", interval="1d")
    wb_cfg = WhiteBoxConfig(arima_order=(1, 0, 1), garch_pq=(1, 1), rolling_window=30, z_score=1.96)
    hawkes_cfg = HawkesConfig(quantile=0.9, signed_events=True, alpha_risk=1.0, time_unit="auto")
    sig_cfg = SignalConfig(position_cap=1.0)
    bt_cfg = BacktestConfig(fee_bps=2.0, slippage_bps=1.0, bars_per_year=252)
    out_cfg = OutputConfig(table_dir="reports/tables", figure_dir="reports/figures")

    # Sync market meta from filename convention when available.
    meta = parse_market_from_csv_path(
        csv_path=data_cfg.csv_path,
        fallback_symbol=data_cfg.symbol,
        fallback_interval=data_cfg.interval,
    )
    data_cfg.symbol = meta["symbol"]
    data_cfg.interval = meta["interval"]

    profile = apply_interval_policy(
        interval=data_cfg.interval,
        wb_cfg=wb_cfg,
        bt_cfg=bt_cfg,
        hawkes_cfg=hawkes_cfg,
    )
    print(
        "[AUTO-CONFIG]",
        f"interval={data_cfg.interval}",
        f"rolling_window={wb_cfg.rolling_window}",
        f"bars_per_year={bt_cfg.bars_per_year}",
        f"hawkes_time_unit={hawkes_cfg.time_unit}",
    )

    # black-box is connected through external table protocol
    ext_cfg = ExternalForecastConfig(
        enabled=False,
        path="data/external_forecasts/blackbox_predictions_template.csv",
        column_map={},
    )

    Path(out_cfg.table_dir).mkdir(parents=True, exist_ok=True)
    Path(out_cfg.figure_dir).mkdir(parents=True, exist_ok=True)

    print("[RUN] Experiment 1: Forecast evaluation")
    exp1 = run_exp1_forecast_eval(data_cfg=data_cfg, wb_cfg=wb_cfg, out_cfg=out_cfg, ext_cfg=ext_cfg)
    print(exp1)

    print("[RUN] Experiment 2: Hawkes ablation backtest")
    exp2 = run_exp2_hawkes_ablation(
        data_cfg=data_cfg,
        wb_cfg=wb_cfg,
        hawkes_cfg=hawkes_cfg,
        sig_cfg=sig_cfg,
        bt_cfg=bt_cfg,
        out_cfg=out_cfg,
        ext_cfg=ext_cfg,
    )
    print(exp2)


if __name__ == "__main__":
    main()
