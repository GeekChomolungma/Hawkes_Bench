from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from config import (
    BacktestConfig,
    DataConfig,
    ExternalForecastConfig,
    HawkesConfig,
    OutputConfig,
    SignalConfig,
    SplitConfig,
    WhiteBoxConfig,
)
from experiments.exp1_forecast_eval import run_exp1_forecast_eval
from experiments.exp2_hawkes_ablation import run_exp2_hawkes_ablation
from utils.interval_policy import apply_interval_policy
from utils.market_meta import parse_market_from_csv_path


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_float_tuple(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return ()
    vals: list[float] = []
    for x in raw.split(","):
        s = x.strip()
        if not s:
            continue
        vals.append(float(s))
    return tuple(vals)


def resolve_market_csv(market_dir: str, symbol: str, interval: str) -> str:
    p = Path(market_dir) / f"{symbol}_{interval}_Binance.csv"
    if not p.exists():
        raise FileNotFoundError(f"Market csv not found: {p}")
    return str(p)


def resolve_external_csv(external_dir: str, symbol: str, interval: str) -> str:
    p = Path(external_dir) / f"zeroshot_{symbol}_{interval}_logreturn_predictions_decision_aligned.csv"
    return str(p)


def build_configs_for_market(
    market_csv: str,
    symbol: str,
    interval: str,
    train_end: str,
    val_end: str,
    enable_blackbox: bool,
    external_csv: str | None,
    hawkes_quantiles: tuple[float, ...],
    hawkes_online_update_enabled: bool,
    exp1_debug_tables: bool,
    exp2_debug_tables: bool,
) -> tuple[DataConfig, WhiteBoxConfig, HawkesConfig, SignalConfig, BacktestConfig, OutputConfig, ExternalForecastConfig]:
    data_cfg = DataConfig(
        csv_path=market_csv,
        symbol=symbol,
        interval=interval,
        split=SplitConfig(train_end=train_end, val_end=val_end),
    )
    wb_cfg = WhiteBoxConfig(arima_order=(1, 0, 1), garch_pq=(1, 1), rolling_window=30, z_score=1.96)
    hawkes_cfg = HawkesConfig(
        quantile=0.9,
        quantile_grid=hawkes_quantiles,
        online_update_enabled=hawkes_online_update_enabled,
        signed_events=True,
        alpha_risk=1.0,
        time_unit="auto",
    )
    sig_cfg = SignalConfig(position_cap=1.0)
    bt_cfg = BacktestConfig(fee_bps=2.0, slippage_bps=1.0, bars_per_year=252)
    out_cfg = OutputConfig(
        table_dir="reports/tables",
        figure_dir="reports/figures",
        exp1_save_debug_tables=exp1_debug_tables,
        exp2_save_debug_tables=exp2_debug_tables,
    )

    meta = parse_market_from_csv_path(
        csv_path=data_cfg.csv_path,
        fallback_symbol=data_cfg.symbol,
        fallback_interval=data_cfg.interval,
    )
    data_cfg.symbol = meta["symbol"]
    data_cfg.interval = meta["interval"]

    apply_interval_policy(
        interval=data_cfg.interval,
        wb_cfg=wb_cfg,
        bt_cfg=bt_cfg,
        hawkes_cfg=hawkes_cfg,
    )

    ext_cfg = ExternalForecastConfig(enabled=False, path="", column_map={})
    if enable_blackbox and external_csv:
        ext_path = Path(external_csv)
        if ext_path.exists():
            ext_cfg = ExternalForecastConfig(enabled=True, path=str(ext_path), column_map={})
        else:
            print(f"[WARN] external forecast not found, black-box disabled: {ext_path}")

    return data_cfg, wb_cfg, hawkes_cfg, sig_cfg, bt_cfg, out_cfg, ext_cfg


def run_pipeline_for_market(
    mode: str,
    market_csv: str,
    symbol: str,
    interval: str,
    train_end: str,
    val_end: str,
    enable_blackbox: bool,
    external_csv: str | None,
    hawkes_quantiles: tuple[float, ...],
    hawkes_online_update_enabled: bool,
    exp1_debug_tables: bool,
    exp2_debug_tables: bool,
) -> dict:
    data_cfg, wb_cfg, hawkes_cfg, sig_cfg, bt_cfg, out_cfg, ext_cfg = build_configs_for_market(
        market_csv=market_csv,
        symbol=symbol,
        interval=interval,
        train_end=train_end,
        val_end=val_end,
        enable_blackbox=enable_blackbox,
        external_csv=external_csv,
        hawkes_quantiles=hawkes_quantiles,
        hawkes_online_update_enabled=hawkes_online_update_enabled,
        exp1_debug_tables=exp1_debug_tables,
        exp2_debug_tables=exp2_debug_tables,
    )

    print(
        "[AUTO-CONFIG]",
        f"symbol={data_cfg.symbol}",
        f"interval={data_cfg.interval}",
        f"rolling_window={wb_cfg.rolling_window}",
        f"bars_per_year={bt_cfg.bars_per_year}",
        f"hawkes_time_unit={hawkes_cfg.time_unit}",
    )

    out: dict = {
        "meta": {
            "mode": mode,
            "symbol": data_cfg.symbol,
            "interval": data_cfg.interval,
            "market_csv": data_cfg.csv_path,
            "external_csv": ext_cfg.path if ext_cfg.enabled else None,
            "blackbox_enabled": ext_cfg.enabled,
            "split": asdict(data_cfg.split),
        }
    }

    if mode in {"full", "exp1"}:
        print(f"[RUN][{data_cfg.symbol}] Experiment 1")
        out["exp1"] = run_exp1_forecast_eval(
            data_cfg=data_cfg,
            wb_cfg=wb_cfg,
            out_cfg=out_cfg,
            ext_cfg=ext_cfg,
        )

    if mode in {"full", "exp2"}:
        print(f"[RUN][{data_cfg.symbol}] Experiment 2")
        out["exp2"] = run_exp2_hawkes_ablation(
            data_cfg=data_cfg,
            wb_cfg=wb_cfg,
            hawkes_cfg=hawkes_cfg,
            sig_cfg=sig_cfg,
            bt_cfg=bt_cfg,
            out_cfg=out_cfg,
            ext_cfg=ext_cfg,
        )

    return out


def run_pipeline_batch(
    mode: str,
    symbols: Iterable[str],
    interval: str,
    market_dir: str,
    train_end: str,
    val_end: str,
    enable_blackbox: bool,
    external_dir: str,
    hawkes_quantiles: tuple[float, ...],
    hawkes_online_update_enabled: bool,
    exp1_debug_tables: bool,
    exp2_debug_tables: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for symbol in symbols:
        market_csv = resolve_market_csv(market_dir=market_dir, symbol=symbol, interval=interval)
        external_csv = resolve_external_csv(external_dir=external_dir, symbol=symbol, interval=interval)
        print(f"\n[PIPELINE] {symbol} {interval}")
        results[symbol] = run_pipeline_for_market(
            mode=mode,
            market_csv=market_csv,
            symbol=symbol,
            interval=interval,
            train_end=train_end,
            val_end=val_end,
            enable_blackbox=enable_blackbox,
            external_csv=external_csv,
            hawkes_quantiles=hawkes_quantiles,
            hawkes_online_update_enabled=hawkes_online_update_enabled,
            exp1_debug_tables=exp1_debug_tables,
            exp2_debug_tables=exp2_debug_tables,
        )
    return results


def parse_cli_list(raw: str | None) -> list[str]:
    return _parse_csv_list(raw)


def parse_cli_quantiles(raw: str | None) -> tuple[float, ...]:
    return _parse_float_tuple(raw)
