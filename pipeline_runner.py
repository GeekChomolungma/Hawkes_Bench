from __future__ import annotations

import re
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

_NEW_EXTERNAL_RE = re.compile(
    r"^predictions_decision_aligned__target_(?P<symbol>[a-z0-9]+)__init_(?P<init_mode>[a-z0-9]+)__loss_(?P<loss_mode>[a-z0-9]+)__tag_(?P<tag>.+)\.csv$",
    flags=re.IGNORECASE,
)


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
    p = Path(market_dir) / f"{symbol}_{interval}_Binance_cleaned.csv"
    if not p.exists():
        raise FileNotFoundError(f"Market csv not found: {p}")
    return str(p)


def _sanitize_tag(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return out or "model"


def resolve_external_csv_candidates(
    external_dir: str,
    symbol: str,
    interval: str,
    family: str,
    run_id: str,
) -> list[tuple[str, str]]:
    """
    Discover external forecast files for one (symbol, interval) pair.

    Expected on-disk layout:
        {external_dir}/{family}/{interval}/{run_id}/predictions_decision_aligned__target_<symbol>__init_<init_mode>__loss_<loss_mode>__tag_<tag>.csv

    Where:
    - family: model family folder (e.g. "ft", "hf")
    - interval: time interval folder (e.g. "1d", "4h")
    - run_id: arbitrary run folder name, usually a number/string (e.g. "1", "2", "expA")
      This function treats run_id as opaque text and does not enforce numeric parsing.

    Simple example:
        data/external_forecasts/ft/1d/1/predictions_decision_aligned__target_bchusdt__init_pretrained__loss_native__tag_batch_bchusdt_to_bchusdt.csv

    Returned value:
    - List of tuples: (output_subdir_key, csv_path)
    - output_subdir_key is mirrored under reports/tables and reports/figures.
      Example key: "ft/1d/1"
    - If multiple files for the same (family/interval/run_id/symbol) exist, a model suffix
      is appended to avoid output overwrite:
      "ft/1d/1/init_pretrained__loss_native__tag_xxx"
    """
    root = Path(external_dir)
    sym_u = symbol.upper()
    itv_l = interval.lower()
    fam = family.strip()
    rid = run_id.strip()

    # Collected entries:
    # (base_subdir, model_slug, path)
    raw_entries: list[tuple[str, str, str]] = []

    # Explicit business rule:
    #   {external_dir}/{family}/{interval}/{run_id}/predictions_decision_aligned__target_...csv
    run_dir = root / fam / interval / rid
    if not run_dir.exists():
        return []
    if not run_dir.is_dir():
        return []

    for p in sorted(run_dir.glob("predictions_decision_aligned__target_*__init_*__loss_*__tag_*.csv")):
        m = _NEW_EXTERNAL_RE.match(p.name)
        if not m:
            continue
        if m.group("symbol").upper() != sym_u:
            continue

        init_mode = _sanitize_tag(m.group("init_mode").lower())
        loss_mode = _sanitize_tag(m.group("loss_mode").lower())
        tag = _sanitize_tag(m.group("tag").lower())
        model_slug = f"init_{init_mode}__loss_{loss_mode}__tag_{tag}"

        # Mirror explicit external folder hierarchy under reports.
        base_subdir = f"{fam}/{interval}/{rid}"
        raw_entries.append((base_subdir, model_slug, str(p)))

    # Build stable output keys:
    # - If one model under same base_subdir, use base_subdir directly.
    # - If multiple models under same base_subdir, append model_slug to avoid overwrite.
    counts: dict[str, int] = {}
    for base_subdir, _, _ in raw_entries:
        counts[base_subdir] = counts.get(base_subdir, 0) + 1

    out: list[tuple[str, str]] = []
    for base_subdir, model_slug, path in raw_entries:
        key = base_subdir if counts[base_subdir] == 1 else f"{base_subdir}/{model_slug}"
        out.append((key, path))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


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
    output_subdir: str | None = None,
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
    table_dir = Path("reports/tables")
    figure_dir = Path("reports/figures")
    if output_subdir:
        table_dir = table_dir / output_subdir
        figure_dir = figure_dir / output_subdir

    out_cfg = OutputConfig(
        table_dir=str(table_dir),
        figure_dir=str(figure_dir),
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
    output_subdir: str | None = None,
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
        output_subdir=output_subdir,
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
            "output_subdir": output_subdir,
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
    external_family: str,
    external_run_id: str,
    hawkes_quantiles: tuple[float, ...],
    hawkes_online_update_enabled: bool,
    exp1_debug_tables: bool,
    exp2_debug_tables: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for symbol in symbols:
        market_csv = resolve_market_csv(market_dir=market_dir, symbol=symbol, interval=interval)
        print(f"\n[PIPELINE] {symbol} {interval}")
        results[symbol] = {}
        candidates = (
            resolve_external_csv_candidates(
                external_dir=external_dir,
                symbol=symbol,
                interval=interval,
                family=external_family,
                run_id=external_run_id,
            )
            if enable_blackbox
            else []
        )

        if enable_blackbox and not candidates:
            print(f"[WARN] no external forecast candidates found for {symbol} {interval}, run white-box only.")
            results[symbol]["whitebox_only"] = run_pipeline_for_market(
                mode=mode,
                market_csv=market_csv,
                symbol=symbol,
                interval=interval,
                train_end=train_end,
                val_end=val_end,
                enable_blackbox=False,
                external_csv=None,
                hawkes_quantiles=hawkes_quantiles,
                hawkes_online_update_enabled=hawkes_online_update_enabled,
                exp1_debug_tables=exp1_debug_tables,
                exp2_debug_tables=exp2_debug_tables,
                output_subdir="whitebox_only",
            )
            continue

        if not enable_blackbox:
            results[symbol]["whitebox_only"] = run_pipeline_for_market(
                mode=mode,
                market_csv=market_csv,
                symbol=symbol,
                interval=interval,
                train_end=train_end,
                val_end=val_end,
                enable_blackbox=False,
                external_csv=None,
                hawkes_quantiles=hawkes_quantiles,
                hawkes_online_update_enabled=hawkes_online_update_enabled,
                exp1_debug_tables=exp1_debug_tables,
                exp2_debug_tables=exp2_debug_tables,
                output_subdir="whitebox_only",
            )
            continue

        for prefix, external_csv in candidates:
            print(f"[PIPELINE][{symbol}] external model prefix={prefix} file={external_csv}")
            results[symbol][prefix] = run_pipeline_for_market(
                mode=mode,
                market_csv=market_csv,
                symbol=symbol,
                interval=interval,
                train_end=train_end,
                val_end=val_end,
                enable_blackbox=True,
                external_csv=external_csv,
                hawkes_quantiles=hawkes_quantiles,
                hawkes_online_update_enabled=hawkes_online_update_enabled,
                exp1_debug_tables=exp1_debug_tables,
                exp2_debug_tables=exp2_debug_tables,
                output_subdir=prefix,
            )
    return results


def parse_cli_list(raw: str | None) -> list[str]:
    return _parse_csv_list(raw)


def parse_cli_quantiles(raw: str | None) -> tuple[float, ...]:
    return _parse_float_tuple(raw)
