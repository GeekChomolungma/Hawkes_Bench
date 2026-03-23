from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_runner import (
    parse_cli_list,
    parse_cli_quantiles,
    resolve_external_csv_candidates,
    run_pipeline_batch,
    run_pipeline_for_market,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hawkes Bench pipeline runner")
    p.add_argument("--mode", choices=["full", "exp1", "exp2"], default="full")

    # single-market mode
    p.add_argument("--market-csv", default="")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1d")
    p.add_argument("--external-csv", default="")

    # batch mode
    p.add_argument("--symbols", default="")
    p.add_argument("--market-dir", default="market_info/cleaned")
    p.add_argument("--external-dir", default="data/external_forecasts")
    p.add_argument("--external-family", default="ft")
    p.add_argument("--external-run-id", default="1")

    # shared experiment controls
    p.add_argument("--enable-blackbox", action="store_true", default=True)
    p.add_argument("--disable-blackbox", action="store_true")
    p.add_argument("--whitebox-mode", choices=["always", "first", "off"], default="off")

    # legacy compatibility flags (mapped to whitebox-mode)
    p.add_argument("--enable-whitebox", action="store_true", default=False)
    p.add_argument("--whitebox-first-only", action="store_true", default=False)

    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--val-end", default="2024-12-31")
    p.add_argument("--hawkes-quantiles", default="0.9", help="comma-separated, e.g. 0.85,0.9,0.95")
    p.add_argument("--execution-mode", choices=["stateful_all_in", "target_continuous"], default="stateful_all_in")
    p.add_argument("--entry-threshold", type=float, default=0.0)
    p.add_argument("--hawkes-online-update", action="store_true", default=False)
    p.add_argument("--exp1-debug-tables", action="store_true", default=False)
    p.add_argument("--exp2-debug-tables", action="store_true", default=False)

    # optional output summary for batch orchestration
    p.add_argument("--save-run-summary", default="")
    return p


def _resolve_whitebox_mode(args: argparse.Namespace) -> str:
    mode = (args.whitebox_mode or "off").strip().lower()

    # Legacy override behavior:
    # --enable-whitebox --whitebox-first-only => first
    # --enable-whitebox => always
    # --whitebox-first-only => first
    if args.enable_whitebox and args.whitebox_first_only:
        return "first"
    if args.enable_whitebox:
        return "always"
    if args.whitebox_first_only:
        return "first"
    return mode


def main() -> None:
    args = build_parser().parse_args()
    if args.entry_threshold < 0:
        raise ValueError("--entry-threshold must be >= 0")

    enable_blackbox = bool(args.enable_blackbox and not args.disable_blackbox)
    hawkes_quantiles = parse_cli_quantiles(args.hawkes_quantiles)
    whitebox_mode = _resolve_whitebox_mode(args)

    symbols = parse_cli_list(args.symbols)
    if symbols:
        results = run_pipeline_batch(
            mode=args.mode,
            symbols=symbols,
            interval=args.interval,
            market_dir=args.market_dir,
            train_end=args.train_end,
            val_end=args.val_end,
            enable_blackbox=enable_blackbox,
            whitebox_mode=whitebox_mode,
            external_dir=args.external_dir,
            external_family=args.external_family,
            external_run_id=args.external_run_id,
            hawkes_quantiles=hawkes_quantiles,
            execution_mode=args.execution_mode,
            entry_threshold=args.entry_threshold,
            hawkes_online_update_enabled=args.hawkes_online_update,
            exp1_debug_tables=args.exp1_debug_tables,
            exp2_debug_tables=args.exp2_debug_tables,
        )
    else:
        market_csv = args.market_csv or f"market_info/cleaned/{args.symbol}_{args.interval}_Binance_cleaned.csv"
        external_csv = args.external_csv
        output_subdir = None

        if enable_blackbox and not external_csv:
            candidates = resolve_external_csv_candidates(
                external_dir=args.external_dir,
                symbol=args.symbol,
                interval=args.interval,
                family=args.external_family,
                run_id=args.external_run_id,
            )
            if candidates:
                output_subdir, external_csv = candidates[0]
                output_subdir = f"{output_subdir}/{args.symbol.lower()}"
                print(f"[AUTO] single-mode external selected: {output_subdir} -> {external_csv}")
            else:
                print(f"[WARN] no external forecast discovered for {args.symbol} {args.interval}, black-box will be disabled.")

        single_enable_whitebox = whitebox_mode != "off"
        if output_subdir is None and enable_blackbox and external_csv:
            output_subdir = f"{args.external_family}/{args.interval}/{args.external_run_id}/{args.symbol.lower()}"
        if output_subdir is None and single_enable_whitebox and not enable_blackbox:
            output_subdir = f"whitebox_only/{args.symbol.lower()}"

        results = {
            args.symbol: run_pipeline_for_market(
                mode=args.mode,
                market_csv=market_csv,
                symbol=args.symbol,
                interval=args.interval,
                train_end=args.train_end,
                val_end=args.val_end,
                enable_blackbox=enable_blackbox,
                enable_whitebox=single_enable_whitebox,
                external_csv=external_csv,
                hawkes_quantiles=hawkes_quantiles,
                execution_mode=args.execution_mode,
                entry_threshold=args.entry_threshold,
                hawkes_online_update_enabled=args.hawkes_online_update,
                exp1_debug_tables=args.exp1_debug_tables,
                exp2_debug_tables=args.exp2_debug_tables,
                output_subdir=output_subdir,
            )
        }

    print("[DONE] symbols:", ", ".join(results.keys()))

    if args.save_run_summary:
        out_path = Path(args.save_run_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVED] run summary: {out_path}")


if __name__ == "__main__":
    main()
