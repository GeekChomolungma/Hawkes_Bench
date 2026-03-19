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

    # shared experiment controls
    p.add_argument("--enable-blackbox", action="store_true", default=True)
    p.add_argument("--disable-blackbox", action="store_true")
    p.add_argument("--train-end", default="2022-12-31")
    p.add_argument("--val-end", default="2024-12-31")
    p.add_argument("--hawkes-quantiles", default="0.9", help="comma-separated, e.g. 0.85,0.9,0.95")
    p.add_argument("--hawkes-online-update", action="store_true", default=False)
    p.add_argument("--exp1-debug-tables", action="store_true", default=False)
    p.add_argument("--exp2-debug-tables", action="store_true", default=False)

    # optional output summary for batch orchestration
    p.add_argument("--save-run-summary", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()

    enable_blackbox = bool(args.enable_blackbox and not args.disable_blackbox)
    hawkes_quantiles = parse_cli_quantiles(args.hawkes_quantiles)

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
            external_dir=args.external_dir,
            hawkes_quantiles=hawkes_quantiles,
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
            )
            if candidates:
                output_subdir, external_csv = candidates[0]
                print(f"[AUTO] single-mode external selected: {output_subdir} -> {external_csv}")
            else:
                print(f"[WARN] no external forecast discovered for {args.symbol} {args.interval}, black-box will be disabled.")
        results = {
            args.symbol: run_pipeline_for_market(
                mode=args.mode,
                market_csv=market_csv,
                symbol=args.symbol,
                interval=args.interval,
                train_end=args.train_end,
                val_end=args.val_end,
                enable_blackbox=enable_blackbox,
                external_csv=external_csv,
                hawkes_quantiles=hawkes_quantiles,
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
