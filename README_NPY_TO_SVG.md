# NPY Meta and Figure Rendering Guide

This document complements [EXP_RESULTS_META.md](./EXP_RESULTS_META.md).
It explains how plotting scripts in `utils/` consume stored results and produce SVG figures.

## 1. Scope

This repo currently has two plotting families:

- `bar_plot_for_*`: manual metric-to-bar plotting (not from `.npy`)
- `npy_*_to_svg`: render `.npy` meta payloads into `.svg`

## 2. Input and Output Conventions

- Meta input root: `reports/exp_results_meta/...`
- Figure output root (experiment figures): `reports/figures/...`
- Some scripts write near input files; some use fixed output paths.

## 3. Bar Plot Scripts (`bar_plot_for_*`)

### 3.1 Files

- [utils/bar_plot_for_metrics_btc.py](./utils/bar_plot_for_metrics_btc.py)
- [utils/bar_plot_for_metrics_doge.py](./utils/bar_plot_for_metrics_doge.py)

### 3.2 Data Input Logic

These scripts do not read `.npy`.
They use hardcoded metric dictionaries for 4 models:

- `ARIMA+GARCH`
- `Chronos2 Zeroshot`
- `Chronos2 Native FT`
- `Chronos2 Proposed FT`

Metric groups:

- Traditional: `mae`, `rmse`
- Risk: `pinball_q10`, `pinball_q90`, derived `Tail Avg=(q10+q90)/2`
- Trend: `sign_accuracy`, `rank_ic_spearman`

### 3.3 Plotting Logic

- Grouped bars by metric block (x-axis block per metric).
- Fixed model color mapping shared across figures.
- Visible zero baseline.
- Manual y-padding controls per figure type (`traditional/risk/trend`).
- Optional value labels and up/down arrows via hardcoded override tables.

### 3.4 Output Products

Running script directly writes SVG files to repo root:

BTC:

- `btc_traditional_metrics.svg`
- `btc_risk_metrics.svg`
- `btc_trend_metrics.svg`

DOGE:

- `doge_traditional_metrics.svg`
- `doge_risk_metrics.svg`
- `doge_trend_metrics.svg`

## 4. NPY-to-SVG Scripts

### 4.1 Forecast Layer

File:

- [utils/npy_forecast_layer_to_svg.py](./utils/npy_forecast_layer_to_svg.py)

Input:

- `forecast_layer` payloads (`kind == "forecast_layer"`)

Output:

- SVG saved in the same folder as each source `.npy`

### 4.2 Return Target Layer

File:

- [utils/npy_return_target_to_svg.py](./utils/npy_return_target_to_svg.py)

Input:

- `return_target_layer` payloads (`kind == "return_target_layer"`)

Output:

- SVG saved in the same folder as each source `.npy`
- Optional suffixes based on flags:
  - `__with_inset`
  - `__with_dual_inset`
  - `__with_scatter`

### 4.3 Backtest Multi-Model Equity

File:

- [utils/npy_backtest_equity_to_svg.py](./utils/npy_backtest_equity_to_svg.py)

Input:

- Hardcoded `MODEL_BACKTEST_PATHS` to `backtest_layer` `.npy`
- Two lines per model: `native_no_hawkes` vs `hawkes_scaled`

Output:

- Fixed single file:
  - `reports/figures/manual/exp2_equity_multi_model.svg`

### 4.4 Backtest Close + 4 Trade Markers

File:

- [utils/npy_backtest_B&S_markers_to_svg.py](./utils/npy_backtest_B&S_markers_to_svg.py)

Input:

- Hardcoded 4 models x 3 strategies (`native_no_hawkes`, `hawkes_scaled_q70`, `hawkes_scaled_q90`)
- Reads `backtest_layer` fields: `close_ts_ns`, `close`,
  `open_long_ts_ns`, `open_short_ts_ns`, `close_short_ts_ns`, `close_long_ts_ns`

Marker policy:

- `open_long`: green up-triangle, below close
- `close_long`: red hollow circle, above close
- `open_short`: red down-triangle, above close
- `close_short`: green hollow circle, below close
- If triangle and circle coincide on same side: circle is closer to close, triangle farther (close-then-open visual ordering)

Output:

- `reports/figures/backtest/btcusdt_1d/*__close_bs_markers.svg`
- Total: 12 figures (4 models x 3 strategies)

### 4.5 Backtest Event + Lambda Recorder

File:

- [utils/npy_backtest_event&lamb_to_svg.py](./utils/npy_backtest_event&lamb_to_svg.py)

Input:

- Hardcoded 4 models x 2 strategies (`hawkes_scaled_q70`, `hawkes_scaled_q90`)
- Reads `event_lambda_recorder` payloads:
  - timeline: `test_ts_ns`
  - events: `event_pos`, `event_neg`, `event_abs`
  - returns: `log_return`
  - lambda series: `lambda_total` (or fallback `lambda_line`), `lambda_pos`, `lambda_neg`, `lambda_abs`
  - parameter snapshots: `theta_*_enabled`, `theta_*_{mu,alpha,beta}`

Plot layout:

- Top panel: event sequence + right-axis `log_return`
- Bottom panel: `lambda_total` plus component lambdas (`lambda_pos/lambda_neg` in signed mode, `lambda_abs` in abs mode)
- Theta text box shown in lambda panel

Output:

- `reports/figures/backtest/btcusdt_1d/*__event_lambda.svg`
- Total: 8 figures (4 models x 2 strategies)

## 5. Quick Commands

```powershell
# Return-target npy -> svg (single file)
python utils/npy_return_target_to_svg.py reports/exp_results_meta/ft/1d/zeroshot/btcusdt/exp1_whitebox_return_target_test_BTCUSDT_1d.npy

# Backtest marker figures (12)
python "utils/npy_backtest_B&S_markers_to_svg.py"

# Event/lambda figures (8)
python "utils/npy_backtest_event&lamb_to_svg.py"

# Manual bar metrics
python utils/bar_plot_for_metrics_btc.py
python utils/bar_plot_for_metrics_doge.py
```

## 6. Notes

- `npy_*` scripts expect payload `kind` to match; mismatch is skipped or raises error by script design.
- Path mappings in hardcoded dicts are the source of truth for batch outputs.
- For protocol field definitions, always refer to [EXP_RESULTS_META.md](./EXP_RESULTS_META.md).
