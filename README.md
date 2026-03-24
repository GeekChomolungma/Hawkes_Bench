# Hawkes_Bench

Hawkes_Bench is a crypto-quant research repository for a thesis pipeline with two layers:

- Forecast layer: evaluate model forecasting quality (white-box and optional black-box input)
- Trading layer: evaluate whether Hawkes-enhanced risk scaling improves strategy robustness

## What This Repo Does

This codebase is built to compare white-box and black-box forecasts under one unified interface, then map both into the same risk-adjusted trading signal.

- White-box branch: ARIMA+GARCH
- Black-box branch: external forecast table (CSV/Parquet) from another repo
- Hawkes branch: market self-excitation intensity used as a risk scaler

## Project Layout

- `main.py`: one-click entry for `exp1 + exp2`
- `config.py`: runtime config dataclasses
- `models/whitebox/arima_garch_core.py`: ARIMA+GARCH core model
- `models/whitebox/arima_garch_adapter.py`: white-box output adapter to unified forecast frame
- `dataio/forecast_loader.py`: external black-box file loader + normalization
- `dataio/validators.py`: protocol validation (required fields, key uniqueness, quantile monotonicity)
- `hawkes/core.py`: Hawkes process implementation
- `hawkes/threshold.py`: return-event thresholding and Hawkes fitting helpers
- `hawkes/lambda_online.py`: train-time theta fit + online lambda evaluation
- `risk/native.py`: native risk inference from sigma or quantile bands
- `risk/hawkes_scaler.py`: Hawkes risk scaling
- `strategy_signal/unified_signal.py`: unified signal and position construction
- `backtest/engine.py`: strict time-aligned backtest engine
- `backtest/metrics.py`: forecast metrics + backtest metrics
- `experiments/exp1_forecast_eval.py`: Experiment 1 (forecast layer)
- `experiments/exp2_hawkes_ablation.py`: Experiment 2 (trading layer)
- `utils/visual.py`: forecast and backtest plots
- `utils/persist.py`: save metrics/tables
- `utils/load_meta.py`: load and preview `.npy` figure metadata payloads
- `schemas/forecast_protocol.md`: external black-box data contract
- `reports/tables`, `reports/figures`: outputs
- [EXP_RESULTS_META.md](EXP_RESULTS_META.md): `.npy` metadata protocol for figure reproduction

## Core Experimental Workflow

### Experiment 1: Forecast Layer

Goal: evaluate predictive quality under a reproducible split.

Steps:
1. Load and preprocess market data.
2. Produce white-box forecasts (`mu_pred`, `sigma_pred`, and prediction bands).
3. Evaluate metrics by `train/val/test` split.
4. Use `test` as primary reporting split.
5. Compute a naive baseline on `test` for sanity check.
6. Optionally evaluate black-box external forecasts on `test`.

Typical metrics:
- MSE
- MAE
- RMSE
- Pinball loss (if quantile outputs exist)

### Experiment 2: Trading Layer (Hawkes Ablation)

Goal: test whether Hawkes-enhanced risk scaling improves strategy performance.

Steps:
1. Build forecast frame.
2. Fit Hawkes theta on train split only.
3. Run online lambda across all decision times.
4. Build two strategy variants per branch:
   - Native risk
   - Hawkes-enhanced risk
5. Backtest and compare metrics.

Typical metrics:
- total_return
- cagr
- sharpe
- max_drawdown
- calmar
- hit_rate

## Environment Setup

```powershell
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Data Requirements

### Market data

Default path in `main.py`:
- `market_info/cleaned/BTCUSDT_1d_Binance_cleaned.csv`

Required columns:
- `starttime` or `eventtime` (ms timestamp)
- `close`

### Black-box external forecast files

Template:
- `data/external_forecasts/blackbox_predictions_template.csv`

Current discovery layout (auto-scanned):
- `data/external_forecasts/<family>/<interval>/<run_id>/predictions_decision_aligned__target_<symbol>__init_<init_mode>__loss_<loss_mode>__tag_<run_tag>.csv`
- `<family>`: usually `ft` or `hf`
- `<interval>`: e.g. `1d`, `4h`
- `<run_id>`: arbitrary string (e.g. `1`, `2`, `expA`)

Example:
- `data/external_forecasts/ft/1d/1/predictions_decision_aligned__target_bchusdt__init_pretrained__loss_native__tag_batch_bchusdt_to_bchusdt.csv`

Required columns:
- `ts`, `symbol`, `horizon`, `close_t`

Optional columns:
- Point: `mu_pred`, `sigma_pred`
- Quantile: `q05,q10,q25,q50,q75,q90,q95`

If your external names differ, set `ExternalForecastConfig.column_map`.

## How To Run

This repo has 3 main runnable Python entry files:

- `main.py`: run experiment pipeline (`exp1`, `exp2`, or both)
- `data/clean_market_info.py`: clean and repair raw market csv files
- `data/verify_cleaned_market_info.py`: validate cleaned market csv files

### 1) Pipeline entry (`main.py`)

Show CLI help:

```powershell
env\Scripts\python main.py --help
```

Run full pipeline for one symbol (auto-discover external file):

```powershell
env\Scripts\python main.py --mode full --symbol BTCUSDT --interval 1d --external-family ft --external-run-id 1 --enable-blackbox --whitebox-mode off
```

Run one symbol with an explicit external csv:

```powershell
env\Scripts\python main.py --mode full --symbol BCHUSDT --interval 1d --external-csv "data/external_forecasts/ft/1d/1/predictions_decision_aligned__target_bchusdt__init_pretrained__loss_native__tag_batch_bchusdt_to_bchusdt.csv"
```

Run only one experiment:

```powershell
env\Scripts\python main.py --mode exp1 --symbol BTCUSDT --interval 1d --enable-blackbox --whitebox-mode always
env\Scripts\python main.py --mode exp2 --symbol BTCUSDT --interval 1d --enable-blackbox --whitebox-mode first
```

Run batch symbols:

```powershell
env\Scripts\python main.py --mode full --symbols BTCUSDT,ETHUSDT,LTCUSDT --interval 1d --external-family ft --external-run-id 1 --enable-blackbox --whitebox-mode first
```

Batch + auto-discovered black-box models:

```powershell
env\Scripts\python main.py --mode full --symbols BTCUSDT,ETHUSDT --interval 1d --external-family ft --external-run-id 1 --enable-blackbox --whitebox-mode first
```

White-box mode control (`--whitebox-mode`):

- `always`: white-box always enabled
- `first`: white-box only for the first discovered external file in batch
- `off`: white-box fully disabled

Black-box filename convention:

- `predictions_decision_aligned__target_<symbol>__init_<init_mode>__loss_<loss_mode>__tag_<run_tag>.csv`
- Example: `predictions_decision_aligned__target_bchusdt__init_pretrained__loss_native__tag_batch_bchusdt_to_bchusdt.csv`

When black-box is enabled, outputs are mirrored to external folder hierarchy:

- `reports/tables/<family>/<interval>/<run_id>/<symbol>/...`
- `reports/figures/<family>/<interval>/<run_id>/<symbol>/...`

If no external file is found (or black-box is disabled), outputs go to:

- `reports/tables/whitebox_only/<symbol>/...`
- `reports/figures/whitebox_only/<symbol>/...`

### 2) Data cleaning entry (`data/clean_market_info.py`)

Default run (input `market_info/`, output `market_info/cleaned/`):

```powershell
env\Scripts\python data/clean_market_info.py
```

Notes:

- Timestamp columns (`starttime`/`eventtime`) are treated as millisecond timestamps.
- The script repairs missing bars by reindexing to expected interval and interpolation.
- It writes `*_cleaned.csv` and `clean_summary.csv`.

### 3) Cleaned data verification entry (`data/verify_cleaned_market_info.py`)

Default run (verify `market_info/cleaned/`):

```powershell
env\Scripts\python data/verify_cleaned_market_info.py
```

It writes `verify_summary.csv` and reports PASS/FAIL for:

- timestamp parse errors
- duplicate timestamps
- monotonicity
- interval gaps
- missing close values

### 4) Wrapper scripts

- `scripts/run_batch_full.ps1` (PowerShell)
- `scripts/run_batch_full.sh` (bash-like environments)

PowerShell example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch_full.ps1 -Symbols "BTCUSDT,ETHUSDT" -WhiteboxMode first
```

Bash example:

```bash
WHITEBOX_MODE=first bash scripts/run_batch_full.sh "BTCUSDT,ETHUSDT"
```

### Auto interval policy

The pipeline auto-adapts key runtime parameters by interval (`1d`, `4h`, `1h`, `15m`, `5m`):

- `WhiteBoxConfig.rolling_window`
- `BacktestConfig.bars_per_year`
- `HawkesConfig.time_unit` (`D` or `s`)

The interval is parsed from the market filename pattern:

- `{SYMBOL}_{INTERVAL}_Binance.csv` (legacy)
- `{SYMBOL}_{INTERVAL}_Binance_cleaned.csv`

## Output Artifacts

### Forecast outputs

- White-box only: `reports/tables/whitebox_only/<symbol>/...`
- Black-box enabled: `reports/tables/<family>/<interval>/<run_id>/<symbol>/...`
- Includes `exp1_summary_metrics_*.json`, split/test metrics, optional black-box metrics

### Trading outputs

- White-box only: `reports/tables/whitebox_only/<symbol>/...`
- Black-box enabled: `reports/tables/<family>/<interval>/<run_id>/<symbol>/...`
- Includes `exp2_summary_metrics_*.json` and per-variant `exp2_*` files

### Figures

- White-box only: `reports/figures/whitebox_only/<symbol>/...`
- Black-box enabled: `reports/figures/<family>/<interval>/<run_id>/<symbol>/...`
- Includes forecast figures and backtest figures:
  - top panel: price + buy/sell markers
  - bottom panel: strategy equity + buy-and-hold reference

### Figure Metadata (.npy)

- White-box only: `reports/exp_results_meta/whitebox_only/<symbol>/...`
- Black-box enabled: `reports/exp_results_meta/<family>/<interval>/<run_id>/<symbol>/...`
- Protocol and field definitions:
  - [EXP_RESULTS_META.md](EXP_RESULTS_META.md)

## Results Showcase (How to Read the Plots)

This section highlights the key interpretation logic using the generated figures (for example, `BTCUSDT_1d` runs).

### 1) Why close-price forecast plots can look deceptively good

Relevant figure:
- `reports/figures/demo/exp1_whitebox_forecast_{SYMBOL}_{INTERVAL}.png`

Example:

![White-box close reconstruction](reports/figures/demo/exp1_whitebox_forecast_BTCUSDT_1d.png)
![White-box close reconstruction details](reports/figures/demo/showcase_1.png)

The white-box model predicts next-bar return, then reconstructs next-bar price from current observed close:
- `pred_price_next = close_t * exp(pred_return)`

Because this is rolling one-step reconstruction anchored to the latest true `close_t`, the predicted close path can visually track the real close even when return prediction quality is limited.

### 2) Use return-target plots as the primary forecast diagnostic

Relevant figure:
- `reports/figures/demo/exp1_whitebox_return_target_test_{SYMBOL}_{INTERVAL}.png`

Example:

![White-box return target test](reports/figures/demo/exp1_whitebox_return_target_test_BTCUSDT_1d.png)

How to read:
- Top panel: predicted next return vs realized next return over time.
- Bottom panel: scatter of `(predicted return, realized return)` with a 45° line (`y=x`).

Interpretation:
- Points near the 45° line: accurate predictions.
- Tight vertical cloud around `x≈0` with wider `y`: model is conservative (small predicted amplitude), while real returns are more volatile.
- Large off-line points: poor capture of extreme moves.

This is why return-level plots and metrics (`MSE/MAE/RMSE`) are more informative than close-only visual overlap.

### 3) How to interpret Hawkes ablation in trading layer (exp2)

Relevant figures:
- `reports/figures/demo/exp2_white_native_{SYMBOL}_{INTERVAL}.png`
- `reports/figures/demo/exp2_white_hawkes_{SYMBOL}_{INTERVAL}.png`
- (if enabled) black-box counterparts: `exp2_black_native_*`, `exp2_black_hawkes_*`

Examples:

![Exp2 white native](reports/figures/demo/exp2_white_native_BTCUSDT_1d.png)

![Exp2 white hawkes](reports/figures/demo/exp2_white_hawkes_BTCUSDT_1d.png)

Relevant metrics:
- `reports/tables/demo/exp2_white_native_metrics_{SYMBOL}_{INTERVAL}.json`
- `reports/tables/demo/exp2_white_hawkes_metrics_{SYMBOL}_{INTERVAL}.json`
- `reports/tables/demo/exp2_summary_metrics_{SYMBOL}_{INTERVAL}.json`

Reading guide:
- Compare equity curves against buy-and-hold in the same panel.
- Check whether Hawkes-enhanced version improves risk-adjusted performance:
  - Higher Sharpe/Calmar
  - Lower max drawdown
  - Similar or acceptable turnover
- If total return improves but drawdown worsens materially, treat as unstable improvement rather than robust enhancement.

## Equity Definition in Backtest Plots

`equity` is the strategy equity index, initialized at `1.0`.

- It is not raw coin price.
- It is not a position series.
- It is cumulative strategy value from per-step strategy PnL in the backtest engine.

The buy-and-hold line is a normalized price benchmark:

- `buy_and_hold = close / close[0]`

## Filename Auto-Tagging

Output filenames and figure titles are auto-tagged by parsing the market file name with:

- `{SYMBOL}_{INTERVAL}_Binance.csv` (legacy)
- `{SYMBOL}_{INTERVAL}_Binance_cleaned.csv`

Example:

- `BTCUSDT_1d_Binance.csv` -> `SYMBOL=BTCUSDT`, `INTERVAL=1d`
- `BTCUSDT_1d_Binance_cleaned.csv` -> `SYMBOL=BTCUSDT`, `INTERVAL=1d`

## Notes

- Current default execution mode is stateful full-notional style (`stateful_all_in`) to avoid repeated same-side orders.
- You can switch to continuous target position mode via config if needed.






