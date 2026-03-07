# Forecast Protocol (v0.1)

This protocol is the exchange format between external black-box model repos and this trading repo.

## Required columns
- ts: decision timestamp (UTC)
- symbol: instrument identifier, e.g. BTCUSDT
- horizon: forecast horizon in bars, usually 1
- close_t: close price at decision timestamp t

## Optional point forecast columns
- mu_pred: predicted next-step log return E[r_{t+h}|t]
- sigma_pred: predicted volatility scale for r_{t+h}

## Optional quantile forecast columns
- q05, q10, q25, q50, q75, q90, q95: forecast quantiles of next-step log return

## Optional metadata columns
- model_name
- model_version
- inference_run_id
- pred_for_ts

## Alignment convention
- All rows are indexed by decision time t.
- Forecast target is t+h where h == horizon.
- Do not use any data from t+h in feature construction.
