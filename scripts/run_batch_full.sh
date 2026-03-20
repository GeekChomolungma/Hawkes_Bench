#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/run_batch_full.sh "BTCUSDT,ETHUSDT,LTCUSDT"
#
# Optional env overrides:
#   MODE=full
#   INTERVAL=1d
#   TRAIN_END=2025-08-31
#   VAL_END=2025-11-30
#   HAWKES_Q=0.85,0.90,0.95
#   EXTERNAL_FAMILY=ft
#   EXTERNAL_RUN_ID=1
#   ENABLE_BLACKBOX=1
#   WHITEBOX_MODE=first   # always | first | off
#   PYTHON_BIN=python

SYMBOLS="${1:-BTCUSDT,ETHUSDT,LTCUSDT,DOGEUSDT,XRPUSDT,BNBUSDT,BCHUSDT,ZECUSDT}"
MODE="${MODE:-full}"
INTERVAL="${INTERVAL:-1d}"
TRAIN_END="${TRAIN_END:-2025-08-31}"
VAL_END="${VAL_END:-2025-11-30}"
HAWKES_Q="${HAWKES_Q:-0.9}"
EXTERNAL_FAMILY="${EXTERNAL_FAMILY:-ft}"
EXTERNAL_RUN_ID="${EXTERNAL_RUN_ID:-1}"
ENABLE_BLACKBOX="${ENABLE_BLACKBOX:-1}"
WHITEBOX_MODE="${WHITEBOX_MODE:-first}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BLACKBOX_FLAG="--disable-blackbox"
if [[ "${ENABLE_BLACKBOX}" == "1" ]]; then
  BLACKBOX_FLAG="--enable-blackbox"
fi

${PYTHON_BIN} main.py \
  --mode "${MODE}" \
  --symbols "${SYMBOLS}" \
  --interval "${INTERVAL}" \
  --train-end "${TRAIN_END}" \
  --val-end "${VAL_END}" \
  --external-family "${EXTERNAL_FAMILY}" \
  --external-run-id "${EXTERNAL_RUN_ID}" \
  --hawkes-quantiles "${HAWKES_Q}" \
  --whitebox-mode "${WHITEBOX_MODE}" \
  ${BLACKBOX_FLAG}
