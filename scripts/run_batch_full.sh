#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/run_batch_full.sh "BTCUSDT,ETHUSDT,LTCUSDT"
#
# Optional env overrides:
#   MODE=full
#   INTERVAL=1d
#   TRAIN_END=2022-12-31
#   VAL_END=2024-12-31
#   HAWKES_Q=0.85,0.90,0.95
#   EXTERNAL_PREFIXES=zeroshot,newLoss1,finetuned
#   ENABLE_BLACKBOX=1
#   PYTHON_BIN=python

SYMBOLS="${1:-BTCUSDT,ETHUSDT,LTCUSDT,DOGEUSDT,XRPUSDT,BNBUSDT,BCHUSDT,ZECUSDT}"
MODE="${MODE:-full}"
INTERVAL="${INTERVAL:-1d}"
TRAIN_END="${TRAIN_END:-2022-12-31}"
VAL_END="${VAL_END:-2024-12-31}"
HAWKES_Q="${HAWKES_Q:-0.9}"
EXTERNAL_PREFIXES="${EXTERNAL_PREFIXES:-zeroshot,newLoss1,finetuned}"
ENABLE_BLACKBOX="${ENABLE_BLACKBOX:-1}"
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
  --external-prefixes "${EXTERNAL_PREFIXES}" \
  --hawkes-quantiles "${HAWKES_Q}" \
  ${BLACKBOX_FLAG}
