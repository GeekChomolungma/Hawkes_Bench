param(
  [string]$Symbols = "BTCUSDT,ETHUSDT,LTCUSDT,DOGEUSDT,XRPUSDT,BNBUSDT,BCHUSDT,ZECUSDT",
  [string]$Mode = "full",
  [string]$Interval = "1d",
  [string]$TrainEnd = "2025-08-31",
  [string]$ValEnd = "2025-11-30",
  [string]$HawkesQ = "0.9",
  [string]$ExternalFamily = "ft",
  [string]$ExternalRunId = "1",
  [switch]$DisableBlackbox
)

$blackboxFlag = "--enable-blackbox"
if ($DisableBlackbox) { $blackboxFlag = "--disable-blackbox" }

env\Scripts\python main.py `
  --mode $Mode `
  --symbols $Symbols `
  --interval $Interval `
  --train-end $TrainEnd `
  --val-end $ValEnd `
  --external-family $ExternalFamily `
  --external-run-id $ExternalRunId `
  --hawkes-quantiles $HawkesQ `
  $blackboxFlag
