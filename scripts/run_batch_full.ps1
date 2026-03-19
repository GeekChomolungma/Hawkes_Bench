param(
  [string]$Symbols = "BTCUSDT,ETHUSDT,LTCUSDT,DOGEUSDT,XRPUSDT,BCHUSDT",
  [string]$Mode = "full",
  [string]$Interval = "1d",
  [string]$TrainEnd = "2025-08-31",
  [string]$ValEnd = "2025-11-30",
  [string]$HawkesQ = "0.9",
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
  --hawkes-quantiles $HawkesQ `
  $blackboxFlag
