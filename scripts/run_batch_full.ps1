param(
  [string]$Symbols = "BTCUSDT,ETHUSDT,LTCUSDT,DOGEUSDT,XRPUSDT,BNBUSDT,BCHUSDT,ZECUSDT",
  [string]$Mode = "full",
  [string]$Interval = "1d",
  [string]$TrainEnd = "2022-12-31",
  [string]$ValEnd = "2024-12-31",
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
