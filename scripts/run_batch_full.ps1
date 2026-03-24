param(
  [string]$Symbols = "BTCUSDT,ETHUSDT,BCHUSDT,BNBUSDT,DOGEUSDT,LTCUSDT,XRPUSDT,ZECUSDT",
  [string]$Mode = "full",
  [string]$Interval = "1d",
  [string]$TrainEnd = "2025-10-31",
  [string]$ValEnd = "2025-11-30",
  [string]$TestStart = "2025-12-10",
  [string]$TestEnd = "2026-01-25",
  [string]$HawkesQ = "0.6,0.7,0.8,0.9",
  [ValidateSet("stateful_all_in", "target_continuous")]
  [string]$ExecutionMode = "stateful_all_in",
  [double]$EntryThreshold = 0.05,
  [string]$ExternalFamily = "ft",
  [string]$ExternalRunId = "pretrained_QuExTime_BTC_Based",
  [ValidateSet("always", "first", "off")]
  [string]$WhiteboxMode = "off",
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
  --test-start $TestStart `
  --test-end $TestEnd `
  --external-family $ExternalFamily `
  --external-run-id $ExternalRunId `
  --hawkes-quantiles $HawkesQ `
  --execution-mode $ExecutionMode `
  --entry-threshold $EntryThreshold `
  --whitebox-mode $WhiteboxMode `
  $blackboxFlag
