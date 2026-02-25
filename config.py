# config.py
from dataclasses import dataclass

@dataclass
class DataConfig:
    csv_path: str
    symbol: str = "BCHUSDT"
    interval: str = "1d"
    tz: str = "UTC"

@dataclass
class EventConfig:
    # The first round: event thresholds based on quantiles of returns
    quantiles: tuple[float, ...] = (0.95, 0.97, 0.99)
    signed_events: bool = True   # positive and negative events are treated separately
    min_events: int = 200        # minimum number of events to consider a threshold valid

@dataclass
class TrendConfig:
    arima_order: tuple[int, int, int] = (1, 0, 1)
    garch_pq: tuple[int, int] = (1, 1)

@dataclass
class SignalConfig:
    # The second round: alpha values for SA candidates
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    position_cap: float = 1.0

@dataclass
class BacktestConfig:
    fee_bps: float = 2.0
    slippage_bps: float = 1.0
    walkforward_train_days: int = 365 * 2
    walkforward_test_days: int = 90