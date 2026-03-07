from dataclasses import dataclass, field


@dataclass
class DataConfig:
    csv_path: str = "market_info/BTCUSDT_1d_Binance.csv"
    symbol: str = "BTCUSDT"
    interval: str = "1d"


@dataclass
class WhiteBoxConfig:
    arima_order: tuple[int, int, int] = (1, 0, 1)
    garch_pq: tuple[int, int] = (1, 1)
    rolling_window: int = 30
    z_score: float = 1.96


@dataclass
class HawkesConfig:
    quantile: float = 0.9
    signed_events: bool = True
    alpha_risk: float = 1.0
    time_unit: str = "auto"  # "auto" | "D" | "s"


@dataclass
class SignalConfig:
    position_cap: float = 1.0
    execution_mode: str = "stateful_all_in"  # "stateful_all_in" | "target_continuous"
    entry_threshold: float = 0.0


@dataclass
class BacktestConfig:
    fee_bps: float = 2.0
    slippage_bps: float = 1.0
    bars_per_year: int = 252


@dataclass
class ExternalForecastConfig:
    enabled: bool = False
    path: str = "data/external_forecasts/blackbox_predictions.csv"
    # standard_name -> external_name, can be empty for best-effort auto inference
    column_map: dict[str, str] = field(default_factory=dict)


@dataclass
class OutputConfig:
    table_dir: str = "reports/tables"
    figure_dir: str = "reports/figures"
