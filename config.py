from dataclasses import dataclass, field


@dataclass
class SplitConfig:
    # Inclusive boundary: decision ts <= train_end belongs to train.
    train_end: str | None = None
    # Inclusive boundary: decision ts <= val_end belongs to val.
    # test is (val_end, +inf). If omitted, test is (train_end, +inf).
    val_end: str | None = None
    # Optional inclusive lower bound for test window.
    # If omitted, keep the default split-derived start.
    test_start: str | None = None
    # Optional inclusive upper bound for test window.
    # If omitted, test window extends to the end of data.
    test_end: str | None = None


@dataclass
class DataConfig:
    csv_path: str = "market_info/BTCUSDT_1d_Binance_cleaned.csv"
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class WhiteBoxConfig:
    arima_order: tuple[int, int, int] = (1, 0, 1)
    garch_pq: tuple[int, int] = (1, 1)
    rolling_window: int = 30
    z_score: float = 1.96


@dataclass
class HawkesConfig:
    quantile: float = 0.9
    # Optional multi-threshold sweep. If empty, fall back to `quantile`.
    quantile_grid: tuple[float, ...] = ()
    # If False (default), fit Hawkes on train+val; if True, fit on train only.
    # Online re-training is reserved for future work.
    online_update_enabled: bool = False
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
    # Keep only core exp1 artifacts by default.
    # If True, also dump intermediate frames/rows/all-split metric files.
    exp1_save_debug_tables: bool = False
    # Keep only core exp2 artifacts by default.
    # If True, also dump backtest rows and additional diagnostics.
    exp2_save_debug_tables: bool = False
