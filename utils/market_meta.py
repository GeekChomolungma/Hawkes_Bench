from __future__ import annotations

import re
from pathlib import Path


_BINANCE_NAME_RE = re.compile(
    r"(?P<symbol>[A-Za-z0-9]+)_(?P<interval>[0-9]+[A-Za-z]+)_Binance\.csv$",
    flags=re.IGNORECASE,
)


def parse_market_from_csv_path(csv_path: str, fallback_symbol: str, fallback_interval: str) -> dict[str, str]:
    """
    Parse market metadata from filename pattern:
      {SYMBOL}_{INTERVAL}_Binance.csv
    """
    name = Path(csv_path).name
    m = _BINANCE_NAME_RE.match(name)
    if m:
        symbol = m.group("symbol").upper()
        interval = m.group("interval")
    else:
        symbol = fallback_symbol.upper()
        interval = fallback_interval

    key = f"{symbol}_{interval}"
    title_label = f"{symbol} {interval}"
    return {"symbol": symbol, "interval": interval, "key": key, "title_label": title_label}

