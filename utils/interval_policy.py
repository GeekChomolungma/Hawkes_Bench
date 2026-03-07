from __future__ import annotations

import re

from config import BacktestConfig, HawkesConfig, WhiteBoxConfig


_INTERVAL_RE = re.compile(r"^(?P<n>\d+)(?P<u>[a-zA-Z]+)$")


def parse_interval_to_minutes(interval: str) -> int:
    s = interval.strip().lower()
    m = _INTERVAL_RE.match(s)
    if not m:
        raise ValueError(f"Unsupported interval format: {interval}")

    n = int(m.group("n"))
    u = m.group("u")
    if u == "d":
        return n * 24 * 60
    if u == "h":
        return n * 60
    if u == "m":
        return n
    raise ValueError(f"Unsupported interval unit: {interval}")


def auto_profile_for_interval(interval: str) -> dict[str, int | str]:
    """
    Built-in profiles for commonly used thesis bars:
    1d, 4h, 1h, 15m, 5m.
    """
    key = interval.strip().lower()
    mapping = {
        "1d": {"bars_per_year": 365, "rolling_window": 30, "hawkes_time_unit": "D"},
        "4h": {"bars_per_year": 365 * 6, "rolling_window": 180, "hawkes_time_unit": "s"},
        "1h": {"bars_per_year": 365 * 24, "rolling_window": 360, "hawkes_time_unit": "s"},
        "15m": {"bars_per_year": 365 * 24 * 4, "rolling_window": 480, "hawkes_time_unit": "s"},
        "5m": {"bars_per_year": 365 * 24 * 12, "rolling_window": 576, "hawkes_time_unit": "s"},
    }
    if key in mapping:
        return mapping[key]

    # fallback for other valid intervals
    mins = parse_interval_to_minutes(key)
    bars_per_year = max(1, int(round(365 * 24 * 60 / mins)))
    rolling_window = min(max(30, bars_per_year // 24), 720)
    hawkes_time_unit = "D" if mins >= 24 * 60 else "s"
    return {
        "bars_per_year": bars_per_year,
        "rolling_window": rolling_window,
        "hawkes_time_unit": hawkes_time_unit,
    }


def apply_interval_policy(
    interval: str,
    wb_cfg: WhiteBoxConfig,
    bt_cfg: BacktestConfig,
    hawkes_cfg: HawkesConfig,
) -> dict[str, int | str]:
    profile = auto_profile_for_interval(interval)
    wb_cfg.rolling_window = int(profile["rolling_window"])
    bt_cfg.bars_per_year = int(profile["bars_per_year"])

    if hawkes_cfg.time_unit == "auto":
        hawkes_cfg.time_unit = str(profile["hawkes_time_unit"])
    return profile

