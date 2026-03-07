from __future__ import annotations

import pandas as pd

from hawkes.core import eventize_returns, to_relative_time


def build_hawkes_events(r: pd.Series, tau: float, signed: bool = True):
    return eventize_returns(r=r, tau=tau, signed=signed)


def event_times_relative(event_index: pd.DatetimeIndex, origin: pd.Timestamp, unit: str = "D"):
    return to_relative_time(event_index=event_index, origin=origin, unit=unit)
