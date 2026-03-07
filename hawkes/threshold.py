import numpy as np
import pandas as pd

from hawkes.core import eventize_returns, to_relative_time, HawkesIntensityModel

def choose_threshold_by_quantile(r: pd.Series, q: float) -> float:
    return float(np.quantile(np.abs(r.values), q))

def fit_hawkes_for_threshold(r: pd.Series, tau: float, signed=True, unit: str = "D"):
    origin = r.index[0]
    if unit == "D":
        T_end = (r.index[-1] - origin).total_seconds() / 86400.0
    elif unit == "s":
        T_end = (r.index[-1] - origin).total_seconds()
    else:
        raise ValueError("unit must be 'D' or 's'")

    events = eventize_returns(r, tau=tau, signed=signed)

    models = {}
    for k, idxs in events.items():
        t = to_relative_time(idxs, origin=origin, unit=unit)
        if len(t) < 10:
            models[k] = None
            continue
        m = HawkesIntensityModel(kernel="exp").fit(t, T_end=T_end)
        models[k] = m
    return models
