import numpy as np
import pandas as pd

from models.hawkes import eventize_returns, to_relative_time, HawkesIntensityModel

def choose_threshold_by_quantile(r: pd.Series, q: float) -> float:
    return float(np.quantile(np.abs(r.values), q))

def fit_hawkes_for_threshold(r: pd.Series, tau: float, signed=True):
    origin = r.index[0]
    events = eventize_returns(r, tau=tau, signed=signed)

    models = {}
    for k, idx in events.items():
        t = to_relative_time(idx, origin=origin, unit="D")
        if len(t) < 10:
            models[k] = None
            continue
        m = HawkesIntensityModel(kernel="exp").fit(t)
        models[k] = m
    return models