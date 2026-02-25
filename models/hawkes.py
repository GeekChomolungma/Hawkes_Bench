import numpy as np
import pandas as pd

class HawkesIntensityModel:
    """
    把 Hawkes 做成可插拔接口：
      - fit(event_times)
      - intensity(t_grid) 或 intensity_at(event_index)
    你后面可以接入 tick/hawkeslib/自己写MLE。
    """
    def __init__(self, kernel: str = "exp"):
        self.kernel = kernel
        self._fitted = False

    def fit(self, event_times: np.ndarray):
        # event_times: seconds or days from origin, strictly increasing
        if len(event_times) < 5:
            raise ValueError("Too few events to fit Hawkes")
        self.event_times = np.asarray(event_times, dtype=float)
        self._fitted = True
        # placeholder params
        self.params_ = {"mu": 0.1, "alpha": 0.5, "beta": 1.0}
        return self

    def intensity_on_grid(self, t: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Hawkes not fitted")
        # Placeholder: constant intensity
        return np.full_like(t, fill_value=self.params_["mu"], dtype=float)

def eventize_returns(
    r: pd.Series,
    tau: float,
    signed: bool = True,
) -> dict[str, pd.DatetimeIndex]:
    """
    返回事件时间戳：
      - signed=True: {"pos": idx_pos, "neg": idx_neg}
      - signed=False: {"abs": idx_abs}
    """
    if signed:
        idx_pos = r.index[r > +tau]
        idx_neg = r.index[r < -tau]
        return {"pos": idx_pos, "neg": idx_neg}
    else:
        idx_abs = r.index[np.abs(r) > tau]
        return {"abs": idx_abs}

def to_relative_time(event_index: pd.DatetimeIndex, origin: pd.Timestamp, unit="D") -> np.ndarray:
    dt = (event_index - origin)
    if unit == "D":
        return dt.total_seconds().astype(float) / 86400.0
    if unit == "s":
        return dt.total_seconds().astype(float)
    raise ValueError("unit must be 'D' or 's'")