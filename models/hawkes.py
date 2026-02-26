import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional

from scipy.optimize import minimize


@dataclass
class HawkesExpParams:
    mu: float
    alpha: float
    beta: float


class HawkesIntensityModel:
    """
    Univariate Hawkes with exponential kernel:
        lambda(t) = mu + alpha * sum_{ti < t} exp(-beta*(t-ti))

    Fit by MLE with closed-form compensator for exp kernel.

    Public API:
      - fit(event_times, T_end=None)
      - intensity_on_grid(t_grid)
      - intensity_at_events()  (lambda(t_i) for each event)
    """

    def __init__(self, kernel: str = "exp"):
        if kernel != "exp":
            raise ValueError("Only exp kernel is implemented in this version.")
        self.kernel = kernel
        self._fitted = False
        self.params_: Optional[HawkesExpParams] = None
        self.event_times: Optional[np.ndarray] = None
        self.T_end: Optional[float] = None

    @staticmethod
    def _neg_loglik_exp(params_log: np.ndarray, t: np.ndarray, T: float) -> float:
        """
        params_log: unconstrained, mapped by exp to positive params:
          mu = exp(x0), alpha = exp(x1), beta = exp(x2)
        """
        mu = float(np.exp(params_log[0]))
        alpha = float(np.exp(params_log[1]))
        beta = float(np.exp(params_log[2]))

        # Penalty to discourage explosive processes (branching ratio >= 1)
        # branching ratio n = alpha/beta (for exp kernel)
        n = alpha / (beta + 1e-12)
        penalty = 0.0
        if n >= 0.999:
            penalty += 1e6 * (n - 0.999) ** 2

        # Recurrence for g_i = sum_{j<i} exp(-beta*(t_i - t_j))
        # g_0 = 0
        # g_i = exp(-beta*dt_i) * (1 + g_{i-1})
        g = 0.0
        ll = 0.0
        for i in range(len(t)):
            if i == 0:
                lam = mu  # no past events
            else:
                dt = t[i] - t[i - 1]
                if dt < 0:
                    return 1e12  # invalid ordering
                decay = np.exp(-beta * dt)
                g = decay * (1.0 + g)
                lam = mu + alpha * g

            if lam <= 0 or not np.isfinite(lam):
                return 1e12
            ll += np.log(lam)

        # Compensator integral: ∫_0^T lambda(s) ds = mu*T + (alpha/beta) * Σ (1 - exp(-beta*(T - t_i)))
        # where sum over events
        tail = np.exp(-beta * (T - t))
        compensator = mu * T + (alpha / (beta + 1e-12)) * float(np.sum(1.0 - tail))

        nll = -(ll - compensator) + penalty
        return float(nll)

    def fit(self, event_times: np.ndarray, T_end: Optional[float] = None):
        t = np.asarray(event_times, dtype=float)
        if len(t) < 10:
            raise ValueError("Too few events to fit Hawkes reliably (need >= 10).")
        if np.any(np.diff(t) <= 0):
            raise ValueError("event_times must be strictly increasing.")

        T = float(T_end) if T_end is not None else float(t[-1])
        if T <= t[-1]:
            T = float(t[-1])  # ensure T >= last event time

        # init guess (rough but stable)
        # baseline mu ~ events per unit time / 2
        rate = len(t) / max(T, 1e-12)
        mu0 = max(rate * 0.5, 1e-6)
        alpha0 = max(rate * 0.5, 1e-6)
        beta0 = 1.0

        x0 = np.log([mu0, alpha0, beta0])

        res = minimize(
            fun=self._neg_loglik_exp,
            x0=x0,
            args=(t, T),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if not res.success:
            raise RuntimeError(f"Hawkes MLE failed: {res.message}")

        mu = float(np.exp(res.x[0]))
        alpha = float(np.exp(res.x[1]))
        beta = float(np.exp(res.x[2]))

        self.params_ = HawkesExpParams(mu=mu, alpha=alpha, beta=beta)
        self.event_times = t
        self.T_end = T
        self._fitted = True
        return self

    def intensity_at_events(self) -> np.ndarray:
        """
        lambda(t_i) evaluated right after processing history up to i-1
        (standard likelihood convention).
        """
        if not self._fitted:
            raise RuntimeError("Hawkes not fitted")

        mu = self.params_.mu
        alpha = self.params_.alpha
        beta = self.params_.beta
        t = self.event_times

        out = np.zeros_like(t, dtype=float)
        g = 0.0
        for i in range(len(t)):
            if i == 0:
                out[i] = mu
            else:
                dt = t[i] - t[i - 1]
                decay = np.exp(-beta * dt)
                g = decay * (1.0 + g)
                out[i] = mu + alpha * g
        return out

    def intensity_on_grid(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Compute lambda(t) on an arbitrary increasing time grid.
        Complexity: O(N + M) using a running excitation state.
        """
        if not self._fitted:
            raise RuntimeError("Hawkes not fitted")

        mu = self.params_.mu
        alpha = self.params_.alpha
        beta = self.params_.beta
        events = self.event_times

        tg = np.asarray(t_grid, dtype=float)
        if np.any(np.diff(tg) < 0):
            raise ValueError("t_grid must be non-decreasing.")

        lam = np.zeros_like(tg, dtype=float)

        # Running state: S(t) = sum_{ti < t} exp(-beta*(t - ti))
        # We update between grid points with decay, and add 1 at each event crossing.
        S = 0.0
        ev_ptr = 0
        last_t = tg[0]

        # bring ptr to events <= first grid time
        while ev_ptr < len(events) and events[ev_ptr] <= last_t:
            # event at exactly last_t contributes to future times, not current, conventionally.
            # If you want right-continuous, you can include it here.
            ev_ptr += 1

        lam[0] = mu + alpha * S

        for i in range(1, len(tg)):
            t_now = tg[i]
            dt = t_now - last_t
            if dt < 0:
                raise ValueError("t_grid must be non-decreasing.")
            if dt > 0:
                S *= np.exp(-beta * dt)

            # Add events in (last_t, t_now]
            while ev_ptr < len(events) and events[ev_ptr] <= t_now:
                # event causes instantaneous +1 to S at its time,
                # but if multiple events in the interval, we just add them (approx within step)
                S += 1.0
                ev_ptr += 1

            lam[i] = mu + alpha * S
            last_t = t_now

        return lam


def eventize_returns(r: pd.Series, tau: float, signed: bool = True) -> Dict[str, pd.DatetimeIndex]:
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


def intensity_series_on_index(
    model: HawkesIntensityModel,
    index: pd.DatetimeIndex,
    origin: pd.Timestamp,
    unit: str = "D",
) -> pd.Series:
    """
    Evaluate lambda(t) on the given bar index (e.g., mu.index decision times).
    Returns a Series aligned to `index`, which is the intensity by decision time i, 
    This intensity can be used as a feature indicator for next time, i+1.
    """
    if unit == "D":
        t_grid = (index - origin).total_seconds().astype(float) / 86400.0
    elif unit == "s":
        t_grid = (index - origin).total_seconds().astype(float)
    else:
        raise ValueError("unit must be 'D' or 's'")

    lam = model.intensity_on_grid(np.asarray(t_grid, dtype=float))
    return pd.Series(lam, index=index)