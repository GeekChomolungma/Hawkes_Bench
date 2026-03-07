# arima_garch.py
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


@dataclass
class ArimaGarchFitResult:
    mu_next: float
    sigma_next: float
    arima_params: dict
    garch_params: dict


class ArimaGarchModel:
    """
    Real rolling ARIMA + GARCH forecasting.

    For each rolling window:
      1) Fit ARIMA(p,d,q) on returns r_t
      2) Forecast next-step mean: mu_{t+1}
      3) Compute ARIMA residuals e_t = r_t - fitted_mean_t
      4) Fit GARCH(p,q) on residuals (mean=0)
      5) Forecast next-step variance: var_{t+1}, sigma_{t+1} = sqrt(var_{t+1})

    Output:
      rolling_forecast() returns DataFrame indexed by timestamp with columns:
        - mu: forecast return mu_{t+1}
        - sigma: forecast vol sigma_{t+1}
        - mu_annualized (optional helper)
        - sigma_annualized (optional helper)
    """

    def __init__(
        self,
        arima_order=(1, 0, 1),
        garch_pq=(1, 1),
        dist: str = "t",  # "t" is often better for crypto returns; can use "normal"
        rescale: bool = True,
        annualization_factor: float | None = None,  # e.g., 252 for daily, 365 for crypto-daily; None disables
        max_arima_iter: int = 200,
        max_garch_iter: int = 200,
        fallback_to_naive: bool = True,
        verbose: bool = False,
    ):
        self.arima_order = arima_order
        self.garch_pq = garch_pq
        self.dist = dist
        self.rescale = rescale
        self.annualization_factor = annualization_factor
        self.max_arima_iter = max_arima_iter
        self.max_garch_iter = max_garch_iter
        self.fallback_to_naive = fallback_to_naive
        self.verbose = verbose

    def _fit_one_window(self, r: pd.Series) -> ArimaGarchFitResult:
        r = r.astype(float).dropna()
        if len(r) < 30:
            raise ValueError("Too few observations in window to fit ARIMA+GARCH reliably")

        # ---------- ARIMA ----------
        # Suppress convergence warnings to keep rolling loop clean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(r, order=self.arima_order)
            arima_res = arima.fit(method_kwargs={"maxiter": self.max_arima_iter})

        # Forecast mean for next step
        mu_next = float(arima_res.forecast(steps=1)[0])

        # ARIMA in-sample fitted values -> residuals for GARCH
        # statsmodels aligns fittedvalues with index, but early terms may be NaN due to differencing / initialization
        # in-sample fitted values (ndarray) -> align back to index for residuals
        fitted_arr = np.asarray(arima_res.fittedvalues, dtype=float)
        fitted = pd.Series(fitted_arr, index=r.index)

        resid = (r - fitted).dropna()

        if len(resid) < 30:
            # If residuals got too short due to NaNs, fallback to demeaned returns
            resid = (r - r.mean()).dropna()

        # ---------- GARCH on residuals ----------
        p, q = self.garch_pq

        # arch_model has an internal scaling option. Using rescale=True helps stability.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            garch = arch_model(
                resid,
                mean="Zero",
                vol="GARCH",
                p=p,
                q=q,
                dist=self.dist,
                rescale=self.rescale,
            )
            garch_res = garch.fit(disp="off", options={"maxiter": self.max_garch_iter})

        # One-step ahead variance forecast
        # forecast returns a variance df with shape (horizon x ...). We need the last row, horizon=1.
        f = garch_res.forecast(horizon=1, reindex=False)
        var_next = float(f.variance.values[-1, 0])
        sigma_next = float(np.sqrt(max(var_next, 1e-18)))

        arima_params = dict(arima_res.params) if hasattr(arima_res, "params") else {}
        garch_params = dict(garch_res.params) if hasattr(garch_res, "params") else {}

        return ArimaGarchFitResult(
            mu_next=mu_next,
            sigma_next=sigma_next,
            arima_params=arima_params,
            garch_params=garch_params,
        )

    @staticmethod
    def _naive_mu_sigma(r: pd.Series) -> tuple[float, float]:
        mu = float(r.tail(50).mean())
        sigma = float(r.tail(50).std(ddof=1) + 1e-12)
        return mu, sigma

    def rolling_forecast(self, returns: pd.Series, window: int = 500) -> pd.DataFrame:
        """
        Rolling forecasts with "decision-time" alignment.

        Assumption (per your pipeline):
        - `returns` is aligned with `close` index (same timestamps)
        - the first return (at t0) is kept as 0.0 (not NaN)

        At decision time i:
        - we fit on returns[i-window+1 : i] (inclusive of r_i)
        - we forecast next-step return r_{i+1}
        - we store forecast at timestamp t_i (because you act at i for i+1)

        Output index: t_i
        Columns:
        - mu:   forecast mean of r_{i+1}
        - sigma: forecast std  of r_{i+1}
        - pred_for_ts: timestamp of the predicted return (t_{i+1}) for clarity
        """
        r = returns.astype(float)

        if len(r) < window + 2:
            raise ValueError("Not enough returns for the requested rolling window (+1 step ahead).")

        out = []
        # i is the decision-time index; we need i+1 to exist, so i <= len(r)-2
        for i in range(window, len(r) - 1):
            hist = r.iloc[i - window : i + 1]  # includes r_i
            ts_decision = r.index[i]
            ts_pred_for = r.index[i + 1]

            try:
                res = self._fit_one_window(hist) # i+1 step forecast
                mu, sigma = res.mu_next, res.sigma_next
            except Exception as e:
                if not self.fallback_to_naive:
                    raise
                mu, sigma = self._naive_mu_sigma(hist)
                if self.verbose:
                    print(f"[rolling_forecast] fallback at {ts_decision}: {type(e).__name__}: {e}")

            out.append((ts_decision, mu, sigma, ts_pred_for))

        df = pd.DataFrame(out, columns=["ts", "mu", "sigma", "pred_for_ts"]).set_index("ts")

        # Optional helpers: annualized versions (useful for paper plots)
        if self.annualization_factor is not None:
            af = float(self.annualization_factor)
            df["mu_annualized"] = df["mu"] * af
            df["sigma_annualized"] = df["sigma"] * np.sqrt(af)

        return df