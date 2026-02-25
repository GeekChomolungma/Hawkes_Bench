import numpy as np
import pandas as pd

class ArimaGarchModel:
    """
    output of forecast_one_step():
      - mu_hat: forecast return (mu_{t+1})
      - sigma_hat: forecast volatility (sigma_{t+1})
    rolling_forecast() will return a DataFrame with columns:
    """
    def __init__(self, arima_order=(1,0,1), garch_pq=(1,1)):
        self.arima_order = arima_order
        self.garch_pq = garch_pq
        self._fitted = False

    def fit(self, returns: pd.Series):
        self.returns = returns.astype(float)
        self._fitted = True
        return self

    def forecast_one_step(self) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        # NOTE: for simplicity we just use the sample mean and std of the last 50 returns as the forecast
        mu = float(self.returns.tail(50).mean())
        sigma = float(self.returns.tail(50).std(ddof=1) + 1e-12)
        return mu, sigma

    def rolling_forecast(self, returns: pd.Series, window: int = 500) -> pd.DataFrame:
        out = []
        r = returns.astype(float)
        for i in range(window, len(r) - 1):
            hist = r.iloc[i-window:i]
            self.fit(hist)
            mu, sig = self.forecast_one_step()
            out.append((r.index[i], mu, sig))
        return pd.DataFrame(out, columns=["ts", "mu", "sigma"]).set_index("ts")