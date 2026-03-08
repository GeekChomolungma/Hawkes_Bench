from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.whitebox.arima_garch_core import ArimaGarchModel


@dataclass
class WhiteBoxConfig:
    arima_order: tuple[int, int, int] = (1, 0, 1)
    garch_pq: tuple[int, int] = (1, 1)
    rolling_window: int = 30
    z_score: float = 1.96


class WhiteBoxForecaster:
    def __init__(self, cfg: WhiteBoxConfig):
        self.cfg = cfg
        self.model = ArimaGarchModel(arima_order=cfg.arima_order, garch_pq=cfg.garch_pq)

    def forecast_frame(self, close: pd.Series, returns: pd.Series, symbol: str, horizon: int = 1) -> pd.DataFrame:
        pred = self.model.rolling_forecast(returns, window=self.cfg.rolling_window) # "pred_for_ts" is the timestamp of the next step forecasting for, but it is stored at the current step t.

        df = pred.rename(columns={"mu": "mu_pred", "sigma": "sigma_pred"}).copy() # pred index is "ts", decision time. 
        df.index.name = "ts" # make sure index is named "ts" for downstream processing
        df["symbol"] = symbol
        df["horizon"] = int(horizon)

        # ground-truth close at decision time, ts
        df["close_t"] = close.reindex(df.index).astype(float)

        z = float(self.cfg.z_score)

        # return-space band, for next step, aligned with the return target at "pred_for_ts" but at decision time ts
        df["ret_pred_lo"] = df["mu_pred"] - z * df["sigma_pred"]
        df["ret_pred_hi"] = df["mu_pred"] + z * df["sigma_pred"]

        # price-space projection band (assuming log-return mapping)
        df["price_pred_median"] = df["close_t"] * np.exp(df["mu_pred"])
        df["price_pred_lo"] = df["close_t"] * np.exp(df["ret_pred_lo"])
        df["price_pred_hi"] = df["close_t"] * np.exp(df["ret_pred_hi"])

        return df.reset_index()
