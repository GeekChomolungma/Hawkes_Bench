# main.py
import numpy as np
import pandas as pd

from config import DataConfig, EventConfig, TrendConfig, SignalConfig, BacktestConfig
from data.loader import load_kline_csv
from data.preprocess import align_features, compute_log_return
from models.arima_garch import ArimaGarchModel
from research.select_threshold import choose_threshold_by_quantile, fit_hawkes_for_threshold
from research.sensitivity_alpha import alpha_sensitivity_study

def main():
    data_cfg = DataConfig(csv_path="your.csv", symbol="BCHUSDT", interval="1d")
    event_cfg = EventConfig(quantiles=(0.95, 0.97, 0.99), signed_events=True)
    trend_cfg = TrendConfig(arima_order=(1,0,1), garch_pq=(1,1))
    sig_cfg = SignalConfig(alpha_grid=(0.0, 0.1, 0.2, 0.5, 1, 2, 5, 10), position_cap=1.0)
    bt_cfg = BacktestConfig()

    df = load_kline_csv(data_cfg.csv_path)
    df = align_features(df)

    close = df["close"]
    r = compute_log_return(close)

    # 1) 趋势/波动预测（先 rolling naive，后面替换成真正 ARIMA+GARCH rolling fit）
    trend = ArimaGarchModel(arima_order=trend_cfg.arima_order, garch_pq=trend_cfg.garch_pq)
    pred = trend.rolling_forecast(r, window=500)  # index=ts
    mu = pred["mu"]
    sigma = pred["sigma"]

    # 2) 第一轮：阈值候选 + Hawkes 拟合
    for q in event_cfg.quantiles:
        tau = choose_threshold_by_quantile(r.loc[mu.index], q=q)
        hawkes_models = fit_hawkes_for_threshold(r.loc[mu.index], tau=tau, signed=event_cfg.signed_events)
        print(f"[q={q}] tau={tau:.6f} models={list(hawkes_models.keys())}")

        # 这里先用占位：lambda_t = 常数 or 你从模型算 intensity
        # 实际你会用 hawkes_models 输出每个时刻的 lambda_t，再合成（pos/neg可加权）
        lam = pd.Series(0.1, index=mu.index)  # TODO: replace with real intensity series

        # 3) 第二轮：alpha SA
        res = alpha_sensitivity_study(
            close=close.loc[mu.index],
            mu=mu,
            sigma=sigma,
            lam=lam,
            alpha_grid=list(sig_cfg.alpha_grid),
            position_cap=sig_cfg.position_cap,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
        )
        print(res.head(10))

if __name__ == "__main__":
    main()