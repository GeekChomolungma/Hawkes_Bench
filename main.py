# main.py
import numpy as np
import pandas as pd

from utils.visual import plot_hawkes_lambda_splits, plot_prediction, plot_alpha_sensitivity
from config import DataConfig, EventConfig, TrendConfig, SignalConfig, BacktestConfig
from data.loader import load_kline_csv, time_split_df
from data.preprocess import align_features, compute_log_return
from models.arima_garch import ArimaGarchModel
from models.hawkes import hawkes_lambda_suite_fixed_theta
from research.select_threshold import choose_threshold_by_quantile, fit_hawkes_for_threshold
from research.sensitivity_alpha import alpha_sensitivity_study

def main():
    data_cfg = DataConfig(csv_path="market_info/BTCUSDT_1d_Binance.csv", symbol="BTCUSDT", interval="1d")
    event_cfg = EventConfig(quantiles=(0.8, 0.9), signed_events=True)
    trend_cfg = TrendConfig(arima_order=(1,0,1), garch_pq=(1,1))
    sig_cfg = SignalConfig(alpha_risk_grid=(0.0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20), position_cap=1.0)
    bt_cfg = BacktestConfig()

    df_all = load_kline_csv(data_cfg.csv_path)
    df_all = align_features(df_all)
    df_train, df_val, df_test = time_split_df(df_all, ratios=(0.7, 0.1, 0.2))

    # close inherits df's datetime index automatically
    close_all = df_all["close"]  # index is datetime from df
    r_all = compute_log_return(close_all)  # r.index == close.index (first value filled as 0)
    close_test = close_all.loc[df_test.index]
    r_test = r_all.loc[df_test.index]

    # 1) Trend forecasting with ARIMA-GARCH, mean and volatility predictions
    trend = ArimaGarchModel(arima_order=trend_cfg.arima_order, garch_pq=trend_cfg.garch_pq)
    pred_all = trend.rolling_forecast(r_all, window=30)  # index=decision time i
    mu_all = pred_all["mu"]
    sigma_all = pred_all["sigma"]
    idx = pred_all.index
    idx_train = idx[idx <= df_train.index[-1]]
    idx_val   = idx[(idx > df_train.index[-1]) & (idx <= df_val.index[-1])]
    idx_test  = idx[idx > df_val.index[-1]]

    pred_test = pred_all.loc[idx_test]

    # plot_prediction(close_all, r_all, pred_all)
    plot_prediction(close_test, r_test, pred_test)

    # 2) threshold selection for events and Hawkes modeling
    for q in event_cfg.quantiles:
        # get tau on train set only
        tau = choose_threshold_by_quantile(r_all.loc[idx_train], q=q)

        # fit Hawkes on train set to get θ (mu, alpha, beta)
        hawkes_models_train = fit_hawkes_for_threshold(
            r_all.loc[idx_train],
            tau=tau,
            signed=event_cfg.signed_events
        )

        # fix θ
        theta = {}
        for k, m in hawkes_models_train.items():
            if m is None:
                continue
            theta[k] = m.params_   # HawkesExpParams(mu, alpha, beta) :contentReference[oaicite:8]{index=8}

        # online update lambda on test set
        origin = idx[0]  # or idx_train[0], but make sure train and test use the same origin for relative time
        
        lam_all_online = hawkes_lambda_suite_fixed_theta(
            r=r_all,
            index=idx,
            origin=origin,
            tau=tau,
            theta_by_key=theta,
            signed=event_cfg.signed_events,
            unit="D",
        )

        plot_hawkes_lambda_splits(
            close=close_all,
            lam=lam_all_online,   # or lam_test / lam_full
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            title=f"BTCUSDT 1D: Price & Hawkes Lambda (splits)",
            smooth_span=20,
        )

        # -------------------------
        # 3) Alpha SA ON TRAIN ONLY
        # -------------------------

        # Align everything to decision-time index (mu_all.index)
        # We'll pass only train slice
        close_train = close_all.reindex(idx_train)
        mu_train = mu_all.reindex(idx_train)
        sigma_train = sigma_all.reindex(idx_train)
        lam_train = lam_all_online.reindex(idx_train)

        res_sa = alpha_sensitivity_study(
            close=close_train,
            mu=mu_train,
            sigma=sigma_train,
            lam=lam_train,
            alpha_risk_grid=list(sig_cfg.alpha_risk_grid),
            position_cap=sig_cfg.position_cap,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
            bars_per_year=252,
        )
        print(res_sa.head(10))
        plot_alpha_sensitivity(res_sa, title=f"Alpha Risk SA on TRAIN | q={q}, tau={tau:.6f}")
        alpha_risk_star = float(res_sa.iloc[0]["alpha_risk"])
        print(f"quantile = {q}, Selected alpha_risk* = {alpha_risk_star} (train-best by Sharpe/Calmar)")

if __name__ == "__main__":
    main()