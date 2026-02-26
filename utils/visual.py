import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(close, returns, pred):
    # close and returns are indexed by all datetime
    # pred is indexed by decision times (subset of close/returns index)
    pred_df = pred.copy()

    # dicision time
    # pred index is decision time t_i
    pred_df["close_t"] = close.loc[pred_df.index]

    # pred_for_ts is the timestamp of the predicted return (t_{i+1})
    pred_df["real_return_next"] = returns.loc[pred_df["pred_for_ts"]].values

    # predicted next price = close_t * exp(predicted return) at decision time t_i for predicted return at t_{i+1}
    pred_df["pred_price_next"] = pred_df["close_t"] * np.exp(pred_df["mu"])

    # real next price
    pred_df["real_price_next"] = close.loc[pred_df["pred_for_ts"]].values

    # ==============================
    # Figure 1：price gt vs predicted next price
    # ==============================

    plt.figure(figsize=(14,6))

    plt.plot(close, label="Close Price", alpha=0.6)

    plt.plot(
        pred_df["pred_for_ts"],
        pred_df["pred_price_next"],
        label="Predicted Next Price (ARIMA)",
        linewidth=2
    )

    plt.title("Price vs ARIMA Predicted Next Price")
    plt.legend()
    plt.grid()
    plt.show()


    # ==============================
    # Figure 2：log return gt vs predicted return
    # ==============================

    plt.figure(figsize=(14,6))

    plt.plot(
        pred_df.index,
        pred_df["mu"],
        label="Predicted Return",
        alpha=0.8
    )

    plt.plot(
        pred_df.index,
        pred_df["real_return_next"],
        label="Real Next Return",
        alpha=0.6
    )

    plt.title("Predicted vs Real Next Returns")
    plt.legend()
    plt.grid()
    plt.show()


    # ==============================
    # Figure 3：volatility (sigma) over time
    # ==============================

    plt.figure(figsize=(14,4))

    plt.plot(pred_df.index, pred_df["sigma"], label="Predicted Volatility (GARCH)")
    plt.title("Predicted Volatility")
    plt.legend()
    plt.grid()
    plt.show()