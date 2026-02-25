import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(close, returns, pred):

    # ===== align data =====
    # pred index is decision time t_i
    # pred_for_ts is the timestamp of the predicted return (t_{i+1})

    df = pred.copy()

    # dicision time
    df["close_t"] = close.loc[df.index]

    # true return r_{i+1}
    df["real_return_next"] = returns.loc[df["pred_for_ts"]].values

    # predicted next price = close_t * exp(predicted return) at decision time t_i for predicted return at t_{i+1}
    df["pred_price_next"] = df["close_t"] * np.exp(df["mu"])

    # real next price
    df["real_price_next"] = close.loc[df["pred_for_ts"]].values

    # ==============================
    # Figure 1：price gt vs predicted next price
    # ==============================

    plt.figure(figsize=(14,6))

    plt.plot(close, label="Close Price", alpha=0.6)

    plt.plot(
        df["pred_for_ts"],
        df["pred_price_next"],
        label="Predicted Next Price (ARIMA)",
        linewidth=2
    )

    plt.title("Price vs ARIMA Predicted Next Price")
    plt.legend()
    plt.grid()
    plt.show()


    # ==============================
    # Figure 2：return gt vs predicted return
    # ==============================

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["mu"],
        label="Predicted Return",
        alpha=0.8
    )

    plt.plot(
        df.index,
        df["real_return_next"],
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

    plt.plot(df.index, df["sigma"], label="Predicted Volatility (GARCH)")
    plt.title("Predicted Volatility")
    plt.legend()
    plt.grid()
    plt.show()