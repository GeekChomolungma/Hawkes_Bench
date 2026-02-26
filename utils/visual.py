import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_hawkes_lambda_suite(
    close: pd.Series,
    returns: pd.Series,
    pred: pd.DataFrame,     # indexed by decision time, contains pred_for_ts
    lam: pd.Series,         # indexed by decision time (same as mu.index)
    tau: float,
    q: float,
    signed: bool = True,
    smooth_span: int = 20,  # EWMA smoothing for readability
):
    """
    Suite of plots for Hawkes intensity (lambda) that are thesis-friendly.

    Alignment:
      - pred.index: decision time t_i
      - pred["pred_for_ts"]: target time t_{i+1}
      - lam is aligned to decision time t_i (no look-ahead)
      - returns is aligned to full timeline; returns[t] is r_t
    """

    # ----- align everything onto decision time index -----
    idx = pred.index
    df = pd.DataFrame(index=idx).copy()
    df["lam"] = lam.reindex(idx).astype(float)
    df["lam_smooth"] = df["lam"].ewm(span=smooth_span, adjust=False).mean()

    df["close"] = close.reindex(idx).astype(float)

    # realized next return at t_{i+1}
    df["pred_for_ts"] = pred["pred_for_ts"].values
    df["r_next"] = returns.reindex(df["pred_for_ts"]).values

    # event indicators for next step (for "predictive" diagnostics)
    if signed:
        df["event_next_pos"] = (df["r_next"] > +tau).astype(int)
        df["event_next_neg"] = (df["r_next"] < -tau).astype(int)
        df["event_next_abs"] = ((np.abs(df["r_next"]) > tau)).astype(int)
    else:
        df["event_next_abs"] = ((np.abs(df["r_next"]) > tau)).astype(int)

    # =========================
    # Plot A: Price + Lambda (two panels)
    # =========================
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df["close"])
    ax1.set_title(f"Close Price (q={q}, tau={tau:.6f})")
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df.index, df["lam"], alpha=0.35, label="lambda (raw)")
    ax2.plot(df.index, df["lam_smooth"], linewidth=2, label=f"lambda (EWMA span={smooth_span})")
    ax2.set_title("Hawkes Conditional Intensity (Risk Temperature)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # Plot B: Lambda with next-step extreme events markers
    # (thesis-friendly: shows whether lambda spikes before events)
    # =========================
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df["lam_smooth"], linewidth=2, label="lambda (smoothed)")
    # mark next-step abs extreme events
    ev_idx = df.index[df["event_next_abs"] == 1]
    plt.scatter(ev_idx, df.loc[ev_idx, "lam_smooth"], marker="x", s=30, label="next-step extreme event")
    plt.title(f"Lambda and Next-Step Extreme Events (q={q}, |r_next|>tau)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # Plot C: "Does lambda predict large moves?" (binning / calibration-like plot)
    # bucket lambda into quantiles and show event rate
    # =========================
    try:
        df["lam_bucket"] = pd.qcut(df["lam"], q=10, duplicates="drop")
        grp = df.groupby("lam_bucket")
        event_rate = grp["event_next_abs"].mean()
        lam_mid = grp["lam"].mean()

        plt.figure(figsize=(10, 5))
        plt.plot(lam_mid.values, event_rate.values, marker="o")
        plt.title("Event Rate vs Lambda Level (Binned)")
        plt.xlabel("Average lambda in bucket")
        plt.ylabel("P(next-step extreme event)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[plot_hawkes_lambda_suite] qcut failed: {e}")

    # =========================
    # Plot D: Compare distributions of lambda conditioned on next-step event
    # =========================
    lam_event = df.loc[df["event_next_abs"] == 1, "lam"]
    lam_noev = df.loc[df["event_next_abs"] == 0, "lam"]

    plt.figure(figsize=(12, 4))
    plt.hist(lam_noev.values, bins=50, alpha=0.6, label="no event next-step")
    plt.hist(lam_event.values, bins=50, alpha=0.6, label="event next-step")
    plt.title("Lambda Distribution: next-step event vs no-event")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: print simple stats for paper
    if len(lam_event) > 0:
        print(f"[q={q}] tau={tau:.6f}  "
              f"lambda mean(event)={lam_event.mean():.4f}, mean(no-event)={lam_noev.mean():.4f}, "
              f"event_rate={df['event_next_abs'].mean():.4f}")