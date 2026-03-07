from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_layer(
    close: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Close vs Forecast",
    out_path: str | None = None,
) -> None:
    """
    Visualize close price against forecast projection at decision times.
    Expected columns in forecast_df: ts, close_t, and either price_pred_median
    or mu_pred (which will be mapped to price projection).
    Optional columns: price_pred_lo, price_pred_hi
    """
    df = forecast_df.copy()
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)

    if "price_pred_median" not in df.columns:
        if "mu_pred" not in df.columns:
            raise ValueError("forecast_df requires price_pred_median or mu_pred")
        
        # close ground-truth at decision time t
        df["close_t"] = close.reindex(df.index).astype(float)

        # ATTENTION: 
        # Using the predicted return rate in conjunction with the true value of the previous close to reconstruct a predicted close value for the next moment presents a problem: 
        # when the return rate is very small, the model's close will be too tight, making the overall prediction appear very accurate.
        df["price_pred_median"] = df["close_t"] * np.exp(df["mu_pred"].astype(float))

    plt.figure(figsize=(14, 6))
    plt.plot(close.index, close.values, label="Close (GT)", alpha=0.65)
    plt.plot(df.index, df["price_pred_median"].values, label="Pred Price (median)", linewidth=2)

    if "price_pred_lo" in df.columns and "price_pred_hi" in df.columns:
        lo = df["price_pred_lo"].astype(float)
        hi = df["price_pred_hi"].astype(float)
        plt.fill_between(df.index, lo, hi, alpha=0.2, label="Pred Band")

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.show()


def plot_return_target_layer(
    returns: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Return Forecast vs Next Return (GT)",
    out_path: str | None = None,
) -> None:
    """
    Compare model target directly:
    - predicted next-bar return (mu_pred or q50)
    - realized next-bar return r_{t+1}
    """
    df = forecast_df.copy()
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)

    pred_col = "mu_pred" if "mu_pred" in df.columns else ("q50" if "q50" in df.columns else None)
    if pred_col is None:
        raise ValueError("forecast_df requires mu_pred or q50 for return target plot")

    pred = df[pred_col].astype(float)
    real_next = returns.shift(-1).reindex(df.index).astype(float)
    valid = pred.notna() & real_next.notna()
    pred = pred[valid]
    real_next = real_next[valid]

    fig = plt.figure(figsize=(15, 9))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    ax1.plot(pred.index, pred.values, label=f"Pred Next Return ({pred_col})", linewidth=1.8, alpha=0.9)
    ax1.plot(real_next.index, real_next.values, label="Real Next Return (GT)", linewidth=1.4, alpha=0.75)
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(pred.values, real_next.values, s=14, alpha=0.55, label="points")
    lo = float(min(pred.min(), real_next.min()))
    hi = float(max(pred.max(), real_next.max()))
    ax2.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, label="45° line")
    ax2.set_xlabel("Predicted next return")
    ax2.set_ylabel("Real next return")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.show()


def plot_backtest_layer(
    close: pd.Series,
    bt: pd.DataFrame,
    title: str = "Backtest (Price + Buy/Sell + Equity)",
    out_path: str | None = None,
) -> None:
    """
    Trading-view-like buy/sell markers based on position changes.
    bt must include: pos, dpos, equity and share the same index as decision times.
    equity is the strategy equity index, starting at 1.0.
    """
    df = bt.copy()
    idx = df.index
    px = close.reindex(idx).astype(float)

    dpos = df["pos"].diff().fillna(0.0)
    buy_idx = idx[dpos > 0]
    sell_idx = idx[dpos < 0]

    fig = plt.figure(figsize=(15, 9))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    ax1.plot(idx, px.values, label="Close", color="tab:blue", linewidth=1.4)
    if len(buy_idx) > 0:
        ax1.scatter(buy_idx, px.reindex(buy_idx).values, marker="^", s=50, label="Buy", color="tab:green")
    if len(sell_idx) > 0:
        ax1.scatter(sell_idx, px.reindex(sell_idx).values, marker="v", s=50, label="Sell", color="tab:red")

    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(idx, df["equity"].values, label="Equity", color="tab:orange", linewidth=1.8)
    buy_hold = (px / (px.iloc[0] + 1e-12)).astype(float)
    ax2.plot(idx, buy_hold.values, label="Buy & Hold", color="tab:gray", linewidth=1.4, linestyle="--")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.show()


def plot_hawkes_lambda_splits(
    close: pd.Series,
    lam: pd.Series,
    idx_train: pd.DatetimeIndex,
    idx_val: pd.DatetimeIndex,
    idx_test: pd.DatetimeIndex,
    title: str = "Price and Hawkes Lambda by Splits",
    smooth_span: int = 20,
    figsize=(16, 9),
):
    idx_all = pd.DatetimeIndex([])
    for seg in (idx_train, idx_val, idx_test):
        if seg is not None:
            idx_all = idx_all.union(pd.DatetimeIndex(seg))
    if len(idx_all) == 0:
        raise ValueError("Empty split indices.")

    idx_all = idx_all.sort_values()
    df = pd.DataFrame(index=idx_all)
    df["close"] = close.reindex(idx_all).astype(float)
    df["lam"] = lam.reindex(idx_all).astype(float)
    df["lam_smooth"] = df["lam"].ewm(span=smooth_span, adjust=False).mean()

    c_train, c_val, c_test = "tab:blue", "tab:orange", "tab:green"

    def _plot_segment(ax, y: pd.Series, seg_idx: pd.DatetimeIndex, color: str, label: str, **kwargs):
        seg = pd.DatetimeIndex(seg_idx)
        if len(seg) == 0:
            return
        ys = y.reindex(seg)
        ax.plot(seg, ys.values, color=color, label=label, **kwargs)

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    _plot_segment(ax1, df["close"], idx_train, c_train, "train")
    _plot_segment(ax1, df["close"], idx_val, c_val, "val")
    _plot_segment(ax1, df["close"], idx_test, c_test, "test")
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    _plot_segment(ax2, df["lam_smooth"], idx_train, c_train, "lambda train")
    _plot_segment(ax2, df["lam_smooth"], idx_val, c_val, "lambda val")
    _plot_segment(ax2, df["lam_smooth"], idx_test, c_test, "lambda test")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
