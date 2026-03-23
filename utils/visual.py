from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_GT = "tab:orange"
COLOR_PRED = "tab:blue"
COLOR_BAND = "lightblue"


def _get_target_x(index: pd.DatetimeIndex, anchor: pd.Series | pd.DataFrame, pred_for_ts: pd.Series | None = None) -> pd.Series:
    if pred_for_ts is not None:
        x = pd.to_datetime(pred_for_ts, utc=True, errors="coerce")
        return pd.Series(x, index=index)

    anchor_idx = anchor.index
    pos = anchor_idx.get_indexer(index)
    next_pos = pos + 1
    valid_shift = (pos >= 0) & (next_pos < len(anchor_idx))
    out = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    out.loc[valid_shift] = anchor_idx[next_pos[valid_shift]]
    return out


def plot_forecast_layer(
    close: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Close vs Forecast",
    out_path: str | None = None,
    band_low_col: str | None = None,
    band_high_col: str | None = None,
    band_label: str | None = None,
    z_score: float = 1.96,
) -> None:
    """
    Visualize close price against forecast projection on target timestamp (t+1).

    Default behavior:
    - median line uses price_pred_median if available, else close_t * exp(mu_pred)
    - band uses price_pred_lo/hi if available, else mu_pred +/- z*sigma_pred

    Custom quantile band (for black-box):
    - pass band_low_col and band_high_col as return-space quantiles (e.g. q10/q90)
      and they will be mapped to price by close_t * exp(return_quantile).
    """
    df = forecast_df.copy()
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)

    close_aligned = close.reindex(df.index).astype(float)
    df["close_t"] = close_aligned

    if "price_pred_median" not in df.columns:
        if "mu_pred" not in df.columns:
            raise ValueError("forecast_df requires price_pred_median or mu_pred")
        df["price_pred_median"] = df["close_t"] * np.exp(df["mu_pred"].astype(float))

    target_x = _get_target_x(
        index=df.index,
        anchor=close,
        pred_for_ts=df["pred_for_ts"] if "pred_for_ts" in df.columns else None,
    )

    lo_price = None
    hi_price = None
    if band_low_col and band_high_col and band_low_col in df.columns and band_high_col in df.columns:
        # custom return-space quantiles -> price-space band
        lo_price = df["close_t"] * np.exp(df[band_low_col].astype(float))
        hi_price = df["close_t"] * np.exp(df[band_high_col].astype(float))
        band_name = band_label or f"Pred Band ({band_low_col}-{band_high_col}, t+1)"
    elif "price_pred_lo" in df.columns and "price_pred_hi" in df.columns:
        lo_price = df["price_pred_lo"].astype(float)
        hi_price = df["price_pred_hi"].astype(float)
        band_name = band_label or "Pred Band (t+1)"
    elif "mu_pred" in df.columns and "sigma_pred" in df.columns:
        mu = df["mu_pred"].astype(float)
        sigma = df["sigma_pred"].astype(float)
        lo_price = df["close_t"] * np.exp(mu - float(z_score) * sigma)
        hi_price = df["close_t"] * np.exp(mu + float(z_score) * sigma)
        band_name = band_label or f"Pred Band (+/-{z_score:.2f}sigma, t+1)"
    else:
        band_name = band_label or "Pred Band"

    valid = target_x.notna() & df["price_pred_median"].notna()
    plt.figure(figsize=(14, 6))
    plt.plot(close.index, close.values, label="Close (GT)", alpha=0.8, color=COLOR_GT)
    plt.plot(
        target_x[valid],
        df.loc[valid, "price_pred_median"].values,
        label="Pred Price (median, t+1)",
        linewidth=2,
        color=COLOR_PRED,
    )

    if lo_price is not None and hi_price is not None:
        lo = lo_price[valid].astype(float)
        hi = hi_price[valid].astype(float)
        plt.fill_between(target_x[valid], lo, hi, alpha=0.25, label=band_name, color=COLOR_BAND)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    # plt.show()  # disabled to avoid blocking in batch runs
    plt.close()


def plot_return_target_layer(
    returns: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Return Forecast vs Next Return (GT)",
    out_path: str | None = None,
    z_score: float = 1.96,
    band_low_col: str | None = None,
    band_high_col: str | None = None,
    band_label: str | None = None,
) -> None:
    """
    Compare model target directly:
    - predicted next-bar return (mu_pred or q50)
    - realized next-bar return r_{t+1}

    Custom band option for black-box: pass band_low_col / band_high_col
    such as q10 / q90.
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

    target_x = _get_target_x(
        index=df.index,
        anchor=returns,
        pred_for_ts=df["pred_for_ts"] if "pred_for_ts" in df.columns else None,
    )

    if band_low_col and band_high_col and band_low_col in df.columns and band_high_col in df.columns:
        band_lo = df[band_low_col].astype(float)
        band_hi = df[band_high_col].astype(float)
        resolved_band_label = band_label or f"Pred Band ({band_low_col}-{band_high_col})"
    elif "ret_pred_lo" in df.columns and "ret_pred_hi" in df.columns:
        band_lo = df["ret_pred_lo"].astype(float)
        band_hi = df["ret_pred_hi"].astype(float)
        resolved_band_label = band_label or "Pred Band"
    elif "mu_pred" in df.columns and "sigma_pred" in df.columns:
        mu = df["mu_pred"].astype(float)
        sigma = df["sigma_pred"].astype(float)
        band_lo = mu - float(z_score) * sigma
        band_hi = mu + float(z_score) * sigma
        resolved_band_label = band_label or f"Pred Band (+/-{z_score:.2f}sigma)"
    else:
        band_lo = None
        band_hi = None
        resolved_band_label = band_label or "Pred Band"

    valid = pred.notna() & real_next.notna() & target_x.notna()
    pred = pred[valid]
    real_next = real_next[valid]
    target_x = target_x[valid]
    if band_lo is not None and band_hi is not None:
        band_lo = band_lo[valid]
        band_hi = band_hi[valid]

    fig = plt.figure(figsize=(15, 9))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    ax1.plot(target_x, pred.values, label=f"Pred Next Return ({pred_col})", linewidth=1.8, alpha=0.9, color=COLOR_PRED)
    ax1.plot(target_x, real_next.values, label="Real Next Return (GT)", linewidth=1.4, alpha=0.85, color=COLOR_GT)
    if band_lo is not None and band_hi is not None:
        ax1.fill_between(target_x, band_lo.values, band_hi.values, alpha=0.25, label=resolved_band_label, color=COLOR_BAND)
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(pred.values, real_next.values, s=14, alpha=0.55, label="points")
    lo = float(min(pred.min(), real_next.min()))
    hi = float(max(pred.max(), real_next.max()))
    ax2.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, label="45-degree line")
    ax2.set_xlabel("Predicted next return")
    ax2.set_ylabel("Real next return")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    # plt.show()  # disabled to avoid blocking in batch runs
    plt.close()


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
    settle_idx = df.index
    close_full = close.astype(float)

    # Use decision-time turnover for markers when available.
    if "decision_ts" in df.columns:
        decision_ts = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
        marker_ts = pd.DatetimeIndex(decision_ts.dropna())
        dpos = pd.Series(df["dpos"].to_numpy(), index=marker_ts).sort_index()
        pos_marker = pd.Series(df["pos"].to_numpy(), index=marker_ts).sort_index()
    else:
        # Fallback to settlement-time inference.
        marker_ts = settle_idx
        dpos = df["pos"].diff()
        if len(dpos) > 0:
            dpos.iloc[0] = df["pos"].iloc[0]
        dpos = dpos.fillna(0.0)
        pos_marker = df["pos"].reindex(marker_ts).fillna(0.0).astype(float)

    # Marker policy:
    # Use regime transitions to distinguish open/close actions.
    # - Open Long:   <=0 -> >0
    # - Open Short:  >=0 -> <0
    # - Close Short: <0  -> >=0
    # - Close Long:  >0  -> <=0
    pos_values = pos_marker.fillna(0.0).astype(float)
    prev = pos_values.shift(1).fillna(0.0)
    open_long_idx = pos_values.index[(pos_values > 0) & (prev <= 0)]
    open_short_idx = pos_values.index[(pos_values < 0) & (prev >= 0)]
    close_short_idx = pos_values.index[(pos_values >= 0) & (prev < 0)]
    close_long_idx = pos_values.index[(pos_values <= 0) & (prev > 0)]

    fig = plt.figure(figsize=(15, 9))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    ax1.plot(close_full.index, close_full.values, label="Close", color="tab:blue", linewidth=1.4)
    if len(open_long_idx) > 0:
        ax1.scatter(open_long_idx, close_full.reindex(open_long_idx).values, marker="^", s=50, label="Open Long", color="tab:green")
    if len(open_short_idx) > 0:
        ax1.scatter(open_short_idx, close_full.reindex(open_short_idx).values, marker="v", s=50, label="Open Short", color="tab:red")
    if len(close_short_idx) > 0:
        y_cover = close_full.reindex(close_short_idx).astype(float) * 1.006
        ax1.scatter(
            close_short_idx,
            y_cover.values,
            marker="o",
            s=34,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.2,
            label="Close Short",
        )
    if len(close_long_idx) > 0:
        y_flat_long = close_full.reindex(close_long_idx).astype(float) * 0.994
        ax1.scatter(
            close_long_idx,
            y_flat_long.values,
            marker="o",
            s=34,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.2,
            label="Close Long",
        )

    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    # Prepend initial equity point at first decision timestamp with value 1.0.
    if "decision_ts" in df.columns:
        first_decision_ts = pd.to_datetime(df["decision_ts"].iloc[0], utc=True, errors="coerce")
    else:
        first_decision_ts = close_full.index.min()

    eq_idx = settle_idx
    eq_val = df["equity"].astype(float)
    if pd.notna(first_decision_ts):
        eq_idx = pd.DatetimeIndex([first_decision_ts]).append(pd.DatetimeIndex(settle_idx))
        eq_val = pd.concat([pd.Series([1.0], index=[first_decision_ts]), eq_val])

    ax2.plot(eq_idx, eq_val.values, label="Equity", color="tab:orange", linewidth=1.8)

    base0 = close_full.loc[first_decision_ts] if pd.notna(first_decision_ts) and first_decision_ts in close_full.index else close_full.iloc[0]
    buy_hold_settle = (close_full.reindex(settle_idx).astype(float) / (base0 + 1e-12)).astype(float)
    if pd.notna(first_decision_ts):
        buy_hold = pd.concat([pd.Series([1.0], index=[first_decision_ts]), buy_hold_settle])
        bh_idx = eq_idx
    else:
        buy_hold = buy_hold_settle
        bh_idx = settle_idx
    ax2.plot(bh_idx, buy_hold.values, label="Buy & Hold", color="tab:gray", linewidth=1.4, linestyle="--")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    # plt.show()  # disabled to avoid blocking in batch runs
    plt.close()


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
    # plt.show()  # disabled to avoid blocking in batch runs
    plt.close(fig)


