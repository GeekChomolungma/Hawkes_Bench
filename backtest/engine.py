# backtest/engine.py
import numpy as np
import pandas as pd

def run_backtest(
    close: pd.Series,
    position: pd.Series,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest with strict timing:
      - position[t] is decided at time t using info available up to t
      - realized return uses next bar simple return: r_next[t] = close[t+1]/close[t] - 1
      - pnl[t] = position[t] * r_next[t] - turnover_cost[t]
      - turnover_cost[t] = abs(position[t] - position[t-1]) * (fee+slippage)
      - output index is settlement timestamp (t+1) for clearer visualization
    """

    # Align to common index (decision time index)
    idx = close.index.intersection(position.index)
    close = close.reindex(idx).astype(float)
    pos = position.reindex(idx).fillna(0.0).astype(float)

    # next simple return (t -> t+1), last point is NaN
    r_next = close.pct_change().shift(-1)

    # signed position change at decision time t (from t-1 to t).
    # At the first decision, previous position is assumed flat (0.0).
    dpos = pos.diff()
    if len(dpos) > 0:
        dpos.iloc[0] = float(pos.iloc[0])
    dpos = dpos.fillna(0.0)
    turnover_abs = dpos.abs()
    cost = turnover_abs * (fee_bps + slippage_bps) * 1e-4

    pnl = pos * r_next - cost

    out_decision = pd.DataFrame(
        {
            "close": close,
            "pos": pos,
            "dpos": dpos,
            "ret_next": r_next,
            "cost": cost,
            "pnl": pnl,
        },
        index=idx,
    )

    # Drop last decision row where ret_next is NaN (cannot realize),
    # then relabel each row to settlement timestamp (t+1).
    out = out_decision.iloc[:-1].copy()
    out["decision_ts"] = out.index
    out.index = idx[1:]
    out.index.name = "ts"
    out["close"] = close.reindex(out.index).astype(float)

    out["equity"] = (1.0 + out["pnl"].fillna(0.0)).cumprod()
    return out
