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
      - realized return uses next bar: r_next[t] = log(close[t+1]/close[t])
      - pnl[t] = position[t] * r_next[t] - turnover_cost[t]
      - turnover_cost[t] = abs(position[t] - position[t-1]) * (fee+slippage)
    """

    # Align to common index (decision time index)
    idx = close.index.intersection(position.index)
    close = close.reindex(idx).astype(float)
    pos = position.reindex(idx).fillna(0.0).astype(float)

    # next return (t -> t+1), last point is NaN
    r_next = np.log(close).diff().shift(-1)

    # turnover uses change at time t (from t-1 to t)
    dpos = pos.diff().abs().fillna(0.0)
    cost = dpos * (fee_bps + slippage_bps) * 1e-4

    pnl = pos * r_next - cost

    out = pd.DataFrame(
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

    # drop last row where ret_next is NaN (cannot realize)
    out = out.iloc[:-1].copy()

    out["equity"] = (1.0 + out["pnl"].fillna(0.0)).cumprod()
    return out