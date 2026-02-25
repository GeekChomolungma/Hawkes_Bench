# backtest/engine.py
import numpy as np
import pandas as pd

def run_backtest(close: pd.Series, position: pd.Series, fee_bps=2.0, slippage_bps=1.0) -> pd.DataFrame:
    """
    simple backtest engine:
      - PnL_t = pos_t * r_{t+1} - cost_t
      - cost_t = abs(pos_t - pos_{t-1}) * (fee +slippage)
    """
    r = np.log(close).diff().shift(-1)  # next return
    pos = position.reindex(r.index).fillna(0.0)
    dpos = pos.diff().abs().fillna(0.0)

    cost = dpos * (fee_bps + slippage_bps) * 1e-4
    pnl = pos * r - cost

    out = pd.DataFrame({
        "ret_next": r,
        "pos": pos,
        "cost": cost,
        "pnl": pnl,
        "equity": (1.0 + pnl.fillna(0.0)).cumprod(),
    })
    return out.dropna()