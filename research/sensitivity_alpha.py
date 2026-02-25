import pandas as pd

from signals.risk import composite_risk, signal_er
from signals.position import signal_to_position
from backtest.engine import run_backtest

def alpha_sensitivity_study(
    close: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    lam: pd.Series,
    alpha_grid: list[float],
    position_cap: float,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    rows = []
    for a in alpha_grid:
        risk = composite_risk(sigma=sigma, lam=lam, alpha=a)
        sig = signal_er(mu=mu, risk=risk)
        pos = signal_to_position(sig, cap=position_cap)
        bt = run_backtest(close=close, position=pos, fee_bps=fee_bps, slippage_bps=slippage_bps)

        # 你后面可以换成更严谨 metrics
        total = bt["equity"].iloc[-1] - 1.0
        vol = bt["pnl"].std(ddof=1) * (252 ** 0.5)
        sharpe = (bt["pnl"].mean() * 252) / (bt["pnl"].std(ddof=1) * (252 ** 0.5) + 1e-12)

        rows.append({"alpha": a, "total_return": total, "ann_vol": vol, "sharpe": sharpe})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)