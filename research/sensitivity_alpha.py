import pandas as pd

from signals.risk import composite_risk, signal_er
from signals.position import signal_to_position
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics

def alpha_sensitivity_study(
    close: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    lam: pd.Series,
    alpha_risk_grid: list[float],
    position_cap: float,
    fee_bps: float,
    slippage_bps: float,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """
    Sensitivity analysis over alpha_risk on TRAIN set only (caller should slice).

    Returns a DataFrame with metrics per alpha.
    """
    # Align all inputs to a common decision-time index
    idx = close.index.intersection(mu.index).intersection(sigma.index).intersection(lam.index)
    close = close.reindex(idx).astype(float)
    mu = mu.reindex(idx).astype(float)
    sigma = sigma.reindex(idx).astype(float)
    lam = lam.reindex(idx).astype(float)

    rows = []
    for a in alpha_risk_grid:
        risk = composite_risk(sigma=sigma, lam=lam, alpha=a)
        sig = signal_er(mu=mu, risk=risk)
        pos = signal_to_position(sig, cap=position_cap)    

        bt = run_backtest(close=close, position=pos, fee_bps=fee_bps, slippage_bps=slippage_bps)
        m = compute_metrics(bt, bars_per_year=bars_per_year)

        rows.append(
            {
                "alpha_risk": a,
                **m,
            }
        )

    res = pd.DataFrame(rows)

    # A common choice: rank by Sharpe then Calmar as tiebreaker
    res = res.sort_values(["sharpe", "calmar"], ascending=False).reset_index(drop=True)
    return res