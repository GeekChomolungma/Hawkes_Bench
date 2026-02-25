import numpy as np
import pandas as pd

def composite_risk(sigma: pd.Series, lam: pd.Series, alpha: float) -> pd.Series:
    # Risk_t = sigma_{t+1} * (1 + alpha * lambda_t)
    return sigma * (1.0 + alpha * lam)

def signal_er(mu: pd.Series, risk: pd.Series, eps: float = 1e-12) -> pd.Series:
    return mu / (risk + eps)