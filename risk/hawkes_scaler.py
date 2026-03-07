from __future__ import annotations

import pandas as pd


def apply_hawkes_risk_scaling(native_risk: pd.Series, lam: pd.Series, alpha_risk: float) -> pd.Series:
    native = native_risk.astype(float)
    lam_aligned = lam.reindex(native.index).fillna(0.0).astype(float)
    return native * (1.0 + float(alpha_risk) * lam_aligned)
