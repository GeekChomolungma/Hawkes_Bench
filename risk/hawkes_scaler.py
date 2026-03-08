from __future__ import annotations

import pandas as pd


def apply_hawkes_risk_scaling(native_risk: pd.Series, lam: pd.Series, alpha_risk: float) -> pd.Series:
    native = native_risk.astype(float) # ts, sigma_pred binded to pred_for_ts
    lam_aligned = lam.reindex(native.index).fillna(0.0).astype(float)

    # ts lam, which means the return in ts has been revealed
    # So use ts lam to scale t+1 risk(pred_for_ts's sigma_pred)
    return native * (1.0 + float(alpha_risk) * lam_aligned)
