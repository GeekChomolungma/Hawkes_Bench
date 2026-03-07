from __future__ import annotations

import numpy as np
import pandas as pd


def build_unified_signal(expected_return: pd.Series, adjusted_risk: pd.Series, eps: float = 1e-12) -> pd.Series:
    er = expected_return.astype(float)
    rk = adjusted_risk.astype(float)
    return er / (rk + eps)


def signal_to_position(signal: pd.Series, cap: float = 1.0) -> pd.Series:
    return signal.clip(-float(cap), float(cap))


def build_position(expected_return: pd.Series, adjusted_risk: pd.Series, cap: float = 1.0) -> pd.Series:
    sig = build_unified_signal(expected_return, adjusted_risk)
    return signal_to_position(sig, cap=cap)


def build_stateful_all_in_position(
    expected_return: pd.Series,
    adjusted_risk: pd.Series,
    cap: float = 1.0,
    entry_threshold: float = 0.0,
) -> pd.Series:
    """
    Stateful execution for full-notional style trading.
    Position takes values in {-cap, 0, +cap}.

    Rules:
    - if signal > +threshold and current != +cap: go long
    - if signal < -threshold and current != -cap: go short
    - if |signal| <= threshold: flatten to 0
    - otherwise hold current position (no repeated same-side order)
    """
    sig = build_unified_signal(expected_return, adjusted_risk).astype(float)
    idx = sig.index
    out = np.zeros(len(sig), dtype=float)
    cur = 0.0
    c = float(cap)
    th = float(entry_threshold)

    for i, s in enumerate(sig.to_numpy()):
        if s > th:
            if cur != c:
                cur = c
        elif s < -th:
            if cur != -c:
                cur = -c
        else:
            cur = 0.0
        out[i] = cur

    return pd.Series(out, index=idx)
