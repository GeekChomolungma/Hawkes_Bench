from __future__ import annotations

import pandas as pd

from hawkes.core import HawkesExpParams, hawkes_lambda_suite_fixed_theta
from hawkes.threshold import choose_threshold_by_quantile, fit_hawkes_for_threshold


def fit_hawkes_theta_from_train(
    returns_train: pd.Series,
    quantile: float = 0.9,
    signed: bool = True,
    unit: str = "D",
) -> tuple[float, dict[str, HawkesExpParams]]:
    tau = choose_threshold_by_quantile(returns_train, q=quantile)
    fitted = fit_hawkes_for_threshold(returns_train, tau=tau, signed=signed, unit=unit)

    theta: dict[str, HawkesExpParams] = {}
    for key, model in fitted.items():
        if model is None:
            continue
        theta[key] = model.params_
    return tau, theta


def hawkes_lambda_online(
    returns: pd.Series,
    index: pd.DatetimeIndex,
    origin: pd.Timestamp,
    tau: float,
    theta_by_key: dict[str, HawkesExpParams],
    signed: bool = True,
    unit: str = "D",
) -> pd.Series:
    return hawkes_lambda_suite_fixed_theta(
        r=returns,
        index=index,
        origin=origin,
        tau=tau,
        theta_by_key=theta_by_key,
        signed=signed,
        unit=unit,
    )
