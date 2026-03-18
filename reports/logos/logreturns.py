import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_crypto_log_returns(
    n_points: int = 500,
    seed: int = 42,
    base_vol: float = 0.012,
    shock_prob: float = 0.03,
    shock_scale: float = 0.05,
):
    """
    Generate a synthetic crypto-style daily log return series.
    It includes:
      - mild volatility clustering
      - occasional jumps / shocks
      - weak local drift
    """
    rng = np.random.default_rng(seed)

    returns = np.zeros(n_points)
    vol = np.zeros(n_points)

    vol[0] = base_vol
    for t in range(1, n_points):
        # simple volatility clustering
        vol[t] = 0.92 * vol[t - 1] + 0.08 * (
            base_vol + 0.8 * abs(returns[t - 1])
        )

        # weak time-varying drift
        drift = 0.0005 * np.sin(2 * np.pi * t / 120.0)

        # heavy-tailed innovation
        eps = rng.standard_t(df=5) * vol[t]

        # occasional market shock
        jump = 0.0
        if rng.random() < shock_prob:
            jump = rng.normal(0.0, shock_scale)

        returns[t] = drift + eps + jump

    return returns


def build_forecast_demo(
    real_returns: np.ndarray,
    rolling_window: int = 20,
    z_value: float = 1.96,
):
    """
    Build a demo forecast:
      - median prediction: rolling mean shifted by 1 step
      - band: mu ± z * rolling std
    """
    s = pd.Series(real_returns)

    mu_pred = s.rolling(rolling_window).mean().shift(1)
    sigma_pred = s.rolling(rolling_window).std(ddof=0).shift(1)

    # fill initial NaNs for display
    mu_pred = mu_pred.bfill()
    sigma_pred = sigma_pred.bfill()

    lower = mu_pred - z_value * sigma_pred
    upper = mu_pred + z_value * sigma_pred

    return mu_pred.to_numpy(), lower.to_numpy(), upper.to_numpy()


def discretize_returns_by_threshold(
    real_returns: np.ndarray,
    tau: float,
    signed: bool = True,
):
    """
    Convert continuous log returns into a discrete event series using threshold tau.

    signed=True:
        +1 if r_t >  tau
        -1 if r_t < -tau
         0 otherwise

    signed=False:
         1 if |r_t| > tau
         0 otherwise
    """
    if signed:
        events = np.zeros_like(real_returns, dtype=int)
        events[real_returns > tau] = 1
        events[real_returns < -tau] = -1
    else:
        events = (np.abs(real_returns) > tau).astype(int)

    return events


def build_hawkes_intensity_from_events(
    events: np.ndarray,
    mu: float = 0.05,
    alpha: float = 0.35,
    beta: float = 0.25,
    signed_mode: str = "abs",
):
    """
    Build a simple discrete-time Hawkes-style intensity sequence from an event series.

    Parameters
    ----------
    events : np.ndarray
        Event sequence, e.g. {-1,0,1} or {0,1}.
    mu : float
        Baseline intensity.
    alpha : float
        Excitation strength.
    beta : float
        Decay rate in discrete time.
    signed_mode : str
        "abs"  -> use abs(events), i.e. both +1 and -1 excite intensity
        "pos"  -> only positive events excite
        "raw"  -> use raw event values directly (less common)

    Returns
    -------
    lam : np.ndarray
        Hawkes-style intensity sequence.
    """
    lam = np.zeros(len(events), dtype=float)

    if signed_mode == "abs":
        trigger = np.abs(events).astype(float)
    elif signed_mode == "pos":
        trigger = (events > 0).astype(float)
    elif signed_mode == "raw":
        trigger = events.astype(float)
    else:
        raise ValueError("signed_mode must be one of {'abs', 'pos', 'raw'}")

    lam[0] = mu + alpha * trigger[0]

    decay = np.exp(-beta)
    for t in range(1, len(events)):
        # discrete-time Hawkes-style recursion:
        # lambda_t = mu + exp(-beta) * (lambda_{t-1} - mu) + alpha * event_{t-1}
        lam[t] = mu + decay * (lam[t - 1] - mu) + alpha * trigger[t - 1]

    return lam


def plot_log_return_series(
    dates: pd.DatetimeIndex,
    real_returns: np.ndarray,
    figsize=(10, 2.2),
):
    """
    Plot screenshot-1 style deterministic log-return series.
    """
    plt.figure(figsize=figsize)
    plt.plot(dates, real_returns, linewidth=1.2)
    plt.title("Synthetic Crypto Log Return Series")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.show()

def plot_event_series(
    dates: pd.DatetimeIndex,
    events: np.ndarray,
    tau: float,
    signed: bool = True,
    figsize=(12, 2.4),
):
    """
    Plot discretized event series after thresholding returns by tau.
    """
    plt.figure(figsize=figsize)

    if signed:
        plt.step(dates, events, where="mid", linewidth=1.2)
        plt.yticks([-1, 0, 1], ["Negative Event", "No Event", "Positive Event"])
        plt.ylim(-1.5, 1.5)
        plt.title(f"Discretized Event Series (signed, tau={tau:.4f})")
    else:
        plt.step(dates, events, where="mid", linewidth=1.2)
        plt.yticks([0, 1], ["No Event", "Event"])
        plt.ylim(-0.2, 1.2)
        plt.title(f"Discretized Event Series (unsigned, tau={tau:.4f})")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_forecast_band(
    dates: pd.DatetimeIndex,
    real_returns: np.ndarray,
    mu_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "BTCUSDT 1d | Forecasted Return with Quantile-style Band",
    figsize=(15, 4.8),
):
    """
    Plot screenshot-2 style figure:
      - median prediction line
      - real return line
      - uncertainty band
    """
    plt.figure(figsize=figsize)

    plt.plot(dates, mu_pred, label="Pred Median Return", linewidth=1.6)
    # plt.plot(dates, real_returns, label="Real Next Return (GT)", linewidth=1.2, alpha=0.9)
    plt.fill_between(
        dates,
        lower,
        upper,
        alpha=0.2,
        label="Pred Band (approx. quantile interval)"
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend(loc="lower left")
    # plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_hawkes_intensity(
    dates: pd.DatetimeIndex,
    lam: np.ndarray,
    events: np.ndarray | None = None,
    tau: float | None = None,
    figsize=(12, 2.8),
):
    """
    Plot Hawkes-style intensity sequence.
    Optionally overlay event markers.
    """
    plt.figure(figsize=figsize)
    plt.plot(dates, lam, linewidth=1.5, label="Hawkes Intensity $\\lambda_t$")

    if events is not None:
        idx = np.where(np.abs(events) > 0)[0]
        if len(idx) > 0:
            plt.scatter(
                dates[idx],
                lam[idx],
                s=14,
                alpha=0.8,
                label="Event Times",
                zorder=3,
            )

    title = "Hawkes-style Intensity Sequence"
    if tau is not None:
        title += f" (tau={tau:.4f})"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) generate synthetic crypto log return data
    n_points = 600
    dates = pd.date_range("2024-06-01", periods=n_points, freq="D")
    real_returns = generate_crypto_log_returns(n_points=n_points, seed=123)

    # 1.1) discretize returns into events with threshold tau
    tau = 0.03
    event_series = discretize_returns_by_threshold(
        real_returns,
        tau=tau,
        signed=True,   # True: {-1,0,1}, False: {0,1}
    )

    # 1.2) build a Hawkes-style intensity sequence from the event series
    lambda_series = build_hawkes_intensity_from_events(
        event_series,
        mu=0.05,
        alpha=0.30,
        beta=0.35,
        signed_mode="abs",   # both positive/negative shocks increase risk
    )

    # 2) build demo forecast band and median line
    mu_pred, lower, upper = build_forecast_demo(
        real_returns,
        rolling_window=20,
        z_value=1.96,
    )

    # 3) plot screenshot-1 style log return figure
    plot_log_return_series(dates, real_returns)
    
    # 3.1) plot discretized event series
    plot_event_series(
        dates,
        event_series,
        tau=tau,
        signed=True,
    )

    # 3.2) plot Hawkes-style intensity sequence
    plot_hawkes_intensity(
        dates,
        lambda_series,
        events=event_series,
        tau=tau,
    )

    # 4) plot screenshot-2 style forecast figure
    plot_forecast_band(
        dates,
        real_returns,
        mu_pred,
        lower,
        upper,
        title="BTCUSDT 1d | White-box Return Target (Demo)",
    )