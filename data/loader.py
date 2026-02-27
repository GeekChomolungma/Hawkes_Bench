import pandas as pd

OHLVC_COLS = ["open", "high", "low", "close", "volume"]

def load_kline_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "starttime" in df.columns:
        ts = pd.to_datetime(df["starttime"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["eventtime"], unit="ms", utc=True)

    df["ts"] = ts
    df = df.sort_values("ts").set_index("ts")

    for c in OHLVC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df

def time_split_df(
    df: pd.DataFrame,
    ratios=(0.7, 0.1, 0.2),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split (NO shuffle).
    ratios must sum to 1.0 (approximately).
    """
    r_train, r_val, r_test = ratios
    s = r_train + r_val + r_test
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={s})")

    n = len(df)
    if n < 100:
        raise ValueError("Too few rows to split.")

    n_train = int(n * r_train)
    n_val = int(n * r_val)
    # remainder goes to test to avoid empty tail due to rounding
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split resulted in empty part: n={n}, ratios={ratios}")

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()
    return df_train, df_val, df_test