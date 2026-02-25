# data/loader.py
import pandas as pd

OHLVC_COLS = ["open", "high", "low", "close", "volume"]

def load_kline_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # eventtime / starttime is in ms, convert to datetime and set as index
    if "starttime" in df.columns:
        ts = pd.to_datetime(df["starttime"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["eventtime"], unit="ms", utc=True)

    df["ts"] = ts
    df = df.sort_values("ts").set_index("ts")

    # covert OHLVC columns to numeric, coerce errors to NaN, and drop rows with NaN in 'close'
    for c in OHLVC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df