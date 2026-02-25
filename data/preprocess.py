import numpy as np
import pandas as pd

def compute_log_return(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    return r.dropna()

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    # filter out non-final rows if 'isfinal' column exists
    if "isfinal" in df.columns:
        df = df[df["isfinal"].astype(str).str.lower().isin(["true", "1"])]
    return df