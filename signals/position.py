import numpy as np
import pandas as pd

def signal_to_position(sig: pd.Series, cap: float = 1.0) -> pd.Series:
    # this is the simple mapping: pos_t = clip(sig_t, -cap, cap)
    pos = sig.clip(-cap, cap)
    return pos