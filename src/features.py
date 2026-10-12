import pandas as pd

FEATURES = ["z", "roll_mean", "roll_std", "diff"]
REQUIRED = ["TimeStamp", "Value", "Series"]

def make_features(df, window=12):
    """Per-series normalised features (series have very different scales)."""
    out = df.sort_values(["Series", "TimeStamp"]).reset_index(drop=True).copy()
    g = out.groupby("Series")["Value"]
    sd = g.transform("std").replace(0, 1).fillna(1)
    out["z"] = (out["Value"] - g.transform("mean")) / sd
    zg = out.groupby("Series")["z"]
    out["roll_mean"] = zg.transform(lambda s: s.rolling(window, min_periods=1).mean())
    out["roll_std"] = zg.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
    out["diff"] = zg.diff().fillna(0)
    return out
