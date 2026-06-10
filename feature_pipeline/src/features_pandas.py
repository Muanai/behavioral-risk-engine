import pandas as pd


# Pandas reference implementation (readable, slower)
def rolling_features(df, window):
    return (
        df
        .groupby("user_id")["bill_amt"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
