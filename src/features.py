"""Time-series feature engineering."""
import pandas as pd

FEATURES = ["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_24", "lag_168", "rolling_mean_24"]

def make_features(df):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").set_index("timestamp")
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    out["lag_1"] = out["energy_kwh"].shift(1)
    out["lag_24"] = out["energy_kwh"].shift(24)
    out["lag_168"] = out["energy_kwh"].shift(168)
    out["rolling_mean_24"] = out["energy_kwh"].shift(1).rolling(24).mean()
    return out.dropna(subset=FEATURES + ["energy_kwh"]).reset_index()
