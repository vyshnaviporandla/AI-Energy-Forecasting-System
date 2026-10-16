"""Dataset utilities. The default dataset is clearly labeled synthetic demo data."""
import numpy as np
import pandas as pd
from src.config import DATA_PATH

def generate_demo_dataset(days=730, seed=42, save=True):
    """Create realistic-pattern hourly demo data; this is simulated, not utility-meter data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    day_of_year = idx.dayofyear.to_numpy()

    daily = 1.0 + 0.22 * np.sin(2 * np.pi * (hour - 7) / 24)
    morning_peak = 0.30 * np.exp(-((hour - 8) / 2.2) ** 2)
    evening_peak = 0.42 * np.exp(-((hour - 19) / 3.0) ** 2)
    weekend = np.where(dow >= 5, -0.12, 0.08)
    seasonal = 0.16 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    temperature = 25 + 8 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25) + 4 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 1.7, len(idx))
    cooling_load = np.maximum(temperature - 24, 0) * 0.035
    base = 95 * (daily + morning_peak + evening_peak + weekend + seasonal + cooling_load)
    trend = np.linspace(0, 7, len(idx))
    noise = rng.normal(0, 4.5, len(idx))
    energy = np.maximum(base + trend + noise, 8)
    df = pd.DataFrame({"timestamp": idx, "energy_kwh": np.round(energy, 3), "temperature_c": np.round(temperature, 2)})
    if save:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df

def load_dataset(path=DATA_PATH):
    if not path.exists():
        return generate_demo_dataset()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "energy_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "energy_kwh"])
    df = df[df["energy_kwh"] >= 0]
    df = df.set_index("timestamp").resample("h").mean(numeric_only=True)
    df["energy_kwh"] = df["energy_kwh"].interpolate(limit=3).ffill().bfill()
    if df.empty or df["energy_kwh"].isna().any():
        raise ValueError("No usable energy data after cleaning.")
    return df.reset_index()
