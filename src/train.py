"""Train, evaluate, save a model, and create reproducible visual outputs."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import MODEL_DIR, OUTPUT_DIR, IMAGE_DIR, MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, FORECAST_PATH
from src.data import load_dataset
from src.features import make_features, FEATURES

def train():
    for folder in [MODEL_DIR, OUTPUT_DIR, IMAGE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    raw = load_dataset()
    featured = make_features(raw)
    split = int(len(featured) * 0.8)
    train_df, test_df = featured.iloc[:split], featured.iloc[split:]
    X_train, y_train = train_df[FEATURES], train_df["energy_kwh"]
    X_test, y_test = test_df[FEATURES], test_df["energy_kwh"]

    model = RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = np.maximum(model.predict(X_test), 0)
    # A simple persistence baseline predicts the previous hour's usage.
    baseline = test_df["lag_1"].to_numpy()
    metrics = {
        "project": "AI-Powered Energy Consumption Forecasting System",
        "dataset_type": "synthetic demo dataset" if not (raw["timestamp"].min() < pd.Timestamp("2024-01-01")) else "user-provided or custom dataset",
        "rows_used": int(len(featured)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split": "chronological 80/20 (no random shuffle)",
        "model": "RandomForestRegressor",
        "mae_kwh": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse_kwh": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "baseline_mae_kwh": round(float(mean_absolute_error(y_test, baseline)), 3),
        "note": "Demo results are not a utility-grade guarantee. Synthetic data is simulated."
    }
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    results = test_df[["timestamp", "energy_kwh"]].copy()
    results["predicted_kwh"] = pred
    results["baseline_lag1_kwh"] = baseline
    results.to_csv(PREDICTIONS_PATH, index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(results["timestamp"].tail(24*7), results["energy_kwh"].tail(24*7), label="Actual", linewidth=1.8)
    plt.plot(results["timestamp"].tail(24*7), results["predicted_kwh"].tail(24*7), label="Forecast", linewidth=1.5)
    plt.title("Actual vs Predicted Energy Consumption — Test Period")
    plt.xlabel("Time"); plt.ylabel("Energy (kWh)"); plt.legend(); plt.tight_layout()
    plt.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(["Random Forest", "Previous-hour baseline"], [metrics["mae_kwh"], metrics["baseline_mae_kwh"]])
    plt.title("Model Comparison (lower MAE is better)")
    plt.ylabel("Mean Absolute Error (kWh)"); plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150); plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(raw["timestamp"].tail(24*14), raw["energy_kwh"].tail(24*14))
    plt.title("Energy Consumption History — Last 14 Days")
    plt.xlabel("Time"); plt.ylabel("Energy (kWh)"); plt.tight_layout()
    plt.savefig(IMAGE_DIR / "energy_history.png", dpi=150); plt.close()

    generate_forecast(model, raw)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model: {MODEL_PATH}")
    print(f"Charts: {IMAGE_DIR}")
    return metrics

def generate_forecast(model=None, raw=None, hours=24):
    if raw is None:
        raw = load_dataset()
    if model is None:
        bundle = joblib.load(MODEL_PATH); model = bundle["model"]
    history = raw.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.sort_values("timestamp").reset_index(drop=True)
    future_rows = []
    values = history["energy_kwh"].astype(float).tolist()
    last_ts = history["timestamp"].iloc[-1]
    for step in range(1, hours + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        hour = ts.hour; dow = ts.dayofweek
        row = {
            "hour": hour, "day_of_week": dow, "month": ts.month, "is_weekend": int(dow >= 5),
            "lag_1": values[-1], "lag_24": values[-24], "lag_168": values[-168],
            "rolling_mean_24": float(np.mean(values[-24:]))
        }
        value = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        values.append(value)
        future_rows.append({"timestamp": ts, "predicted_kwh": round(value, 3)})
    forecast = pd.DataFrame(future_rows)
    forecast.to_csv(FORECAST_PATH, index=False)
    plt.figure(figsize=(11, 4))
    plt.plot(forecast["timestamp"], forecast["predicted_kwh"], marker="o")
    plt.title("Next 24 Hours — Simulated Energy Forecast")
    plt.xlabel("Forecast time"); plt.ylabel("Predicted energy (kWh)"); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(IMAGE_DIR / "next_24_hours_forecast.png", dpi=150); plt.close()
    return forecast

if __name__ == "__main__":
    train()
