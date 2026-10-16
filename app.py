"""Local Flask API for forecasting energy use."""
from flask import Flask, jsonify, request
import joblib
import pandas as pd
from src.config import MODEL_PATH
from src.features import FEATURES

app = Flask(__name__)

def get_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python main.py")
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "service": "AI Energy Forecast API",
        "status": "running",
        "endpoints": {"GET /health": "API and model status", "POST /predict": "Predict one hour using recent usage history"},
        "note": "Educational demo; not for operational grid control."
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})

@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    required = ["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_24", "lag_168", "rolling_mean_24"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400
    try:
        row = {k: float(body[k]) for k in required}
        if not 0 <= row["hour"] <= 23 or not 0 <= row["day_of_week"] <= 6 or not 1 <= row["month"] <= 12:
            return jsonify({"error": "hour must be 0-23, day_of_week 0-6, month 1-12"}), 400
        model = get_bundle()["model"]
        value = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        return jsonify({"predicted_energy_kwh": round(value, 3), "unit": "kWh", "model": "RandomForestRegressor"})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
