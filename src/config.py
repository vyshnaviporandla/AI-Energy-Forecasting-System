from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "images"
DATA_PATH = DATA_DIR / "energy_consumption.csv"
MODEL_PATH = MODEL_DIR / "energy_forecast_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
FORECAST_PATH = OUTPUT_DIR / "next_24_hours.csv"
