
import json
import joblib
import pandas as pd
import streamlit as st

from src.config import (
    IMAGE_DIR, METRICS_PATH, PREDICTIONS_PATH,
    FORECAST_PATH, MODEL_PATH
)
from src.features import FEATURES

st.set_page_config(
    page_title="WattWise AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom UI theme ----------


# ---------- Custom UI theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg: #07111f;
    --panel: #102137;
    --mint: #67f0bd;
    --text: #edf5ff;
    --muted: #a8bbd0;
}

.stApp {
    background: radial-gradient(ellipse at top left, #18354a, #07111f 58%);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.96);
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0b1727;
    border-right: 1px solid #26394d;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

h1, h2, h3, h4, p, label, li, span {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
}

p, label, .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    color: var(--text) !important;
    border-radius: 8px;
    padding: 5px 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: var(--text) !important;
    font-size: 0.95rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #172c42 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #142940, #0d1b2c);
    border: 1px solid #294158;
    padding: 18px;
    border-radius: 17px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #b6c8dc !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: var(--mint) !important;
}

.stTextInput input,
.stNumberInput input,
[data-baseweb="select"] > div {
    background-color: #102137 !important;
    color: var(--text) !important;
    border-color: #36506a !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #102137 !important;
}

[data-baseweb="menu"] * {
    color: var(--text) !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: var(--mint) !important;
    color: #07111f !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #91f8d2 !important;
    color: #07111f !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #294158;
    border-radius: 12px;
}

hr {
    border-color: #26394d;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load project data ----------
if not METRICS_PATH.exists():
    st.error("Metrics are missing. Run python main.py first.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

forecast = (
    pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    if FORECAST_PATH.exists() else pd.DataFrame()
)
predictions = (
    pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    if PREDICTIONS_PATH.exists() else pd.DataFrame()
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("# ⚡ WattWise AI")
    st.caption("ENERGY INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Forecast Explorer",
            "Make a Prediction",
            "Model Insights"
        ]
    )

    st.divider()
    st.success("Model artifacts loaded")
    st.caption("Random Forest Regression")
    st.caption("Hourly energy forecasting")
    st.caption("Synthetic demo data")

# ---------- Hero header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">ENERGY ANALYTICS PLATFORM</div>
  <div class="hero-title">Understand usage.<br>Forecast what's next.</div>
  <div class="hero-copy">
    Explore consumption patterns, review machine-learning performance,
    and generate energy estimates with your trained model.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Your energy snapshot")

    a, b, c, d = st.columns(4)
    a.metric("MAE", f'{metrics["mae_kwh"]:.3f} kWh')
    b.metric("RMSE", f'{metrics["rmse_kwh"]:.3f} kWh')
    c.metric("R² Score", f'{metrics["r2"]:.4f}')
    d.metric("Test Rows", f'{metrics["test_rows"]:,}')

    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Next 24 hours")
        if not forecast.empty:
            chart = forecast.set_index("timestamp")[["predicted_kwh"]]
            chart.columns = ["Forecast kWh"]
            st.line_chart(chart, color="#67f0bd", height=320)
        else:
            st.warning("Forecast file not found.")

    with right:
        st.subheader("Model summary")
        st.write("**Algorithm:** RandomForestRegressor")
        st.write(f'**Training rows:** {metrics["train_rows"]:,}')
        st.write(f'**Test rows:** {metrics["test_rows"]:,}')
        st.write(f'**Dataset:** {metrics["dataset_type"]}')
        st.write(f'**Baseline MAE:** {metrics["baseline_mae_kwh"]:.3f} kWh')

    if not predictions.empty:
        st.subheader("Actual vs predicted energy")
        chart = predictions.tail(168).set_index("timestamp")[
            ["energy_kwh", "predicted_kwh"]
        ]
        chart.columns = ["Actual kWh", "Predicted kWh"]
        st.line_chart(chart, height=320)

    st.info(
        "Educational portfolio demo using simulated data. "
        "Results do not guarantee real-world grid performance."
    )

# ---------- Forecast explorer ----------
elif page == "Forecast Explorer":
    st.subheader("Forecast explorer")

    if forecast.empty:
        st.warning("Run python main.py to generate forecasts.")
    else:
        horizon = st.slider(
            "Hours to display",
            min_value=6,
            max_value=min(24, len(forecast)),
            value=min(24, len(forecast))
        )

        visible = forecast.head(horizon).copy()
        chart = visible.set_index("timestamp")[["predicted_kwh"]]
        chart.columns = ["Predicted kWh"]
        st.line_chart(chart, color="#67f0bd", height=350)

        a, b, c = st.columns(3)
        a.metric("Average", f'{visible["predicted_kwh"].mean():.2f} kWh')
        b.metric("Peak", f'{visible["predicted_kwh"].max():.2f} kWh')

        peak = visible.loc[visible["predicted_kwh"].idxmax()]
        c.metric("Peak hour", peak["timestamp"].strftime("%H:%M"))

        st.subheader("Forecast records")
        st.dataframe(visible, use_container_width=True, hide_index=True)

        st.download_button(
            "Download forecast CSV",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name="wattwise_forecast.csv",
            mime="text/csv"
        )

# ---------- Interactive prediction ----------
elif page == "Make a Prediction":
    st.subheader("Prediction lab")
    st.write("Enter recent consumption values to estimate the next hour.")

    with st.form("prediction_form"):
        a, b, c, d = st.columns(4)

        hour = a.slider("Hour", 0, 23, 15)
        day = b.selectbox(
            "Day of week",
            list(range(7)),
            index=4,
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ][x]
        )
        month = c.selectbox("Month", list(range(1, 13)), index=9)
        weekend = d.selectbox(
            "Weekend?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        e, f, g, h = st.columns(4)
        lag1 = e.number_input("Previous hour (kWh)", min_value=0.0, value=175.0)
        lag24 = f.number_input("24 hours ago (kWh)", min_value=0.0, value=160.0)
        lag168 = g.number_input("168 hours ago (kWh)", min_value=0.0, value=150.0)
        mean24 = h.number_input("24-hour average (kWh)", min_value=0.0, value=145.0)

        submitted = st.form_submit_button(
            "⚡ Generate prediction",
            use_container_width=True
        )

    if submitted:
        if not MODEL_PATH.exists():
            st.error("Trained model missing. Run python main.py first.")
        else:
            try:
                bundle = joblib.load(MODEL_PATH)
                model = bundle["model"]

                row = {
                    "hour": float(hour),
                    "day_of_week": float(day),
                    "month": float(month),
                    "is_weekend": float(weekend),
                    "lag_1": float(lag1),
                    "lag_24": float(lag24),
                    "lag_168": float(lag168),
                    "rolling_mean_24": float(mean24)
                }

                result = max(
                    0.0,
                    float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
                )

                st.success("Prediction generated successfully")
                st.metric("Estimated energy consumption", f"{result:.3f} kWh")
                st.caption("This is a model estimate based on the values entered.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ---------- Model insights ----------
elif page == "Model Insights":
    st.subheader("Model evaluation")

    a, b, c = st.columns(3)
    a.metric("Model MAE", f'{metrics["mae_kwh"]:.3f} kWh')
    b.metric("Baseline MAE", f'{metrics["baseline_mae_kwh"]:.3f} kWh')
    c.metric("R² Score", f'{metrics["r2"]:.4f}')

    if metrics.get("baseline_mae_kwh", 0):
        improvement = (
            1 - metrics["mae_kwh"] / metrics["baseline_mae_kwh"]
        ) * 100
        st.write(f"MAE improvement over baseline: **{improvement:.1f}%**")

    st.write(f'**Rows used:** {metrics["rows_used"]:,}')
    st.write(f'**Training rows:** {metrics["train_rows"]:,}')
    st.write(f'**Test rows:** {metrics["test_rows"]:,}')
    st.write(f'**Split:** {metrics["split"]}')

    chart_path = IMAGE_DIR / "actual_vs_predicted.png"
    if chart_path.exists():
        st.subheader("Actual vs predicted")
        st.image(str(chart_path), use_container_width=True)

    comparison_path = IMAGE_DIR / "model_comparison.png"
    if comparison_path.exists():
        st.subheader("Model comparison")
        st.image(str(comparison_path), use_container_width=True)

    with st.expander("Raw metrics JSON"):
        st.json(metrics)

st.divider()
st.caption(
    "WattWise AI · Machine-learning portfolio demo · "
    "Synthetic data, not intended for operational grid control."
)
