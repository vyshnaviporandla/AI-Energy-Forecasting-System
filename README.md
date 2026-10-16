# ⚡ WattWise AI — AI-Powered Energy Consumption Forecasting System

**WattWise AI** is a machine learning project that forecasts electricity consumption using historical energy patterns and time-based features. It combines a Random Forest regression model, a Flask REST API, and an interactive Streamlit dashboard to demonstrate how AI can support energy monitoring and planning.

<p align="center">
  <a href="https://wattwise-ai-energy.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Dashboard-brightgreen?style=for-the-badge&logo=streamlit" alt="Live Dashboard"/>
  </a>
  <a href="https://github.com/vyshnaviporandla/AI-Energy-Forecasting-System">
    <img src="https://img.shields.io/badge/GitHub-View%20Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository"/>
  </a>
  <a href="https://ai-energy-forecast-api.onrender.com/health">
    <img src="https://img.shields.io/badge/API-Health%20Check-blue?style=for-the-badge" alt="API Health Check"/>
  </a>
</p>

## 🚀 Live Project Links

| Resource | Link |
|---|---|
| 🌐 Interactive Dashboard | [Open WattWise AI](https://wattwise-ai-energy.streamlit.app/) |
| 💻 GitHub Repository | [View Source Code](https://github.com/vyshnaviporandla/AI-Energy-Forecasting-System) |
| 🔌 Deployed Flask API | [Open API](https://ai-energy-forecast-api.onrender.com/) |
| ❤️ API Health Check | [Check API Status](https://ai-energy-forecast-api.onrender.com/health) |

## 🎯 Project Objectives

- Forecast future electricity consumption from historical patterns.
- Apply machine learning to an energy forecasting problem.
- Provide an interactive dashboard for exploring predictions and model performance.
- Expose forecasting functionality through a REST API.
- Demonstrate an end-to-end machine learning deployment workflow.

## ✨ Features

- **AI-powered forecasting:** Uses a trained Random Forest Regressor.
- **Interactive dashboard:** Explore energy forecasting results through Streamlit.
- **REST API:** Access prediction functionality through Flask.
- **Time-based features:** Uses hour, day of week, month, and weekend information.
- **Historical consumption features:** Includes lagged consumption and rolling-average inputs.
- **Model evaluation:** Reports MAE, RMSE, and R².
- **Prediction outputs:** Includes generated CSV files for forecasts and test predictions.
- **Cloud deployment:** Public dashboard and API endpoints.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Model | Random Forest Regressor |
| Backend | Flask REST API |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Version Control | Git and GitHub |
| API Deployment | Render |
| Dashboard Deployment | Streamlit Community Cloud |

## 🧠 How It Works

1. **Data preparation:** Creates and processes hourly energy-consumption data.
2. **Feature engineering:** Uses time features, historical lags, and rolling averages.
3. **Model training:** Trains a Random Forest regression model.
4. **Evaluation:** Tests performance on a chronological holdout dataset.
5. **API integration:** Makes prediction functionality available through Flask.
6. **Dashboard visualization:** Presents forecasting results through Streamlit.
7. **Deployment:** Publishes the dashboard and API online.

## 📊 Model Performance

The model was evaluated on a chronological holdout set using a **synthetic demonstration dataset**.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 5.493 kWh |
| Root Mean Squared Error (RMSE) | 7.044 kWh |
| R² Score | 0.9447 |
| Baseline MAE | 9.507 kWh |
| Total Synthetic Hourly Records | 17,520 |
| Training Samples | 13,881 |
| Test Samples | 3,471 |

**Important:** These results describe performance on synthetic data. They do not establish accuracy on real electricity-meter data or guarantee real-world forecasting performance.

## 📁 Project Structure

```text
AI-Energy-Forecasting-System/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── models/
│   └── energy_forecast_model.joblib
├── outputs/
│   ├── metrics.json
│   ├── next_24_hours.csv
│   └── test_predictions.csv
└── images/
    └── project screenshots and charts
```

*Note: The exact filenames and directories may vary slightly with your local project version.*

## 💻 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/vyshnaviporandla/AI-Energy-Forecasting-System.git
cd AI-Energy-Forecasting-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the machine learning pipeline

```bash
python main.py
```

This runs the training and evaluation workflow, provided the script and required dependencies are available.

### 5. Launch the dashboard

If your Streamlit dashboard entry point is `app.py`, run:

```bash
streamlit run app.py
```

### 6. Start the Flask API

Run the Flask entry-point script used by your project. For example, if it is named `api.py`:

```bash
python api.py
```

The API filename and startup command should match the actual backend file in your repository.

## 🔌 API Example

**Base URL:**  
`https://ai-energy-forecast-api.onrender.com/`

**Health endpoint:**

```http
GET /health
```

**Example prediction request body:**

```json
{
  "hour": 14,
  "day_of_week": 3,
  "month": 10,
  "is_weekend": 0,
  "lag_1": 150,
  "lag_24": 145,
  "lag_168": 160,
  "rolling_mean_24": 155
}
```

The prediction endpoint expects the feature fields required by the deployed API. Check the API source code for the exact endpoint path and request schema.

Example response:

```json
{
  "model": "RandomForestRegressor",
  "predicted_energy_kwh": 160.257,
  "unit": "kWh"
}
```

The result above is an example prediction, not a guaranteed forecast for an actual building or household.

## 📸 Screenshots

Add your real project screenshots to the `images/` folder and update the filenames below.

```markdown
![WattWise AI Dashboard](images/dashboard.png)
![Forecast Results](images/forecast-results.png)
![Model Evaluation](images/model-evaluation.png)
```

## 🔮 Future Improvements

- Integrate real electricity-meter datasets.
- Add weather and holiday features.
- Compare Random Forest with other forecasting algorithms.
- Add forecast confidence intervals and improved error analysis.
- Support configurable forecasting periods.
- Improve input validation, API documentation, and automated testing.

## 🎓 Learning Outcomes

This project provided practical experience with:

- Data preprocessing and feature engineering.
- Machine learning model training and evaluation.
- Time-aware train/test splitting.
- REST API development with Flask.
- Interactive dashboard development with Streamlit.
- Model deployment and version control.
- Debugging and end-to-end project integration.

## 👩‍💻 Author

**Vyshnavi Porandla**

- GitHub: [@vyshnaviporandla](https://github.com/vyshnaviporandla)
- Project Repository: [AI-Energy-Forecasting-System](https://github.com/vyshnaviporandla/AI-Energy-Forecasting-System)
- Live Demo: [WattWise AI Dashboard](https://wattwise-ai-energy.streamlit.app/)

---

⭐ If you find this project interesting, consider starring the repository!

*WattWise AI is an educational machine learning demonstration and should not be treated as a utility-grade energy forecasting or decision-making system.*
