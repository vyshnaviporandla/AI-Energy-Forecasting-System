# Project Report: AI-Powered Energy Consumption Forecasting System

## Abstract
This project demonstrates a machine-learning pipeline for hourly energy-consumption forecasting. It generates or loads a time-series dataset, cleans the observations, constructs temporal and lagged features, trains a Random Forest regression model, evaluates it chronologically, and creates a 24-hour recursive forecast. A Flask API and Streamlit dashboard provide simple ways to inspect predictions.

## Problem statement
Energy consumption varies over time. Without estimates of upcoming demand, building teams may find it harder to plan operations and examine peak-use periods. The project demonstrates how historical data can be transformed into forecasts that support analysis.

## Objectives
1. Prepare hourly time-series energy data.
2. Engineer calendar and lag features without randomly shuffling time.
3. Train and evaluate a regression model.
4. Compare model MAE with a previous-hour baseline.
5. Export visualizations, forecast CSV, and machine-readable metrics.
6. Expose a simple prediction API and dashboard.

## Dataset
The default dataset is synthetic and generated locally. It includes timestamp, simulated energy consumption in kWh, and a simulated temperature column. It contains designed daily, weekly, seasonal, and noise patterns. It is not sourced from a real utility and should not be represented as real meter data.

## Methodology
The data is sorted by timestamp, deduplicated, resampled hourly, and cleaned. Features include hour, day of week, month, weekend flag, previous-hour usage, usage 24 hours earlier, usage 168 hours earlier, and the previous 24-hour rolling mean. The earliest 80% of rows train a Random Forest; the latest 20% are held out for evaluation. The model is assessed using MAE, RMSE, and R², and compared to a simple lag-1 persistence baseline.

## System modules
- `src/data.py`: synthetic data generation and data validation.
- `src/features.py`: time-series feature engineering.
- `src/train.py`: model training, metrics, forecasts, and charts.
- `app.py`: Flask API.
- `dashboard.py`: optional Streamlit dashboard.

## Outputs
Running `python main.py` writes the model, metrics JSON, prediction CSV, next-24-hour forecast CSV, and four charts. Actual values are calculated on the machine where the project is run; record the output from your own execution in the final submission.

## Limitations
Synthetic patterns do not represent every real building. Performance metrics on simulated data cannot be generalized to real sites. Forecast quality depends on data quality, consistent units, and representative history. Multi-step recursive predictions may compound errors.

## Conclusion
The project provides an end-to-end student portfolio example of time-series feature engineering, regression, evaluation, forecasting, visualization, and lightweight API integration. A next step is to validate the pipeline on a public, appropriately licensed real energy dataset.
