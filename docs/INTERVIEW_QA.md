# Interview Q&A

**1. What is the objective?**  
To demonstrate a pipeline that forecasts hourly energy use from past observations and temporal patterns.

**2. Why use a chronological split?**  
Forecasting must respect time. Training on future rows and testing on earlier rows can cause unrealistic evaluation.

**3. Why Random Forest?**  
It is a strong beginner-friendly nonlinear regression baseline that handles interactions between engineered features without requiring feature scaling.

**4. What are lag features?**  
They provide previous observations, such as consumption one hour, one day, or one week earlier.

**5. What do MAE and RMSE mean?**  
MAE is the average absolute error. RMSE gives more weight to larger errors. Both are expressed in the target unit, kWh.

**6. Why compare with a baseline?**  
A model is more useful when it improves on a simple, transparent prediction such as using the previous hour's consumption.

**7. Is this real electricity data?**  
The default is synthetic simulated data. I would use a properly licensed real dataset and re-evaluate before claiming real-world accuracy.

**8. What does the API do?**  
It accepts engineered time and lag features as JSON and returns a predicted energy value.

**9. What are the limitations?**  
Synthetic data, limited features, recursive forecast error, and no production monitoring or prediction intervals.

**10. What would you improve?**  
Add weather and holiday features, seasonal-naive benchmarks, prediction intervals, model tracking, and evaluation on real data.
