import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. Prepare Hourly Demand Time-Series Data
# -------------------------------------------------------------
df_orders = pd.read_csv("data/raw/orders.csv")
df_orders['order_timestamp'] = pd.to_datetime(df_orders['order_timestamp'])

# Resample to hourly counts, filling missing hours with 0
df_hourly = df_orders.set_index('order_timestamp').resample('h').size().reset_index()
df_hourly.columns = ['ds', 'y']

df_hourly.to_csv("data/processed/hourly_demand.csv", index=False)

# -------------------------------------------------------------
# 2. Stationarity Check (ADF Test via standard tuple indexing)
# -------------------------------------------------------------
adf_out = adfuller(df_hourly['y'])
adf_stat = adf_out[0]
p_value = adf_out[1]

print("=== Augmented Dickey-Fuller (ADF) Test ===")
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4e}")
print("Stationary: True" if p_value < 0.05 else "Stationary: False")

# -------------------------------------------------------------
# 3. Chronological Train / Test Split
# -------------------------------------------------------------
split_idx = int(len(df_hourly) * 0.8)
train_df = df_hourly.iloc[:split_idx].copy()
test_df = df_hourly.iloc[split_idx:].copy()

# -------------------------------------------------------------
# 4. ARIMA Modeling
# -------------------------------------------------------------
arima_model = ARIMA(train_df['y'], order=(2, 0, 2))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=len(test_df))

# -------------------------------------------------------------
# 5. Prophet Modeling
# -------------------------------------------------------------
prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
prophet_model.fit(train_df)

future = prophet_model.make_future_dataframe(periods=len(test_df), freq='h')
forecast = prophet_model.predict(future)
prophet_preds = forecast.iloc[split_idx:]['yhat'].values

# -------------------------------------------------------------
# 6. Evaluation & Comparison
# -------------------------------------------------------------
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

metrics = []
for name, preds in [('ARIMA', arima_preds), ('Prophet', prophet_preds)]:
    mae = mean_absolute_error(test_df['y'], preds)
    rmse = np.sqrt(mean_squared_error(test_df['y'], preds))
    mape = calculate_mape(test_df['y'], preds)
    metrics.append({'Model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2)})

df_metrics = pd.DataFrame(metrics)
print("\n=== Forecast Model Comparison ===")
print(df_metrics.to_string(index=False))

# -------------------------------------------------------------
# 7. Visualization
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(test_df['ds'].values[:168], test_df['y'].values[:168], label='Actual Demand', color='black', alpha=0.7)
plt.plot(test_df['ds'].values[:168], arima_preds[:168], label='ARIMA Forecast', color='blue', linestyle='--')
plt.plot(test_df['ds'].values[:168], prophet_preds[:168], label='Prophet Forecast', color='green', linestyle='-')
plt.title('7-Day Hourly Demand Forecast Comparison (Actual vs ARIMA vs Prophet)')
plt.xlabel('Timestamp')
plt.ylabel('Order Volume')
plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/hourly_demand_forecast.png')
plt.close()

print("\nSUCCESS: Day 5 Demand Forecasting Pipeline Execution Completed.")