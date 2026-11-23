import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

# 1. Load Data
for cand in ['data/clean_daily_dataset.csv', 'data/cardiac_data.csv', 'clean_daily_dataset.csv', 'cardiac_data.csv']:
    if os.path.exists(cand):
        data_path = cand
        break
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Feature Engineering
def build_features(dates, is_holidays=None):
    feat = pd.DataFrame({'date': dates})
    feat['day_of_year'] = feat['date'].dt.dayofyear
    feat['day_of_week'] = feat['date'].dt.dayofweek
    feat['sin_year'] = np.sin(2 * np.pi * feat['day_of_year'] / 365.25)
    feat['cos_year'] = np.cos(2 * np.pi * feat['day_of_year'] / 365.25)
    
    # DOW dummies (0=Monday to 5=Saturday, 6=Sunday as base)
    for d in range(6):
        feat[f'dow_{d}'] = (feat['day_of_week'] == d).astype(float)
        
    if is_holidays is not None:
        feat['is_public_holiday'] = is_holidays.values if hasattr(is_holidays, 'values') else is_holidays
    else:
        feat['is_public_holiday'] = 0.0
    feat['is_public_holiday'] = feat['is_public_holiday'].fillna(0).astype(float)
    
    dow_cols = [f'dow_{d}' for d in range(6)]
    reg_cols = dow_cols + ['sin_year', 'cos_year', 'is_public_holiday']
    return feat, reg_cols

train_feat, reg_cols = build_features(df['date'], df['is_public_holiday'])
X_train = train_feat[reg_cols].values
y_train = df['cardiac_admissions'].values

print("Training dataset shape:", y_train.shape)
print("Regressors used:", reg_cols)

# 3. Fit optimal SARIMAX Model
# Order: (2, 0, 1) x (1, 0, 0, 7)
order = (2, 0, 1)
seasonal_order = (1, 0, 0, 7)

print(f"\nFitting SARIMAX{order}x{seasonal_order} on all 730 days...")
model = SARIMAX(
    y_train,
    exog=X_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit_res = model.fit(disp=False, maxiter=300)
print(fit_res.summary())

# 4. Diagnostics on Residuals
residuals = fit_res.resid[14:] # skip warmup lags

# Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(residuals, lags=[7, 14, 21, 28], return_df=True)
print("\n--- Ljung-Box Test on Residuals ---")
print(lb_test)

# Jarque-Bera test for normality
jb_stat, jb_p = jarque_bera(residuals)
print(f"\nJarque-Bera Normality Test: stat={jb_stat:.4f}, p-value={jb_p:.4e}")

# In-sample metrics
in_sample_pred = fit_res.fittedvalues
in_sample_mae = np.mean(np.abs(y_train[14:] - in_sample_pred[14:]))
in_sample_rmse = np.sqrt(np.mean((y_train[14:] - in_sample_pred[14:])**2))
print(f"In-sample MAE (post-warmup): {in_sample_mae:.2f}")
print(f"In-sample RMSE (post-warmup): {in_sample_rmse:.2f}")

# 5. Build 14-day upcoming forecast horizon: 2019-04-01 to 2019-04-14
future_dates = pd.date_range(start='2019-04-01', periods=14, freq='D')

# In 2019:
# 2019-04-08: Guru Nabha Dass's Birthday (Monday)
# 2019-04-13: Ram Navami / Vaisakhi (Saturday)
# 2019-04-14: Dr. B. R. Ambedkar's Birthday (Sunday)
future_holidays = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]

future_feat, _ = build_features(future_dates, future_holidays)
X_future = future_feat[reg_cols].values

# Generate forecast and confidence intervals
forecast_res = fit_res.get_forecast(steps=14, exog=X_future)
point_forecast = np.maximum(forecast_res.predicted_mean, 0)
ci_95 = forecast_res.conf_int(alpha=0.05)
ci_80 = forecast_res.conf_int(alpha=0.20)

lower_95 = np.maximum(ci_95[:, 0], 0)
upper_95 = np.maximum(ci_95[:, 1], 0)
lower_80 = np.maximum(ci_80[:, 0], 0)
upper_80 = np.maximum(ci_80[:, 1], 0)

forecast_df = pd.DataFrame({
    'date': future_dates.strftime('%Y-%m-%d'),
    'day_of_week': future_dates.day_name(),
    'is_public_holiday': future_holidays,
    'predicted_cardiac_admissions': np.round(point_forecast, 2),
    'lower_ci_80': np.round(lower_80, 2),
    'upper_ci_80': np.round(upper_80, 2),
    'lower_ci_95': np.round(lower_95, 2),
    'upper_ci_95': np.round(upper_95, 2)
})

print("\n" + "="*60)
print("14-DAY UPCOMING FORECAST (2019-04-01 to 2019-04-14):")
print("="*60)
print(forecast_df.to_string(index=False))

# Save forecast to CSV
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'src' else os.getcwd()
data_dir = os.path.join(root_dir, 'data')
models_dir = os.path.join(root_dir, 'models')
figures_dir = os.path.join(root_dir, 'figures')

os.makedirs(os.path.join(data_dir, 'forecasts'), exist_ok=True)
forecast_path = os.path.join(data_dir, 'forecasts', 'cardiac_admissions_14day_forecast.csv')
forecast_df.to_csv(forecast_path, index=False)
print(f"\nSaved 14-day forecast to '{forecast_path}'")

# 6. Save Model Artifact
model_data = {
    'model_fit': fit_res,
    'reg_cols': reg_cols,
    'order': order,
    'seasonal_order': seasonal_order,
    'last_date': df['date'].max(),
    'in_sample_mae': in_sample_mae,
    'in_sample_rmse': in_sample_rmse
}
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'cardiac_admissions_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Saved model artifact to '{model_path}'")

# 7. Plots
os.makedirs(figures_dir, exist_ok=True)
plt.figure(figsize=(14, 6))
# Plot last 60 days of historical data
recent_df = df.iloc[-60:].copy()
plt.plot(recent_df['date'], recent_df['cardiac_admissions'], label='Historical Actuals (Last 60 Days)', color='#1f77b4', marker='o', markersize=4)

# Plot Forecast
plt.plot(future_dates, point_forecast, label='SARIMAX 14-Day Forecast', color='#d62728', marker='s', linewidth=2.5)
plt.fill_between(future_dates, lower_80, upper_80, color='#d62728', alpha=0.25, label='80% Prediction Interval')
plt.fill_between(future_dates, lower_95, upper_95, color='#d62728', alpha=0.12, label='95% Prediction Interval')

plt.title('Daily Cardiac Admissions: Historical & Upcoming 14-Day Forecast', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cardiac Admissions', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
chart_path = os.path.join(figures_dir, 'forecast_chart.png')
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Saved forecast plot to '{chart_path}'")

# Diagnostics plot
fig = fit_res.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
diag_path = os.path.join(figures_dir, 'diagnostic_chart.png')
plt.savefig(diag_path, dpi=300)
plt.close()
print(f"Saved diagnostic plot to '{diag_path}'")
