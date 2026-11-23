import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import pmdarima as pm

df = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Feature engineering
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['sin_year2'] = np.sin(4 * np.pi * df['day_of_year'] / 365.25)
df['cos_year2'] = np.cos(4 * np.pi * df['day_of_year'] / 365.25)

# Day of week dummies
dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True, dtype=float)
df = pd.concat([df, dow_dummies], axis=1)

# Holdout split: last 14 days
HORIZON = 14
train_df = df.iloc[:-HORIZON].copy()
test_df = df.iloc[-HORIZON:].copy()

y_train = train_df['cardiac_admissions'].values
y_test = test_df['cardiac_admissions'].values

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
print(f"Test actuals (14 days): {y_test}")
print(f"Test dates: {test_df['date'].dt.strftime('%Y-%m-%d').tolist()}")

def evaluate_pred(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{name}]")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.3f}")
    return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'pred': y_pred}

results = []

# --- 1. Pure SARIMA auto-fit ---
print("\nFitting auto_arima for pure SARIMA...")
auto_sarima = pm.auto_arima(
    y_train,
    seasonal=True,
    m=7,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    d=None, D=None,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print("Auto SARIMA best order:", auto_sarima.order, auto_sarima.seasonal_order)
pred_sarima = auto_sarima.predict(n_periods=HORIZON)
results.append(evaluate_pred(y_test, pred_sarima, f"Pure SARIMA {auto_sarima.order}x{auto_sarima.seasonal_order}"))

# --- 2. SARIMAX with Calendar Features (DOW + Annual Fourier) ---
calendar_cols = [c for c in dow_dummies.columns] + ['sin_year', 'cos_year', 'is_public_holiday']
X_train_cal = train_df[calendar_cols].astype(float)
X_test_cal = test_df[calendar_cols].astype(float)

print("\nFitting auto_arima for SARIMAX with Calendar Regressors...")
auto_sarimax_cal = pm.auto_arima(
    y_train,
    X=X_train_cal,
    seasonal=True,
    m=7,
    max_p=3, max_q=3,
    max_P=1, max_Q=1,
    d=None, D=None,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print("Auto SARIMAX (Calendar) best order:", auto_sarimax_cal.order, auto_sarimax_cal.seasonal_order)
pred_sarimax_cal = auto_sarimax_cal.predict(n_periods=HORIZON, X=X_test_cal)
results.append(evaluate_pred(y_test, pred_sarimax_cal, f"SARIMAX (Calendar) {auto_sarimax_cal.order}x{auto_sarimax_cal.seasonal_order}"))

# --- 3. SARIMAX with Weekend + Annual Fourier ---
simple_cal_cols = ['is_weekend', 'sin_year', 'cos_year', 'is_public_holiday']
X_train_simple = train_df[simple_cal_cols].astype(float)
X_test_simple = test_df[simple_cal_cols].astype(float)

auto_sarimax_simple = pm.auto_arima(
    y_train,
    X=X_train_simple,
    seasonal=True,
    m=7,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    trace=False,
    error_action='ignore',
    suppress_warnings=True
)
print("Auto SARIMAX (Weekend+Fourier) best order:", auto_sarimax_simple.order, auto_sarimax_simple.seasonal_order)
pred_sarimax_simple = auto_sarimax_simple.predict(n_periods=HORIZON, X=X_test_simple)
results.append(evaluate_pred(y_test, pred_sarimax_simple, f"SARIMAX (Weekend+Fourier) {auto_sarimax_simple.order}x{auto_sarimax_simple.seasonal_order}"))

# --- 4. SARIMAX with Weather & AQI (Oracle / Actuals benchmark) ---
env_cols = ['is_weekend', 'sin_year', 'cos_year', 'temp_mean_c', 'humidity_mean_pct', 'pm25_avg']
X_train_env = train_df[env_cols].astype(float)
X_test_env = test_df[env_cols].astype(float)

auto_sarimax_env = pm.auto_arima(
    y_train,
    X=X_train_env,
    seasonal=True,
    m=7,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    trace=False,
    error_action='ignore',
    suppress_warnings=True
)
print("Auto SARIMAX (Env Oracle) best order:", auto_sarimax_env.order, auto_sarimax_env.seasonal_order)
pred_sarimax_env = auto_sarimax_env.predict(n_periods=HORIZON, X=X_test_env)
results.append(evaluate_pred(y_test, pred_sarimax_env, f"SARIMAX (Env Oracle) {auto_sarimax_env.order}x{auto_sarimax_env.seasonal_order}"))

# Comparison summary
res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'pred'} for r in results])
print("\n" + "=" * 60)
print("SUMMARY COMPARISON ON 14-DAY HOLDOUT SET:")
print("=" * 60)
print(res_df.to_string(index=False))
