import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

df = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()} (total days: {len(df)})")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

target = df['cardiac_admissions']
print("\n--- Target Summary (cardiac_admissions) ---")
print(target.describe())

# Stationarity test
adf_res = adfuller(target.dropna())
print(f"\nADF Statistic: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4e}")
print(f"Critical Values: {adf_res[4]}")
if adf_res[1] < 0.05:
    print("Series is stationary according to ADF test (reject unit root H0).")
else:
    print("Series has a unit root according to ADF test (non-stationary).")

# Weekly pattern
df['day_name'] = df['date'].dt.day_name()
weekly_mean = df.groupby(['day_of_week', 'day_name'])['cardiac_admissions'].agg(['mean', 'std', 'count'])
print("\n--- Admissions by Day of Week ---")
print(weekly_mean)

# Monthly pattern
monthly_mean = df.groupby('month')['cardiac_admissions'].agg(['mean', 'std', 'count'])
print("\n--- Admissions by Month ---")
print(monthly_mean)

# Correlations with numeric features
num_cols = df.select_dtypes(include=[np.number]).columns
corrs = df[num_cols].corr()['cardiac_admissions'].sort_values(ascending=False)
print("\n--- Correlation with numeric features ---")
print(corrs)
