import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Calendar features
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# DOW dummies
dow_cols = []
for d in range(6):
    col = f'dow_{d}'
    df[col] = (df['day_of_week'] == d).astype(float)
    dow_cols.append(col)

df['is_public_holiday'] = df['is_public_holiday'].fillna(0).astype(float)

feature_configs = {
    'DOW + Fourier': dow_cols + ['sin_year', 'cos_year', 'is_public_holiday'],
    'DOW + Holiday': dow_cols + ['is_public_holiday'],
    'Weekend + Fourier': ['is_weekend', 'sin_year', 'cos_year', 'is_public_holiday']
}

model_orders = [
    ((2, 0, 0), (0, 0, 0, 7)),
    ((1, 0, 1), (0, 0, 0, 7)),
    ((2, 0, 1), (1, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 1, 7)),
    ((1, 1, 1), (0, 0, 1, 7))
]

# 4 folds of 14 days (last 56 days)
folds = [
    (730 - 56, 730 - 42),
    (730 - 42, 730 - 28),
    (730 - 28, 730 - 14),
    (730 - 14, 730)
]

cv_results = []

for feat_name, feat_cols in feature_configs.items():
    for order, s_order in model_orders:
        fold_maes = []
        fold_rmses = []
        fold_mapes = []
        
        for train_end, test_end in folds:
            train_sub = df.iloc[:train_end]
            test_sub = df.iloc[train_end:test_end]
            
            y_tr = train_sub['cardiac_admissions'].values
            y_te = test_sub['cardiac_admissions'].values
            X_tr = train_sub[feat_cols].values
            X_te = test_sub[feat_cols].values
            
            try:
                mod = SARIMAX(
                    y_tr, exog=X_tr,
                    order=order, seasonal_order=s_order,
                    enforce_stationarity=False, enforce_invertibility=False
                )
                res = mod.fit(disp=False, maxiter=200)
                pred = res.forecast(steps=len(y_te), exog=X_te)
                pred = np.maximum(pred, 0)
                
                fold_maes.append(mean_absolute_error(y_te, pred))
                fold_rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
                fold_mapes.append(mean_absolute_percentage_error(y_te, pred) * 100)
            except Exception as e:
                pass
                
        if len(fold_maes) == len(folds):
            cv_results.append({
                'features': feat_name,
                'order': f"{order}x{s_order}",
                'cv_mae': np.mean(fold_maes),
                'cv_rmse': np.mean(fold_rmses),
                'cv_mape': np.mean(fold_mapes)
            })

cv_df = pd.DataFrame(cv_results).sort_values('cv_mae').reset_index(drop=True)
print("--- 4-FOLD ROLLING WINDOW (14-DAY) CROSS VALIDATION ---")
print(cv_df.to_string(index=False))
