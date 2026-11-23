import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Engineering features
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['sin_year2'] = np.sin(4 * np.pi * df['day_of_year'] / 365.25)
df['cos_year2'] = np.cos(4 * np.pi * df['day_of_year'] / 365.25)

# DOW dummies (Monday to Saturday, Sunday is baseline)
dow_cols = []
for d in range(6):
    col = f'dow_{d}'
    df[col] = (df['day_of_week'] == d).astype(float)
    dow_cols.append(col)

# Additional calendar indicators
df['is_public_holiday'] = df['is_public_holiday'].fillna(0).astype(float)
df['is_festival'] = df['is_festival'].fillna(0).astype(float)

HORIZON = 14
train_df = df.iloc[:-HORIZON].copy()
test_df = df.iloc[-HORIZON:].copy()

y_train = train_df['cardiac_admissions'].values
y_test = test_df['cardiac_admissions'].values

feature_sets = {
    'DOW + Fourier_1': dow_cols + ['sin_year', 'cos_year', 'is_public_holiday'],
    'DOW + Fourier_2': dow_cols + ['sin_year', 'cos_year', 'sin_year2', 'cos_year2', 'is_public_holiday'],
    'Weekend + Fourier_1': ['is_weekend', 'sin_year', 'cos_year', 'is_public_holiday'],
    'DOW + Holiday': dow_cols + ['is_public_holiday', 'is_festival']
}

candidate_orders = [
    # (p, d, q), (P, D, Q, s)
    ((1, 0, 0), (0, 0, 0, 7)),
    ((1, 0, 1), (0, 0, 0, 7)),
    ((2, 0, 0), (0, 0, 0, 7)),
    ((2, 0, 1), (0, 0, 0, 7)),
    ((1, 0, 2), (0, 0, 0, 7)),
    ((2, 0, 2), (0, 0, 0, 7)),
    ((1, 0, 1), (1, 0, 0, 7)),
    ((1, 0, 1), (0, 0, 1, 7)),
    ((1, 0, 1), (1, 0, 1, 7)),
    ((2, 0, 1), (1, 0, 0, 7)),
    ((1, 1, 1), (0, 0, 0, 7)),
    ((1, 1, 1), (1, 0, 1, 7)),
    ((2, 1, 1), (0, 0, 0, 7)),
]

experiments = []

for f_name, f_cols in feature_sets.items():
    X_train = train_df[f_cols].values
    X_test = test_df[f_cols].values
    
    # Also test log-transform vs linear
    for transform in ['none', 'log1p', 'sqrt']:
        if transform == 'none':
            yt = y_train
        elif transform == 'log1p':
            yt = np.log1p(y_train)
        elif transform == 'sqrt':
            yt = np.sqrt(y_train)
            
        for order, s_order in candidate_orders:
            try:
                model = SARIMAX(
                    yt,
                    exog=X_train,
                    order=order,
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit(disp=False, maxiter=200)
                
                pred_raw = fit.forecast(steps=HORIZON, exog=X_test)
                if transform == 'log1p':
                    pred = np.expm1(pred_raw)
                elif transform == 'sqrt':
                    pred = np.square(np.maximum(pred_raw, 0))
                else:
                    pred = pred_raw
                    
                # Clip negative values if any
                pred = np.maximum(pred, 0)
                
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mape = mean_absolute_percentage_error(y_test, pred) * 100
                r2 = r2_score(y_test, pred)
                
                experiments.append({
                    'features': f_name,
                    'transform': transform,
                    'order': f"{order}x{s_order}",
                    'aic': fit.aic,
                    'bic': fit.bic,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'pred': pred
                })
            except Exception as e:
                pass

exp_df = pd.DataFrame([{k: v for k, v in e.items() if k != 'pred'} for e in experiments])
exp_df = exp_df.sort_values(by=['mae', 'rmse']).reset_index(drop=True)

print("TOP 15 MODELS SORTED BY MAE:")
print(exp_df.head(15).to_string())

# Save best model details
best = exp_df.iloc[0]
print("\n--- BEST MODEL ---")
print(best)

# Get predictions of top model
best_pred = experiments[exp_df.index[0]]['pred']
eval_comp = pd.DataFrame({
    'date': test_df['date'].dt.strftime('%Y-%m-%d'),
    'actual': y_test,
    'predicted': np.round(best_pred, 1),
    'abs_error': np.round(np.abs(y_test - best_pred), 1)
})
print("\n14-Day Validation Breakdown:")
print(eval_comp.to_string(index=False))
