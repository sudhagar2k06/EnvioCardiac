"""
CLI & API script to load the trained SARIMAX model and generate forecasts
Usage:
    python predict.py --days 14
    python predict.py --start-date 2019-04-01 --days 14 --output custom_forecast.csv
"""
import os
import argparse
import pickle
import pandas as pd
import numpy as np

def build_features(dates, is_holidays=None):
    feat = pd.DataFrame({'date': pd.to_datetime(dates)})
    feat['day_of_year'] = feat['date'].dt.day_of_year
    feat['day_of_week'] = feat['date'].dt.day_of_week
    feat['sin_year'] = np.sin(2 * np.pi * feat['day_of_year'] / 365.25)
    feat['cos_year'] = np.cos(2 * np.pi * feat['day_of_year'] / 365.25)
    
    for d in range(6):
        feat[f'dow_{d}'] = (feat['day_of_week'] == d).astype(float)
        
    if is_holidays is not None:
        feat['is_public_holiday'] = is_holidays
    else:
        feat['is_public_holiday'] = 0.0
    feat['is_public_holiday'] = feat['is_public_holiday'].fillna(0).astype(float)
    
    dow_cols = [f'dow_{d}' for d in range(6)]
    reg_cols = dow_cols + ['sin_year', 'cos_year', 'is_public_holiday']
    return feat, reg_cols

def forecast(days=14, start_date='2019-04-01', model_path=None):
    if model_path is None:
        for cand in [
            'models/cardiac_admissions_model.pkl',
            '../models/cardiac_admissions_model.pkl',
            'cardiac_admissions_model.pkl'
        ]:
            if os.path.exists(cand):
                model_path = cand
                break
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train_final_model.py first.")
        
    with open(model_path, 'rb') as f:
        model_meta = pickle.load(f)
        
    model_fit = model_meta['model_fit']
    reg_cols = model_meta['reg_cols']
    
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Standard holiday calendar lookup for April
    holidays = []
    for dt in future_dates:
        # Check standard holiday dates
        if dt.month == 4 and dt.day in [8, 13, 14]:
            holidays.append(1.0)
        else:
            holidays.append(0.0)
            
    feat_df, _ = build_features(future_dates, holidays)
    X_future = feat_df[reg_cols].values
    
    fc = model_fit.get_forecast(steps=days, exog=X_future)
    mean_pred = np.maximum(fc.predicted_mean, 0)
    ci_95 = fc.conf_int(alpha=0.05)
    ci_80 = fc.conf_int(alpha=0.20)
    
    out_df = pd.DataFrame({
        'date': future_dates.strftime('%Y-%m-%d'),
        'day_of_week': future_dates.day_name(),
        'is_public_holiday': holidays,
        'predicted_admissions': np.round(mean_pred, 1),
        'lower_80': np.round(np.maximum(ci_80[:, 0], 0), 1),
        'upper_80': np.round(np.maximum(ci_80[:, 1], 0), 1),
        'lower_95': np.round(np.maximum(ci_95[:, 0], 0), 1),
        'upper_95': np.round(np.maximum(ci_95[:, 1], 0), 1)
    })
    return out_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forecast upcoming cardiac admissions using SARIMAX.")
    parser.add_argument('--days', type=int, default=14, help="Number of forecast days (default: 14)")
    parser.add_argument('--start-date', type=str, default='2019-04-01', help="Start date (YYYY-MM-DD)")
    parser.add_argument('--output', type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    
    res = forecast(days=args.days, start_date=args.start_date)
    print(res.to_string(index=False))
    
    if args.output:
        res.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
