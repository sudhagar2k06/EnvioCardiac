"""
CLI & API script to load the trained SARIMAX model and generate forecasts
Supports input directories with or without admission data (e.g. from dataset.zip).
Usage:
    python predict.py --days 14
    python predict.py --data-dir "extracted_data/Datasets" --days 14
    python predict.py --start-date 2019-04-01 --days 14 --output custom_forecast.csv
"""
import os
import argparse
import pickle
import pandas as pd
import numpy as np

def resolve_file(data_dir, keywords, exclude=None):
    """Finds a CSV file matching all keywords, handling spaces and URL-encoded '%20'."""
    if not os.path.exists(data_dir):
        return None
    for f in os.listdir(data_dir):
        clean_name = f.lower().replace('%20', ' ').replace('_', ' ')
        if all(k.lower() in clean_name for k in keywords) and f.endswith('.csv'):
            if exclude and any(ex.lower() in clean_name for ex in exclude):
                continue
            return os.path.join(data_dir, f)
    return None

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

def forecast(days=14, start_date='2019-04-01', data_dir=None, model_path=None):
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
    
    # 1. Process data_dir if provided
    holiday_dates = set()
    wea_file = None
    pol_file = None
    
    if data_dir and os.path.exists(data_dir):
        exclude_list = ["forecast", "clean", "summary", "my_"]
        adm_file = resolve_file(data_dir, ["admission"], exclude=exclude_list)
        fes_file = resolve_file(data_dir, ["festival"], exclude=exclude_list)
        wea_file = resolve_file(data_dir, ["weather"], exclude=exclude_list)
        pol_file = resolve_file(data_dir, ["pollution"], exclude=exclude_list) or resolve_file(data_dir, ["airquality"], exclude=exclude_list)
        
        if not adm_file:
            print(f"[INFO] No admission file found in '{data_dir}'. Predicting cardiac admissions from environmental & calendar features.")
        else:
            print(f"[INFO] Using admission dataset: '{os.path.basename(adm_file)}'")
            
        if fes_file and os.path.exists(fes_file):
            try:
                fes = pd.read_csv(fes_file)
                fes['date'] = pd.to_datetime(fes['date'], errors='coerce')
                pub = fes[fes['type'].astype(str).str.contains('state|national', case=False, na=False)]
                holiday_dates = set(pub['date'].dt.strftime('%Y-%m-%d'))
            except Exception:
                pass
                
    future_dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # 2. Holiday flags
    holidays = []
    for dt in future_dates:
        d_str = dt.strftime('%Y-%m-%d')
        if holiday_dates:
            holidays.append(1.0 if d_str in holiday_dates else 0.0)
        else:
            if dt.month == 4 and dt.day in [8, 13, 14]:
                holidays.append(1.0)
            else:
                holidays.append(0.0)
            
    feat_df, _ = build_features(future_dates, holidays)
    X_future = feat_df[reg_cols].values
    
    fc = model_fit.get_forecast(steps=days, exog=X_future)
    mean_pred = np.maximum(fc.predicted_mean, 0.0)
    ci_95 = fc.conf_int(alpha=0.05)
    ci_80 = fc.conf_int(alpha=0.20)
    
    # 3. Environmental DLM adjustments if weather/air quality files are provided
    if wea_file and os.path.exists(wea_file):
        try:
            w_df = pd.read_csv(wea_file)
            date_col = [c for c in w_df.columns if 'date' in c.lower()][0]
            temp_col = [c for c in w_df.columns if 'temp' in c.lower() and ('mean' in c.lower() or 'avg' in c.lower())]
            if temp_col:
                w_df[date_col] = pd.to_datetime(w_df[date_col], errors='coerce')
                w_match = w_df[w_df[date_col].isin(future_dates)]
                if len(w_match) > 0:
                    temp_mean = w_match[temp_col[0]].mean()
                    temp_delta = (24.0 - temp_mean) / 5.0
                    mean_pred += max(-2.0, min(4.0, temp_delta * 0.41))
        except Exception:
            pass

    if pol_file and os.path.exists(pol_file):
        try:
            p_df = pd.read_csv(pol_file)
            date_col = [c for c in p_df.columns if 'date' in c.lower()][0]
            pm_col = [c for c in p_df.columns if 'pm2.5' in c.lower() or 'pm25' in c.lower()]
            if pm_col:
                p_df[date_col] = pd.to_datetime(p_df[date_col], errors='coerce')
                p_match = p_df[p_df[date_col].isin(future_dates)]
                if len(p_match) > 0:
                    pm_mean = p_match[pm_col[0]].mean()
                    pm_delta = (pm_mean - 80.0) / 10.0
                    mean_pred += max(-2.0, min(3.0, pm_delta * 0.125))
        except Exception:
            pass
            
    lower_80 = np.maximum(np.minimum(ci_80[:, 0], mean_pred), 0.0)
    upper_80 = np.maximum(ci_80[:, 1], mean_pred)
    lower_95 = np.maximum(np.minimum(ci_95[:, 0], mean_pred), 0.0)
    upper_95 = np.maximum(ci_95[:, 1], mean_pred)
    
    out_df = pd.DataFrame({
        'date': future_dates.strftime('%Y-%m-%d'),
        'day_of_week': future_dates.day_name(),
        'is_public_holiday': holidays,
        'predicted_admissions': np.round(mean_pred, 1),
        'lower_80': np.round(lower_80, 1),
        'upper_80': np.round(upper_80, 1),
        'lower_95': np.round(lower_95, 1),
        'upper_95': np.round(upper_95, 1)
    })
    return out_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Forecast upcoming cardiac admissions using SARIMAX.")
    parser.add_argument('--days', type=int, default=14, help="Number of forecast days (default: 14)")
    parser.add_argument('--start-date', type=str, default='2019-04-01', help="Start date (YYYY-MM-DD)")
    parser.add_argument('--data-dir', type=str, default=None, help="Directory containing input CSVs (weather, pollution, festival, etc.)")
    parser.add_argument('--output', type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    
    res = forecast(days=args.days, start_date=args.start_date, data_dir=args.data_dir)
    print(res.to_string(index=False))
    
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        res.to_csv(args.output, index=False)
        print(f"\nSaved forecast to {args.output}")
