"""
submission.py
Official Submission for PS-06 Time Series Analysis of Hospital Admissions
Defines forecast(cutoff_date) according to the HARNESS.md contract.
"""
import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

def forecast(cutoff_date: str) -> pd.DataFrame:
    """
    cutoff_date: 'YYYY-MM-DD'.
    Returns exactly 14 rows: date, mean, lower_80, upper_80 for cutoff_date .. cutoff_date + 13 days.
    """
    data_dir = os.environ.get("PS06_DATA_DIR", "")
    exclude_list = ["forecast", "clean", "summary", "my_"]
    
    # 1. Resolve active data directory
    if not data_dir or not os.path.exists(data_dir) or not resolve_file(data_dir, ["admission"], exclude=exclude_list):
        for candidate_dir in [
            os.path.join(os.getcwd(), "extracted_data", "Datasets"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_data", "Datasets"),
            os.getcwd()
        ]:
            if os.path.exists(candidate_dir) and resolve_file(candidate_dir, ["admission"], exclude=exclude_list):
                data_dir = candidate_dir
                break
                
    if not data_dir:
        data_dir = os.getcwd()
        
    adm_file = resolve_file(data_dir, ["admission"], exclude=exclude_list)
    fes_file = resolve_file(data_dir, ["festival"], exclude=exclude_list)
    
    if not adm_file:
        raise FileNotFoundError(f"Could not find admission file in {data_dir}. Set PS06_DATA_DIR or place HDHI Admission data.csv in directory.")
        
    # 2. Parse admissions with month-year disambiguation (resolves M/D/Y vs D/M/Y)
    adm = pd.read_csv(adm_file)
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    def parse_doa(row):
        doa_str = str(row['D.O.A']).strip()
        my_str = str(row.get('month year', '')).strip()
        try:
            m_name, y_short = my_str.split('-')
            expected_month = month_map[m_name]
            expected_year = 2000 + int(y_short)
        except:
            expected_month = None
            expected_year = None
            
        parts = doa_str.split('/')
        if len(parts) == 3:
            p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
            if p2 < 100:
                p2 += 2000
            if expected_month is not None:
                if p0 == expected_month:
                    month, day, year = p0, p1, p2
                elif p1 == expected_month:
                    day, month, year = p0, p1, p2
                else:
                    month, day, year = p0, p1, p2
            else:
                if p0 > 12:
                    day, month, year = p0, p1, p2
                else:
                    month, day, year = p0, p1, p2
            return pd.Timestamp(year=year, month=month, day=day)
        return pd.NaT

    adm['date'] = adm.apply(parse_doa, axis=1)
    adm_clean = adm.dropna(subset=['date']).copy()
    
    cutoff_dt = pd.to_datetime(cutoff_date)
    # Strictly only observed data before cutoff_date
    adm_clean = adm_clean[adm_clean['date'] < cutoff_dt]
    
    daily_adm = adm_clean.groupby('date').size().rename('cardiac_admissions')
    
    # Create complete date index from min date to cutoff_dt - 1 day
    min_date = daily_adm.index.min()
    max_train_date = cutoff_dt - pd.Timedelta(days=1)
    train_dates = pd.date_range(start=min_date, end=max_train_date, freq='D')
    
    df_train = pd.DataFrame({'date': train_dates})
    df_train = df_train.merge(daily_adm.reset_index(), on='date', how='left')
    df_train['cardiac_admissions'] = df_train['cardiac_admissions'].fillna(0).astype(float)
    
    # 3. Process Public Holidays from Festival Calendar
    holiday_dates = set()
    if fes_file and os.path.exists(fes_file):
        fes = pd.read_csv(fes_file)
        fes['date'] = pd.to_datetime(fes['date'], errors='coerce')
        pub_holidays = fes[fes['type'].astype(str).str.contains('state|national', case=False, na=False)]
        holiday_dates = set(pub_holidays['date'].dt.strftime('%Y-%m-%d'))
        
    def build_features(dates_series):
        feat = pd.DataFrame({'date': pd.to_datetime(dates_series)})
        feat['day_of_year'] = feat['date'].dt.dayofyear
        feat['day_of_week'] = feat['date'].dt.dayofweek
        feat['sin_year'] = np.sin(2 * np.pi * feat['day_of_year'] / 365.25)
        feat['cos_year'] = np.cos(2 * np.pi * feat['day_of_year'] / 365.25)
        
        # Day of week dummies (0=Monday to 5=Saturday, 6=Sunday as base)
        for d in range(6):
            feat[f'dow_{d}'] = (feat['day_of_week'] == d).astype(float)
            
        feat['is_public_holiday'] = feat['date'].dt.strftime('%Y-%m-%d').isin(holiday_dates).astype(float)
        
        dow_cols = [f'dow_{d}' for d in range(6)]
        cols = dow_cols + ['sin_year', 'cos_year', 'is_public_holiday']
        return feat[cols].values
        
    X_train = build_features(df_train['date'])
    y_train = df_train['cardiac_admissions'].values
    
    # 4. Fit SARIMAX(2, 0, 1) x (1, 0, 0, 7)
    order = (2, 0, 1)
    seasonal_order = (1, 0, 0, 7)
    
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = model.fit(disp=False, maxiter=200)
    
    # 5. Build 14-day forecast horizon [cutoff_dt .. cutoff_dt + 13 days]
    future_dates = pd.date_range(start=cutoff_dt, periods=14, freq='D')
    X_future = build_features(future_dates)
    
    fc = fit_res.get_forecast(steps=14, exog=X_future)
    mean_pred = np.maximum(fc.predicted_mean, 0.0)
    ci_80 = fc.conf_int(alpha=0.20)
    
    lower_80 = np.maximum(ci_80[:, 0], 0.0)
    upper_80 = np.maximum(ci_80[:, 1], 0.0)
    
    # Guarantee lower_80 <= mean <= upper_80
    lower_80 = np.minimum(lower_80, mean_pred)
    upper_80 = np.maximum(upper_80, mean_pred)
    
    result_df = pd.DataFrame({
        'date': future_dates.strftime('%Y-%m-%d'),
        'mean': np.round(mean_pred, 2),
        'lower_80': np.round(lower_80, 2),
        'upper_80': np.round(upper_80, 2)
    })
    
    return result_df
