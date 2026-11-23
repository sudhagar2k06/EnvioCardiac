"""
build_clean_dataset.py
Joins raw hospital admissions, weather, air quality, and Punjab festival calendar
into a clean, daily time-series dataset.
"""
import os
import re
import pandas as pd
import numpy as np

def resolve_file(data_dir, keywords):
    for f in os.listdir(data_dir):
        clean_name = f.lower().replace('%20', ' ').replace('_', ' ')
        if all(k.lower() in clean_name for k in keywords) and f.endswith('.csv'):
            return os.path.join(data_dir, f)
    return None

def build_dataset(data_dir=None, output_path=None):
    if data_dir is None:
        for cand in [r"extracted_data\Datasets", r"..\extracted_data\Datasets"]:
            if os.path.exists(cand):
                data_dir = cand
                break
    if output_path is None:
        out_base = "data" if os.path.exists("data") or not os.path.exists("../data") else "../data"
        output_path = os.path.join(out_base, "clean_daily_dataset.csv")
    # 1. Locate files
    adm_file = resolve_file(data_dir, ["admission"])
    pol_file = resolve_file(data_dir, ["pollution"])
    wea_file = resolve_file(data_dir, ["weather"])
    fes_file = resolve_file(data_dir, ["festival"])
    mor_file = resolve_file(data_dir, ["mortality"])
    
    if not (adm_file and wea_file and pol_file and fes_file):
        raise FileNotFoundError(f"Missing required CSV files in {data_dir}")
        
    print(f"Loading Admission data: {adm_file}")
    adm = pd.read_csv(adm_file)
    
    print(f"Loading Weather data: {wea_file}")
    wea = pd.read_csv(wea_file)
    
    print(f"Loading Pollution data: {pol_file}")
    pol = pd.read_csv(pol_file)
    
    print(f"Loading Festival data: {fes_file}")
    fes = pd.read_csv(fes_file)
    
    # 2. Parse admission dates reliably using 'month year' disambiguation
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    def parse_doa(row):
        doa_str = str(row['D.O.A']).strip()
        my_str = str(row['month year']).strip()
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
    
    # Aggregate admissions daily: overall + clinical subcategories
    daily_total = adm_clean.groupby('date').size().rename('cardiac_admissions')
    
    # Optional clinical sub-breakdown
    has_emergency = 'TYPE OF ADMISSION-EMERGENCY/OPD' in adm_clean.columns
    if has_emergency:
        daily_emerg = adm_clean[adm_clean['TYPE OF ADMISSION-EMERGENCY/OPD'] == 'E'].groupby('date').size().rename('emergency_admissions')
        daily_opd = adm_clean[adm_clean['TYPE OF ADMISSION-EMERGENCY/OPD'] == 'O'].groupby('date').size().rename('opd_admissions')
    else:
        daily_emerg = pd.Series(dtype=float)
        daily_opd = pd.Series(dtype=float)
        
    daily_adm = pd.concat([daily_total, daily_emerg, daily_opd], axis=1).fillna(0).reset_index()
    
    # 3. Process Weather Data
    wea['date'] = pd.to_datetime(wea['date'])
    weather_cols = ['date', 'temp_mean_c', 'temp_min_c', 'temp_max_c', 'precip_mm', 'rel_humidity_mean_pct', 'wind_mean_kmh', 'wind_max_kmh']
    wea_clean = wea[[c for c in weather_cols if c in wea.columns]].copy()
    wea_clean = wea_clean.rename(columns={'rel_humidity_mean_pct': 'humidity_mean_pct'})
    
    # 4. Process Pollution Data
    pol['date'] = pd.to_datetime(pol['DATE'])
    pol_map = {
        'AQI': 'aqi',
        'PM2.5 AVG': 'pm25_avg',
        'PM10 AVG': 'pm10_avg',
        'NO2 AVG': 'no2_avg',
        'NH3 AVG': 'nh3_avg',
        'SO2 AVG': 'so2_avg',
        'CO AVG': 'co_avg',
        'OZONE AVG': 'ozone_avg'
    }
    avail_pol_cols = ['date'] + [orig for orig in pol_map if orig in pol.columns]
    pol_clean = pol[avail_pol_cols].rename(columns=pol_map)
    
    # 5. Process Festival & Holiday Calendar
    fes['date'] = pd.to_datetime(fes['date'])
    fes_clean = fes.groupby('date').agg({
        'name': lambda s: ' | '.join(sorted(set(str(x) for x in s if pd.notnull(x)))),
        'type': lambda s: ' | '.join(sorted(set(str(x) for x in s if pd.notnull(x))))
    }).reset_index()
    fes_clean.columns = ['date', 'calendar_event_names', 'calendar_event_types']
    
    # 6. Join All Datasets
    # Determine date range
    start_date = max(daily_adm['date'].min(), wea_clean['date'].min(), pol_clean['date'].min())
    end_date = min(daily_adm['date'].max(), wea_clean['date'].max(), pol_clean['date'].max())
    
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    master = pd.DataFrame({'date': full_dates})
    
    master = master.merge(daily_adm, on='date', how='left')
    master['cardiac_admissions'] = master['cardiac_admissions'].fillna(0).astype(int)
    
    master = master.merge(wea_clean, on='date', how='left')
    master = master.merge(pol_clean, on='date', how='left')
    master = master.merge(fes_clean, on='date', how='left')
    
    # Calendar flags
    master['is_festival'] = master['calendar_event_types'].str.contains('festival', case=False, na=False).astype(int)
    master['is_public_holiday'] = master['calendar_event_types'].str.contains('state|national', case=False, na=False).astype(int)
    master['day_of_week'] = master['date'].dt.day_of_week
    master['month'] = master['date'].dt.month
    master['is_weekend'] = master['day_of_week'].isin([5, 6]).astype(int)
    
    # Forward fill missing weather/pollution if minor single-day gaps exist
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].interpolate(method='linear').bfill().ffill()
    
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        master.to_csv(output_path, index=False)
        print(f"\nSaved joined dataset to {output_path} ({len(master)} rows, {len(master.columns)} columns)")
        
    return master

if __name__ == '__main__':
    df = build_dataset()
    print("Head:\n", df[['date', 'cardiac_admissions', 'temp_mean_c', 'aqi', 'pm25_avg', 'is_public_holiday', 'is_weekend']].head())
    print("\nSummary of cardiac admissions:")
    print(df['cardiac_admissions'].describe())
