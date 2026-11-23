import os
import glob
import pandas as pd
import numpy as np

datasets_dir = r"extracted_data\Datasets"

# Check filenames (both decoded and raw)
def find_file(pattern):
    for f in os.listdir(datasets_dir):
        if pattern.lower() in f.lower().replace('%20', ' '):
            return os.path.join(datasets_dir, f)
    return None

adm_path = find_file("admission")
pol_path = find_file("pollution")
wea_path = find_file("weather")
fes_path = find_file("festival")
mor_path = find_file("mortality")

print("Found files:")
print("Admission:", adm_path)
print("Pollution:", pol_path)
print("Weather:  ", wea_path)
print("Festival: ", fes_path)
print("Mortality:", mor_path)

# Load Admission Data
adm = pd.read_csv(adm_path)
print("\nAdmission shape:", adm.shape)
print("Admission D.O.A sample:", adm['D.O.A'].head().tolist())

# Convert D.O.A to datetime
adm['date'] = pd.to_datetime(adm['D.O.A'], format='%m/%d/%Y', errors='coerce')
if adm['date'].isnull().any():
    adm['date'] = pd.to_datetime(adm['D.O.A'], errors='coerce')
    
daily_adm = adm.groupby('date').size().reset_index(name='cardiac_admissions')
print(f"Daily admissions range: {daily_adm['date'].min()} to {daily_adm['date'].max()}, count: {len(daily_adm)}")
print("Daily admissions head:\n", daily_adm.head())

# Compare with our previous cardiac_data.csv
prev_data = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
prev_data['date'] = pd.to_datetime(prev_data['date'])
merged = pd.merge(prev_data[['date', 'cardiac_admissions']], daily_adm, on='date', suffixes=('_prev', '_new'))
match = (merged['cardiac_admissions_prev'] == merged['cardiac_admissions_new']).all()
print("\nDo daily admissions match cardiac_data.csv exactly?", match)
if not match:
    diff = merged[merged['cardiac_admissions_prev'] != merged['cardiac_admissions_new']]
    print("Differences:\n", diff.head())
