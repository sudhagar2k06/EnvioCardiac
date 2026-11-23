import pandas as pd
import numpy as np

adm = pd.read_csv(r"extracted_data\Datasets\HDHI%20Admission%20data.csv")
mor = pd.read_csv(r"extracted_data\Datasets\HDHI%20Mortality%20Data.csv")
prev = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
prev['date'] = pd.to_datetime(prev['date'])

# Notice date format: Some dates might be M/D/YYYY
def parse_dates_with_month_year(df, date_col, month_col=None):
    # D.O.A is M/D/YYYY
    return pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')

adm['date'] = parse_dates_with_month_year(adm, 'D.O.A')
mor['date'] = parse_dates_with_month_year(mor, 'DATE OF BROUGHT DEAD')

adm_counts = adm.groupby('date').size().rename('adm_count')
mor_counts = mor.groupby('date').size().rename('mor_count')

comb = pd.concat([adm_counts, mor_counts], axis=1).fillna(0)
comb['total'] = comb['adm_count'] + comb['mor_count']

merged = pd.merge(prev[['date', 'cardiac_admissions']], comb.reset_index(), on='date', how='left')

print("Match between total (adm + mor) and cardiac_admissions:")
merged['diff'] = merged['cardiac_admissions'] - merged['total']
print("Number of exact matches:", (merged['diff'] == 0).sum(), "out of", len(merged))
if (merged['diff'] != 0).any():
    print("Mismatched dates (first 10):")
    print(merged[merged['diff'] != 0][['date', 'cardiac_admissions', 'adm_count', 'mor_count', 'total', 'diff']].head(10))
