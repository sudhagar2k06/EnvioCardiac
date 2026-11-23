import pandas as pd
import numpy as np

adm = pd.read_csv(r"extracted_data\Datasets\HDHI%20Admission%20data.csv")
prev = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
prev['date'] = pd.to_datetime(prev['date'])

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def parse_doa(row):
    doa_str = str(row['D.O.A']).strip()
    my_str = str(row['month year']).strip()
    
    # Extract expected month and year from 'month year' e.g. 'Apr-17'
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
                # M/D/Y
                month, day, year = p0, p1, p2
            elif p1 == expected_month:
                # D/M/Y
                day, month, year = p0, p1, p2
            else:
                # fallback
                month, day, year = p0, p1, p2
        else:
            if p0 > 12:
                day, month, year = p0, p1, p2
            else:
                month, day, year = p0, p1, p2
        return pd.Timestamp(year=year, month=month, day=day)
    return pd.NaT

adm['parsed_date'] = adm.apply(parse_doa, axis=1)
daily_adm = adm.groupby('parsed_date').size().rename('parsed_count')

merged = pd.merge(prev[['date', 'cardiac_admissions']], daily_adm.reset_index(), left_on='date', right_on='parsed_date', how='left')

matches = (merged['cardiac_admissions'] == merged['parsed_count']).sum()
print(f"Exact matches with cardiac_data.csv: {matches} / {len(merged)} ({(matches/len(merged))*100:.1f}%)")

diff = merged[merged['cardiac_admissions'] != merged['parsed_count']]
print("Differences (first 10):")
print(diff.head(10))
