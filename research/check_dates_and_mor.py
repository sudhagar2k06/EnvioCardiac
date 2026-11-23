import pandas as pd

adm = pd.read_csv(r"extracted_data\Datasets\HDHI%20Admission%20data.csv")
mor = pd.read_csv(r"extracted_data\Datasets\HDHI%20Mortality%20Data.csv")

print("Mortality columns:", mor.columns)
print("Mortality head:\n", mor.head())

# Check date format in admissions
# Look at month year column in adm!
print("\nSample DOA with month year:")
print(adm[['D.O.A', 'month year']].head(10))

# If D.O.A is '4/1/2017' and month year is 'Apr-17', then 4 is Month, 1 is Day!
# But what about '1/4/2017'?
sample_jan = adm[adm['month year'] == 'Apr-17']
print("April 1, 2017 rows in adm:", len(adm[adm['D.O.A'] == '4/1/2017']))
print("April 2, 2017 rows in adm:", len(adm[adm['D.O.A'] == '4/2/2017']))

# Check brought dead on 4/1/2017
mor['mor_date'] = pd.to_datetime(mor['DATE OF BROUGHT DEAD'], format='%m/%d/%Y', errors='coerce')
print("Brought dead on 4/1/2017:", len(mor[mor['DATE OF BROUGHT DEAD'] == '4/1/2017']))
print("Brought dead on 4/2/2017:", len(mor[mor['DATE OF BROUGHT DEAD'] == '4/2/2017']))
print("Brought dead on 4/5/2017:", len(mor[mor['DATE OF BROUGHT DEAD'] == '4/5/2017']))
print("Brought dead on 4/7/2017:", len(mor[mor['DATE OF BROUGHT DEAD'] == '4/7/2017']))
print("Brought dead on 4/8/2017:", len(mor[mor['DATE OF BROUGHT DEAD'] == '4/8/2017']))
