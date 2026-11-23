import pandas as pd
import numpy as np

wea = pd.read_csv(r"extracted_data\Datasets\weather_ludhiana_daily.csv")
pol = pd.read_csv(r"extracted_data\Datasets\HDHI%20Pollution%20Data.csv")
fes = pd.read_csv(r"extracted_data\Datasets\festival_calendar_punjab.csv")

wea['date'] = pd.to_datetime(wea['date'])
pol['date'] = pd.to_datetime(pol['DATE'])
fes['date'] = pd.to_datetime(fes['date'])

print(f"Weather date range: {wea['date'].min()} to {wea['date'].max()} ({len(wea)} rows)")
print(f"Pollution date range: {pol['date'].min()} to {pol['date'].max()} ({len(pol)} rows)")
print(f"Festival date range: {fes['date'].min()} to {fes['date'].max()} ({len(fes)} rows)")

print("\nWeather cols:", list(wea.columns))
print("Pollution cols:", list(pol.columns))
print("Festival cols:", list(fes.columns))
