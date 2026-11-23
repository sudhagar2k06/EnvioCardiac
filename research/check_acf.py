import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

df = pd.read_csv(('data/cardiac_data.csv' if os.path.exists('data/cardiac_data.csv') else ('../data/cardiac_data.csv' if os.path.exists('../data/cardiac_data.csv') else 'cardiac_data.csv')))
y = df['cardiac_admissions']

acf_vals, acf_conf = acf(y, nlags=28, alpha=0.05)
pacf_vals, pacf_conf = pacf(y, nlags=28, alpha=0.05)

print("Lag | ACF | PACF")
print("-" * 25)
for lag in range(1, 29):
    star_acf = " *" if (acf_vals[lag] < acf_conf[lag, 0] - acf_vals[lag] or acf_vals[lag] > acf_conf[lag, 1] - acf_vals[lag]) else ""
    star_pacf = " *" if (pacf_vals[lag] < pacf_conf[lag, 0] - pacf_vals[lag] or pacf_vals[lag] > pacf_conf[lag, 1] - pacf_vals[lag]) else ""
    print(f"{lag:2d}  | {acf_vals[lag]:.4f} | {pacf_vals[lag]:.4f}")
