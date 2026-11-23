"""
delayed_effects_analysis.py
Performs distributed lag statistical analysis to evaluate:
- Which conditions (PM2.5, Temperature, Humidity, Day of Week, Holidays) affect cardiac admissions
- By how much (effect size per unit/IQR change)
- After what delay (Lags 0 through 7)
- Cumulative multi-day effect with 95% confidence intervals and p-values.
Separates overlapping calendar and environmental effects.
"""
import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import norm

data_path = 'data/clean_daily_dataset.csv' if os.path.exists('data/clean_daily_dataset.csv') else 'clean_daily_dataset.csv'
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Calendar features to control for confounding
df['day_of_year'] = df['date'].dt.day_of_year
df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Day of week dummies (Sunday is reference)
for d in range(6):
    df[f'dow_{d}'] = (df['day_of_week'] == d).astype(float)
dow_cols = [f'dow_{d}' for d in range(6)]

# Generate Lags for Environmental Exposures (Lags 0 to 7)
MAX_LAG = 7

# PM2.5 (per 10 ug/m3 increment for interpretability)
df['pm25_scaled'] = df['pm25_avg'] / 10.0
for l in range(MAX_LAG + 1):
    df[f'pm25_lag{l}'] = df['pm25_scaled'].shift(l)

# Temperature (per 5 C drop: invert sign so positive coefficient means admissions increase with cold)
df['cold_scaled'] = -df['temp_mean_c'] / 5.0
for l in range(MAX_LAG + 1):
    df[f'cold_lag{l}'] = df['cold_scaled'].shift(l)

# Humidity (per 10% increment)
df['hum_scaled'] = df['humidity_mean_pct'] / 10.0
for l in range(MAX_LAG + 1):
    df[f'hum_lag{l}'] = df['hum_scaled'].shift(l)

control_cols = dow_cols + ['sin_year', 'cos_year', 'is_public_holiday']
pm25_lag_cols = [f'pm25_lag{l}' for l in range(MAX_LAG + 1)]
cold_lag_cols = [f'cold_lag{l}' for l in range(MAX_LAG + 1)]
hum_lag_cols = [f'hum_lag{l}' for l in range(MAX_LAG + 1)]

all_exog = ['const'] + control_cols + pm25_lag_cols + cold_lag_cols + hum_lag_cols
df['const'] = 1.0

# Drop only the first MAX_LAG rows where lagged columns are NaN
df_reg = df.iloc[MAX_LAG:].copy().reset_index(drop=True)

print(f"Regression dataset sample size: {len(df_reg)} days")

# Fit OLS with Newey-West HAC standard errors (autocorrelation and heteroskedasticity robust)
model_dlm = sm.OLS(df_reg['cardiac_admissions'], df_reg[all_exog])
results_dlm = model_dlm.fit(cov_type='HAC', cov_kwds={'maxlags': 7})

print("\n" + "="*75)
print("DISTRIBUTED LAG MODEL (DLM) ESTIMATION RESULTS (HAC Robust SE)")
print("="*75)

# Extract lag-specific effects
def extract_lag_table(results, prefix, label, scale_desc):
    rows = []
    lag_coeffs = []
    lag_ses = []
    for l in range(MAX_LAG + 1):
        col = f'{prefix}_lag{l}'
        coef = results.params[col]
        se = results.bse[col]
        pval = results.pvalues[col]
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        lag_coeffs.append(coef)
        lag_ses.append(se)
        rows.append({
            'Exposure': label,
            'Lag (Days)': l,
            'Effect Size': coef,
            'Std. Error': se,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'p-value': pval
        })
        
    # Cumulative Effect
    lag_cols = [f'{prefix}_lag{l}' for l in range(MAX_LAG + 1)]
    sub_cov = results.cov_params().loc[lag_cols, lag_cols].values
    cum_coef = np.sum(lag_coeffs)
    cum_se = np.sqrt(np.sum(sub_cov))
    cum_pval = 2 * (1 - norm.cdf(abs(cum_coef / cum_se)))
    
    rows.append({
        'Exposure': f"{label} (CUMULATIVE Lags 0-7)",
        'Lag (Days)': '0-7',
        'Effect Size': cum_coef,
        'Std. Error': cum_se,
        '95% CI Lower': cum_coef - 1.96 * cum_se,
        '95% CI Upper': cum_coef + 1.96 * cum_se,
        'p-value': cum_pval
    })
    return pd.DataFrame(rows)

pm25_table = extract_lag_table(results_dlm, 'pm25', 'PM2.5 (+10 ug/m3)', 'per +10 ug/m3')
cold_table = extract_lag_table(results_dlm, 'cold', 'Temperature Drop (+5 C Cold)', 'per 5 C drop')

print("\n--- PM2.5 Lag Structure (Change in Daily Admissions per +10 ug/m3) ---")
print(pm25_table.to_string(index=False))

print("\n--- Cold Temperature Lag Structure (Change in Daily Admissions per 5 C Drop) ---")
print(cold_table.to_string(index=False))

# Confounding & Day of Week Effects
print("\n--- Calendar & Confounder Effects (Separated from Weather/Pollution) ---")
cal_rows = []
for c in control_cols:
    coef = results_dlm.params[c]
    se = results_dlm.bse[c]
    pval = results_dlm.pvalues[c]
    cal_rows.append({
        'Variable': c,
        'Effect (Admissions)': coef,
        'Std. Error': se,
        '95% CI': f"[{coef - 1.96*se:.2f}, {coef + 1.96*se:.2f}]",
        'p-value': pval
    })
print(pd.DataFrame(cal_rows).to_string(index=False))

# 2. Plot Lag Response Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PM2.5 Plot
pm_lags = pm25_table.iloc[:MAX_LAG + 1].copy()
lags_x = pm_lags['Lag (Days)'].astype(int).values
ax1.plot(lags_x, pm_lags['Effect Size'].astype(float).values, marker='o', color='#d62728', linewidth=2.5, label='Estimated Lag Effect')
ax1.fill_between(lags_x, pm_lags['95% CI Lower'].astype(float).values, pm_lags['95% CI Upper'].astype(float).values, color='#d62728', alpha=0.2, label='95% Confidence Interval')
ax1.axhline(0, color='black', linestyle='--', alpha=0.6)
ax1.set_title('Air Pollution (PM2.5): Lag-Response Curve\nEffect of +10 $\\mu$g/m$^3$ on Admissions', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Lag Delay (Days after Exposure)', fontsize=11)
ax1.set_ylabel('Change in Cardiac Admissions / Day', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Temperature Plot
cold_lags = cold_table.iloc[:MAX_LAG + 1].copy()
cold_x = cold_lags['Lag (Days)'].astype(int).values
ax2.plot(cold_x, cold_lags['Effect Size'].astype(float).values, marker='s', color='#1f77b4', linewidth=2.5, label='Estimated Lag Effect')
ax2.fill_between(cold_x, cold_lags['95% CI Lower'].astype(float).values, cold_lags['95% CI Upper'].astype(float).values, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')
ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
ax2.set_title('Cold Exposure: Lag-Response Curve\nEffect of 5$^{\\circ}$C Temperature Drop on Admissions', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Lag Delay (Days after Exposure)', fontsize=11)
ax2.set_ylabel('Change in Cardiac Admissions / Day', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig_path = os.path.join('figures', 'lag_response_curves.png')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"\nSaved lag-response curves to '{fig_path}'")

# Save tables to CSV for reference
combined_table = pd.concat([pm25_table, cold_table], axis=0)
os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'delayed_effects_summary.csv')
combined_table.to_csv(csv_path, index=False)
print(f"Saved delayed effects table to '{csv_path}'")
