"""
test_harness_simulation.py
Simulates the exact evaluation process defined in HARNESS.md:
- Tests 4 hidden cut-off dates between January and March 2019 (56 evaluation days).
- Compares against seasonal-naive baseline (same weekday in the last observed week).
- Computes MAE ratio and Error Points (out of 25).
- Computes 80% interval coverage and Coverage Points (out of 15).
- Total forecast score out of 40.
"""
import os
import pandas as pd
import numpy as np
from submission import forecast

# Set environment variable to data dir
os.environ["PS06_DATA_DIR"] = os.path.abspath(r"extracted_data\Datasets")

# Load ground truth daily admissions
clean_path = 'data/clean_daily_dataset.csv' if os.path.exists('data/clean_daily_dataset.csv') else ('../data/clean_daily_dataset.csv' if os.path.exists('../data/clean_daily_dataset.csv') else 'clean_daily_dataset.csv')
clean_df = pd.read_csv(clean_path)
clean_df['date'] = pd.to_datetime(clean_df['date'])
ground_truth = dict(zip(clean_df['date'].dt.strftime('%Y-%m-%d'), clean_df['cardiac_admissions']))

# 4 cut-off dates between January and March 2019
CUTOFF_DATES = [
    '2019-01-15',
    '2019-02-01',
    '2019-02-15',
    '2019-03-01'
]

print("="*75)
print("SIMULATING OFFICIAL HARNESS SCORING (4 CUT-OFF DATES, 56 EVALUATION DAYS)")
print("="*75)

all_preds = []
all_actuals = []
all_baselines = []
all_lowers = []
all_uppers = []
all_baseline_intervals = []

cutoff_summaries = []

for cutoff in CUTOFF_DATES:
    print(f"\nEvaluating Cut-off Date: {cutoff} ...")
    
    # Run candidate forecast
    pred_df = forecast(cutoff)
    
    # 1. Contract checks
    assert len(pred_df) == 14, f"Expected 14 rows, got {len(pred_df)}"
    assert list(pred_df.columns) == ['date', 'mean', 'lower_80', 'upper_80'], f"Invalid columns: {pred_df.columns}"
    assert (pred_df['lower_80'] <= pred_df['mean']).all(), "Violation: lower_80 must be <= mean"
    assert (pred_df['mean'] <= pred_df['upper_80']).all(), "Violation: mean must be <= upper_80"
    assert np.isfinite(pred_df['mean']).all(), "Non-finite values found"
    
    # 2. Get ground truth for the 14 days
    actuals = np.array([ground_truth[d] for d in pred_df['date']])
    means = pred_df['mean'].values
    lowers = pred_df['lower_80'].values
    uppers = pred_df['upper_80'].values
    
    # 3. Compute seasonal-naive baseline:
    # "the value of the same weekday in the last observed week"
    # For day k in 0..13: baseline is ground_truth[cutoff_date - 7 + (k % 7)]
    cutoff_dt = pd.to_datetime(cutoff)
    baseline_vals = []
    for k in range(14):
        target_dt = pd.to_datetime(pred_df['date'].iloc[k])
        lag7_dt = target_dt - pd.Timedelta(days=7)
        # If lag7_dt is >= cutoff_dt, use lag14_dt
        while lag7_dt >= cutoff_dt:
            lag7_dt -= pd.Timedelta(days=7)
        baseline_vals.append(ground_truth[lag7_dt.strftime('%Y-%m-%d')])
    baseline_vals = np.array(baseline_vals)
    
    # Baseline interval width (estimated from rolling historical weekday residuals)
    # Typically baseline interval width is about +/- 1.28 * std(residuals)
    cutoff_idx = clean_df[clean_df['date'] < cutoff_dt].index
    hist_adm = clean_df.loc[cutoff_idx, 'cardiac_admissions'].values
    hist_s_naive_err = hist_adm[7:] - hist_adm[:-7]
    baseline_half_width = 1.28 * np.std(hist_s_naive_err)
    
    cutoff_mae = np.mean(np.abs(actuals - means))
    baseline_mae = np.mean(np.abs(actuals - baseline_vals))
    coverage_count = np.sum((actuals >= lowers) & (actuals <= uppers))
    
    cutoff_summaries.append({
        'Cut-off': cutoff,
        'Our MAE': cutoff_mae,
        'Base MAE': baseline_mae,
        'MAE Ratio': cutoff_mae / baseline_mae,
        'Coverage': f"{coverage_count}/14 ({coverage_count/14*100:.1f}%)",
        'Avg Interval Width': np.mean(uppers - lowers)
    })
    
    all_preds.extend(means)
    all_actuals.extend(actuals)
    all_baselines.extend(baseline_vals)
    all_lowers.extend(lowers)
    all_uppers.extend(uppers)
    all_baseline_intervals.append(baseline_half_width * 2)

print("\n--- Per-Cutoff Evaluation Breakdown ---")
print(pd.DataFrame(cutoff_summaries).to_string(index=False))

# --- Overall Harness Scoring ---
all_preds = np.array(all_preds)
all_actuals = np.array(all_actuals)
all_baselines = np.array(all_baselines)
all_lowers = np.array(all_lowers)
all_uppers = np.array(all_uppers)

overall_our_mae = np.mean(np.abs(all_actuals - all_preds))
overall_base_mae = np.mean(np.abs(all_actuals - all_baselines))
overall_ratio = overall_our_mae / overall_base_mae

# Rule from HARNESS.md:
# Points = 25 * clip(1 - ratio/2, 0, 1)
error_points = 25.0 * np.clip(1.0 - (overall_ratio / 2.0), 0.0, 1.0)

# Coverage rule:
# Share of the 56 true values inside interval
# 80% or more: 15 points. Less: falls linearly to 0 at 40%.
# Intervals wider than twice the baseline interval lose points in proportion.
n_inside = np.sum((all_actuals >= all_lowers) & (all_actuals <= all_uppers))
coverage_share = n_inside / 56.0

if coverage_share >= 0.80:
    base_cov_points = 15.0
elif coverage_share <= 0.40:
    base_cov_points = 0.0
else:
    base_cov_points = 15.0 * (coverage_share - 0.40) / (0.80 - 0.40)

# Check width penalty
mean_our_width = np.mean(all_uppers - all_lowers)
mean_base_width = np.mean(all_baseline_intervals)
width_ratio = mean_our_width / mean_base_width
if width_ratio > 2.0:
    coverage_points = base_cov_points * (2.0 / width_ratio)
else:
    coverage_points = base_cov_points

total_score = error_points + coverage_points

print("\n" + "="*75)
print("FINAL HARNESS SCORE (OUT OF 40 POINTS):")
print("="*75)
print(f"Overall Candidate MAE:      {overall_our_mae:.3f} admissions")
print(f"Overall Seasonal-Naive MAE: {overall_base_mae:.3f} admissions")
print(f"Error Ratio (Our / Base):   {overall_ratio:.3f} (Our error is {(1-overall_ratio)*100:.1f}% lower than baseline!)")
print(f"Error Points:               {error_points:.2f} / 25.00")
print(f"\nInterval Coverage (56 pts): {n_inside}/56 ({coverage_share*100:.1f}%)")
print(f"Average Interval Width:     {mean_our_width:.2f} (Baseline Width: {mean_base_width:.2f})")
print(f"Coverage Points:            {coverage_points:.2f} / 15.00")
print("-" * 75)
print(f"TOTAL FORECAST SCORE:       {total_score:.2f} / 40.00 POINTS")
print("="*75)
