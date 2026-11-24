"""
evaluate_accuracy.py
Standalone program to evaluate and visualize model forecast accuracy.

Computes:
- Actual vs. Predicted Admissions across 56 evaluation days (4 hidden cutoffs)
- Comparison against Seasonal-Naive Baseline
- Accuracy metrics: MAE, RMSE, MAPE, R², 80% and 95% Coverage
- Official PS-06 Competition Score breakdown (Error points + Coverage points out of 40)
- Multi-panel publication-quality visualization saved to figures/model_accuracy_evaluation.png

Usage:
    python evaluate_accuracy.py
    python evaluate_accuracy.py --output figures/custom_accuracy.png
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Set matplotlib cache directory to avoid permissions issues
os.environ['MPLCONFIGDIR'] = os.path.abspath('.mpl')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, pearsonr

# Add src and root to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

from submission import forecast

def evaluate_model_accuracy(data_dir=None, clean_dataset_path=None):
    # 1. Resolve data paths
    if data_dir is None:
        cand_dirs = [
            os.path.abspath(r"extracted_data\Datasets"),
            os.path.abspath(os.path.join(ROOT_DIR, r"extracted_data\Datasets")),
            os.getcwd()
        ]
        for cd in cand_dirs:
            if os.path.exists(cd) and any('admission' in f.lower() for f in os.listdir(cd)):
                data_dir = cd
                break
        if not data_dir:
            data_dir = os.path.abspath(r"extracted_data\Datasets")
    os.environ["PS06_DATA_DIR"] = data_dir

    if clean_dataset_path is None:
        cand_clean = [
            os.path.join(ROOT_DIR, 'data', 'clean_daily_dataset.csv'),
            'data/clean_daily_dataset.csv',
            'clean_daily_dataset.csv'
        ]
        for cc in cand_clean:
            if os.path.exists(cc):
                clean_dataset_path = cc
                break
        if not clean_dataset_path or not os.path.exists(clean_dataset_path):
            raise FileNotFoundError("Could not find clean_daily_dataset.csv. Please run build_clean_dataset.py first.")

    # Load ground truth daily admissions
    clean_df = pd.read_csv(clean_dataset_path)
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    ground_truth = dict(zip(clean_df['date'].dt.strftime('%Y-%m-%d'), clean_df['cardiac_admissions']))

    # 4 competition cutoffs (Jan 15, Feb 1, Feb 15, Mar 1, 2019)
    cutoffs = ['2019-01-15', '2019-02-01', '2019-02-15', '2019-03-01']

    results = []
    cutoff_summaries = []

    print("=" * 80)
    print("RUNNING MODEL ACCURACY EVALUATION (56 EVALUATION DAYS ACROSS 4 CUTOFFS)")
    print("=" * 80)

    for c_idx, cutoff in enumerate(cutoffs, 1):
        print(f"[{c_idx}/4] Forecasting Cut-off: {cutoff} ...")
        pred_df = forecast(cutoff)

        cutoff_dt = pd.to_datetime(cutoff)

        # Baseline: seasonal-naive (same weekday in the last observed week)
        cutoff_idx = clean_df[clean_df['date'] < cutoff_dt].index
        hist_adm = clean_df.loc[cutoff_idx, 'cardiac_admissions'].values
        hist_s_naive_err = hist_adm[7:] - hist_adm[:-7]
        baseline_half_width = 1.28 * np.std(hist_s_naive_err)

        for k in range(len(pred_df)):
            row = pred_df.iloc[k]
            d_str = str(row['date'])
            pred_mean = float(row['mean'])
            low_80 = float(row['lower_80'])
            up_80 = float(row['upper_80'])
            
            actual = ground_truth.get(d_str, np.nan)

            # Target date
            target_dt = pd.to_datetime(d_str)
            lag7_dt = target_dt - pd.Timedelta(days=7)
            while lag7_dt >= cutoff_dt:
                lag7_dt -= pd.Timedelta(days=7)
            base_val = ground_truth[lag7_dt.strftime('%Y-%m-%d')]

            is_inside_80 = (actual >= low_80) and (actual <= up_80)

            results.append({
                'cutoff': cutoff,
                'cutoff_num': f"Cut-off {c_idx}",
                'day_index': k + 1,
                'date': d_str,
                'day_of_week': target_dt.day_name(),
                'actual': actual,
                'pred_mean': pred_mean,
                'lower_80': low_80,
                'upper_80': up_80,
                'baseline': base_val,
                'model_err': actual - pred_mean,
                'model_abs_err': abs(actual - pred_mean),
                'base_abs_err': abs(actual - base_val),
                'inside_80': is_inside_80,
                'base_half_width': baseline_half_width
            })

    df_res = pd.DataFrame(results)

    # Compute summary stats per cutoff
    for c_idx, cutoff in enumerate(cutoffs, 1):
        sub = df_res[df_res['cutoff'] == cutoff]
        c_mae = sub['model_abs_err'].mean()
        b_mae = sub['base_abs_err'].mean()
        c_rmse = np.sqrt(np.mean(sub['model_err'] ** 2))
        c_cov = sub['inside_80'].mean() * 100
        cutoff_summaries.append({
            'Cut-off': cutoff,
            'Window': f"{sub['date'].iloc[0]} to {sub['date'].iloc[-1]}",
            'Model MAE': c_mae,
            'Base MAE': b_mae,
            'MAE Ratio': c_mae / b_mae,
            'RMSE': c_rmse,
            '80% Coverage': f"{sub['inside_80'].sum()}/14 ({c_cov:.1f}%)"
        })

    # Overall metrics
    overall_our_mae = df_res['model_abs_err'].mean()
    overall_base_mae = df_res['base_abs_err'].mean()
    overall_ratio = overall_our_mae / overall_base_mae
    overall_rmse = np.sqrt(np.mean(df_res['model_err'] ** 2))
    overall_mape = np.mean(df_res['model_abs_err'] / np.maximum(df_res['actual'], 1)) * 100
    corr, _ = pearsonr(df_res['actual'], df_res['pred_mean'])
    r2 = 1.0 - (np.sum(df_res['model_err'] ** 2) / np.sum((df_res['actual'] - df_res['actual'].mean()) ** 2))

    # Competition scoring rules
    error_points = 25.0 * np.clip(1.0 - (overall_ratio / 2.0), 0.0, 1.0)
    n_inside = df_res['inside_80'].sum()
    coverage_share = n_inside / 56.0

    if coverage_share >= 0.80:
        base_cov_points = 15.0
    elif coverage_share <= 0.40:
        base_cov_points = 0.0
    else:
        base_cov_points = 15.0 * (coverage_share - 0.40) / (0.80 - 0.40)

    mean_our_width = np.mean(df_res['upper_80'] - df_res['lower_80'])
    mean_base_width = np.mean(df_res['base_half_width'] * 2.0)
    width_ratio = mean_our_width / mean_base_width
    coverage_points = base_cov_points * (2.0 / width_ratio) if width_ratio > 2.0 else base_cov_points
    total_score = error_points + coverage_points

    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY BY CUT-OFF:")
    print("=" * 80)
    print(pd.DataFrame(cutoff_summaries).to_string(index=False))

    print("\n" + "=" * 80)
    print("OVERALL MODEL ACCURACY BENCHMARK:")
    print("=" * 80)
    print(f"Overall SARIMAX Model MAE:   {overall_our_mae:.3f} admissions")
    print(f"Overall Seasonal-Naive MAE:  {overall_base_mae:.3f} admissions")
    print(f"Error Ratio (Model / Base):  {overall_ratio:.3f} (Error is {(1-overall_ratio)*100:.1f}% lower than baseline)")
    print(f"RMSE (Root Mean Sq Error):   {overall_rmse:.3f} admissions")
    print(f"MAPE (Mean Abs % Error):     {overall_mape:.1f}%")
    print(f"Correlation (r):             {corr:.3f} (R² = {r2:.3f})")
    print(f"80% Interval Coverage:       {n_inside}/56 ({coverage_share*100:.1f}%)")
    print(f"Avg Interval Width:          {mean_our_width:.2f} (Baseline: {mean_base_width:.2f})")
    print("-" * 80)
    print(f"OFFICIAL COMPETITION SCORE:  {total_score:.2f} / 40.00 POINTS")
    print(f"  • Error Points:            {error_points:.2f} / 25.00")
    print(f"  • Coverage Points:         {coverage_points:.2f} / 15.00")
    print("=" * 80)

    metrics = {
        'our_mae': overall_our_mae,
        'base_mae': overall_base_mae,
        'error_ratio': overall_ratio,
        'rmse': overall_rmse,
        'mape': overall_mape,
        'r2': r2,
        'corr': corr,
        'n_inside': n_inside,
        'coverage_share': coverage_share,
        'error_points': error_points,
        'coverage_points': coverage_points,
        'total_score': total_score,
        'cutoff_summaries': cutoff_summaries
    }

    return df_res, metrics

def plot_accuracy_graph(df_res, metrics, output_path=None):
    if output_path is None:
        out_dir = os.path.join(ROOT_DIR, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'model_accuracy_evaluation.png')
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Style configuration
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.edgecolor'] = '#cbd5e1'
    plt.rcParams['axes.linewidth'] = 1.0

    fig = plt.figure(figsize=(16, 11), facecolor='#ffffff')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # PANEL 1: 56-Day Evaluation Timeline (Actual vs Pred + 80% CI)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    x_indices = np.arange(len(df_res))

    # Shaded 80% Prediction Interval
    ax1.fill_between(
        x_indices,
        df_res['lower_80'],
        df_res['upper_80'],
        color='#93c5fd',
        alpha=0.35,
        label='SARIMAX 80% Prediction Interval'
    )

    # Baseline line
    ax1.plot(
        x_indices,
        df_res['baseline'],
        color='#94a3b8',
        linestyle=':',
        linewidth=1.8,
        label='Seasonal-Naive Baseline'
    )

    # Model Predicted Line
    ax1.plot(
        x_indices,
        df_res['pred_mean'],
        color='#1d4ed8',
        linestyle='-',
        linewidth=2.4,
        marker='s',
        markersize=4.5,
        label='SARIMAX Model Forecast (Mean)'
    )

    # Actual Admissions Points
    ax1.plot(
        x_indices,
        df_res['actual'],
        color='#0f172a',
        linestyle='-',
        linewidth=2.0,
        marker='o',
        markersize=5.5,
        label='Actual Daily Admissions'
    )

    # Highlight inside vs outside interval points
    inside_pts = df_res[df_res['inside_80']]
    outside_pts = df_res[~df_res['inside_80']]
    ax1.scatter(
        inside_pts.index,
        inside_pts['actual'],
        color='#16a34a',
        s=36,
        zorder=5,
        label=f"Inside 80% CI ({len(inside_pts)}/56 = {len(inside_pts)/56*100:.1f}%)"
    )
    if len(outside_pts) > 0:
        ax1.scatter(
            outside_pts.index,
            outside_pts['actual'],
            color='#dc2626',
            s=55,
            marker='x',
            linewidths=2,
            zorder=5,
            label=f"Outside 80% CI ({len(outside_pts)}/56)"
        )

    # Vertical divider lines between the 4 cutoffs
    cutoff_changes = [0, 14, 28, 42]
    cutoff_labels = ['Cut-off 1 (Jan 15)', 'Cut-off 2 (Feb 01)', 'Cut-off 3 (Feb 15)', 'Cut-off 4 (Mar 01)']
    for i, c_idx in enumerate(cutoff_changes):
        if c_idx > 0:
            ax1.axvline(c_idx - 0.5, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.text(
            c_idx + 6.5,
            ax1.get_ylim()[1] if i > 0 else 46,
            cutoff_labels[i],
            ha='center',
            va='top',
            fontsize=10.5,
            fontweight='bold',
            color='#1e293b',
            backgroundcolor='#f1f5f9'
        )

    # Format X ticks with dates
    tick_step = 2
    ax1.set_xticks(x_indices[::tick_step])
    ax1.set_xticklabels([d[5:] for d in df_res['date'].iloc[::tick_step]], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Daily Cardiac Admissions', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_title(
        '56-Day Evaluation Timeline: Actual Admissions vs. SARIMAX Forecast with 80% Prediction Band\n'
        f"Overall Model MAE: {metrics['our_mae']:.2f} vs. Baseline MAE: {metrics['base_mae']:.2f} | Error Reduction: {(1-metrics['error_ratio'])*100:.1f}%",
        fontsize=13,
        fontweight='bold',
        pad=12,
        color='#0f172a'
    )
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9, ncol=3)
    ax1.set_xlim(-0.5, 55.5)

    # -------------------------------------------------------------
    # PANEL 2: Parity Scatter Plot (Actual vs. Predicted)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    cutoff_colors = ['#2563eb', '#7c3aed', '#059669', '#d97706']
    cutoffs = df_res['cutoff'].unique()

    for idx, c in enumerate(cutoffs):
        sub = df_res[df_res['cutoff'] == c]
        ax2.scatter(
            sub['actual'],
            sub['pred_mean'],
            color=cutoff_colors[idx],
            label=f"Cut-off {idx+1}",
            alpha=0.85,
            s=45,
            edgecolors='white',
            linewidths=0.5
        )

    # Identity 45-degree line
    min_val = min(df_res['actual'].min(), df_res['pred_mean'].min()) - 2
    max_val = max(df_res['actual'].max(), df_res['pred_mean'].max()) + 2
    ax2.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=1.8, label='Perfect Parity (y=x)')

    # Fit regression line
    m_slope, b_inter = np.polyfit(df_res['actual'], df_res['pred_mean'], 1)
    x_line = np.linspace(min_val, max_val, 100)
    ax2.plot(x_line, m_slope * x_line + b_inter, color='#1e40af', linestyle='-', linewidth=1.5, label=f'Fit (slope={m_slope:.2f})')

    ax2.set_xlabel('Actual Admissions', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Predicted Admissions', fontsize=10.5, fontweight='bold')
    ax2.set_title('Forecast Parity (Actual vs. Predicted)', fontsize=11.5, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=8.5, frameon=True)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    # Parity stats annotation box
    ax2.text(
        0.95,
        0.05,
        f"Pearson r = {metrics['corr']:.3f}\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f}\n"
        f"MAPE = {metrics['mape']:.1f}%",
        transform=ax2.transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )

    # -------------------------------------------------------------
    # PANEL 3: Error Comparison by Cut-off (Model MAE vs. Baseline MAE)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    summaries = metrics['cutoff_summaries']
    labels = [f"Cut {i+1}\n({s['Cut-off'][5:]})" for i, s in enumerate(summaries)] + ['OVERALL\n(56 Days)']
    model_maes = [s['Model MAE'] for s in summaries] + [metrics['our_mae']]
    base_maes = [s['Base MAE'] for s in summaries] + [metrics['base_mae']]

    x = np.arange(len(labels))
    width = 0.36

    b_rects1 = ax3.bar(x - width/2, model_maes, width, label='SARIMAX Model', color='#2563eb', edgecolor='#1e40af')
    b_rects2 = ax3.bar(x + width/2, base_maes, width, label='Seasonal-Naive Baseline', color='#cbd5e1', edgecolor='#94a3b8')

    # Add error reduction labels above bars
    for idx, (m, b) in enumerate(zip(model_maes, base_maes)):
        pct = (1.0 - m / b) * 100
        ax3.text(
            x[idx],
            max(m, b) + 0.35,
            f"-{pct:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#15803d'
        )

    ax3.set_ylabel('Mean Absolute Error (Admissions)', fontsize=10.5, fontweight='bold')
    ax3.set_title('Forecast Error Reduction vs. Baseline', fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9.5)
    ax3.set_ylim(0, max(base_maes) + 2.0)
    ax3.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax3.legend(loc='upper right', fontsize=8.5, frameon=True)

    # -------------------------------------------------------------
    # PANEL 4: Error Residual Distribution & Official Score Card
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    errors = df_res['model_err']

    # Residual histogram + normal density curve
    n_bins = 14
    counts, bins, _ = ax4.hist(errors, bins=n_bins, density=True, color='#60a5fa', edgecolor='#2563eb', alpha=0.7, label='Error Histogram')
    mu, std = norm.fit(errors)
    x_pdf = np.linspace(errors.min() - 2, errors.max() + 2, 100)
    ax4.plot(x_pdf, norm.pdf(x_pdf, mu, std), color='#1e3a8a', linewidth=2.0, label=f'Normal Fit (μ={mu:.1f}, σ={std:.1f})')
    ax4.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5, label='Zero Error')

    ax4.set_xlabel('Forecast Error (Actual - Predicted)', fontsize=10.5, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=10.5, fontweight='bold')
    ax4.set_title('Residual Distribution & Contest Score', fontsize=11.5, fontweight='bold', pad=10)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper left', fontsize=8, frameon=True)

    # Official Score Callout Box
    score_text = (
        f"★ CONTEST BENCHMARK SCORE\n"
        f"----------------------------------\n"
        f"• Error Points:    {metrics['error_points']:.2f} / 25.00\n"
        f"• Coverage Points: {metrics['coverage_points']:.2f} / 15.00\n"
        f"• TOTAL SCORE:     {metrics['total_score']:.2f} / 40.00\n"
        f"----------------------------------\n"
        f"• 80% CI Coverage: {metrics['coverage_share']*100:.1f}%\n"
        f"• Error Reduction: {(1-metrics['error_ratio'])*100:.1f}%"
    )
    ax4.text(
        0.95,
        0.95,
        score_text,
        transform=ax4.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        fontfamily='monospace',
        fontweight='bold',
        color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#fef08a', edgecolor='#ca8a04', alpha=0.95)
    )

    # Save figure
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.96, hspace=0.35, wspace=0.28)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Accuracy evaluation graph saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SARIMAX model accuracy and output graphical report.")
    parser.add_argument('--output-graph', type=str, default='figures/model_accuracy_evaluation.png', help="Output graph path")
    parser.add_argument('--data-dir', type=str, default=None, help="Directory containing dataset CSVs")
    args = parser.parse_args()

    df_res, metrics = evaluate_model_accuracy(data_dir=args.data_dir)
    plot_accuracy_graph(df_res, metrics, output_path=args.output_graph)
