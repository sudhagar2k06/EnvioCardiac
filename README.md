# EnvioCardiac: Daily Hospital Cardiac Admissions Forecasting & Environmental Risk Analysis

An end-to-end, competition-grade time series forecasting and environmental epidemiology platform for daily hospital cardiac admissions at Hero DMC Heart Institute (HDHI), Ludhiana, Punjab.

Developed for the **PS-06 Hospital Admissions Hackathon**, strictly adhering to the SARIMAX modeling requirement and the official evaluation contract defined in `HARNESS.md`.

---

## 🏆 Official Competition Deliverables & Scores

| Deliverable | Description | Location / Script | Outcome / Score |
| :--- | :--- | :--- | :--- |
| **(Core 1) Data Join** | 100% matched join of 15,757 admissions with Ludhiana weather, air quality, & festivals | [`data/clean_daily_dataset.csv`](data/clean_daily_dataset.csv) | 730 continuous daily rows; 0 missing admissions |
| **(Core 2) Delayed Effects** | Multi-Variable Distributed Lag Model (Lags 0–7) with Newey-West HAC standard errors | [`src/delayed_effects_analysis.py`](src/delayed_effects_analysis.py) | Full delay curves for PM2.5, cold drops, DOW & holidays |
| **(Core 3) 14-Day Forecast** | SARIMAX(2,0,1)×(1,0,0)₇ engine conforming strictly to `HARNESS.md` contract | [`submission.py`](submission.py) / [`team_stub.py`](team_stub.py) | **31.86 / 40.00 pts** (MAE ratio 0.651, 87.5% 80%-coverage) |
| **(Core 4) Planning View** | Interactive HTML dashboard with staffing tables & "What-If" scenario stress testing | [`dashboard/hospital_admin_planning_view.html`](dashboard/hospital_admin_planning_view.html) | Standalone interactive visualization & simulation view |

---

## 📁 Repository Directory Structure

```text
EnvioCardiac/
├── submission.py                     # Primary contest entry: defines forecast(cutoff_date) under HARNESS.md
├── team_stub.py                       # Submission verification & quick self-test runner
├── evaluate_accuracy.py               # Standalone program to compute accuracy & render 4-panel evaluation graph
├── predict.py                         # Standalone CLI inference tool for custom dates
├── build_clean_dataset.py             # Root launcher for data cleaning & 4-way join pipeline
├── delayed_effects_analysis.py        # Root launcher for distributed lag statistical analysis
├── test_harness_simulation.py         # Official 4-cutoff evaluation harness simulator (Scores 31.86/40)
├── train_final_model.py               # Model training & serialization script
├── README.md                          # Repository documentation
│
├── src/                               # Modular production source code
│   ├── __init__.py
│   ├── evaluate_accuracy.py           # Model accuracy evaluation and multi-panel graph generator
│   ├── build_clean_dataset.py         # Data cleaning, date disambiguation & 4-way join pipeline
│   ├── delayed_effects_analysis.py    # Multi-Variable Distributed Lag Model (Lags 0 to 7)
│   ├── train_final_model.py           # Trains SARIMAX model and writes artifacts
│   ├── predict.py                     # Programmatic and CLI prediction interface
│   └── test_harness_simulation.py     # Competition evaluation benchmark runner
│
├── data/                              # Structured datasets
│   ├── clean_daily_dataset.csv        # Master joined daily dataset (730 days: 2017-04-01 to 2019-03-31)
│   ├── delayed_effects_summary.csv    # Statistical table of lag coefficients, SEs, p-values, 95% CIs
│   ├── cardiac_data.csv               # Aggregated daily admissions time series
│   └── forecasts/                     # Exported forecast CSVs
│       ├── cardiac_admissions_14day_forecast.csv
│       └── my_forecast.csv
│
├── models/                            # Trained model artifacts
│   └── cardiac_admissions_model.pkl   # Serialized SARIMAX model fit & exog metadata
│
├── figures/                           # Visualizations and diagnostic plots
│   ├── model_accuracy_evaluation.png  # 4-panel comprehensive accuracy evaluation chart
│   ├── lag_response_curves.png        # 4-panel impulse response curves (PM2.5, Cold, DOW, Holidays)
│   ├── forecast_chart.png             # Historical actuals (last 60 days) + 14-day forecast with 80% & 95% CIs
│   └── diagnostic_chart.png           # 4-panel SARIMAX residual diagnostics (Standardized residuals, Q-Q, Correlogram)
│
├── dashboard/                         # Hospital administrator planning interface
│   └── hospital_admin_planning_view.html # Interactive HTML UI with Chart.js & "What-If" scenario sliders
│
├── research/                          # Exploratory analysis & experimental validation
│   ├── eda.py                         # Exploratory data analysis & summary statistics
│   ├── check_acf.py                   # Autocorrelation (ACF) & partial autocorrelation (PACF) analysis
│   ├── experiment_models.py           # Benchmark comparison of baseline vs SARIMAX specifications
│   ├── tune_sarimax.py                # Hyperparameter grid search for (p,d,q) and (P,D,Q)s
│   ├── evaluate_cv.py                 # Rolling-origin cross-validation
│   ├── check_date_format.py           # Investigation of mixed date formats (M/D/Y vs D/M/Y)
│   ├── test_parse_doa.py              # Month-year disambiguation parser validation (100% match)
│   └── verify_admission_aggregation.py# Reconciliation of patient rows vs daily aggregated counts
│
└── extracted_data/Datasets/           # Original raw competition datasets & HARNESS.md
    ├── HARNESS.md
    ├── HDHI Admission data.csv
    ├── HDHI Mortality Data.csv
    ├── AirQuality_Hourly.csv
    ├── Ludhiana_hourly_weather.csv
    └── punjab_festivals_2017_2019.csv
```

---

## 🚀 Execution Guide

### Prerequisites
Run commands using the local Python 3.11 virtual environment (`.venv`):
```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1
```

### 1. Test Competition Submission (`forecast(cutoff_date)`)
Runs the official submission contract for any cutoff date between Jan and Mar 2019:
```powershell
# Run the self-test runner
.venv\Scripts\python.exe team_stub.py

# Or call directly from Python
.venv\Scripts\python.exe -c "from submission import forecast; print(forecast('2019-01-15'))"
```

### 2. Run the Accuracy Evaluation & Multi-Panel Visualization Graph
Runs the 56-day evaluation across all 4 competition cut-offs, calculates accuracy metrics (MAE, RMSE, MAPE, Coverage), and renders the 4-panel accuracy graph:
```powershell
.venv\Scripts\python.exe evaluate_accuracy.py
```
*Outputs to terminal and saves publication-grade plot to [`figures/model_accuracy_evaluation.png`](figures/model_accuracy_evaluation.png).*

### 3. Run the Official Evaluation Harness Simulation (Headless)
Calculates MAE error ratio against seasonal-naive baseline and 80% interval coverage:
```powershell
.venv\Scripts\python.exe test_harness_simulation.py
```

### 4. Rebuild the Clean Master Dataset
Executes the data cleaning, mixed-format date disambiguation, and 4-way join:
```powershell
.venv\Scripts\python.exe build_clean_dataset.py
```
*Outputs to `data/clean_daily_dataset.csv`.*

### 5. Run Delayed Effects & Distributed Lag Analysis
Runs the multi-variable distributed lag model (Lags 0–7) controlling for day-of-week and public holidays:
```powershell
.venv\Scripts\python.exe delayed_effects_analysis.py
```
*Outputs to `data/delayed_effects_summary.csv` and `figures/lag_response_curves.png`.*

### 6. Launch the Hospital Administrator Planning View & CSV Forecaster
Open the interactive dashboard directly in any browser:
```powershell
Start-Process "dashboard/hospital_admin_planning_view.html"
```
*(Features drag-and-drop CSV upload for files from `Datasets.zip`, real-time auto-forecasting of patient admissions when admission CSVs are absent, interactive Chart.js forecasts, nurse/bed allocation formulas, dynamic "What-If" scenario sliders, and one-click "Export Forecast CSV").*

### 7. Generate Custom 14-Day Forecasts (CLI Tool)
```powershell
# Default 14-day forecast from 2019-04-01:
.venv\Scripts\python.exe predict.py --days 14

# Input folder holding CSV files (with or WITHOUT admission data):
.venv\Scripts\python.exe predict.py --data-dir "extracted_data/Datasets" --days 14

# Custom start date and custom output file:
.venv\Scripts\python.exe predict.py --start-date 2019-04-01 --days 14 --output data/forecasts/custom_forecast.csv
```

---

## 📊 Key Analytical Findings

1. **Air Pollution ($\text{PM}_{2.5}$)**:
   - **Acute Surge (Lag 0)**: $+0.125$ cardiac admissions per $+10\ \mu\text{g/m}^3$ spike ($p = 0.030$).
   - **Harvesting Effect (Lag 1)**: $-0.149$ drop ($p = 0.0058$) as fragile patients are hospitalized immediately on exposure day.
   - **Secondary Wave (Lag 5)**: $+0.105$ delayed wave reflecting sustained systemic inflammation.
2. **Cold Exposure**:
   - Delayed peak at **Lag 4** ($+2.05$ admissions per $5^\circ\text{C}$ temperature drop) driven by cold-induced vasoconstriction and blood pressure elevations.
3. **Calendar Confounders**:
   - **Monday Surge**: $+10.75$ admissions over Sunday ($p < 10^{-40}$) due to elective and post-weekend backlog.
   - **Public Holidays**: $-1.99$ dip ($p = 0.0067$) on official state/national holidays.
