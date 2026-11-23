---
license: other
pretty_name: "PS06 Time Series Analysis of Hospital Admissions (student data)"
tags: [hackathon, hospital]
---
# PS06: Time Series Analysis of Hospital Admissions

**Theme:** Data Preparation | 24-hour hackathon, teams of two, AI coding tools allowed.

## What you build
Relate daily cardiac admissions in Ludhiana to weather, air quality and festivals, and forecast the next 14 days with prediction intervals.

## What is in this repository
Hospital admission, pollution and mortality files exactly as published (Apr 2017 to Mar 2019), daily Ludhiana weather, a Punjab festival calendar, `HARNESS.md` (the forecast contract) and `team_stub.py`.

## Output format
A Python file defining `forecast(cutoff_date)` that returns 14 rows `date,mean,lower_80,upper_80` (see `HARNESS.md`).

## Source and licence
Hospital Admissions Data, Hero DMC Heart Institute (Kaggle, CC BY-NC-SA 4.0; cite Bollepalli et al., Diagnostics 2022); weather from Open-Meteo (CC BY 4.0). Use this data only for the event and for non-commercial work, with attribution. Hidden test data and answer keys are not included.
Read the student problem statement document for the full task, scoring and rules.
