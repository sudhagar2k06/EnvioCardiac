# PS-06 forecast harness: contract

You submit one Python file that defines:

```python
def forecast(cutoff_date: str) -> pandas.DataFrame   # cutoff_date is "YYYY-MM-DD"
```

It returns exactly 14 rows with the columns `date, mean, lower_80, upper_80`:

- `date`: the 14 days from `cutoff_date` to `cutoff_date + 13 days`, as `YYYY-MM-DD`, in order.
- `mean`: your forecast of the number of admissions on that day.
- `lower_80`, `upper_80`: an 80% prediction interval. Each value must be finite, and `lower_80 <= mean <= upper_80`.

## What the harness does for each cut-off

1. It makes a new empty folder and copies into it the files that you received. The admission, pollution and weather
   files contain only the days before `cutoff_date`. The festival calendar and `table_headings.csv` are complete.
   For patients still in hospital at the cut-off, the discharge columns are blank.
2. It starts a new Python process with this folder as the working directory and sets the environment variable
   `PS06_DATA_DIR` to the same folder.
3. It imports your file and calls `forecast(cutoff_date)` once. Each call has a time limit of 15 minutes.
   Your code must not read any other data about the future.

Your file may import other modules that are in the same folder as your file. Your code runs once per cut-off in a
fresh process, so keep fitted models in files that you read, or fit inside `forecast`. Use observed weather and air
quality only from before the cut-off, or treat observed values as the forecast, as the problem statement says.
There are about four cut-offs, all between January and March 2019. The cut-off dates are not published.

`team_stub.py` shows the file layout.

## Score for the forecast part (40 points)

| Part | Points | Rule |
| --- | --- | --- |
| Error | 25 | `ratio` = your MAE / MAE of the seasonal-naive baseline (the value of the same weekday in the last observed week). Points = `25 * clip(1 - ratio/2, 0, 1)`. Same as baseline: 12.5. Half the error: 18.75. |
| Coverage | 15 | Share of the 56 true values inside your interval. 80% or more: 15 points. Less: falls linearly to 0 at 40%. Intervals wider than twice the baseline interval lose points in proportion. |

A cut-off where your code fails (error, wrong format, time out) counts as ratio 2 and coverage 0.
The other parts of the score (analysis, data join, demo) are judged.
