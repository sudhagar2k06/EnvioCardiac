"""Your submission: define forecast(cutoff_date). See HARNESS.md for the contract."""
import os
import pandas as pd


def forecast(cutoff_date):
    """cutoff_date is 'YYYY-MM-DD'. Return 14 rows: date, mean, lower_80, upper_80 for cutoff_date .. +13 days."""
    data_dir = os.environ["PS06_DATA_DIR"]  # same as the working directory
    raise NotImplementedError
