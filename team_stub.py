"""Your submission: define forecast(cutoff_date). See HARNESS.md for the contract."""
import os
import pandas as pd
from submission import forecast

if __name__ == '__main__':
    # Quick self-test
    res = forecast('2019-01-15')
    print("Test run on 2019-01-15:")
    print(res)
