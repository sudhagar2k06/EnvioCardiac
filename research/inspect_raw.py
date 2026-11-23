import os
import glob
import pandas as pd

datasets_dir = r"extracted_data\Datasets"
files = os.listdir(datasets_dir)
print("Files in Datasets dir:", files)

for f in sorted(files):
    if f.endswith('.csv'):
        path = os.path.join(datasets_dir, f)
        print("\n" + "="*70)
        print(f"FILE: {f} (size: {os.path.getsize(path):,} bytes)")
        print("="*70)
        try:
            df = pd.read_csv(path, nrows=5)
            print("Columns (first 15):", list(df.columns)[:15])
            print("Total columns:", len(df.columns))
            print("Sample row 0:\n", df.iloc[0].to_dict())
            
            # Count total rows
            full_df = pd.read_csv(path, usecols=[df.columns[0]])
            print("Total rows:", len(full_df))
        except Exception as e:
            print(f"Error reading {f}: {e}")
