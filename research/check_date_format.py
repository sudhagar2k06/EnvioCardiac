import pandas as pd

adm = pd.read_csv(r"extracted_data\Datasets\HDHI%20Admission%20data.csv")
print(adm[['D.O.A', 'month year']].drop_duplicates().head(30))
