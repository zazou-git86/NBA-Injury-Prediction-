import pandas as pd

df = pd.read_parquet("Datasets/nba_features_v3.parquet")
print(df["SEASON"].value_counts())
