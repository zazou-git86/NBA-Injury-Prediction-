import pandas as pd

df = pd.read_parquet("Datasets/nba_features.parquet")
for col in df.columns:
    print(f"{col} — {df[col].dtype} — ex: {df[col].iloc[0]}")