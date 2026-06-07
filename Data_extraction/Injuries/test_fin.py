import pandas as pd

files = [
    "Datasets/nba_injuries_2021_22.parquet",
    "Datasets/nba_injuries_2022_23.parquet",
    "Datasets/nba_injuries_2023_24.parquet",
    "Datasets/nba_injuries_2024_25.parquet",
    "Datasets/nba_injuries_2025_26.parquet",
]

for f in files:
    df = pd.read_parquet(f)
    print(f"{f} → {len(df):,} lignes | {df['player_name'].nunique()} joueurs | {df['report_timestamp'].nunique()} rapports")
    #print(df.head())
    print(df.isnull().sum())



df_ = pd.read_parquet("Datasets/nba_injuries_2025_26.parquet")
print(df_[df_["body_part"].isna()]["reason"].value_counts().head(10))