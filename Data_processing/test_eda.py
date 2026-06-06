import pandas as pd

inj_files = [
    "../Data_extraction/Injuries/Datasets/nba_injuries_2021_22.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2022_23.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2023_24.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2025_26.parquet",
]
inj = pd.concat([pd.read_parquet(f) for f in inj_files], ignore_index=True)

print(inj["season"].value_counts())
print(inj["season"].isna().sum(), "NaN")