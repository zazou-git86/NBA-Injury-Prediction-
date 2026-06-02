import pandas as pd

# Charger les episodes
inj_files = [
    "../Data_extraction/Injuries/Datasets/nba_injuries_2021_22.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2022_23.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2023_24.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet",
    "../Data_extraction/Injuries/Datasets/nba_injuries_2025_26.parquet",
]
inj = pd.concat([pd.read_parquet(f) for f in inj_files], ignore_index=True)
inj["game_date"] = pd.to_datetime(inj["game_date"])
inj["name_key"] = (inj["first_name"] + " " + inj["last_name"]).str.lower().str.strip()

# Regarder les absences de Kawhi Leonard
kawhi = inj[(inj["name_key"] == "kawhi leonard") & (inj["current_status"] == "Out")]
print(kawhi[["game_date", "current_status", "reason"]].sort_values("game_date").to_string())