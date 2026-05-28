""" import pandas as pd

inj = pd.read_parquet("../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet")
logs = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
logs = logs[logs["SEASON"] == "2024-25"]

# Format des noms
print("=== Injuries ===")
print(inj[["player_name", "last_name", "first_name", "game_date"]].head(5).to_string())

print("\n=== Game logs ===")
print(logs[["PLAYER_NAME", "PLAYER_ID", "GAME_DATE"]].head(5).to_string())

# Tester la jointure sur un joueur connu
print("\n=== Test jointure LeBron ===")
print(inj[inj["last_name"] == "James"][["player_name", "game_date"]].head(3).to_string())
print(logs[logs["PLAYER_NAME"] == "LeBron James"][["PLAYER_NAME", "GAME_DATE"]].head(3).to_string()) """


import pandas as pd

inj = pd.read_parquet("../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet")
logs = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
logs = logs[logs["SEASON"] == "2024-25"]

# Normalisation : "Harden, James" → "james harden"
inj["name_key"] = (inj["first_name"] + " " + inj["last_name"]).str.lower().str.strip()

# Normalisation : "James Harden" → "james harden"
logs["name_key"] = logs["PLAYER_NAME"].str.lower().str.strip()

# Joueurs présents dans injuries mais absents des gamelogs
inj_names  = set(inj["name_key"].unique())
logs_names = set(logs["name_key"].unique())

only_inj  = inj_names - logs_names
only_logs = logs_names - inj_names

print(f"Joueurs dans injuries uniquement : {len(only_inj)}")
print(sorted(list(only_inj))[:20])

print(f"\nJoueurs dans gamelogs uniquement : {len(only_logs)}")
print(sorted(list(only_logs))[:20])