import pandas as pd

df = pd.read_parquet("Datasets/nba_features_v5.parquet")

print("NaN par colonne :")
nans = df.isnull().sum()
print(nans[nans > 0].to_string() if len(nans[nans > 0]) > 0 else "  Aucun NaN ✓")

print(f"\nFeatures ML ({len([c for c in df.columns if c not in ['PLAYER_ID','PLAYER_NAME','name_key','GAME_DATE','MATCHUP','split','injury_next_10d','SEASON']])}) :")
for c in df.columns:
    if c not in ['PLAYER_ID','PLAYER_NAME','name_key','GAME_DATE','MATCHUP','split','injury_next_10d','SEASON']:
        print(f"  {c}")