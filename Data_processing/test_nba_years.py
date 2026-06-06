import pandas as pd
import numpy as np

df      = pd.read_parquet("Datasets/nba_features_v5.parquet")
info    = pd.read_parquet("Datasets/nba_player_info.parquet")
logs    = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

# ── Fix nba_years ──
# Calculer la première apparition dans les game logs pour chaque joueur
first_game = logs.groupby("PLAYER_ID")["GAME_DATE"].min().reset_index()
first_game.columns = ["PLAYER_ID", "FIRST_GAME_DATE"]

df = df.merge(first_game, on="PLAYER_ID", how="left")

# Recalculer nba_years depuis la première apparition pour les NaN
mask = df["nba_years"].isna()
df.loc[mask, "nba_years"] = (
    (df.loc[mask, "GAME_DATE"] - df.loc[mask, "FIRST_GAME_DATE"]).dt.days / 365.25
).round(1)
df = df.drop(columns=["FIRST_GAME_DATE"])

# ── Fix weight_lbs ──
# Remplacer par la médiane du poste
median_weight = df.groupby("position_code")["weight_lbs"].median()
for pos, median in median_weight.items():
    mask = df["weight_lbs"].isna() & (df["position_code"] == pos)
    df.loc[mask, "weight_lbs"] = median

# ── Vérification ──
print("NaN restants :")
nans = df.isnull().sum()
nans = nans[nans > 0]
print(nans.to_string() if len(nans) > 0 else "  Aucun NaN ✓")

print(f"\nnba_years : min={df['nba_years'].min():.1f} max={df['nba_years'].max():.1f} mean={df['nba_years'].mean():.1f}")
print(f"weight_lbs : min={df['weight_lbs'].min():.0f} max={df['weight_lbs'].max():.0f} mean={df['weight_lbs'].mean():.1f}")

# ── Sauvegarde ──
df.to_parquet("Datasets/nba_features_v5.parquet", index=False)
print("\nnba_features_v5.parquet mis à jour ✓")