import pandas as pd
import numpy as np

df = pd.read_parquet("Datasets/nba_features_v4.parquet")

print("=" * 55)
print("  VÉRIFICATION DATASET v4")
print("=" * 55)

# 1. Saisons
print("\n[1] Saisons présentes :")
print(df.groupby("SEASON")["PLAYER_ID"].count().rename("lignes").to_string())

# 2. Split
print("\n[2] Split :")
print(df.groupby("split")["PLAYER_ID"].count().rename("lignes").to_string())

# 3. Cible
print("\n[3] Cible :")
print(df["injury_next_10d"].value_counts().to_string())
print(f"  NaN : {df['injury_next_10d'].isna().sum()}")

# 4. Correction b2b
print("\n[4] b2b (vrai back-to-back) :")
print(df["b2b"].value_counts().to_string())

# 5. Correction ACWR
print("\n[5] ACWR :")
print(f"  min={df['acwr'].min():.2f} max={df['acwr'].max():.2f} mean={df['acwr'].mean():.2f}")
print(f"  Lignes ACWR > 1.5 (zone risque) : {(df['acwr'] > 1.5).sum():,}")
print(f"  Lignes ACWR = 0 (pas de charge chronique) : {(df['acwr'] == 0).sum():,}")

# 6. NaN par colonne
print("\n[6] Valeurs manquantes :")
nans = df.isnull().sum()
nans = nans[nans > 0]
if len(nans) == 0:
    print("  Aucun NaN ✓")
else:
    print(nans.to_string())

# 7. Cohérence temporelle — pas de leakage
print("\n[7] Cohérence temporelle :")
print(f"  Date min : {df['GAME_DATE'].min().date()}")
print(f"  Date max : {df['GAME_DATE'].max().date()}")
train_max = df[df['split']=='train']['GAME_DATE'].max()
val_min   = df[df['split']=='val']['GAME_DATE'].min()
test_min  = df[df['split']=='test']['GAME_DATE'].min()
print(f"  Train max : {train_max.date()} | Val min : {val_min.date()} | Test min : {test_min.date()}")
print(f"  Pas de chevauchement : {'✓' if train_max < val_min else '✗ PROBLÈME'}")

# 8. Valeurs aberrantes
print("\n[8] Valeurs aberrantes :")
print(f"  MIN > 60 : {(df['MIN'] > 60).sum()} lignes")
print(f"  games_10d > 6 : {(df['games_10d'] > 6).sum()} lignes")
print(f"  acwr > 3 : {(df['acwr'] > 3).sum()} lignes")
print(f"  days_since_last_inj = 999 (jamais blessé) : {(df['days_since_last_inj'] == 999).sum():,}")

# 9. Features à ne jamais mettre dans le modèle
print("\n[9] Colonnes identifiants (à exclure du modèle ML) :")
id_cols = ["PLAYER_ID", "PLAYER_NAME", "name_key", "GAME_DATE", "MATCHUP", "split"]
print(f"  {id_cols}")

# 10. Features ML disponibles
feature_cols = [c for c in df.columns if c not in id_cols + ["injury_next_10d", "SEASON"]]
print(f"\n[10] Features ML ({len(feature_cols)}) :")
for c in feature_cols:
    print(f"  {c}")