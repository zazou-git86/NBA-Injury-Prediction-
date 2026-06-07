"""
Preprocessing ML — Prédiction de blessures NBA
Appliqué sur nba_features_v5.parquet avant modélisation.

Corrections :
    1. ACWR clippé à 2.0 (valeurs physiologiquement impossibles)
    2. days_since_last_inj = 999 → NaN + never_injured (binaire)
    3. acwr = 0 → NaN + no_chronic_load (binaire)
    4. has_previous_injury (binaire lisible)
    5. delta_min_7d_28d : variation charge aiguë vs chronique
    6. min_avg_vs_season : rapport charge 10j vs moyenne saison

Output : Datasets/nba_features_ml.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH  = Path("Data_processing/Datasets/nba_features_v5.parquet")
OUTPUT_PATH = Path("Dataset_ML/nba_features_ml.parquet")

print("=" * 55)
print("  Preprocessing ML — NBA injury prediction")
print("=" * 55)

df = pd.read_parquet(INPUT_PATH)
print(f"\nDataset chargé : {len(df):,} lignes | {len(df.columns)} colonnes")

# ── 1. ACWR clippé à 2.0 ─────────────────────────────────────────────────────
print("\n[1] Clipping ACWR...")
n_above = (df["acwr"] > 2.0).sum()
df["acwr"] = df["acwr"].clip(upper=2.0)
print(f"  {n_above:,} valeurs clippées à 2.0")
print(f"  acwr : min={df['acwr'].min():.2f} max={df['acwr'].max():.2f} mean={df['acwr'].mean():.2f}")

# ── 2. days_since_last_inj = 999 → NaN + never_injured ───────────────────────
print("\n[2] Traitement days_since_last_inj...")
n_999 = (df["days_since_last_inj"] == 999).sum()
df["never_injured"]      = (df["days_since_last_inj"] == 999).astype(int)
df["days_since_last_inj"] = df["days_since_last_inj"].replace(999, np.nan)
print(f"  {n_999:,} valeurs 999 → NaN")
print(f"  never_injured : {df['never_injured'].sum():,} joueurs jamais blessés")
print(f"  days_since_last_inj NaN : {df['days_since_last_inj'].isna().sum():,}")

# ── 3. acwr = 0 → NaN + no_chronic_load ──────────────────────────────────────
print("\n[3] Traitement acwr = 0...")
n_zero = (df["acwr"] == 0).sum()
df["no_chronic_load"] = (df["acwr"] == 0).astype(int)
df["acwr"]            = df["acwr"].replace(0, np.nan)
print(f"  {n_zero:,} valeurs 0 → NaN (pas de charge chronique)")
print(f"  no_chronic_load : {df['no_chronic_load'].sum():,} matchs sans charge chronique")
print(f"  acwr NaN : {df['acwr'].isna().sum():,}")

# ── 4. has_previous_injury ────────────────────────────────────────────────────
print("\n[4] Ajout has_previous_injury...")
df["has_previous_injury"] = (df["never_injured"] == 0).astype(int)
print(f"  has_previous_injury = 1 : {df['has_previous_injury'].sum():,} lignes")
print(f"  has_previous_injury = 0 : {(df['has_previous_injury']==0).sum():,} lignes")

# ── 5. delta_min_7d_28d ───────────────────────────────────────────────────────
print("\n[5] Ajout delta_min_7d_28d...")
# Différence entre charge aiguë (7j) et charge chronique (28j)
# Valeur positive = surcharge aiguë, négative = sous-charge récente
df["delta_min_7d_28d"] = df["acute_load_7d"] - df["chronic_load_28d"]
print(f"  delta_min_7d_28d : min={df['delta_min_7d_28d'].min():.1f} "
      f"max={df['delta_min_7d_28d'].max():.1f} "
      f"mean={df['delta_min_7d_28d'].mean():.2f}")
print(f"  Surcharges aiguës (delta > 5) : {(df['delta_min_7d_28d'] > 5).sum():,}")

# ── 6. min_avg_vs_season ──────────────────────────────────────────────────────
print("\n[6] Ajout min_avg_vs_season...")
# Rapport entre la moyenne des minutes sur 10j et la moyenne saison
# > 1 = joue plus que d'habitude récemment, < 1 = moins que d'habitude
df["min_avg_vs_season"] = np.where(
    df["min_season_avg"] > 0,
    (df["min_avg_10d"] / df["min_season_avg"]).round(3),
    np.nan
)
# Clipper les valeurs extrêmes (début de saison avec peu de données)
df["min_avg_vs_season"] = df["min_avg_vs_season"].clip(upper=3.0)
print(f"  min_avg_vs_season : min={df['min_avg_vs_season'].min():.2f} "
      f"max={df['min_avg_vs_season'].max():.2f} "
      f"mean={df['min_avg_vs_season'].mean():.2f}")
print(f"  NaN (début saison) : {df['min_avg_vs_season'].isna().sum():,}")

# ── Vérification finale ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Vérification finale")
print("="*55)

ID_COLS      = ["PLAYER_ID", "PLAYER_NAME", "name_key", "GAME_DATE",
                "MATCHUP", "SEASON", "split"]
TARGET_COL   = "injury_next_10d"
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]

print(f"\nFeatures ML : {len(FEATURE_COLS)}")
print(f"  Nouvelles : never_injured, no_chronic_load, has_previous_injury,")
print(f"              delta_min_7d_28d, min_avg_vs_season")

print("\nNaN par colonne (features ML) :")
nans = df[FEATURE_COLS].isnull().sum()
nans_nonzero = nans[nans > 0]
if len(nans_nonzero) == 0:
    print("  Aucun NaN ✓ (hors features avec NaN intentionnels)")
else:
    for col, n in nans_nonzero.items():
        print(f"  {col:<30} : {n:,} NaN ({n/len(df)*100:.1f}%)")

print(f"\nCible :")
print(f"  injury_next_10d = 1 : {(df[TARGET_COL]==1).sum():,} ({(df[TARGET_COL]==1).mean()*100:.1f}%)")
print(f"  injury_next_10d = 0 : {(df[TARGET_COL]==0).sum():,} ({(df[TARGET_COL]==0).mean()*100:.1f}%)")

print(f"\nSplit :")
print(df.groupby("split")[TARGET_COL].count().rename("lignes").to_string())

# ── Sauvegarde ────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\n{'='*55}")
print(f"  Dataset sauvegardé : {OUTPUT_PATH}")
print(f"{'='*55}")
print(f"  Lignes   : {len(df):,}")
print(f"  Colonnes : {len(df.columns)}")
print(f"  Features ML : {len(FEATURE_COLS)}")