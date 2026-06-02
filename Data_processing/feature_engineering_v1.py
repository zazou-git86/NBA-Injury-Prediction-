"""
Feature Engineering — Prédiction de blessures NBA
Fenêtre de prédiction : 10 jours

Pipeline :
    1. Chargement des game logs et des rapports de blessures
    2. Normalisation des noms (jointure injuries ↔ gamelogs)
    3. Construction des features de charge glissantes (10 jours)
    4. Construction de la cible : blessure dans les 10 prochains jours
    5. Sauvegarde du dataset final

Lancement :
    python feature_engineering.py

Output : Datasets/nba_features.parquet
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_DAYS   = 10      # fenêtre de prédiction (jours)
MIN_GAMES     = 3       # minimum de matchs joués dans la fenêtre pour inclure la ligne
OUTPUT_PATH   = Path("Datasets/nba_features.parquet")

# Saisons d'entraînement uniquement (pas de leakage)
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]
VAL_SEASON    = "2024-25"
TEST_SEASON   = "2025-26"

# Mapping des noms différents entre les deux sources
NAME_MAPPING = {
    "alexandre sarr"    : "alex sarr",
    "carlton carrington": "bub carrington",
    "jimmy butler"      : "jimmy butler iii",
}


# ── Normalisation des noms ────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    return NAME_MAPPING.get(name, name)


# ── Chargement des données ────────────────────────────────────────────────────

def load_data():
    print("Chargement des données...")

    # Game logs
    logs = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["name_key"]  = logs["PLAYER_NAME"].apply(normalize_name)
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    print(f"  Game logs : {len(logs):,} lignes | {logs['PLAYER_ID'].nunique()} joueurs")

    # Injuries (toutes saisons)
    inj_files = [
        "../Data_extraction/Injuries/Datasets/nba_injuries_2021_22.parquet",
        "../Data_extraction/Injuries/Datasets/nba_injuries_2022_23.parquet",
        "../Data_extraction/Injuries/Datasets/nba_injuries_2023_24.parquet",
        "../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet",
        "../Data_extraction/Injuries/Datasets/nba_injuries_2025_26.parquet",
    ]
    inj = pd.concat([pd.read_parquet(f) for f in inj_files], ignore_index=True)
    inj["game_date"] = pd.to_datetime(inj["game_date"])
    inj["name_key"]  = (inj["first_name"] + " " + inj["last_name"]).apply(normalize_name)

    # Garder uniquement les vraies blessures (exclure repos, suspensions)
    inj = inj[inj["injury_category"].isin([
        "Lower Body", "Upper Body", "Head/Neck", "Other", "Reconditioning"
    ])]

    # Un rapport par joueur par date (garder le statut le plus grave)
    inj = inj.sort_values("status_code").drop_duplicates(
        subset=["name_key", "game_date"], keep="first"
    )
    print(f"  Injuries  : {len(inj):,} lignes | {inj['name_key'].nunique()} joueurs")

    return logs, inj


# ── Features glissantes (rolling) ────────────────────────────────────────────

def build_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match joué, calcule les features de charge
    sur les 10 jours PRÉCÉDENTS (pas de leakage).
    """
    print("\nConstruction des features glissantes...")

    all_features = []

    for player_id, group in logs.groupby("PLAYER_ID"):
        group = group.sort_values("GAME_DATE").reset_index(drop=True)

        rows = []
        for i, row in group.iterrows():
            current_date = row["GAME_DATE"]
            window_start = current_date - pd.Timedelta(days=WINDOW_DAYS)

            # Matchs dans la fenêtre des 10 jours AVANT ce match (exclu)
            window = group[
                (group["GAME_DATE"] >= window_start) &
                (group["GAME_DATE"] < current_date)
            ]

            # Matchs dans les 2 jours avant (back-to-back)
            b2b_window = group[
                (group["GAME_DATE"] >= current_date - pd.Timedelta(days=2)) &
                (group["GAME_DATE"] < current_date)
            ]

            n_games = len(window)

            features = {
                # Identifiants
                "PLAYER_ID"        : player_id,
                "PLAYER_NAME"      : row["PLAYER_NAME"],
                "name_key"         : row["name_key"],
                "GAME_DATE"        : current_date,
                "SEASON"           : row["SEASON"],
                "MATCHUP"          : row["MATCHUP"],
                "IS_AWAY"          : row["IS_AWAY"],

                # Stats du match actuel
                "MIN"              : row["MIN"],
                "PTS"              : row["PTS"],
                "REB"              : row["REB"],
                "AST"              : row["AST"],

                # Features de charge sur 10 jours
                "games_10d"        : n_games,
                "min_total_10d"    : window["MIN"].sum(),
                "min_avg_10d"      : window["MIN"].mean() if n_games > 0 else 0,
                "min_max_10d"      : window["MIN"].max() if n_games > 0 else 0,
                "away_games_10d"   : window["IS_AWAY"].sum(),
                "away_ratio_10d"   : window["IS_AWAY"].mean() if n_games > 0 else 0,
                "b2b_10d"          : 1 if len(b2b_window) > 0 else 0,

                # Fatigue cumulée sur la saison
                "min_season_avg"   : group[group["GAME_DATE"] < current_date]["MIN"].mean()
                                     if i > 0 else 0,
                "games_season"     : len(group[group["GAME_DATE"] < current_date]),
            }
            rows.append(features)

        all_features.append(pd.DataFrame(rows))

    df_features = pd.concat(all_features, ignore_index=True)
    print(f"  Features construites : {len(df_features):,} lignes")
    return df_features


# ── Construction de la cible ──────────────────────────────────────────────────

def build_target(df_features: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """
    Cible binaire : le joueur sera-t-il blessé (Out) dans les 10 prochains jours ?
    """
    print("\nConstruction de la cible...")

    # Index des blessures Out par joueur et date
    inj_out = inj[inj["current_status"] == "Out"][["name_key", "game_date"]].copy()

    targets = []
    for _, row in df_features.iterrows():
        future_start = row["GAME_DATE"] + pd.Timedelta(days=1)
        future_end   = row["GAME_DATE"] + pd.Timedelta(days=WINDOW_DAYS)

        # Y a-t-il une blessure Out dans les 10 prochains jours ?
        future_injuries = inj_out[
            (inj_out["name_key"] == row["name_key"]) &
            (inj_out["game_date"] >= future_start) &
            (inj_out["game_date"] <= future_end)
        ]
        targets.append(1 if len(future_injuries) > 0 else 0)

    df_features["injury_next_10d"] = targets

    pos = sum(targets)
    neg = len(targets) - pos
    print(f"  Blessés (1)     : {pos:,} ({pos/len(targets)*100:.1f}%)")
    print(f"  Non blessés (0) : {neg:,} ({neg/len(targets)*100:.1f}%)")

    return df_features


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Feature Engineering — Prédiction blessures NBA")
    print("=" * 55)

    logs, inj = load_data()

    # Features glissantes
    df_features = build_rolling_features(logs)

    # Cible
    df_features = build_target(df_features, inj)

    # Tag train / val / test
    df_features["split"] = df_features["SEASON"].map({
        "2021-22": "train",
        "2022-23": "train",
        "2023-24": "train",
        "2024-25": "val",
        "2025-26": "test",
    })

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"  Dataset sauvegardé : {OUTPUT_PATH}")
    print(f"{'='*55}")
    print(f"  Lignes totales : {len(df_features):,}")
    print(f"  Colonnes       : {len(df_features.columns)}")
    print(f"\n  Répartition split :")
    print(df_features.groupby("split")["PLAYER_ID"].count().rename("lignes").to_string())
    print(f"\n  Distribution cible :")
    print(df_features["injury_next_10d"].value_counts().to_string())


if __name__ == "__main__":
    main()