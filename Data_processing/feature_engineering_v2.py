"""
Feature Engineering V2 — Prédiction de blessures NBA
Fenêtre de prédiction : 10 jours

Nouveautés v2 : features d'historique de blessures ajoutées comme variables explicatives
    - was_injured_10d        : était-il Out sur blessure dans les 10 derniers jours ?
    - days_since_last_injury : jours depuis la dernière blessure Out
    - injury_count_season    : nombre de fois Out sur blessure depuis le début de saison
    - last_injury_category   : catégorie de la dernière blessure (encodée)
    - last_injury_body_part  : partie du corps de la dernière blessure (encodée)
    - is_returning           : revenait-il de blessure ce match ?

Lancement :
    python feature_engineering_v2.py

Output : Datasets/nba_features_v2.parquet
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_DAYS = 10
OUTPUT_PATH = Path("Datasets/nba_features_v2.parquet")

NAME_MAPPING = {
    "alexandre sarr"    : "alex sarr",
    "carlton carrington": "bub carrington",
    "jimmy butler"      : "jimmy butler iii",
}

# Encodage ordinal des catégories de blessures
CATEGORY_ENCODING = {
    "Lower Body"    : 1,
    "Upper Body"    : 2,
    "Head/Neck"     : 3,
    "Other"         : 4,
    "Reconditioning": 5,
    "Unknown"       : 0,
}

# Encodage ordinal des parties du corps les plus fréquentes
BODY_PART_ENCODING = {
    "Knee"      : 1,
    "Ankle"     : 2,
    "Hamstring" : 3,
    "Foot"      : 4,
    "Shoulder"  : 5,
    "Back"      : 6,
    "Calf"      : 7,
    "Hip"       : 8,
    "Wrist"     : 9,
    "Hand"      : 10,
    "Head"      : 11,
    "Achilles"  : 12,
    "Groin"     : 13,
    "Elbow"     : 14,
    "Quad"      : 15,
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

    # Injuries
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

    # Garder uniquement les vraies blessures
    inj = inj[inj["injury_category"].isin([
        "Lower Body", "Upper Body", "Head/Neck", "Other", "Reconditioning"
    ])]

    # Un rapport par joueur par date (statut le plus grave en premier)
    inj = inj.sort_values("status_code").drop_duplicates(
        subset=["name_key", "game_date"], keep="first"
    )
    print(f"  Injuries  : {len(inj):,} lignes | {inj['name_key'].nunique()} joueurs")

    return logs, inj


# ── Préparation de l'index de blessures ──────────────────────────────────────

def build_injury_index(inj: pd.DataFrame) -> dict:
    """
    Construit un index {name_key -> DataFrame} pour accès rapide
    aux blessures passées d'un joueur, trié par date.
    """
    index = {}
    for name_key, group in inj.groupby("name_key"):
        index[name_key] = group.sort_values("game_date").reset_index(drop=True)
    return index


# ── Features glissantes + historique blessures ────────────────────────────────

def build_rolling_features(logs: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match joué, calcule :
    - Les features de charge sur les 10 jours précédents (game logs)
    - Les features d'historique de blessures (rapports injuries)
    Aucune donnée future n'est utilisée — pas de leakage.
    """
    print("\nConstruction des features glissantes + historique blessures...")

    inj_index = build_injury_index(inj)
    inj_out   = inj[inj["current_status"] == "Out"]

    all_features = []

    for player_id, group in logs.groupby("PLAYER_ID"):
        group    = group.sort_values("GAME_DATE").reset_index(drop=True)
        name_key = group["name_key"].iloc[0]

        # Blessures de ce joueur (historique complet)
        player_inj     = inj_index.get(name_key, pd.DataFrame())
        player_inj_out = inj_out[inj_out["name_key"] == name_key] if len(player_inj) > 0 else pd.DataFrame()

        rows = []
        for i, row in group.iterrows():
            current_date = row["GAME_DATE"]
            window_start = current_date - pd.Timedelta(days=WINDOW_DAYS)

            # ── Features de charge (game logs) ──
            window = group[
                (group["GAME_DATE"] >= window_start) &
                (group["GAME_DATE"] < current_date)
            ]
            b2b_window = group[
                (group["GAME_DATE"] >= current_date - pd.Timedelta(days=2)) &
                (group["GAME_DATE"] < current_date)
            ]
            n_games = len(window)

            # ── Features d'historique de blessures ──
            # Blessures Out strictement AVANT ce match
            if len(player_inj_out) > 0:
                past_out = player_inj_out[player_inj_out["game_date"] < current_date]
            else:
                past_out = pd.DataFrame()

            # était-il Out dans les 10 derniers jours ?
            if len(past_out) > 0:
                recent_out = past_out[past_out["game_date"] >= window_start]
                was_injured_10d = 1 if len(recent_out) > 0 else 0
            else:
                was_injured_10d = 0

            # jours depuis la dernière blessure Out
            if len(past_out) > 0:
                last_out_date       = past_out["game_date"].max()
                days_since_last_inj = (current_date - last_out_date).days
            else:
                days_since_last_inj = 999  # jamais blessé → valeur sentinelle

            # nombre de fois Out sur blessure depuis le début de la saison
            current_season = row["SEASON"]
            if len(past_out) > 0:
                season_start_approx = pd.Timestamp(f"{current_season[:4]}-10-01")
                season_out = past_out[past_out["game_date"] >= season_start_approx]
                injury_count_season = season_out["game_date"].nunique()
            else:
                injury_count_season = 0

            # catégorie et partie du corps de la dernière blessure
            if len(past_out) > 0:
                last_row             = past_out.loc[past_out["game_date"].idxmax()]
                last_inj_category    = CATEGORY_ENCODING.get(last_row.get("injury_category", ""), 0)
                raw_body_part        = str(last_row.get("body_part", "")).strip().title()
                last_inj_body_part   = BODY_PART_ENCODING.get(raw_body_part, 0)
            else:
                last_inj_category  = 0
                last_inj_body_part = 0

            # revenait-il d'une blessure ? (était Out dans les 5 derniers jours)
            if len(past_out) > 0:
                very_recent = past_out[past_out["game_date"] >= current_date - pd.Timedelta(days=5)]
                is_returning = 1 if len(very_recent) > 0 else 0
            else:
                is_returning = 0

            features = {
                # Identifiants
                "PLAYER_ID"           : player_id,
                "PLAYER_NAME"         : row["PLAYER_NAME"],
                "name_key"            : name_key,
                "GAME_DATE"           : current_date,
                "SEASON"              : current_season,
                "MATCHUP"             : row["MATCHUP"],
                "IS_AWAY"             : row["IS_AWAY"],

                # Stats du match actuel
                "MIN"                 : row["MIN"],
                "PTS"                 : row["PTS"],
                "REB"                 : row["REB"],
                "AST"                 : row["AST"],

                # Features de charge sur 10 jours
                "games_10d"           : n_games,
                "min_total_10d"       : window["MIN"].sum(),
                "min_avg_10d"         : window["MIN"].mean() if n_games > 0 else 0,
                "min_max_10d"         : window["MIN"].max() if n_games > 0 else 0,
                "away_games_10d"      : window["IS_AWAY"].sum(),
                "away_ratio_10d"      : window["IS_AWAY"].mean() if n_games > 0 else 0,
                "b2b_10d"             : 1 if len(b2b_window) > 0 else 0,

                # Fatigue cumulée sur la saison
                "min_season_avg"      : group[group["GAME_DATE"] < current_date]["MIN"].mean()
                                        if i > 0 else 0,
                "games_season"        : len(group[group["GAME_DATE"] < current_date]),

                # Historique de blessures (NOUVELLES FEATURES)
                "was_injured_10d"     : was_injured_10d,
                "days_since_last_inj" : days_since_last_inj,
                "injury_count_season" : injury_count_season,
                "last_inj_category"   : last_inj_category,
                "last_inj_body_part"  : last_inj_body_part,
                "is_returning"        : is_returning,
            }
            rows.append(features)

        all_features.append(pd.DataFrame(rows))

    df_features = pd.concat(all_features, ignore_index=True)
    print(f"  Features construites : {len(df_features):,} lignes | {len(df_features.columns)} colonnes")
    return df_features


# ── Construction de la cible ──────────────────────────────────────────────────

def build_target(df_features: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """
    Cible binaire : le joueur sera-t-il Out sur blessure dans les 10 prochains jours ?
    """
    print("\nConstruction de la cible...")

    inj_out = inj[inj["current_status"] == "Out"][["name_key", "game_date"]].copy()

    targets = []
    for _, row in df_features.iterrows():
        future_start = row["GAME_DATE"] + pd.Timedelta(days=1)
        future_end   = row["GAME_DATE"] + pd.Timedelta(days=WINDOW_DAYS)

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
    print("  Feature Engineering v2 — Prédiction blessures NBA")
    print("=" * 55)

    logs, inj = load_data()

    # Features glissantes + historique blessures
    df_features = build_rolling_features(logs, inj)

    # Cible
    df_features = build_target(df_features, inj)

    # Split temporel
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
    print(f"\n  Nouvelles features de blessures :")
    for col in ["was_injured_10d","days_since_last_inj","injury_count_season",
                "last_inj_category","last_inj_body_part","is_returning"]:
        print(f"    {col} : {df_features[col].value_counts().to_dict()}" if df_features[col].nunique() < 6
              else f"    {col} : min={df_features[col].min():.0f} max={df_features[col].max():.0f} mean={df_features[col].mean():.1f}")
    print(f"\n  Répartition split :")
    print(df_features.groupby("split")["PLAYER_ID"].count().rename("lignes").to_string())
    print(f"\n  Distribution cible :")
    print(df_features["injury_next_10d"].value_counts().to_string())


if __name__ == "__main__":
    main()