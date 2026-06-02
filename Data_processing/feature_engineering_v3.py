"""
Feature Engineering V3 — Prédiction de blessures NBA
Fenêtre de prédiction : 10 jours

Features v3 — ajout de la gravité des blessures :
    - last_injury_duration_days : durée de la dernière blessure Out (jours)
    - last_injury_severity      : encodage ordinal court/moyen/long/très long
    - total_days_out_season     : total jours d'absence sur la saison en cours

Lancement :
    python feature_engineering_v3.py

Output : Datasets/nba_features_v3.parquet
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_DAYS = 10
OUTPUT_PATH = Path("Datasets/nba_features_v3.parquet")

NAME_MAPPING = {
    "alexandre sarr"    : "alex sarr",
    "carlton carrington": "bub carrington",
    "jimmy butler"      : "jimmy butler iii",
}

CATEGORY_ENCODING = {
    "Lower Body"    : 1,
    "Upper Body"    : 2,
    "Head/Neck"     : 3,
    "Other"         : 4,
    "Reconditioning": 5,
    "Unknown"       : 0,
}

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

def severity_label(days: int) -> int:
    """
    Encodage ordinal de la gravité d'une blessure selon sa durée :
        0 = jamais blessé
        1 = court       (1–3 jours)
        2 = moyen       (4–14 jours)
        3 = long        (15–30 jours)
        4 = très long   (> 30 jours)
    """
    if days <= 0:   return 0
    if days <= 3:   return 1
    if days <= 14:  return 2
    if days <= 30:  return 3
    return 4


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

    logs = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["name_key"]  = logs["PLAYER_NAME"].apply(normalize_name)
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    print(f"  Game logs : {len(logs):,} lignes | {logs['PLAYER_ID'].nunique()} joueurs")

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

    inj = inj[inj["injury_category"].isin([
        "Lower Body", "Upper Body", "Head/Neck", "Other", "Reconditioning"
    ])]
    inj = inj.sort_values("status_code").drop_duplicates(
        subset=["name_key", "game_date"], keep="first"
    )
    print(f"  Injuries  : {len(inj):,} lignes | {inj['name_key'].nunique()} joueurs")

    return logs, inj


# ── Calcul des durées de blessures ────────────────────────────────────────────

def compute_injury_durations(inj: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque épisode de blessure par joueur, calcule la durée réelle en jours.

    Règles de regroupement — même épisode si :
        1. Gap entre deux rapports Out <= 10 jours (couvre road trips, All-Star break,
           jours sans match où aucun rapport n'est publié)
        2. Même body_part normalisé (évite de fusionner deux blessures distinctes
           qui se succèdent chez un joueur fragile comme Kawhi)

    Si l'une des deux conditions n'est pas remplie → nouvel épisode.
    """
    inj_out = inj[inj["current_status"] == "Out"].copy()
    inj_out = inj_out.sort_values(["name_key", "game_date"])

    # Normalisation du body_part pour la comparaison
    inj_out["body_part_norm"] = inj_out["body_part"].fillna("").str.strip().str.lower()

    episodes = []

    for name_key, group in inj_out.groupby("name_key"):
        group = group.sort_values("game_date").reset_index(drop=True)

        start      = group.iloc[0]["game_date"]
        prev       = group.iloc[0]["game_date"]
        prev_bp    = group.iloc[0]["body_part_norm"]
        start_cat  = group.iloc[0].get("injury_category", "")
        start_bp   = group.iloc[0].get("body_part", "")

        for _, row in group.iloc[1:].iterrows():
            d      = row["game_date"]
            gap    = (d - prev).days
            cur_bp = row["body_part_norm"]

            # Même épisode : gap <= 10j ET même partie du corps
            if gap <= 10 and cur_bp == prev_bp:
                prev    = d
                prev_bp = cur_bp
            else:
                episodes.append({
                    "name_key"       : name_key,
                    "injury_start"   : start,
                    "injury_end"     : prev,
                    "duration_days"  : (prev - start).days + 1,
                    "injury_category": start_cat,
                    "body_part"      : start_bp,
                })
                start     = d
                prev      = d
                prev_bp   = cur_bp
                start_cat = row.get("injury_category", "")
                start_bp  = row.get("body_part", "")

        # Dernier épisode
        episodes.append({
            "name_key"       : name_key,
            "injury_start"   : start,
            "injury_end"     : prev,
            "duration_days"  : (prev - start).days + 1,
            "injury_category": start_cat,
            "body_part"      : start_bp,
        })

    df_episodes = pd.DataFrame(episodes)
    print(f"  Épisodes de blessure détectés : {len(df_episodes):,}")
    print(f"  Durée max : {df_episodes['duration_days'].max()} jours")
    print(f"  Durée moyenne : {df_episodes['duration_days'].mean():.1f} jours")
    return df_episodes


# ── Index de blessures ────────────────────────────────────────────────────────

def build_injury_index(inj: pd.DataFrame) -> dict:
    index = {}
    for name_key, group in inj.groupby("name_key"):
        index[name_key] = group.sort_values("game_date").reset_index(drop=True)
    return index

def build_episode_index(episodes: pd.DataFrame) -> dict:
    index = {}
    for name_key, group in episodes.groupby("name_key"):
        index[name_key] = group.sort_values("injury_start").reset_index(drop=True)
    return index


# ── Features glissantes + historique + gravité ────────────────────────────────

def build_rolling_features(logs: pd.DataFrame, inj: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    print("\nConstruction des features...")

    inj_index     = build_injury_index(inj)
    episode_index = build_episode_index(episodes)
    inj_out       = inj[inj["current_status"] == "Out"]

    all_features = []
    total_players = logs["PLAYER_ID"].nunique()

    for idx, (player_id, group) in enumerate(logs.groupby("PLAYER_ID")):
        group    = group.sort_values("GAME_DATE").reset_index(drop=True)
        name_key = group["name_key"].iloc[0]

        player_inj_out = inj_out[inj_out["name_key"] == name_key] if name_key in inj_index else pd.DataFrame()
        player_episodes = episode_index.get(name_key, pd.DataFrame())

        rows = []
        for i, row in group.iterrows():
            current_date = row["GAME_DATE"]
            window_start = current_date - pd.Timedelta(days=WINDOW_DAYS)
            current_season = row["SEASON"]
            season_start   = pd.Timestamp(f"{current_season[:4]}-10-01")

            # ── Charge physique ──
            window = group[
                (group["GAME_DATE"] >= window_start) &
                (group["GAME_DATE"] < current_date)
            ]
            b2b_window = group[
                (group["GAME_DATE"] >= current_date - pd.Timedelta(days=2)) &
                (group["GAME_DATE"] < current_date)
            ]
            n_games = len(window)

            # ── Historique blessures ──
            if len(player_inj_out) > 0:
                past_out    = player_inj_out[player_inj_out["game_date"] < current_date]
                recent_out  = past_out[past_out["game_date"] >= window_start]
                season_out  = past_out[past_out["game_date"] >= season_start]
            else:
                past_out = recent_out = season_out = pd.DataFrame()

            was_injured_10d     = 1 if len(recent_out) > 0 else 0
            injury_count_season = season_out["game_date"].nunique() if len(season_out) > 0 else 0
            if len(past_out) > 0:
                very_recent  = past_out[past_out["game_date"] >= current_date - pd.Timedelta(days=5)]
                is_returning = 1 if len(very_recent) > 0 else 0
            else:
                is_returning = 0

            if len(past_out) > 0:
                last_out_date       = past_out["game_date"].max()
                days_since_last_inj = (current_date - last_out_date).days
                last_row            = past_out.loc[past_out["game_date"].idxmax()]
                last_inj_category   = CATEGORY_ENCODING.get(last_row.get("injury_category", ""), 0)
                raw_bp              = str(last_row.get("body_part", "")).strip().title()
                last_inj_body_part  = BODY_PART_ENCODING.get(raw_bp, 0)
            else:
                days_since_last_inj = 999
                last_inj_category   = 0
                last_inj_body_part  = 0

            # ── Gravité des blessures ──
            if len(player_episodes) > 0:
                past_ep = player_episodes[player_episodes["injury_end"] < current_date]
            else:
                past_ep = pd.DataFrame()

            if len(past_ep) > 0:
                # Dernière blessure : durée et sévérité
                last_ep                   = past_ep.loc[past_ep["injury_end"].idxmax()]
                last_injury_duration      = int(last_ep["duration_days"])
                last_injury_severity      = severity_label(last_injury_duration)

                # Total jours d'absence sur la saison
                season_ep                 = past_ep[past_ep["injury_start"] >= season_start]
                total_days_out_season     = int(season_ep["duration_days"].sum())

                # Pire blessure de la carrière (dans nos données)
                max_injury_duration       = int(past_ep["duration_days"].max())
                max_injury_severity       = severity_label(max_injury_duration)
            else:
                last_injury_duration  = 0
                last_injury_severity  = 0
                total_days_out_season = 0
                max_injury_duration   = 0
                max_injury_severity   = 0

            features = {
                # Identifiants
                "PLAYER_ID"              : player_id,
                "PLAYER_NAME"            : row["PLAYER_NAME"],
                "name_key"               : name_key,
                "GAME_DATE"              : current_date,
                "SEASON"                 : current_season,
                "MATCHUP"                : row["MATCHUP"],
                "IS_AWAY"                : row["IS_AWAY"],

                # Stats du match actuel
                "MIN"                    : row["MIN"],
                "PTS"                    : row["PTS"],
                "REB"                    : row["REB"],
                "AST"                    : row["AST"],

                # Charge sur 10 jours
                "games_10d"              : n_games,
                "min_total_10d"          : window["MIN"].sum(),
                "min_avg_10d"            : window["MIN"].mean() if n_games > 0 else 0,
                "min_max_10d"            : window["MIN"].max() if n_games > 0 else 0,
                "away_games_10d"         : window["IS_AWAY"].sum(),
                "away_ratio_10d"         : window["IS_AWAY"].mean() if n_games > 0 else 0,
                "b2b_10d"                : 1 if len(b2b_window) > 0 else 0,

                # Fatigue saison
                "min_season_avg"         : group[group["GAME_DATE"] < current_date]["MIN"].mean()
                                           if i > 0 else 0,
                "games_season"           : len(group[group["GAME_DATE"] < current_date]),

                # Historique blessures
                "was_injured_10d"        : was_injured_10d,
                "days_since_last_inj"    : days_since_last_inj,
                "injury_count_season"    : injury_count_season,
                "last_inj_category"      : last_inj_category,
                "last_inj_body_part"     : last_inj_body_part,
                "is_returning"           : is_returning,

                # Gravité des blessures (NOUVELLES FEATURES)
                "last_injury_duration"   : last_injury_duration,
                "last_injury_severity"   : last_injury_severity,
                "total_days_out_season"  : total_days_out_season,
                "max_injury_duration"    : max_injury_duration,
                "max_injury_severity"    : max_injury_severity,
            }
            rows.append(features)

        all_features.append(pd.DataFrame(rows))

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{total_players} joueurs traités...")

    df_features = pd.concat(all_features, ignore_index=True)
    print(f"  Features construites : {len(df_features):,} lignes | {len(df_features.columns)} colonnes")
    return df_features


# ── Construction de la cible ──────────────────────────────────────────────────

def build_target(df_features: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
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
    print("  Feature Engineering v3 — Prédiction blessures NBA")
    print("=" * 55)

    logs, inj = load_data()

    # Calcul des épisodes de blessure et leur durée
    print("\nCalcul des durées de blessure...")
    episodes = compute_injury_durations(inj)

    # Features
    df_features = build_rolling_features(logs, inj, episodes)

    # Cible
    df_features = build_target(df_features, inj)

    # Split
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
    print(f"  Lignes   : {len(df_features):,}")
    print(f"  Colonnes : {len(df_features.columns)}")
    print(f"\n  Features de gravité :")
    for col in ["last_injury_duration","last_injury_severity",
                "total_days_out_season","max_injury_duration","max_injury_severity"]:
        print(f"    {col} : min={df_features[col].min():.0f} "
              f"max={df_features[col].max():.0f} "
              f"mean={df_features[col].mean():.1f}")
    print(f"\n  Split :")
    print(df_features.groupby("split")["PLAYER_ID"].count().rename("lignes").to_string())
    print(f"\n  Cible :")
    print(df_features["injury_next_10d"].value_counts().to_string())


if __name__ == "__main__":
    main()