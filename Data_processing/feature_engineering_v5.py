"""
Feature Engineering V5 — Prédiction de blessures NBA
Fenêtre de prédiction : 10 jours

Ajouts v5 :
    - age_at_game      : âge du joueur au moment du match (années décimales)
    - nba_years        : années d'expérience NBA au moment du match
    - position_code    : poste encodé (1=Guard → 5=Center)
    - height_cm        : taille en cm
    - weight_lbs       : poids en lbs

Lancement :
    python feature_engineering_v5.py

Output : Datasets/nba_features_v5.parquet
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_DAYS   = 10
CHRONIC_DAYS  = 28
OUTPUT_PATH   = Path("Datasets/nba_features_v5.parquet")

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
    if days <= 0:  return 0
    if days <= 3:  return 1
    if days <= 14: return 2
    if days <= 30: return 3
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
    print(f"  Game logs    : {len(logs):,} lignes | {logs['PLAYER_ID'].nunique()} joueurs")

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
    print(f"  Injuries     : {len(inj):,} lignes | {inj['name_key'].nunique()} joueurs")

    # Infos joueurs
    player_info = pd.read_parquet("Datasets/nba_player_info.parquet")
    player_info["BIRTHDATE"] = pd.to_datetime(player_info["BIRTHDATE"])
    print(f"  Player info  : {len(player_info):,} joueurs")

    return logs, inj, player_info


# ── Calcul des épisodes de blessure ──────────────────────────────────────────

def compute_injury_durations(inj: pd.DataFrame) -> pd.DataFrame:
    inj_out = inj[inj["current_status"] == "Out"].copy()
    inj_out = inj_out.sort_values(["name_key", "game_date"])
    inj_out["body_part_norm"] = inj_out["body_part"].fillna("").str.strip().str.lower()

    episodes = []
    for name_key, group in inj_out.groupby("name_key"):
        group     = group.sort_values("game_date").reset_index(drop=True)
        start     = group.iloc[0]["game_date"]
        prev      = group.iloc[0]["game_date"]
        prev_bp   = group.iloc[0]["body_part_norm"]
        start_cat = group.iloc[0].get("injury_category", "")
        start_bp  = group.iloc[0].get("body_part", "")

        for _, row in group.iloc[1:].iterrows():
            d      = row["game_date"]
            gap    = (d - prev).days
            cur_bp = row["body_part_norm"]

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

        episodes.append({
            "name_key"       : name_key,
            "injury_start"   : start,
            "injury_end"     : prev,
            "duration_days"  : (prev - start).days + 1,
            "injury_category": start_cat,
            "body_part"      : start_bp,
        })

    df_ep = pd.DataFrame(episodes)
    print(f"  Épisodes : {len(df_ep):,} | Max : {df_ep['duration_days'].max()}j | Moy : {df_ep['duration_days'].mean():.1f}j")
    return df_ep


# ── Index ─────────────────────────────────────────────────────────────────────

def build_injury_index(inj):
    return {k: v.sort_values("game_date").reset_index(drop=True)
            for k, v in inj.groupby("name_key")}

def build_episode_index(episodes):
    return {k: v.sort_values("injury_start").reset_index(drop=True)
            for k, v in episodes.groupby("name_key")}


# ── Features glissantes ───────────────────────────────────────────────────────

def build_rolling_features(logs, inj, episodes, player_info):
    print("\nConstruction des features...")

    inj_index     = build_injury_index(inj)
    episode_index = build_episode_index(episodes)
    inj_out       = inj[inj["current_status"] == "Out"]

    # Index player_info par PLAYER_ID
    info_index = player_info.set_index("PLAYER_ID").to_dict("index")

    all_features  = []
    total_players = logs["PLAYER_ID"].nunique()

    for idx, (player_id, group) in enumerate(logs.groupby("PLAYER_ID")):
        group    = group.sort_values("GAME_DATE").reset_index(drop=True)
        name_key = group["name_key"].iloc[0]

        # Infos statiques du joueur
        pinfo      = info_index.get(player_id, {})
        birthdate  = pinfo.get("BIRTHDATE", None)
        draft_year = pinfo.get("DRAFT_YEAR", None)
        pos_code   = pinfo.get("POSITION_CODE", 3)
        height_cm  = pinfo.get("HEIGHT_CM", None)
        weight_lbs = pinfo.get("WEIGHT_LBS", None)

        player_inj_out  = inj_out[inj_out["name_key"] == name_key] if name_key in inj_index else pd.DataFrame()
        player_episodes = episode_index.get(name_key, pd.DataFrame())

        rows = []
        for i, row in group.iterrows():
            current_date   = row["GAME_DATE"]
            window_start   = current_date - pd.Timedelta(days=WINDOW_DAYS)
            chronic_start  = current_date - pd.Timedelta(days=CHRONIC_DAYS)
            current_season = row["SEASON"]
            season_start   = pd.Timestamp(f"{current_season[:4]}-10-01")

            # ── Âge et expérience NBA (dynamiques par match) ──
            if birthdate is not None and not pd.isna(birthdate):
                age_at_game = (current_date - birthdate).days / 365.25
            else:
                age_at_game = None

            if draft_year is not None and not pd.isna(draft_year):
                nba_years = current_date.year - int(draft_year)
                # Ajustement si on est avant octobre (début de saison)
                if current_date.month < 10:
                    nba_years -= 1
                nba_years = max(0, nba_years)
            else:
                nba_years = None

            # ── Charge physique ──
            window_10d = group[
                (group["GAME_DATE"] >= window_start) &
                (group["GAME_DATE"] < current_date)
            ]
            window_28d = group[
                (group["GAME_DATE"] >= chronic_start) &
                (group["GAME_DATE"] < current_date)
            ]
            window_7d = group[
                (group["GAME_DATE"] >= current_date - pd.Timedelta(days=7)) &
                (group["GAME_DATE"] < current_date)
            ]
            n_games_10d   = len(window_10d)
            n_games_28d   = len(window_28d)
            acute_load    = window_7d["MIN"].sum() / 7
            chronic_load  = window_28d["MIN"].sum() / 28 if n_games_28d > 0 else 0
            acwr          = round(acute_load / chronic_load, 3) if chronic_load > 0 else 0

            # Vrai back-to-back
            if i > 0:
                prev_game_date = group.iloc[i - 1]["GAME_DATE"]
                b2b = 1 if (current_date - prev_game_date).days == 1 else 0
            else:
                b2b = 0

            # ── Historique blessures ──
            if len(player_inj_out) > 0:
                past_out   = player_inj_out[player_inj_out["game_date"] < current_date]
                recent_out = past_out[past_out["game_date"] >= window_start]
                season_out = past_out[past_out["game_date"] >= season_start]
            else:
                past_out = recent_out = season_out = pd.DataFrame()

            was_injured_10d = 1 if len(recent_out) > 0 else 0
            days_out_season = season_out["game_date"].nunique() if len(season_out) > 0 else 0

            if len(past_out) > 0:
                very_recent         = past_out[past_out["game_date"] >= current_date - pd.Timedelta(days=5)]
                is_returning        = 1 if len(very_recent) > 0 else 0
                last_out_date       = past_out["game_date"].max()
                days_since_last_inj = (current_date - last_out_date).days
                last_row            = past_out.loc[past_out["game_date"].idxmax()]
                last_inj_category   = CATEGORY_ENCODING.get(last_row.get("injury_category", ""), 0)
                raw_bp              = str(last_row.get("body_part", "")).strip().title()
                last_inj_body_part  = BODY_PART_ENCODING.get(raw_bp, 0)
            else:
                is_returning        = 0
                days_since_last_inj = 999
                last_inj_category   = 0
                last_inj_body_part  = 0

            # ── Gravité ──
            if len(player_episodes) > 0:
                past_ep = player_episodes[player_episodes["injury_end"] < current_date]
            else:
                past_ep = pd.DataFrame()

            if len(past_ep) > 0:
                last_ep               = past_ep.loc[past_ep["injury_end"].idxmax()]
                last_injury_duration  = int(last_ep["duration_days"])
                last_injury_severity  = severity_label(last_injury_duration)
                season_ep             = past_ep[past_ep["injury_start"] >= season_start]
                total_days_out_season = int(season_ep["duration_days"].sum())
                max_injury_duration   = int(past_ep["duration_days"].max())
                max_injury_severity   = severity_label(max_injury_duration)
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

                # Stats match
                "MIN"                    : row["MIN"],
                "PTS"                    : row["PTS"],
                "REB"                    : row["REB"],
                "AST"                    : row["AST"],

                # Profil joueur (v5)
                "age_at_game"            : round(age_at_game, 2) if age_at_game else None,
                "nba_years"              : nba_years,
                "position_code"          : pos_code,
                "height_cm"              : height_cm,
                "weight_lbs"             : weight_lbs,

                # Charge 10j
                "games_10d"              : n_games_10d,
                "min_total_10d"          : window_10d["MIN"].sum(),
                "min_avg_10d"            : window_10d["MIN"].mean() if n_games_10d > 0 else 0,
                "min_max_10d"            : window_10d["MIN"].max()  if n_games_10d > 0 else 0,
                "away_games_10d"         : window_10d["IS_AWAY"].sum(),
                "away_ratio_10d"         : window_10d["IS_AWAY"].mean() if n_games_10d > 0 else 0,
                "b2b"                    : b2b,

                # ACWR
                "min_total_28d"          : window_28d["MIN"].sum(),
                "games_28d"              : n_games_28d,
                "acute_load_7d"          : round(acute_load, 2),
                "chronic_load_28d"       : round(chronic_load, 2),
                "acwr"                   : acwr,

                # Fatigue saison
                "min_season_avg"         : group[group["GAME_DATE"] < current_date]["MIN"].mean()
                                           if i > 0 else 0,
                "games_season"           : len(group[group["GAME_DATE"] < current_date]),

                # Historique blessures
                "was_injured_10d"        : was_injured_10d,
                "days_since_last_inj"    : days_since_last_inj,
                "days_out_season"        : days_out_season,
                "last_inj_category"      : last_inj_category,
                "last_inj_body_part"     : last_inj_body_part,
                "is_returning"           : is_returning,

                # Gravité
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

def build_target(df_features, inj):
    print("\nConstruction de la cible...")

    inj_out = inj[inj["current_status"] == "Out"][["name_key", "game_date"]].copy()

    out_index = {}
    for name_key, group in inj_out.groupby("name_key"):
        out_index[name_key] = set(group["game_date"].dt.normalize())

    targets  = []
    excluded = 0

    for _, row in df_features.iterrows():
        name_key     = row["name_key"]
        current_date = pd.Timestamp(row["GAME_DATE"]).normalize()
        future_start = current_date + pd.Timedelta(days=1)
        future_end   = current_date + pd.Timedelta(days=WINDOW_DAYS)

        player_out_dates = out_index.get(name_key, set())

        if current_date in player_out_dates:
            targets.append(np.nan)
            excluded += 1
            continue

        future_out = any(future_start <= d <= future_end for d in player_out_dates)
        targets.append(1 if future_out else 0)

    df_features["injury_next_10d"] = targets

    valid = df_features["injury_next_10d"].notna()
    pos   = (df_features["injury_next_10d"] == 1).sum()
    neg   = (df_features["injury_next_10d"] == 0).sum()

    print(f"  Lignes exclues  : {excluded:,} ({excluded/len(df_features)*100:.1f}%)")
    print(f"  Blessés (1)     : {pos:,} ({pos/valid.sum()*100:.1f}%)")
    print(f"  Non blessés (0) : {neg:,} ({neg/valid.sum()*100:.1f}%)")

    return df_features


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Feature Engineering v5 — Prédiction blessures NBA")
    print("=" * 55)

    logs, inj, player_info = load_data()

    print("\nCalcul des durées de blessure...")
    episodes = compute_injury_durations(inj)

    df_features = build_rolling_features(logs, inj, episodes, player_info)
    df_features = build_target(df_features, inj)

    # Supprimer lignes exclues
    df_features = df_features.dropna(subset=["injury_next_10d"]).reset_index(drop=True)
    df_features["injury_next_10d"] = df_features["injury_next_10d"].astype(int)

    df_features["split"] = df_features["SEASON"].map({
        "2021-22": "train",
        "2022-23": "train",
        "2023-24": "train",
        "2024-25": "val",
        "2025-26": "test",
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"  Dataset sauvegardé : {OUTPUT_PATH}")
    print(f"{'='*55}")
    print(f"  Lignes   : {len(df_features):,}")
    print(f"  Colonnes : {len(df_features.columns)}")
    print(f"\n  Profil joueurs :")
    print(f"    age_at_game  : min={df_features['age_at_game'].min():.1f} max={df_features['age_at_game'].max():.1f} mean={df_features['age_at_game'].mean():.1f}")
    print(f"    nba_years    : min={df_features['nba_years'].min():.0f} max={df_features['nba_years'].max():.0f} mean={df_features['nba_years'].mean():.1f}")
    print(f"    position_code: {df_features['position_code'].value_counts().to_dict()}")
    print(f"\n  Split :")
    print(df_features.groupby("split")["PLAYER_ID"].count().rename("lignes").to_string())
    print(f"\n  Cible :")
    print(df_features["injury_next_10d"].value_counts().to_string())


if __name__ == "__main__":
    main()