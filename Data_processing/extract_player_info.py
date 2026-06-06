"""
Extraction des informations joueurs NBA
Source : nba_api (CommonPlayerInfo)

Données extraites :
    - Poste (Forward / Guard / Center)
    - Date de naissance → age calculé par saison
    - Année de draft → années NBA calculées par saison
    - Taille (cm)
    - Poids (lbs)

Lancement :
    python extract_player_info.py

Output : Datasets/nba_player_info.parquet
"""

import pandas as pd
import time
import random
from pathlib import Path
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players as nba_players


# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_PATH = Path("Datasets/nba_player_info.parquet")
DELAY_MIN   = 0.6
DELAY_MAX   = 1.2

# Conversion taille format "6-9" → cm
def height_to_cm(height_str: str) -> float:
    try:
        feet, inches = height_str.split("-")
        return round(int(feet) * 30.48 + int(inches) * 2.54, 1)
    except:
        return None

# Encodage ordinal du poste
POSITION_ENCODING = {
    "Guard"          : 1,
    "Guard-Forward"  : 2,
    "Forward-Guard"  : 2,
    "Forward"        : 3,
    "Forward-Center" : 4,
    "Center-Forward" : 4,
    "Center"         : 5,
}


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_player_info(player_ids: list) -> pd.DataFrame:
    results = []
    failed  = []
    total   = len(player_ids)

    print(f"Extraction de {total} joueurs...")

    for i, pid in enumerate(player_ids):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=pid)
            df   = info.get_data_frames()[0]

            birthdate = pd.to_datetime(df["BIRTHDATE"].values[0], errors="coerce")
            draft_year = df["DRAFT_YEAR"].values[0]
            height_str = df["HEIGHT"].values[0]
            position   = df["POSITION"].values[0]

            results.append({
                "PLAYER_ID"       : pid,
                "PLAYER_NAME"     : df["DISPLAY_FIRST_LAST"].values[0],
                "BIRTHDATE"       : birthdate,
                "DRAFT_YEAR"      : int(draft_year) if str(draft_year).isdigit() else None,
                "POSITION"        : position,
                "POSITION_CODE"   : POSITION_ENCODING.get(position, 3),
                "HEIGHT_CM"       : height_to_cm(str(height_str)),
                "WEIGHT_LBS"      : int(df["WEIGHT"].values[0]) if str(df["WEIGHT"].values[0]).isdigit() else None,
            })

        except Exception as e:
            failed.append((pid, str(e)))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} joueurs traités...")

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if failed:
        print(f"\nAvertissement : {len(failed)} joueurs en échec")
        for pid, err in failed[:5]:
            print(f"  {pid} → {err}")

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Extraction infos joueurs NBA")
    print("=" * 55)

    # Récupérer les PLAYER_ID depuis le dataset v4
    df_feat = pd.read_parquet("Datasets/nba_features_v4.parquet")
    player_ids = df_feat["PLAYER_ID"].unique().tolist()
    print(f"Joueurs à extraire : {len(player_ids)}")

    # Extraction
    df_info = extract_player_info(player_ids)

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_info.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"  Fichier sauvegardé : {OUTPUT_PATH}")
    print(f"{'='*55}")
    print(f"  Joueurs extraits : {len(df_info):,}")
    print(f"\n  Répartition postes :")
    print(df_info["POSITION"].value_counts().to_string())
    print(f"\n  Taille moyenne : {df_info['HEIGHT_CM'].mean():.1f} cm")
    print(f"  Poids moyen    : {df_info['WEIGHT_LBS'].mean():.1f} lbs")
    print(f"  Draft year min : {df_info['DRAFT_YEAR'].min()}")
    print(f"  Draft year max : {df_info['DRAFT_YEAR'].max()}")


if __name__ == "__main__":
    main()