"""
Extraction des game logs NBA — 5 saisons (2021-22 → 2025-26)
Source : nba_api (PlayerGameLog, un appel par joueur par saison)

Lancement :
    pip install nba_api
    python extract_gamelogs.py

Durée estimée : 45-90 min (délais anti-ban inclus)
Output : Datasets/nba_gamelogs_all.parquet
"""

import pandas as pd
import time
import random
from pathlib import Path
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players


# ── Configuration ─────────────────────────────────────────────────────────────

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
OUTPUT_DIR  = Path("Datasets")
OUTPUT_PATH = OUTPUT_DIR / "nba_gamelogs_all.parquet"

# Délai entre chaque appel API (secondes) — ne pas descendre sous 0.6
DELAY_MIN = 0.6
DELAY_MAX = 1.2


# ── Utilitaires ───────────────────────────────────────────────────────────────

def get_all_nba_players():
    """Retourne tous les joueurs actifs + inactifs (couvre toutes les saisons)."""
    return nba_players.get_players()

def is_away(matchup: str) -> int:
    """Retourne 1 si match à l'extérieur (contient '@'), 0 sinon."""
    return 1 if "@" in matchup else 0


# ── Extraction d'une saison ───────────────────────────────────────────────────

def fetch_season(season: str, all_players: list) -> pd.DataFrame:
    """
    Récupère les game logs de tous les joueurs pour une saison donnée.
    Sauvegarde un fichier intermédiaire par saison pour reprendre en cas d'erreur.
    """
    interim_path = OUTPUT_DIR / f"nba_gamelogs_{season.replace('-', '_')}.parquet"

    # Reprise si déjà extrait
    if interim_path.exists():
        print(f"  [SKIP] {season} — fichier déjà présent")
        return pd.read_parquet(interim_path)

    results = []
    failed  = []
    total   = len(all_players)

    print(f"\n  {total} joueurs à traiter pour {season}")

    for i, player in enumerate(all_players):
        pid  = player["id"]
        name = player["full_name"]

        try:
            log = playergamelog.PlayerGameLog(
                player_id=pid,
                season=season,
                season_type_all_star="Regular Season"
            )
            df = log.get_data_frames()[0]

            if len(df) == 0:
                # Joueur sans matchs cette saison (normal pour beaucoup)
                pass
            else:
                df["PLAYER_ID"]   = pid
                df["PLAYER_NAME"] = name
                df["SEASON"]      = season
                df["IS_AWAY"]     = df["MATCHUP"].apply(is_away)
                results.append(df)

        except Exception as e:
            failed.append((name, str(e)))

        # Progression toutes les 100 requêtes
        if (i + 1) % 100 == 0:
            found = sum(len(r) for r in results)
            print(f"    {i+1}/{total} joueurs | {found:,} matchs collectés")

        # Délai anti-ban aléatoire
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if failed:
        print(f"\n  Avertissement : {len(failed)} joueurs en échec")
        for name, err in failed[:3]:
            print(f"    {name} → {err}")

    if not results:
        print(f"  Aucun match trouvé pour {season}")
        return pd.DataFrame()

    df_season = pd.concat(results, ignore_index=True)
    df_season = clean_gamelogs(df_season)

    # Sauvegarde intermédiaire
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_season.to_parquet(interim_path, index=False)
    print(f"\n  Sauvegardé : {interim_path.name}")
    print(f"  Matchs     : {len(df_season):,}")
    print(f"  Joueurs    : {df_season['PLAYER_ID'].nunique()}")

    return df_season


# ── Nettoyage ─────────────────────────────────────────────────────────────────

def clean_gamelogs(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et type les colonnes utiles."""
    df = df.copy()

    # Parse la date (format "Apr 13, 2025")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")

    # Trier par joueur puis par date (chronologique)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Garder uniquement les colonnes utiles pour le projet
    cols = [
        "SEASON", "PLAYER_ID", "PLAYER_NAME", "Game_ID",
        "GAME_DATE", "MATCHUP", "IS_AWAY", "WL",
        "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
        "PLUS_MINUS"
    ]
    cols_present = [c for c in cols if c in df.columns]
    df = df[cols_present]

    # Typage
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Extraction game logs NBA — 5 saisons")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Liste complète des joueurs (active + historique)
    all_players = get_all_nba_players()
    print(f"Joueurs dans la base nba_api : {len(all_players)}")

    # Extraction saison par saison
    all_dfs = []
    for season in SEASONS:
        print(f"\n{'='*55}")
        print(f"  Saison {season}")
        print(f"{'='*55}")
        df = fetch_season(season, all_players)
        if len(df) > 0:
            all_dfs.append(df)

    # Consolidation finale
    print(f"\n{'='*55}")
    print("  Consolidation finale...")
    print(f"{'='*55}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n  Fichier final : {OUTPUT_PATH}")
    print(f"  Lignes totales : {len(df_all):,}")
    print(f"  Joueurs uniques : {df_all['PLAYER_ID'].nunique()}")
    print(f"\n  Par saison :")
    print(df_all.groupby("SEASON")["Game_ID"].count().rename("matchs").to_string())


if __name__ == "__main__":
    main()