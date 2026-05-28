"""
Extraction des données de blessures NBA
Projet : Prédiction de blessures NBA
Source : nbainjuries (https://pypi.org/project/nbainjuries/)

Prérequis :
    pip install nbainjuries pandas aiohttp
    Java 8+ installé et accessible dans le PATH système
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ── 1. REQUÊTE SIMPLE (un seul rapport) ───────────────────────────────────────

def fetch_single_report(year: int, month: int, day: int, hour: int, minute: int = 30) -> pd.DataFrame:
    """
    Récupère un rapport de blessures à un instant précis.
    Les rapports sont publiés toutes les heures, en général à H:00 ou H:30 (ET).
    """
    from nbainjuries import injury

    ts = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    df = injury.get_reportdata(ts, return_df=True)
    print(f"[OK] Rapport du {ts} — {len(df)} joueurs reportés")
    return df


# ── 2. GÉNÉRATION DES TIMESTAMPS D'UNE SAISON ────────────────────────────────

def generate_season_timestamps(
    start: datetime,
    end: datetime,
    hours: list = None,
    minutes: list = None
) -> list:
    """
    Génère la liste des timestamps à requêter entre deux dates.
    Par défaut : 11h00, 17h30 et 19h00 ET (horaires typiques de publication).
    """
    if hours is None:
        hours = [11, 17, 19]
    if minutes is None:
        minutes = [0, 30, 0]

    timestamps = []
    current = start
    while current <= end:
        for h, m in zip(hours, minutes):
            timestamps.append(current.replace(hour=h, minute=m, second=0, microsecond=0))
        current += timedelta(days=1)
    return timestamps


def _get_season(ts: datetime) -> str:
    """Retourne la saison NBA au format '2023-24' à partir d'un timestamp."""
    year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


# ── 3. EXTRACTION EN BATCH (asynchrone) ──────────────────────────────────────

async def fetch_season_async(
    timestamps: list,
    output_path: str = "nba_injuries_raw.parquet"
) -> pd.DataFrame:
    """
    Récupère en parallèle tous les rapports d'une liste de timestamps.
    Utilise injury_asy pour la performance sur des centaines de requêtes.
    """
    from nbainjuries import injury_asy
    import aiohttp

    results = []
    failed = []

    async with aiohttp.ClientSession() as session:
        # Vérification de disponibilité avant requête (évite les erreurs 404)
        valid_ts = []
        print(f"Vérification de {len(timestamps)} timestamps...")
        for ts in timestamps:
            try:
                is_valid = await injury_asy.check_reportvalid(ts, session=session)
                if is_valid:
                    valid_ts.append(ts)
            except Exception:
                pass

        print(f"{len(valid_ts)} rapports disponibles sur {len(timestamps)} vérifiés.")

        # Extraction des rapports valides
        for i, ts in enumerate(valid_ts):
            try:
                df = await injury_asy.get_reportdata(ts, session=session, return_df=True)
                df["report_timestamp"] = ts
                df["season"] = _get_season(ts)
                results.append(df)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(valid_ts)} rapports traités...")
            except Exception as e:
                failed.append((ts, str(e)))

    if failed:
        print(f"\nAvertissement : {len(failed)} rapports en echec")
        for ts, err in failed[:5]:
            print(f"  {ts} -> {err}")

    if not results:
        print("Aucune donnée récupérée.")
        return pd.DataFrame()

    full_df = pd.concat(results, ignore_index=True)
    full_df = clean_and_enrich(full_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(output_path, index=False)
    print(f"\nDataset sauvegardé : {output_path} ({len(full_df)} lignes, {full_df['player_name'].nunique()} joueurs uniques)")
    return full_df


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "reason" in df.columns:
        df["injury_side"] = df["reason"].str.extract(
            r"- (Left|Right|Bilateral)", expand=False
        ).fillna("N/A")

        df["body_part"] = df["reason"].str.extract(
            r"- (?:Left |Right |Bilateral )?([^;]+);", expand=False
        ).str.strip()

        df["injury_detail"] = df["reason"].str.extract(
            r"; (.+)$", expand=False
        ).str.strip()

        lower_body_kws = ["knee", "ankle", "foot", "hamstring", "quad",
                          "achilles", "hip", "groin", "calf", "thigh",
                          "toe", "pelvic", "sacroiliac", "lower leg",
                          "lower back", "patella", "tibia", "fibula"]
        upper_body_kws = ["shoulder", "elbow", "wrist", "hand", "finger",
                          "arm", "back", "chest", "ribs", "abdomen",
                          "forearm", "bicep", "tricep", "rotator",
                          "radius", "ulna", "clavicle"]
        head_kws       = ["head", "concussion", "neck", "eye", "nose", "jaw",
                          "facial", "orbital", "ear", "dental"]

        def categorize(part):
            if pd.isna(part):
                return "Unknown"
            p = part.lower()
            if any(k in p for k in lower_body_kws): return "Lower Body"
            if any(k in p for k in upper_body_kws): return "Upper Body"
            if any(k in p for k in head_kws):       return "Head/Neck"
            return "Other"

        df["injury_category"] = df["body_part"].apply(categorize)

        # Cas spéciaux : Concussion Protocol
        mask_concussion = df["reason"].str.contains("Concussion Protocol", na=False)
        df.loc[mask_concussion, "injury_category"] = "Head/Neck"
        df.loc[mask_concussion, "body_part"]       = "Head"
        df.loc[mask_concussion, "injury_detail"]   = "Concussion Protocol"

        # Cas spéciaux : Return to Competition
        mask_rtc = df["reason"].str.contains("Return to Competition", na=False)
        df.loc[mask_rtc, "injury_category"] = "Reconditioning"
        df.loc[mask_rtc, "body_part"]       = "N/A"
        df.loc[mask_rtc, "injury_detail"]   = "Return to Competition Reconditioning"

    status_map = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Probable": 3, "Available": 4}
    if "current_status" in df.columns:
        df["status_code"] = df["current_status"].map(status_map)

    if "player_name" in df.columns:
        name_split = df["player_name"].str.split(", ", n=1, expand=True)
        df["last_name"]  = name_split[0].str.strip()
        df["first_name"] = name_split[1].str.strip() if 1 in name_split.columns else ""

    return df

# ── 5. PIPELINE PRINCIPAL ─────────────────────────────────────────────────────

def run_extraction(
    season_start: datetime,
    season_end: datetime,
    output_path: str = "data/nba_injuries.parquet"
) -> pd.DataFrame:
    """
    Lance l'extraction complète d'une saison NBA.

    Exemple :
        run_extraction(
            season_start=datetime(2023, 10, 24),
            season_end=datetime(2024, 4, 14),
            output_path="data/nba_injuries_2023_24.parquet"
        )
    """
    timestamps = generate_season_timestamps(season_start, season_end)
    print(f"Saison {_get_season(season_start)} : {len(timestamps)} timestamps générés")
    print(f"Période : {season_start.date()} -> {season_end.date()}\n")

    df = asyncio.run(fetch_season_async(timestamps, output_path=output_path))
    return df


# ── 6. EXEMPLE D'UTILISATION ──────────────────────────────────────────────────

if __name__ == "__main__":

    # A) Requête rapide sur un seul rapport (pour tester l'installation)
    print("=== Test : rapport unique ===")
    df_test = fetch_single_report(2025, 4, 25, 17, 30)
    print(df_test.head())
    print(df_test.columns.tolist())

    # B) Extraction d'une saison complète (décommenter pour lancer)
    #
    # print("\n=== Extraction saison 2023-24 ===")
    # df_season = run_extraction(
    #     season_start=datetime(2023, 10, 24),
    #     season_end=datetime(2024, 4, 14),
    #     output_path="data/nba_injuries_2023_24.parquet"
    # )
    # print(df_season["injury_category"].value_counts())
    # print(df_season["body_part"].value_counts().head(10))

    # C) Multi-saisons (2021-22 à 2023-24 — données dispo depuis 2021-22)
    #
    # seasons = [
    #     (datetime(2021, 10, 19), datetime(2022, 4, 10), "data/nba_injuries_2021_22.parquet"),
    #     (datetime(2022, 10, 18), datetime(2023, 4,  9), "data/nba_injuries_2022_23.parquet"),
    #     (datetime(2023, 10, 24), datetime(2024, 4, 14), "data/nba_injuries_2023_24.parquet"),
    # ]
    # for start, end, path in seasons:
    #     run_extraction(start, end, path)