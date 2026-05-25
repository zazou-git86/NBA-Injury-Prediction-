"""
Extraction des blessures NBA — Saison 2022-23
P�riode : 18 octobre 2022 → 9 avril 2023 (saison régulière)
Output  : nba_injuries_2022_23.parquet

Lancement :
    pip install nbainjuries pandas aiohttp
    python 22_23.py
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

SEASON_START = datetime(2022, 10, 18)
SEASON_END   = datetime(2023, 4, 9)
OUTPUT_PATH  = "Datasets/nba_injuries_2022_23.parquet"

# Horaires ET auxquels la NBA publie ses rapports (ajuster si besoin)
REPORT_HOURS   = [11, 17, 19]
REPORT_MINUTES = [0,  30,  0]


# ── Génération des timestamps ──────────────────────────────────────────────────

def generate_timestamps(start, end, hours, minutes):
    timestamps, current = [], start
    while current <= end:
        for h, m in zip(hours, minutes):
            timestamps.append(current.replace(hour=h, minute=m, second=0, microsecond=0))
        current += timedelta(days=1)
    return timestamps


# ── Nettoyage & enrichissement ────────────────────────────────────────────────

def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Supprime les lignes vides (artefacts PDF)
    df = df.dropna(subset=["player_name"])
    # Catégories non-blessures à exclure du modèle
    NON_INJURY_REASONS = [
    "G League - Two-Way",
    "G League - On Assignment",
    "Not With Team",
    "Rest",
    "Personal Reasons",
    "-",
    "Trade Pending",
    "League Suspension",
    "Team Suspension",
    "Coach's Decision",
    ]

    df = df[~df["reason"].str.strip().isin(NON_INJURY_REASONS)]

    # Concussion Protocol → Head/Neck
    mask_concussion = df["reason"].str.contains("Concussion Protocol", na=False)
    df.loc[mask_concussion, "injury_category"] = "Head/Neck"
    df.loc[mask_concussion, "body_part"] = "Head"

    # Return to Competition → catégorie propre
    mask_rtc = df["reason"].str.contains("Return to Competition", na=False)
    df.loc[mask_rtc, "injury_category"] = "Reconditioning"

    # Probable → status_map mis à jour
    status_map = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Probable": 3, "Available": 4}

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "reason" in df.columns:
        df["injury_side"] = df["reason"].str.extract(
            r"- (Left|Right|Bilateral)", expand=False).fillna("N/A")
        df["body_part"] = df["reason"].str.extract(
            r"- (?:Left |Right |Bilateral )?([^;]+);", expand=False).str.strip()
        df["injury_detail"] = df["reason"].str.extract(
            r"; (.+)$", expand=False).str.strip()

        lower_body_kws = ["knee","ankle","foot","hamstring","quad","achilles",
                          "hip","groin","calf","thigh","toe","pelvic",
                          "sacroiliac","lower leg","lower back","patella",
                          "tibia","fibula"]
        upper_body_kws = ["shoulder","elbow","wrist","hand","finger","arm",
                          "back","chest","ribs","abdomen","forearm","bicep",
                          "tricep","rotator","radius","ulna","clavicle"]
        head_kws       = ["head","concussion","neck","eye","nose","jaw",
                          "facial","orbital","ear","dental"]

        def categorize(part):
            if pd.isna(part): return "Unknown"
            p = part.lower()
            if any(k in p for k in lower_body_kws): return "Lower Body"
            if any(k in p for k in upper_body_kws): return "Upper Body"
            if any(k in p for k in head_kws):       return "Head/Neck"
            return "Other"

        df["injury_category"] = df["body_part"].apply(categorize)

    status_map = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Available": 3}
    if "current_status" in df.columns:
        df["status_code"] = df["current_status"].map(status_map)

    if "player_name" in df.columns:
        split = df["player_name"].str.split(", ", n=1, expand=True)
        df["last_name"]  = split[0].str.strip()
        df["first_name"] = split[1].str.strip() if 1 in split.columns else ""

    return df


# ── Extraction asynchrone ─────────────────────────────────────────────────────

async def fetch_all(timestamps):
    from nbainjuries import injury_asy
    import aiohttp

    results, failed = [], []

    async with aiohttp.ClientSession() as session:

        # 1. Vérification de disponibilité
        print(f"Vérification de {len(timestamps)} timestamps...")
        valid = []
        for ts in timestamps:
            try:
                if await injury_asy.check_reportvalid(ts, session=session):
                    valid.append(ts)
            except Exception:
                pass
        print(f"→ {len(valid)} rapports disponibles\n")

        # 2. Extraction
        for i, ts in enumerate(valid):
            try:
                df = await injury_asy.get_reportdata(ts, session=session, return_df=True)
                df["report_timestamp"] = ts
                df["season"] = "2024-25"
                results.append(df)
            except Exception as e:
                failed.append((ts, str(e)))

            # Progression toutes les 50 requêtes
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(valid)} rapports traités...")

    if failed:
        print(f"\nAvertissement : {len(failed)} rapports en échec")
        for ts, err in failed[:3]:
            print(f"  {ts} → {err}")

    return results


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Extraction blessures NBA — Saison 2024-25")
    print("=" * 55)

    timestamps = generate_timestamps(
        SEASON_START, SEASON_END, REPORT_HOURS, REPORT_MINUTES
    )
    print(f"Période    : {SEASON_START.date()} → {SEASON_END.date()}")
    print(f"Timestamps : {len(timestamps)} générés\n")

    # Extraction
    results = asyncio.run(fetch_all(timestamps))

    if not results:
        print("Aucune donnée récupérée.")
        return

    # Consolidation & enrichissement
    print("\nConsolidation et enrichissement...")
    df = pd.concat(results, ignore_index=True)
    df = clean_and_enrich(df)

    # Sauvegarde
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # Résumé
    print("\n" + "=" * 55)
    print(f"  Fichier sauvegardé : {OUTPUT_PATH}")
    print("=" * 55)
    print(f"Lignes totales     : {len(df):,}")
    print(f"Joueurs uniques    : {df['player_name'].nunique()}")
    print(f"Rapports extraits  : {df['report_timestamp'].nunique()}")
    print(f"\nRépartition injury_category :")
    print(df["injury_category"].value_counts().to_string())
    print(f"\nRépartition current_status :")
    print(df["current_status"].value_counts().to_string())


if __name__ == "__main__":
    main()