from nbainjuries import injury
from datetime import datetime
import pandas as pd

# Copie de clean_and_enrich() depuis le script
def clean_and_enrich(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "reason" in df.columns:
        df["injury_side"] = df["reason"].str.extract(r"- (Left|Right|Bilateral)", expand=False).fillna("N/A")
        df["body_part"] = df["reason"].str.extract(r"- (?:Left |Right |Bilateral )?([^;]+);", expand=False).str.strip()
        df["injury_detail"] = df["reason"].str.extract(r"; (.+)$", expand=False).str.strip()
        lower_body_kws = ["knee","ankle","foot","hamstring","quad","achilles","hip","groin","calf","thigh"]
        upper_body_kws = ["shoulder","elbow","wrist","hand","finger","arm","back","chest","ribs","abdomen"]
        head_kws = ["head","concussion","neck","eye","nose","jaw"]
        def categorize(part):
            if pd.isna(part): return "Unknown"
            p = part.lower()
            if any(k in p for k in lower_body_kws): return "Lower Body"
            if any(k in p for k in upper_body_kws): return "Upper Body"
            if any(k in p for k in head_kws): return "Head/Neck"
            return "Other"
        df["injury_category"] = df["body_part"].apply(categorize)
    status_map = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Available": 3}
    if "current_status" in df.columns:
        df["status_code"] = df["current_status"].map(status_map)
    if "player_name" in df.columns:
        name_split = df["player_name"].str.split(", ", n=1, expand=True)
        df["last_name"] = name_split[0].str.strip()
        df["first_name"] = name_split[1].str.strip() if 1 in name_split.columns else ""
    return df

# Récupération + enrichissement
ts = datetime(2025, 4, 25, 17, 30)
df_raw = injury.get_reportdata(ts, return_df=True)
df = clean_and_enrich(df_raw)

# Vérifications
print("=== Colonnes ===")
print(df.columns.tolist())

print("\n=== Aperçu des nouvelles colonnes ===")
print(df[["player_name","body_part","injury_side","injury_category","status_code"]].to_string())

print("\n=== Répartition injury_category ===")
print(df["injury_category"].value_counts())

print("\n=== Répartition status_code ===")
print(df["current_status"].value_counts())