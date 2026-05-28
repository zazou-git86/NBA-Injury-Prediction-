import pandas as pd
import unicodedata

NAME_MAPPING = {
    "alexandre sarr"    : "alex sarr",
    "carlton carrington": "bub carrington",
    "jimmy butler"      : "jimmy butler iii",
}

def normalize_name(name: str) -> str:
    """Supprime les accents et met en minuscules."""
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()

inj = pd.read_parquet("../Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet")
logs = pd.read_parquet("../Data_extraction/load/Datasets/nba_gamelogs_all.parquet")
logs = logs[logs["SEASON"] == "2024-25"]

inj["name_key"]  = (inj["first_name"] + " " + inj["last_name"]).apply(normalize_name)
logs["name_key"] = logs["PLAYER_NAME"].apply(normalize_name)

# Chercher les vrais noms dans gamelogs pour les 10 non-matchés
unmatched = ['alexandre sarr', 'bojan bogdanovic', 'carlton carrington',
             'christian wood', 'cui yongxi', 'daron holmes ii',
             'jimmy butler', 'nikola topic', 'saddiq bey', 'seth lundy']

logs_names = sorted(logs["name_key"].unique())

for name in unmatched:
    # Chercher par nom de famille
    last = name.split()[-1]
    candidates = [n for n in logs_names if last in n]
    print(f"{name} → {candidates}")