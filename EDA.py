"""
EDA — Prédiction de blessures NBA
Analyses :
    1. Distribution des blessures par type et partie du corps
    2. Joueurs les plus souvent blessés
    3. Relation charge physique / blessures
    4. Évolution des blessures par saison
    5. Corrélations entre les features

Lancement :
    pip install matplotlib seaborn
    python EDA.py

Output : figures_EDA/ (un PNG par analyse)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "Data_processing/Datasets/nba_features_v3.parquet"
INJ_FILES = [
    "Data_extraction/Injuries/Datasets/nba_injuries_2021_22.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2022_23.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2023_24.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2025_26.parquet",
]
OUT_DIR  = Path("figures_EDA_3")
OUT_DIR.mkdir(exist_ok=True)

SEASONS  = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
PALETTE  = ["#1D9E75", "#534AB7", "#BA7517", "#D85A30", "#378ADD"]
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def get_season(date) -> str:
    """Calcule la saison NBA (ex: '2023-24') depuis une date."""
    year = date.year if date.month >= 10 else date.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


# ── Chargement ────────────────────────────────────────────────────────────────

print("Chargement des données...")
df  = pd.read_parquet(DATA_PATH)
inj = pd.concat([pd.read_parquet(f) for f in INJ_FILES], ignore_index=True)
inj["game_date"] = pd.to_datetime(inj["game_date"])

# Recalcul de la saison depuis la date (corrige le bug season=2024-25 partout)
inj["season"] = inj["game_date"].apply(get_season)

# Vraies blessures uniquement
inj_real = inj[inj["injury_category"].isin([
    "Lower Body", "Upper Body", "Head/Neck", "Other", "Reconditioning"
])]
inj_out = inj_real[inj_real["current_status"] == "Out"].copy()

print(f"  Features     : {len(df):,} lignes")
print(f"  Injuries Out : {len(inj_out):,} lignes")
print(f"  Saisons      : {sorted(inj_out['season'].unique())}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Distribution des blessures par type et partie du corps
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Distribution des blessures...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Distribution des blessures NBA (2021-22 → 2025-26)", fontsize=14, fontweight="bold")

# Catégorie anatomique
cat_counts = inj_out["injury_category"].value_counts()
colors_cat = PALETTE[:len(cat_counts)]
axes[0].barh(cat_counts.index, cat_counts.values, color=colors_cat)
axes[0].set_title("Par catégorie anatomique")
axes[0].set_xlabel("Nombre de rapports Out")
for i, v in enumerate(cat_counts.values):
    axes[0].text(v + 200, i, f"{v:,}", va="center", fontsize=10)

# Top 15 parties du corps
bp_counts = inj_out["body_part"].dropna().value_counts().head(15)
axes[1].barh(bp_counts.index[::-1], bp_counts.values[::-1], color=PALETTE[0])
axes[1].set_title("Top 15 parties du corps")
axes[1].set_xlabel("Nombre de rapports Out")
for i, v in enumerate(bp_counts.values[::-1]):
    axes[1].text(v + 50, i, f"{v:,}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "1_distribution_blessures.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA/1_distribution_blessures.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Joueurs les plus souvent blessés
# ═══════════════════════════════════════════════════════════════════════════════

print("[2/5] Joueurs les plus blessés...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Joueurs les plus souvent blessés", fontsize=14, fontweight="bold")

# Top 20 toutes saisons confondues
player_total = (
    inj_out.groupby("player_name")["game_date"]
    .nunique()
    .reset_index()
    .rename(columns={"game_date": "jours_out"})
    .sort_values("jours_out", ascending=False)
    .head(20)
)
axes[0].barh(player_total["player_name"][::-1], player_total["jours_out"][::-1], color=PALETTE[1])
axes[0].set_title("Top 20 — jours Out cumulés (5 saisons)")
axes[0].set_xlabel("Jours Out")
for i, v in enumerate(player_total["jours_out"][::-1]):
    axes[0].text(v + 1, i, str(v), va="center", fontsize=9)

# Top 20 pire saison individuelle
player_season = (
    inj_out.groupby(["player_name", "season"])["game_date"]
    .nunique()
    .reset_index()
    .rename(columns={"game_date": "jours_out"})
    .sort_values("jours_out", ascending=False)
    .head(20)
)
labels = [f"{r['player_name']} ({r['season']})" for _, r in player_season.iterrows()]
axes[1].barh(labels[::-1], player_season["jours_out"].values[::-1], color=PALETTE[2])
axes[1].set_title("Top 20 — pire saison individuelle")
axes[1].set_xlabel("Jours Out")
for i, v in enumerate(player_season["jours_out"].values[::-1]):
    axes[1].text(v + 0.5, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "2_joueurs_blesses.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA/2_joueurs_blesses.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Relation charge physique / blessures
# ═══════════════════════════════════════════════════════════════════════════════

print("[3/5] Relation charge / blessures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Relation charge physique / blessures (injury_next_10d)", fontsize=14, fontweight="bold")

features_to_plot = [
    ("min_total_10d",       "Minutes totales 10j",    0, 0),
    ("games_10d",           "Matchs joués 10j",       0, 1),
    ("b2b_10d",             "Back-to-back 10j",       0, 2),
    ("away_ratio_10d",      "Ratio away 10j",         1, 0),
    ("min_season_avg",      "Moy. minutes saison",    1, 1),
    ("days_since_last_inj", "Jours depuis blessure",  1, 2),
]

for feat, title, r, c in features_to_plot:
    ax = axes[r][c]
    blesses  = df[df["injury_next_10d"] == 1][feat].dropna()
    non_bles = df[df["injury_next_10d"] == 0][feat].dropna()

    # Limiter days_since_last_inj pour lisibilité (exclure valeur sentinelle 999)
    if feat == "days_since_last_inj":
        blesses  = blesses[blesses < 500]
        non_bles = non_bles[non_bles < 500]

    ax.hist(non_bles, bins=30, alpha=0.6, color=PALETTE[0], label="Non blessé (0)", density=True)
    ax.hist(blesses,  bins=30, alpha=0.6, color=PALETTE[2], label="Blessé (1)",     density=True)
    ax.set_title(title)
    ax.set_ylabel("Densité")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "3_charge_vs_blessures.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA/3_charge_vs_blessures.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Évolution des blessures par saison
# ═══════════════════════════════════════════════════════════════════════════════

print("[4/5] Évolution par saison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Évolution des blessures par saison", fontsize=14, fontweight="bold")

# Nombre de blessures (épisodes distincts) par saison
# Un épisode = un joueur blessé sur une période continue
inj_episodes = (
    inj_out.groupby(["season", "player_name"])["game_date"]
    .apply(lambda dates: (dates.sort_values().diff().dt.days.fillna(999) > 10).sum() + 1)
    .reset_index()
    .rename(columns={"game_date": "nb_episodes"})
    .groupby("season")["nb_episodes"]
    .sum()
    .reindex(SEASONS)
    .fillna(0)
)

axes[0].bar(SEASONS, inj_episodes.values, color=PALETTE)
axes[0].set_title("Nombre de blessures par saison")
axes[0].set_ylabel("Épisodes de blessure")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(inj_episodes.values):
    axes[0].text(i, v + 5, f"{int(v):,}", ha="center", fontsize=9)

# Taux de blessure par saison
inj_rate = df.groupby("SEASON")["injury_next_10d"].mean() * 100
inj_rate = inj_rate.reindex(SEASONS)
axes[1].bar(SEASONS, inj_rate.values, color=PALETTE)
axes[1].set_title("Taux prédit de blessure par saison")
axes[1].set_ylabel("% matchs précédant une blessure")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(inj_rate.values):
    axes[1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)

# Catégories de blessures par saison
cat_season = (
    inj_out.groupby(["season", "injury_category"])
    .size()
    .unstack(fill_value=0)
    .reindex(SEASONS)
    .fillna(0)
)
cat_season.plot(kind="bar", ax=axes[2], color=PALETTE, width=0.7)
axes[2].set_title("Catégories de blessures par saison")
axes[2].set_ylabel("Rapports Out")
axes[2].tick_params(axis="x", rotation=30)
axes[2].legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_DIR / "4_evolution_saisons.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA/4_evolution_saisons.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Corrélations entre les features
# ═══════════════════════════════════════════════════════════════════════════════

print("[5/5] Corrélations...")

feature_cols = [
    "MIN", "games_10d", "min_total_10d", "min_avg_10d", "min_max_10d",
    "away_games_10d", "away_ratio_10d", "b2b_10d",
    "min_season_avg", "games_season",
    "was_injured_10d", "days_since_last_inj", "injury_count_season",
    "last_inj_category", "last_inj_body_part", "is_returning",
    "last_injury_duration", "last_injury_severity",
    "total_days_out_season", "max_injury_duration", "max_injury_severity",
    "injury_next_10d"
]

corr = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    annot_kws={"size": 7}, linewidths=0.3,
    cbar_kws={"shrink": 0.6}
)
ax.set_title("Matrice de corrélation — features + cible", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "5_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA/5_correlations.png")


# ── Résumé stats clés ─────────────────────────────────────────────────────────

print("\n" + "="*50)
print("  Stats clés")
print("="*50)
print(f"Taux global de blessure (cible=1)  : {df['injury_next_10d'].mean()*100:.1f}%")
print(f"Corps le plus touché               : {inj_out['body_part'].value_counts().index[0]}")
print(f"Catégorie la plus fréquente        : {inj_out['injury_category'].value_counts().index[0]}")
print(f"\nJours Out par saison :")
print(inj_episodes.apply(lambda x: f"{int(x):,}").to_string())
print(f"\nTop 3 features corrélées à la cible :")
top_corr = corr["injury_next_10d"].drop("injury_next_10d").abs().sort_values(ascending=False)
for feat, val in top_corr.head(3).items():
    print(f"  {feat} : {val:.3f}")
print(f"\nFigures sauvegardées dans : {OUT_DIR.resolve()}")