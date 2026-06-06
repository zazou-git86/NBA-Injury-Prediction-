"""
EDA v5 — Prédiction de blessures NBA
Analyses :
    1. Corrélations complètes (35 features + cible)
    2. Évolution des blessures par saison (enrichie avec features v5)

Lancement :
    python eda_v5.py

Output : figures_EDA_v5/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "Data_processing/Datasets/nba_features_v5.parquet"
INJ_FILES = [
    "Data_extraction/Injuries/Datasets/nba_injuries_2021_22.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2022_23.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2023_24.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2024_25.parquet",
    "Data_extraction/Injuries/Datasets/nba_injuries_2025_26.parquet",
]
OUT_DIR = Path("figures_EDA_v5")
OUT_DIR.mkdir(exist_ok=True)

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
PALETTE = ["#1D9E75", "#534AB7", "#BA7517", "#D85A30", "#378ADD"]
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def get_season(date) -> str:
    year = date.year if date.month >= 10 else date.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


# ── Chargement ────────────────────────────────────────────────────────────────

print("Chargement des données...")
df  = pd.read_parquet(DATA_PATH)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

inj = pd.concat([pd.read_parquet(f) for f in INJ_FILES], ignore_index=True)
inj["game_date"] = pd.to_datetime(inj["game_date"])
inj["season"]    = inj["game_date"].apply(get_season)

inj_real = inj[inj["injury_category"].isin([
    "Lower Body", "Upper Body", "Head/Neck", "Other", "Reconditioning"
])]
inj_out = inj_real[inj_real["current_status"] == "Out"].copy()

print(f"  Features v5  : {len(df):,} lignes | {len(df.columns)} colonnes")
print(f"  Injuries Out : {len(inj_out):,} lignes")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Corrélations complètes v5
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/2] Matrice de corrélations v5...")

feature_cols = [
    # Profil joueur
    "age_at_game", "nba_years", "position_code", "height_cm", "weight_lbs",
    # Stats match
    "MIN", "IS_AWAY", "PTS", "REB", "AST",
    # Charge 10j
    "games_10d", "min_total_10d", "min_avg_10d", "min_max_10d",
    "away_games_10d", "away_ratio_10d", "b2b",
    # ACWR
    "acute_load_7d", "chronic_load_28d", "acwr", "games_28d", "min_total_28d",
    # Fatigue saison
    "min_season_avg", "games_season",
    # Historique blessures
    "was_injured_10d", "days_since_last_inj", "days_out_season",
    "last_inj_category", "last_inj_body_part", "is_returning",
    # Gravité
    "last_injury_duration", "last_injury_severity",
    "total_days_out_season", "max_injury_duration", "max_injury_severity",
    # Cible
    "injury_next_10d"
]

corr = df[feature_cols].corr()

# ── Figure 1a : heatmap complète ──
fig, ax = plt.subplots(figsize=(20, 17))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    annot_kws={"size": 6.5}, linewidths=0.3,
    cbar_kws={"shrink": 0.5}
)
ax.set_title("Matrice de corrélation complète v5 — 35 features + cible",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "1a_correlations_completes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA_v5/1a_correlations_completes.png")

# ── Figure 1b : corrélations avec la cible uniquement (barplot) ──
corr_target = (
    corr["injury_next_10d"]
    .drop("injury_next_10d")
    .sort_values(key=abs, ascending=True)
)

colors = ["#D85A30" if v < 0 else "#1D9E75" for v in corr_target.values]

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(corr_target.index, corr_target.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Corrélation de chaque feature avec injury_next_10d",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Corrélation de Pearson")

for i, v in enumerate(corr_target.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), i,
            f"{v:.3f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "1b_correlations_cible.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA_v5/1b_correlations_cible.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Évolution des blessures par saison — version enrichie
# ═══════════════════════════════════════════════════════════════════════════════

print("[2/2] Évolution des blessures par saison...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Évolution des blessures par saison — dataset v5",
             fontsize=14, fontweight="bold")

# ── 2a : Nombre d'épisodes de blessure par saison ──
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
axes[0][0].bar(SEASONS, inj_episodes.values, color=PALETTE)
axes[0][0].set_title("Nombre d'épisodes de blessure")
axes[0][0].set_ylabel("Épisodes")
axes[0][0].tick_params(axis="x", rotation=30)
for i, v in enumerate(inj_episodes.values):
    axes[0][0].text(i, v + 5, f"{int(v):,}", ha="center", fontsize=9)

# ── 2b : Taux de blessure par saison ──
inj_rate = df.groupby("SEASON")["injury_next_10d"].mean() * 100
inj_rate = inj_rate.reindex(SEASONS)
axes[0][1].bar(SEASONS, inj_rate.values, color=PALETTE)
axes[0][1].set_title("Taux de blessure (cible=1)")
axes[0][1].set_ylabel("% matchs précédant une blessure")
axes[0][1].tick_params(axis="x", rotation=30)
for i, v in enumerate(inj_rate.values):
    axes[0][1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)

# ── 2c : ACWR moyen par saison ──
acwr_season = df.groupby("SEASON")["acwr"].mean().reindex(SEASONS)
axes[0][2].bar(SEASONS, acwr_season.values, color=PALETTE)
axes[0][2].axhline(1.3, color="orange", linestyle="--", linewidth=1.5, label="Seuil 1.3")
axes[0][2].axhline(1.5, color="red",    linestyle="--", linewidth=1.5, label="Seuil risque 1.5")
axes[0][2].set_title("ACWR moyen par saison")
axes[0][2].set_ylabel("ACWR")
axes[0][2].tick_params(axis="x", rotation=30)
axes[0][2].legend(fontsize=9)
for i, v in enumerate(acwr_season.values):
    axes[0][2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

# ── 2d : Âge moyen des blessés vs non-blessés par saison ──
age_injured = df[df["injury_next_10d"] == 1].groupby("SEASON")["age_at_game"].mean().reindex(SEASONS)
age_healthy = df[df["injury_next_10d"] == 0].groupby("SEASON")["age_at_game"].mean().reindex(SEASONS)
x = np.arange(len(SEASONS))
w = 0.35
axes[1][0].bar(x - w/2, age_injured.values, w, label="Blessés",     color=PALETTE[2])
axes[1][0].bar(x + w/2, age_healthy.values, w, label="Non blessés", color=PALETTE[0])
axes[1][0].set_title("Âge moyen : blessés vs non-blessés")
axes[1][0].set_ylabel("Âge (années)")
axes[1][0].set_xticks(x)
axes[1][0].set_xticklabels(SEASONS, rotation=30)
axes[1][0].legend(fontsize=9)
axes[1][0].set_ylim(24, 30)

# ── 2e : Taux de blessure par poste (toutes saisons) ──
pos_labels = {1: "Guard", 2: "G-F", 3: "Forward", 4: "F-C", 5: "Center"}
inj_by_pos = df.groupby("position_code")["injury_next_10d"].mean() * 100
inj_by_pos.index = [pos_labels.get(i, str(i)) for i in inj_by_pos.index]
axes[1][1].bar(inj_by_pos.index, inj_by_pos.values, color=PALETTE)
axes[1][1].set_title("Taux de blessure par poste")
axes[1][1].set_ylabel("% matchs précédant une blessure")
for i, v in enumerate(inj_by_pos.values):
    axes[1][1].text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9)

# ── 2f : Distribution ACWR — blessés vs non-blessés ──
blesses  = df[df["injury_next_10d"] == 1]["acwr"].clip(0, 3)
non_bles = df[df["injury_next_10d"] == 0]["acwr"].clip(0, 3)
axes[1][2].hist(non_bles, bins=40, alpha=0.6, color=PALETTE[0],
                label="Non blessé (0)", density=True)
axes[1][2].hist(blesses,  bins=40, alpha=0.6, color=PALETTE[2],
                label="Blessé (1)",     density=True)
axes[1][2].axvline(1.3, color="orange", linestyle="--", linewidth=1.5, label="Seuil 1.3")
axes[1][2].axvline(1.5, color="red",    linestyle="--", linewidth=1.5, label="Seuil risque 1.5")
axes[1][2].set_title("Distribution ACWR — blessés vs non-blessés")
axes[1][2].set_xlabel("ACWR (clippé à 3)")
axes[1][2].set_ylabel("Densité")
axes[1][2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "2_evolution_saisons_v5.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_EDA_v5/2_evolution_saisons_v5.png")


# ── Résumé ────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  Stats clés v5")
print("="*55)
top_corr = corr["injury_next_10d"].drop("injury_next_10d").abs().sort_values(ascending=False)
print("Top 10 features corrélées à injury_next_10d :")
for feat, val in top_corr.head(10).items():
    direction = "+" if corr["injury_next_10d"][feat] > 0 else "-"
    print(f"  {direction} {feat:<30} : {val:.3f}")

print(f"\nACWR > 1.5 (zone risque) : {(df['acwr'] > 1.5).sum():,} matchs ({(df['acwr'] > 1.5).mean()*100:.1f}%)")
print(f"Taux blessure blessés ACWR>1.5 : {df[df['acwr']>1.5]['injury_next_10d'].mean()*100:.1f}%")
print(f"Taux blessure ACWR<=1.5        : {df[df['acwr']<=1.5]['injury_next_10d'].mean()*100:.1f}%")
print(f"\nFigures sauvegardées dans : {OUT_DIR.resolve()}")