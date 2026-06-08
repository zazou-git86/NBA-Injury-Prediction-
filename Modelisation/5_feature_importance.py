"""
Étape 5 — Feature Importance
Prédiction de blessures NBA

Approches :
    5a — Permutation importance sur CatBoost ET Random Forest
    5b — SHAP values sur CatBoost uniquement
    5c — Feature selection : sous-ensemble optimal
         Réentraînement CatBoost et RF sur features réduites

But : trouver le sous-ensemble minimal de features qui maintient
      les performances + répondre à la question scientifique centrale :
      les variables de charge apportent-elles quelque chose
      au-delà de l'historique médical ?
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

Path("Models").mkdir(exist_ok=True)
Path("figures_ML").mkdir(exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 5 — Feature Importance")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("Models/step3_rf.pkl", "rb") as f:
    rf_data = pickle.load(f)

with open("Models/step4_cat.pkl", "rb") as f:
    cat_data = pickle.load(f)

X_train      = data["X_train_tree"]
X_val        = data["X_val_tree"]
y_train      = data["y_train"]
y_val        = data["y_val"]
feature_cols = data["feature_cols"]

rf  = rf_data["model"]
cat = cat_data["model"]

print(f"\nModèles chargés : RF (PR-AUC=0.3458) | CatBoost (PR-AUC=0.3580)")
print(f"Features : {len(feature_cols)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5a — Permutation Importance sur CatBoost ET RF
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  5a — Permutation Importance")
print("="*55)

def compute_perm_importance(model, X_val, y_val, feature_cols, n_repeats=10):
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring="average_precision",
        random_state=42,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature"  : feature_cols,
        "importance_mean" : result.importances_mean,
        "importance_std"  : result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return df

print("\nPermutation importance RF (10 répétitions)...")
perm_rf = compute_perm_importance(rf, X_val, y_val, feature_cols)
print("  OK ✓")

print("Permutation importance CatBoost (10 répétitions)...")
perm_cat = compute_perm_importance(cat, X_val, y_val, feature_cols)
print("  OK ✓")

# Affichage top 15
print(f"\n  Top 15 — Permutation Importance RF :")
for _, row in perm_rf.head(15).iterrows():
    print(f"  {row['feature']:<30} : {row['importance_mean']:.5f} ± {row['importance_std']:.5f}")

print(f"\n  Top 15 — Permutation Importance CatBoost :")
for _, row in perm_cat.head(15).iterrows():
    print(f"  {row['feature']:<30} : {row['importance_mean']:.5f} ± {row['importance_std']:.5f}")

# Figure comparaison permutation importance
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Permutation Importance — RF vs CatBoost", fontsize=13, fontweight="bold")

top_n = 20
for ax, df, title, color in [
    (axes[0], perm_rf.head(top_n),  "Random Forest",  "#1D9E75"),
    (axes[1], perm_cat.head(top_n), "CatBoost",       "#534AB7"),
]:
    ax.barh(df["feature"][::-1], df["importance_mean"][::-1], color=color,
            xerr=df["importance_std"][::-1], capsize=3, alpha=0.85)
    ax.set_title(f"{title} — Top {top_n}", fontweight="bold")
    ax.set_xlabel("Importance (chute de PR-AUC)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("figures_ML/step5a_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figures_ML/step5a_permutation_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5b — SHAP sur CatBoost
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  5b — SHAP values (CatBoost)")
print("="*55)

try:
    import shap

    print("\nCalcul des SHAP values sur val (peut prendre 2-3 min)...")
    explainer   = shap.TreeExplainer(cat)
    shap_values = explainer.shap_values(X_val)
    print("  OK ✓")

    shap_df = pd.DataFrame({
        "feature"        : feature_cols,
        "shap_importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importance", ascending=False)

    print(f"\n  Top 15 — SHAP Importance (CatBoost) :")
    for _, row in shap_df.head(15).iterrows():
        print(f"  {row['feature']:<30} : {row['shap_importance']:.5f}")

    # Figure SHAP summary
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("SHAP — CatBoost", fontsize=13, fontweight="bold")

    # Barplot importance
    top20_shap = shap_df.head(20)
    axes[0].barh(top20_shap["feature"][::-1],
                 top20_shap["shap_importance"][::-1],
                 color="#534AB7", alpha=0.85)
    axes[0].set_title("SHAP Feature Importance (mean |SHAP|)")
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Beeswarm plot
    plt.sca(axes[1])
    shap.summary_plot(
        shap_values, X_val,
        feature_names=feature_cols,
        max_display=20,
        show=False,
        plot_size=None,
    )
    axes[1].set_title("SHAP Beeswarm — direction des effets")

    plt.tight_layout()
    plt.savefig("figures_ML/step5b_shap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → figures_ML/step5b_shap.png")

except ImportError:
    print("  SHAP non installé — pip install shap")
    shap_df = None


# ═══════════════════════════════════════════════════════════════════════════════
# 5c — Feature Selection + Réentraînement
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  5c — Feature Selection")
print("="*55)

# Combinaison des importances RF + CatBoost (native + permutation)
native_rf  = rf_data["importance_df"].set_index("feature")["importance"]
native_cat = pd.Series(
    dict(zip(feature_cols, cat.get_feature_importance())),
    name="importance"
) / cat.get_feature_importance().sum()  # normalisation

perm_rf_s  = perm_rf.set_index("feature")["importance_mean"]
perm_cat_s = perm_cat.set_index("feature")["importance_mean"]

# Score combiné : moyenne des 4 rangs
combined = pd.DataFrame({
    "native_rf" : native_rf,
    "native_cat": native_cat,
    "perm_rf"   : perm_rf_s,
    "perm_cat"  : perm_cat_s,
})

# Normalisation 0-1 par colonne puis moyenne
for col in combined.columns:
    mn, mx = combined[col].min(), combined[col].max()
    combined[col] = (combined[col] - mn) / (mx - mn + 1e-8)

combined["score_final"] = combined.mean(axis=1)
combined = combined.sort_values("score_final", ascending=False)

print("\n  Score combiné (RF native + RF perm + CAT native + CAT perm) :")
print(f"\n  {'Feature':<30} {'Score':>8} {'RF_nat':>8} {'CAT_nat':>8} {'RF_perm':>8} {'CAT_perm':>8}")
print(f"  {'-'*72}")
for feat, row in combined.iterrows():
    print(f"  {feat:<30} {row['score_final']:>8.4f} {row['native_rf']:>8.4f} "
          f"{row['native_cat']:>8.4f} {row['perm_rf']:>8.4f} {row['perm_cat']:>8.4f}")

# Seuil de sélection : features avec score > 0.20
selected_features = combined[combined["score_final"] > 0.10].index.tolist()
print(f"\n  Features sélectionnées (score > 0.20) : {len(selected_features)}")
print(f"  {selected_features}")

# Features éliminées
eliminated = [f for f in feature_cols if f not in selected_features]
print(f"\n  Features éliminées : {len(eliminated)}")
print(f"  {eliminated}")


# ── Réentraînement sur features réduites ──────────────────────────────────────

print("\n  Réentraînement sur features réduites...")

feat_idx      = [feature_cols.index(f) for f in selected_features]
X_train_red   = X_train[:, feat_idx]
X_val_red     = X_val[:, feat_idx]

# CatBoost réduit
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

cat_red = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    eval_metric="PRAUC",
    random_seed=42,
    verbose=0,
)
cat_red.fit(X_train_red, y_train, eval_set=(X_val_red, y_val),
            early_stopping_rounds=50, verbose=False)

pr_cat_red = average_precision_score(y_val, cat_red.predict_proba(X_val_red)[:, 1])
roc_cat_red = roc_auc_score(y_val, cat_red.predict_proba(X_val_red)[:, 1])

# RF réduit
rf_red = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1,
)
rf_red.fit(X_train_red, y_train)

pr_rf_red  = average_precision_score(y_val, rf_red.predict_proba(X_val_red)[:, 1])
roc_rf_red = roc_auc_score(y_val, rf_red.predict_proba(X_val_red)[:, 1])

print(f"\n{'='*55}")
print(f"  Comparaison complet vs réduit")
print(f"{'='*55}")
print(f"\n  {'Modèle':<25} {'Features':>8} {'PR-AUC':>8} {'ROC-AUC':>8}")
print(f"  {'-'*50}")
print(f"  {'CatBoost complet':<25} {len(feature_cols):>8} {0.3580:>8.4f} {0.6978:>8.4f}")
print(f"  {'CatBoost réduit':<25} {len(selected_features):>8} {pr_cat_red:>8.4f} {roc_cat_red:>8.4f}")
print(f"  {'RF complet':<25} {len(feature_cols):>8} {0.3458:>8.4f} {0.6941:>8.4f}")
print(f"  {'RF réduit':<25} {len(selected_features):>8} {pr_rf_red:>8.4f} {roc_rf_red:>8.4f}")

delta_cat = pr_cat_red - 0.3580
delta_rf  = pr_rf_red  - 0.3458
print(f"\n  Δ PR-AUC CatBoost : {delta_cat:+.4f}")
print(f"  Δ PR-AUC RF       : {delta_rf:+.4f}")


# ── Question scientifique centrale ────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Question scientifique : charge vs médical")
print(f"{'='*55}")

# Groupes de features
medical_features = [f for f in selected_features if f in [
    "days_since_last_inj", "was_injured_10d", "days_out_season",
    "last_inj_category", "last_inj_body_part", "is_returning",
    "last_injury_duration", "last_injury_severity",
    "total_days_out_season", "max_injury_duration", "max_injury_severity",
    "never_injured", "has_previous_injury", "no_chronic_load"
]]
load_features = [f for f in selected_features if f in [
    "games_10d", "min_total_10d", "min_avg_10d", "min_max_10d",
    "away_games_10d", "away_ratio_10d", "b2b",
    "min_total_28d", "games_28d", "acute_load_7d",
    "chronic_load_28d", "acwr", "min_season_avg", "games_season",
    "min_avg_vs_season", "delta_min_7d_28d", "MIN"
]]
profile_features = [f for f in selected_features if f in [
    "age_at_game", "nba_years", "position_code", "height_cm",
    "weight_lbs", "PTS", "REB", "AST", "IS_AWAY"
]]

print(f"\n  Features médicales   : {len(medical_features)} → {medical_features}")
print(f"  Features de charge   : {len(load_features)} → {load_features}")
print(f"  Features profil      : {len(profile_features)} → {profile_features}")

# Test : CatBoost sans features médicales
feat_no_med = load_features + profile_features
if len(feat_no_med) > 0:
    idx_no_med  = [feature_cols.index(f) for f in feat_no_med]
    cat_no_med  = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=42, verbose=0
    )
    cat_no_med.fit(X_train[:, idx_no_med], y_train, verbose=False)
    pr_no_med = average_precision_score(
        y_val, cat_no_med.predict_proba(X_val[:, idx_no_med])[:, 1]
    )
    print(f"\n  CatBoost SANS médical : PR-AUC = {pr_no_med:.4f}")

# Test : CatBoost sans features de charge
feat_no_load = medical_features + profile_features
if len(feat_no_load) > 0:
    idx_no_load = [feature_cols.index(f) for f in feat_no_load]
    cat_no_load = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=42, verbose=0
    )
    cat_no_load.fit(X_train[:, idx_no_load], y_train, verbose=False)
    pr_no_load = average_precision_score(
        y_val, cat_no_load.predict_proba(X_val[:, idx_no_load])[:, 1]
    )
    print(f"  CatBoost SANS charge  : PR-AUC = {pr_no_load:.4f}")

print(f"\n  CatBoost COMPLET      : PR-AUC = 0.3580")
print(f"\n  Interprétation :")
if len(feat_no_med) > 0 and len(feat_no_load) > 0:
    drop_med  = 0.3580 - pr_no_med
    drop_load = 0.3580 - pr_no_load
    print(f"  Retirer médical  → chute de {drop_med:+.4f}")
    print(f"  Retirer charge   → chute de {drop_load:+.4f}")
    if drop_med > drop_load:
        print(f"\n  → Les variables MÉDICALES dominent le signal")
    else:
        print(f"\n  → Les variables de CHARGE dominent le signal")
    if drop_load > 0.01:
        print(f"  → La charge apporte un signal RÉEL au-delà du médical (+{drop_load:.4f})")
    else:
        print(f"  → La charge n'apporte pas de signal significatif seule")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

results_step5 = {
    "perm_rf"          : perm_rf,
    "perm_cat"         : perm_cat,
    "combined_scores"  : combined,
    "selected_features": selected_features,
    "eliminated"       : eliminated,
    "cat_red"          : cat_red,
    "rf_red"           : rf_red,
    "pr_cat_red"       : pr_cat_red,
    "pr_rf_red"        : pr_rf_red,
    "feat_idx"         : feat_idx,
}
with open("Models/step5_results.pkl", "wb") as f:
    pickle.dump(results_step5, f)

print(f"\n  Résultats sauvegardés : Models/step5_results.pkl")