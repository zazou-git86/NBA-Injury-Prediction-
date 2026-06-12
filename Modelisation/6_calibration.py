"""
Étape 6 — Calibration
Prédiction de blessures NBA

But : vérifier et corriger que les probabilités prédites
      sont honnêtes — si le modèle dit 30%, est-ce que
      30% des joueurs dans cette situation se blessent vraiment ?

Méthodes :
    - Platt Scaling (CalibratedClassifierCV sigmoid)
    - Isotonic Regression (CalibratedClassifierCV isotonic)

Métriques de calibration :
    - Brier Score (plus bas = mieux)
    - Courbe de calibration (reliability diagram)
    - ECE (Expected Calibration Error)

Modèles : CatBoost réduit + RF réduit
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, average_precision_score,
    roc_auc_score, recall_score, precision_score
)
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

Path("Models").mkdir(exist_ok=True)
Path("figures_ML").mkdir(exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 6 — Calibration")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("Models/step5_results.pkl", "rb") as f:
    step5 = pickle.load(f)

X_train      = data["X_train_tree"]
X_val        = data["X_val_tree"]
y_train      = data["y_train"]
y_val        = data["y_val"]
feature_cols = data["feature_cols"]

selected_features = step5["selected_features"]
feat_idx          = [feature_cols.index(f) for f in selected_features]
X_train_red       = X_train[:, feat_idx]
X_val_red         = X_val[:, feat_idx]

cat_red = step5["cat_red"]
rf_red  = step5["rf_red"]

print(f"\nFeatures réduites : {len(selected_features)}")
print(f"Train : {X_train_red.shape} | Val : {X_val_red.shape}")


# ── Utilitaires ───────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calcule l'ECE — erreur de calibration attendue."""
    bins        = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])
    ece         = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece     += mask.mean() * abs(bin_acc - bin_conf)
    return ece

def evaluate_calibration(name, y_val, y_proba):
    """Calcule toutes les métriques de calibration."""
    brier = brier_score_loss(y_val, y_proba)
    ece   = expected_calibration_error(y_val, y_proba)
    pr    = average_precision_score(y_val, y_proba)
    roc   = roc_auc_score(y_val, y_proba)
    print(f"  {name:<35} Brier={brier:.4f}  ECE={ece:.4f}  PR-AUC={pr:.4f}  ROC={roc:.4f}")
    return {"name": name, "brier": brier, "ece": ece, "pr_auc": pr, "roc_auc": roc, "proba": y_proba}


# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost — Non calibré + Platt + Isotonic
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  CatBoost réduit — Calibration")
print("="*55)

# Non calibré
y_proba_cat = cat_red.predict_proba(X_val_red)[:, 1]

# Platt Scaling
print("\nEntraînement Platt Scaling (CatBoost)...")
cat_platt = CalibratedClassifierCV(cat_red, method="sigmoid", cv=None, ensemble=False)
cat_platt.fit(X_train_red, y_train)
y_proba_cat_platt = cat_platt.predict_proba(X_val_red)[:, 1]
print("  OK ✓")

# Isotonic Regression
print("Entraînement Isotonic Regression (CatBoost)...")
cat_iso = CalibratedClassifierCV(cat_red, method="isotonic", cv=None, ensemble=False)
cat_iso.fit(X_train_red, y_train)
y_proba_cat_iso = cat_iso.predict_proba(X_val_red)[:, 1]
print("  OK ✓")

print("\n  Métriques de calibration — CatBoost :")
print(f"  {'Modèle':<35} {'Brier':>8} {'ECE':>8} {'PR-AUC':>8} {'ROC':>8}")
print(f"  {'-'*65}")
res_cat_base  = evaluate_calibration("CatBoost non calibré",   y_val, y_proba_cat)
res_cat_platt = evaluate_calibration("CatBoost + Platt",       y_val, y_proba_cat_platt)
res_cat_iso   = evaluate_calibration("CatBoost + Isotonic",    y_val, y_proba_cat_iso)


# ═══════════════════════════════════════════════════════════════════════════════
# RF — Non calibré + Platt + Isotonic
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  Random Forest réduit — Calibration")
print("="*55)

y_proba_rf = rf_red.predict_proba(X_val_red)[:, 1]

print("\nEntraînement Platt Scaling (RF)...")
rf_platt = CalibratedClassifierCV(rf_red, method="sigmoid", cv=None, ensemble=False)
rf_platt.fit(X_train_red, y_train)
y_proba_rf_platt = rf_platt.predict_proba(X_val_red)[:, 1]
print("  OK ✓")

print("Entraînement Isotonic Regression (RF)...")
rf_iso = CalibratedClassifierCV(rf_red, method="isotonic", cv=None, ensemble=False)
rf_iso.fit(X_train_red, y_train)
y_proba_rf_iso = rf_iso.predict_proba(X_val_red)[:, 1]
print("  OK ✓")

print("\n  Métriques de calibration — Random Forest :")
print(f"  {'Modèle':<35} {'Brier':>8} {'ECE':>8} {'PR-AUC':>8} {'ROC':>8}")
print(f"  {'-'*65}")
res_rf_base  = evaluate_calibration("RF non calibré",          y_val, y_proba_rf)
res_rf_platt = evaluate_calibration("RF + Platt",              y_val, y_proba_rf_platt)
res_rf_iso   = evaluate_calibration("RF + Isotonic",           y_val, y_proba_rf_iso)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures — Reliability Diagrams
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGénération des courbes de calibration...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Courbes de calibration — Reliability Diagrams", fontsize=13, fontweight="bold")

configs = [
    # CatBoost
    (axes[0][0], "CatBoost non calibré",  y_proba_cat,       "#534AB7"),
    (axes[0][1], "CatBoost + Platt",      y_proba_cat_platt, "#1D9E75"),
    (axes[0][2], "CatBoost + Isotonic",   y_proba_cat_iso,   "#BA7517"),
    # RF
    (axes[1][0], "RF non calibré",        y_proba_rf,        "#534AB7"),
    (axes[1][1], "RF + Platt",            y_proba_rf_platt,  "#1D9E75"),
    (axes[1][2], "RF + Isotonic",         y_proba_rf_iso,    "#BA7517"),
]

for ax, title, y_proba, color in configs:
    fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Calibration parfaite")
    ax.plot(mean_pred, fraction_pos, "o-", color=color, linewidth=2,
            markersize=6, label=title)

    brier = brier_score_loss(y_val, y_proba)
    ece   = expected_calibration_error(y_val, y_proba)

    ax.set_title(f"{title}\nBrier={brier:.4f} | ECE={ece:.4f}", fontsize=10)
    ax.set_xlabel("Probabilité prédite moyenne")
    ax.set_ylabel("Fraction de positifs observés")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures_ML/step6_calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step6_calibration_curves.png")


# ── Distribution des probabilités ────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle("Distribution des probabilités prédites — blessés vs non-blessés",
             fontsize=13, fontweight="bold")

for ax, title, y_proba, color in configs:
    ax.hist(y_proba[y_val == 0], bins=40, alpha=0.6, color="#1D9E75",
            density=True, label="Non blessé (0)")
    ax.hist(y_proba[y_val == 1], bins=40, alpha=0.6, color="#D85A30",
            density=True, label="Blessé (1)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Densité")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_ML/step6_proba_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step6_proba_distributions.png")


# ── Résumé comparatif ─────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Résumé comparatif — tous les modèles calibrés")
print(f"{'='*55}")

all_res = [
    res_cat_base, res_cat_platt, res_cat_iso,
    res_rf_base,  res_rf_platt,  res_rf_iso,
]
df_res = pd.DataFrame(all_res).sort_values("brier")

print(f"\n  {'Modèle':<35} {'Brier':>8} {'ECE':>8} {'PR-AUC':>8} {'ROC':>8}")
print(f"  {'-'*65}")
for _, row in df_res.iterrows():
    print(f"  {row['name']:<35} {row['brier']:>8.4f} {row['ece']:>8.4f} "
          f"{row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f}")

# Meilleur modèle calibré
best = df_res.iloc[0]
print(f"\n  Meilleur modèle (Brier le plus bas) : {best['name']}")
print(f"  → Brier={best['brier']:.4f} | ECE={best['ece']:.4f} | PR-AUC={best['pr_auc']:.4f}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

best_name  = best["name"]
if "CatBoost" in best_name and "Platt" in best_name:
    best_model = cat_platt
    best_proba = y_proba_cat_platt
elif "CatBoost" in best_name and "Isotonic" in best_name:
    best_model = cat_iso
    best_proba = y_proba_cat_iso
elif "RF" in best_name and "Platt" in best_name:
    best_model = rf_platt
    best_proba = y_proba_rf_platt
elif "RF" in best_name and "Isotonic" in best_name:
    best_model = rf_iso
    best_proba = y_proba_rf_iso
elif "CatBoost" in best_name:
    best_model = cat_red
    best_proba = y_proba_cat
else:
    best_model = rf_red
    best_proba = y_proba_rf

results_step6 = {
    "best_model"     : best_model,
    "best_model_name": best_name,
    "best_proba"     : best_proba,
    "feat_idx"       : feat_idx,
    "selected_features": selected_features,
    "cat_platt"      : cat_platt,
    "cat_iso"        : cat_iso,
    "rf_platt"       : rf_platt,
    "rf_iso"         : rf_iso,
    "df_results"     : df_res,
}

with open("Models/step6_results.pkl", "wb") as f:
    pickle.dump(results_step6, f)

print(f"\n  Résultats sauvegardés : Models/step6_results.pkl")
print(f"\n{'='*55}")
print(f"  Étape 6 terminée — prêt pour l'optimisation du seuil")
print(f"{'='*55}")