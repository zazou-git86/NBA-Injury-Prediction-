"""
Étape 3 — Random Forest
Prédiction de blessures NBA

But : premier modèle non-linéaire, robuste à la multicolinéarité
      et aux features corrélées. Référence pour les modèles suivants.

Métriques : PR-AUC, Recall, Precision, F1, ROC-AUC
Évaluation : sur val 2024-25 uniquement
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    roc_curve, recall_score, precision_score
)

# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 3 — Random Forest")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train      = data["X_train_tree"]
X_val        = data["X_val_tree"]
y_train      = data["y_train"]
y_val        = data["y_val"]
feature_cols = data["feature_cols"]

print(f"\nTrain : {X_train.shape} | Val : {X_val.shape}")


# ── Entraînement ──────────────────────────────────────────────────────────────

print("\nEntraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,        # 500 arbres — bon compromis perf/vitesse
    max_depth=12,            # limite la profondeur pour éviter overfitting
    min_samples_leaf=20,     # au moins 20 exemples par feuille
    max_features="sqrt",     # sqrt(40) ≈ 6 features par split — standard
    class_weight="balanced", # compense le déséquilibre 16/84
    random_state=42,
    n_jobs=-1,               # utilise tous les cores disponibles
)
rf.fit(X_train, y_train)
print("  OK ✓")


# ── Prédictions ───────────────────────────────────────────────────────────────

y_pred_proba = rf.predict_proba(X_val)[:, 1]
y_pred       = rf.predict(X_val)


# ── Métriques ─────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  Métriques sur Val 2024-25")
print("="*55)

roc_auc = roc_auc_score(y_val, y_pred_proba)
pr_auc  = average_precision_score(y_val, y_pred_proba)

print(f"\n  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}  ← métrique principale")
print(f"\n  Classification report (seuil 0.5) :")
print(classification_report(y_val, y_pred, target_names=["Non blessé", "Blessé"]))


# ── Figures ───────────────────────────────────────────────────────────────────

precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_pred_proba)
fpr, tpr, _                               = roc_curve(y_val, y_pred_proba)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest", fontsize=13, fontweight="bold")

# PR Curve
axes[0].plot(recall_curve, precision_curve, color="#1D9E75", linewidth=2)
axes[0].axhline(y=y_val.mean(), color="gray", linestyle="--",
                label=f"Baseline ({y_val.mean():.2f})")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Courbe PR — AUC = {pr_auc:.4f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC Curve
axes[1].plot(fpr, tpr, color="#534AB7", linewidth=2)
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title(f"Courbe ROC — AUC = {roc_auc:.4f}")
axes[1].grid(True, alpha=0.3)

# Precision / Recall vs Seuil
axes[2].plot(thresholds, precision_curve[:-1], color="#BA7517", label="Precision", linewidth=2)
axes[2].plot(thresholds, recall_curve[:-1],    color="#D85A30", label="Recall",    linewidth=2)
axes[2].axvline(x=0.5, color="gray", linestyle="--", label="Seuil 0.5")
axes[2].set_xlabel("Seuil de décision")
axes[2].set_ylabel("Score")
axes[2].set_title("Precision / Recall vs Seuil")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

Path("figures_ML").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("figures_ML/step3_random_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figures_ML/step3_random_forest.png")


# ── Feature importance native ─────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\n  Top 15 features (importance native) :")
for _, row in importance_df.head(15).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"  {row['feature']:<30} : {row['importance']:.4f} {bar}")

# Figure feature importance
fig, ax = plt.subplots(figsize=(10, 10))
top20   = importance_df.head(20)
ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#1D9E75")
ax.set_title("Random Forest — Feature Importance (top 20)", fontweight="bold")
ax.set_xlabel("Importance (MDI)")
plt.tight_layout()
plt.savefig("figures_ML/step3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step3_feature_importance.png")


# ── Seuil optimal ─────────────────────────────────────────────────────────────

f1_scores   = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
best_idx    = np.argmax(f1_scores[:-1])
best_thresh = thresholds[best_idx]
best_f1     = f1_scores[best_idx]

y_pred_opt  = (y_pred_proba >= best_thresh).astype(int)
rec_opt     = recall_score(y_val, y_pred_opt)
prec_opt    = precision_score(y_val, y_pred_opt)

print(f"\n  Seuil optimal (max F1) : {best_thresh:.3f}")
print(f"    Recall    : {rec_opt:.4f}")
print(f"    Precision : {prec_opt:.4f}")
print(f"    F1        : {best_f1:.4f}")


# ── Comparaison avec LR ───────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Comparaison LR vs RF")
print(f"{'='*55}")
print(f"  {'Métrique':<15} {'LR':>10} {'RF':>10} {'Δ':>10}")
print(f"  {'-'*45}")
print(f"  {'PR-AUC':<15} {'0.3410':>10} {pr_auc:>10.4f} {pr_auc-0.3410:>+10.4f}")
print(f"  {'ROC-AUC':<15} {'0.6790':>10} {roc_auc:>10.4f} {roc_auc-0.6790:>+10.4f}")
print(f"  {'Recall':<15} {'0.5872':>10} {rec_opt:>10.4f} {rec_opt-0.5872:>+10.4f}")
print(f"  {'Precision':<15} {'0.2781':>10} {prec_opt:>10.4f} {prec_opt-0.2781:>+10.4f}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

results = {
    "model"        : rf,
    "roc_auc"      : roc_auc,
    "pr_auc"       : pr_auc,
    "best_thresh"  : best_thresh,
    "importance_df": importance_df,
    "y_pred_proba" : y_pred_proba,
}
with open("Models/step3_rf.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Modèle sauvegardé : Models/step3_rf.pkl")
print(f"\n{'='*55}")
print(f"  Résumé Random Forest")
print(f"{'='*55}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"  Recall   : {rec_opt:.4f} (seuil {best_thresh:.3f})")
print(f"  Precision: {prec_opt:.4f} (seuil {best_thresh:.3f})")