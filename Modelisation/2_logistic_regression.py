"""
Étape 2 — Baseline Logistic Regression
Prédiction de blessures NBA

But : établir une baseline linéaire simple pour comparer
      les modèles plus complexes. Si un modèle non-linéaire
      ne fait pas mieux, le problème est dans les features.

Métriques : PR-AUC, Recall, Precision, F1, ROC-AUC
Évaluation : sur val 2024-25 uniquement
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Chargement des données preprocessées ─────────────────────────────────────

print("=" * 55)
print("  Étape 2 — Baseline Logistic Regression")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train_lr"]
X_val   = data["X_val_lr"]
y_train = data["y_train"]
y_val   = data["y_val"]
feature_cols = data["feature_cols"]

print(f"\nTrain : {X_train.shape} | Val : {X_val.shape}")


# ── Entraînement ──────────────────────────────────────────────────────────────

print("\nEntraînement Logistic Regression...")
lr = LogisticRegression(
    class_weight="balanced",   # gère le déséquilibre 16/84
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
    C=1.0,
)
lr.fit(X_train, y_train)
print("  OK ✓")


# ── Prédictions ───────────────────────────────────────────────────────────────

y_pred_proba = lr.predict_proba(X_val)[:, 1]
y_pred       = lr.predict(X_val)


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


# ── Courbe Precision-Recall ───────────────────────────────────────────────────

precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression — Baseline", fontsize=13, fontweight="bold")

# PR Curve
axes[0].plot(recall, precision, color="#1D9E75", linewidth=2)
axes[0].axhline(y=y_val.mean(), color="gray", linestyle="--",
                label=f"Baseline ({y_val.mean():.2f})")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Courbe PR — AUC = {pr_auc:.4f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
axes[1].plot(fpr, tpr, color="#534AB7", linewidth=2)
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title(f"Courbe ROC — AUC = {roc_auc:.4f}")
axes[1].grid(True, alpha=0.3)

# Recall / Precision vs Threshold
axes[2].plot(thresholds, precision[:-1], color="#BA7517", label="Precision", linewidth=2)
axes[2].plot(thresholds, recall[:-1],    color="#D85A30", label="Recall",    linewidth=2)
axes[2].axvline(x=0.5, color="gray", linestyle="--", label="Seuil 0.5")
axes[2].set_xlabel("Seuil de décision")
axes[2].set_ylabel("Score")
axes[2].set_title("Precision / Recall vs Seuil")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

Path("figures_ML").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("figures_ML/step2_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figures_ML/step2_logistic_regression.png")


# ── Top features (coefficients) ───────────────────────────────────────────────

coef_df = pd.DataFrame({
    "feature"    : feature_cols,
    "coefficient": lr.coef_[0],
    "abs_coef"   : np.abs(lr.coef_[0])
}).sort_values("abs_coef", ascending=False)

print(f"\n  Top 15 features (coefficients) :")
for _, row in coef_df.head(15).iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    print(f"  {direction} {row['feature']:<30} : {row['coefficient']:.4f}")


# ── Seuil optimal (maximise F1) ───────────────────────────────────────────────

f1_scores  = 2 * precision * recall / (precision + recall + 1e-8)
best_idx   = np.argmax(f1_scores[:-1])
best_thresh = thresholds[best_idx]
best_f1    = f1_scores[best_idx]

y_pred_opt = (y_pred_proba >= best_thresh).astype(int)
from sklearn.metrics import recall_score, precision_score
rec_opt    = recall_score(y_val, y_pred_opt)
prec_opt   = precision_score(y_val, y_pred_opt)

print(f"\n  Seuil optimal (max F1) : {best_thresh:.3f}")
print(f"    Recall    : {rec_opt:.4f}")
print(f"    Precision : {prec_opt:.4f}")
print(f"    F1        : {best_f1:.4f}")


# ── Sauvegarde du modèle ──────────────────────────────────────────────────────

results = {
    "model"       : lr,
    "roc_auc"     : roc_auc,
    "pr_auc"      : pr_auc,
    "best_thresh" : best_thresh,
    "coef_df"     : coef_df,
}
with open("Models/step2_lr.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\n  Modèle sauvegardé : Models/step2_lr.pkl")
print(f"\n{'='*55}")
print(f"  Résumé Logistic Regression")
print(f"{'='*55}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"  Recall   : {rec_opt:.4f} (seuil {best_thresh:.3f})")
print(f"  Precision: {prec_opt:.4f} (seuil {best_thresh:.3f})")