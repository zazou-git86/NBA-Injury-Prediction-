"""
Étape 8 — Test Final
Prédiction de blessures NBA

ATTENTION : Cette étape ne doit être lancée QU'UNE SEULE FOIS.
            Le dataset test (2025-26) n'a jamais été utilisé
            dans les étapes précédentes.

Modèle : CatBoost réduit (20 features, entraîné sur 2021-24)
Seuil  : 3 scénarios (Conservateur=0.600, Équilibré=0.511, Agressif=0.441)
Test   : Saison 2025-26 (26 409 matchs)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    roc_curve, recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
)

Path("Models").mkdir(exist_ok=True)
Path("figures_ML").mkdir(exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 8 — TEST FINAL (2025-26)")
print("  !! Dataset test utilisé pour la première fois !!")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("Models/step5_results.pkl", "rb") as f:
    step5 = pickle.load(f)

with open("Models/step7_results.pkl", "rb") as f:
    step7 = pickle.load(f)

X_val        = data["X_val_tree"]
X_test       = data["X_test_tree"]
y_val        = data["y_val"]
y_test       = data["y_test"]
feature_cols = data["feature_cols"]

selected_features = step5["selected_features"]
feat_idx          = [feature_cols.index(f) for f in selected_features]
X_val_red         = X_val[:, feat_idx]
X_test_red        = X_test[:, feat_idx]
cat_red           = step5["cat_red"]
scenarios         = step7["scenarios"]

print(f"\nFeatures : {len(selected_features)}")
print(f"Val  2024-25 : {len(y_val):,} matchs | {y_val.sum():,} blessures ({y_val.mean()*100:.1f}%)")
print(f"Test 2025-26 : {len(y_test):,} matchs | {y_test.sum():,} blessures ({y_test.mean()*100:.1f}%)")


# ── Prédictions sur le test ───────────────────────────────────────────────────

y_proba_val  = cat_red.predict_proba(X_val_red)[:, 1]
y_proba_test = cat_red.predict_proba(X_test_red)[:, 1]


# ── Métriques globales ────────────────────────────────────────────────────────

roc_val  = roc_auc_score(y_val,  y_proba_val)
pr_val   = average_precision_score(y_val,  y_proba_val)
roc_test = roc_auc_score(y_test, y_proba_test)
pr_test  = average_precision_score(y_test, y_proba_test)

print(f"\n{'='*55}")
print(f"  Métriques globales — Val vs Test")
print(f"{'='*55}")
print(f"\n  {'Métrique':<15} {'Val 2024-25':>14} {'Test 2025-26':>14} {'Δ':>10}")
print(f"  {'-'*55}")
print(f"  {'PR-AUC':<15} {pr_val:>14.4f} {pr_test:>14.4f} {pr_test-pr_val:>+10.4f}")
print(f"  {'ROC-AUC':<15} {roc_val:>14.4f} {roc_test:>14.4f} {roc_test-roc_val:>+10.4f}")
print(f"  {'Brier':<15} {brier_score_loss(y_val, y_proba_val):>14.4f} "
      f"{brier_score_loss(y_test, y_proba_test):>14.4f} "
      f"{brier_score_loss(y_test, y_proba_test)-brier_score_loss(y_val, y_proba_val):>+10.4f}")


# ── Résultats par scénario ────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Résultats par scénario — Test 2025-26")
print(f"{'='*55}")
print(f"\n  Rappel : {y_test.sum()} vraies blessures dans la période\n")

test_scenarios = {}
for name, sc in scenarios.items():
    thresh      = sc["threshold"]
    y_pred_test = (y_proba_test >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    prec        = precision_score(y_test, y_pred_test)
    rec         = recall_score(y_test, y_pred_test)
    f1          = f1_score(y_test, y_pred_test)
    n_alerts    = int(tp + fp)

    test_scenarios[name] = {
        "threshold": thresh, "precision": prec, "recall": rec,
        "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "tn": int(tn), "n_alerts": n_alerts, "color": sc["color"],
        "y_pred": y_pred_test
    }

    # Comparaison val vs test
    val_prec = sc["precision"]
    val_rec  = sc["recall"]

    print(f"  ── {name} (seuil={thresh:.3f}) ──")
    print(f"  {'Métrique':<12} {'Val 2024-25':>14} {'Test 2025-26':>14} {'Δ':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Precision':<12} {val_prec:>14.3f} {prec:>14.3f} {prec-val_prec:>+10.3f}")
    print(f"  {'Recall':<12} {val_rec:>14.3f} {rec:>14.3f} {rec-val_rec:>+10.3f}")
    print(f"  {'F1':<12} {sc['f1']:>14.3f} {f1:>14.3f} {f1-sc['f1']:>+10.3f}")
    print(f"  Alertes    : {n_alerts:,} | Détectées : {tp:,} | Manquées : {fn:,}")
    print(f"  FA/Blessure: {fp/max(tp,1):.1f}")
    print()


# ── Figures ───────────────────────────────────────────────────────────────────

# Figure 1 : Val vs Test — courbes PR et ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Val 2024-25 vs Test 2025-26 — CatBoost réduit",
             fontsize=13, fontweight="bold")

# PR Curve
prec_val_c, rec_val_c, _   = precision_recall_curve(y_val,  y_proba_val)
prec_test_c, rec_test_c, _ = precision_recall_curve(y_test, y_proba_test)

axes[0].plot(rec_val_c,  prec_val_c,  color="#1D9E75", linewidth=2,
             label=f"Val 2024-25  (AUC={pr_val:.4f})")
axes[0].plot(rec_test_c, prec_test_c, color="#D85A30", linewidth=2,
             linestyle="--", label=f"Test 2025-26 (AUC={pr_test:.4f})")
axes[0].axhline(y=y_test.mean(), color="gray", linestyle=":",
                label=f"Baseline ({y_test.mean():.2f})")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Courbes PR")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# ROC Curve
fpr_val,  tpr_val,  _ = roc_curve(y_val,  y_proba_val)
fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)

axes[1].plot(fpr_val,  tpr_val,  color="#1D9E75", linewidth=2,
             label=f"Val 2024-25  (AUC={roc_val:.4f})")
axes[1].plot(fpr_test, tpr_test, color="#D85A30", linewidth=2,
             linestyle="--", label=f"Test 2025-26 (AUC={roc_test:.4f})")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Courbes ROC")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_ML/step8_val_vs_test_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step8_val_vs_test_curves.png")


# Figure 2 : Matrices de confusion test
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Matrices de confusion — Test 2025-26",
             fontsize=13, fontweight="bold")

for ax, (name, sc) in zip(axes, test_scenarios.items()):
    cm   = confusion_matrix(y_test, sc["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non blessé", "Blessé"])
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title(f"{name}\n(seuil={sc['threshold']:.3f})", fontweight="bold")

plt.tight_layout()
plt.savefig("figures_ML/step8_test_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step8_test_confusion_matrices.png")


# Figure 3 : Comparaison Val vs Test par scénario
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparaison Val vs Test — par scénario",
             fontsize=13, fontweight="bold")

metrics_compare = ["precision", "recall", "f1"]
labels_compare  = ["Precision", "Recall", "F1"]

for ax, (name, sc) in zip(axes, test_scenarios.items()):
    val_sc   = scenarios[name]
    val_vals = [val_sc["precision"], val_sc["recall"], val_sc["f1"]]
    test_vals= [sc["precision"],     sc["recall"],     sc["f1"]]

    x     = np.arange(len(metrics_compare))
    width = 0.35

    b1 = ax.bar(x - width/2, val_vals,  width, label="Val 2024-25",
                color="#1D9E75", alpha=0.85)
    b2 = ax.bar(x + width/2, test_vals, width, label="Test 2025-26",
                color="#D85A30", alpha=0.85)

    for bar, val in zip(b1, val_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(b2, test_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Scénario {name}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_compare)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("figures_ML/step8_val_vs_test_scenarios.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step8_val_vs_test_scenarios.png")


# ── Résumé final ─────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  RÉSUMÉ FINAL DU PROJET")
print(f"{'='*55}")
print(f"\n  Modèle    : CatBoost (20 features, entraîné sur 2021-24)")
print(f"  PR-AUC    : {pr_test:.4f} (test 2025-26)")
print(f"  ROC-AUC   : {roc_test:.4f} (test 2025-26)")
print(f"\n  Question scientifique :")
print(f"  → Variables médicales : contribution dominante (-0.083 PR-AUC sans elles)")
print(f"  → Variables de charge : signal réel (+0.021 PR-AUC au-delà du médical)")
print(f"\n  Tableau final — Test 2025-26 :")
print(f"\n  {'Scénario':<15} {'Seuil':>7} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>6} {'Alertes':>8} {'Détectées':>10} {'Manquées':>9}")
print(f"  {'-'*80}")
for name, sc in test_scenarios.items():
    print(f"  {name:<15} {sc['threshold']:>7.3f} {sc['precision']:>10.3f} "
          f"{sc['recall']:>8.3f} {sc['f1']:>6.3f} {sc['n_alerts']:>8,} "
          f"{sc['tp']:>10,} {sc['fn']:>9,}")

print(f"\n  Blessures totales : {y_test.sum():,} sur {len(y_test):,} matchs")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

results_final = {
    "pr_auc_test"     : pr_test,
    "roc_auc_test"    : roc_test,
    "pr_auc_val"      : pr_val,
    "roc_auc_val"     : roc_val,
    "test_scenarios"  : test_scenarios,
    "y_proba_test"    : y_proba_test,
    "y_test"          : y_test,
}

with open("Models/step8_final_results.pkl", "wb") as f:
    pickle.dump(results_final, f)

print(f"\n  Résultats sauvegardés : Models/step8_final_results.pkl")
print(f"\n{'='*55}")
print(f"  Projet terminé.")
print(f"{'='*55}")