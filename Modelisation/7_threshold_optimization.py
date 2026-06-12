"""
Étape 7 — Optimisation du seuil de décision
Prédiction de blessures NBA

But : traduire la performance mathématique en décision actionnable.
      Le staff médical choisit son scénario selon sa tolérance
      aux fausses alarmes.

3 scénarios :
    - Conservateur : Precision maximale, peu d'alertes
    - Équilibré    : Compromis Precision/Recall
    - Agressif     : Recall maximal, beaucoup d'alertes

Modèle : CatBoost réduit (meilleur modèle, PR-AUC=0.3536)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

Path("Models").mkdir(exist_ok=True)
Path("figures_ML").mkdir(exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 7 — Optimisation du seuil")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("Models/step5_results.pkl", "rb") as f:
    step5 = pickle.load(f)

X_val        = data["X_val_tree"]
y_val        = data["y_val"]
feature_cols = data["feature_cols"]

selected_features = step5["selected_features"]
feat_idx          = [feature_cols.index(f) for f in selected_features]
X_val_red         = X_val[:, feat_idx]
cat_red           = step5["cat_red"]

y_proba = cat_red.predict_proba(X_val_red)[:, 1]

print(f"\nModèle : CatBoost réduit (20 features)")
print(f"Val    : {len(y_val):,} matchs | {y_val.sum():,} blessures réelles ({y_val.mean()*100:.1f}%)")


# ── Calcul de la courbe PR complète ──────────────────────────────────────────

precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_proba)
pr_auc = average_precision_score(y_val, y_proba)

# F1 par seuil
f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)

print(f"\nPR-AUC : {pr_auc:.4f}")
print(f"Seuil min : {thresholds.min():.3f} | max : {thresholds.max():.3f}")


# ── Définition des 3 scénarios ────────────────────────────────────────────────

def find_threshold_for_precision(precision_curve, recall_curve, thresholds,
                                  target_precision, mode="min_precision"):
    """
    Trouve le seuil qui maximise le Recall
    tout en respectant une Precision minimale cible.
    """
    valid = precision_curve[:-1] >= target_precision
    if not valid.any():
        return thresholds[0], precision_curve[0], recall_curve[0]
    idx   = np.where(valid)[0]
    best  = idx[np.argmax(recall_curve[:-1][idx])]
    return thresholds[best], precision_curve[best], recall_curve[best]

def find_threshold_max_f1(precision_curve, recall_curve, thresholds):
    """Trouve le seuil qui maximise le F1."""
    f1    = 2 * precision_curve[:-1] * recall_curve[:-1] / \
            (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
    best  = np.argmax(f1)
    return thresholds[best], precision_curve[best], recall_curve[best]

def find_threshold_for_recall(precision_curve, recall_curve, thresholds,
                               target_recall):
    """
    Trouve le seuil qui maximise la Precision
    tout en respectant un Recall minimal cible.
    """
    valid = recall_curve[:-1] >= target_recall
    if not valid.any():
        return thresholds[-1], precision_curve[-1], recall_curve[-1]
    idx  = np.where(valid)[0]
    best = idx[np.argmax(precision_curve[:-1][idx])]
    return thresholds[best], precision_curve[best], recall_curve[best]

# Scénario Conservateur : Precision ≥ 0.35, Recall maximisé
thresh_cons, prec_cons, rec_cons = find_threshold_for_precision(
    precision_curve, recall_curve, thresholds, target_precision=0.35)

# Scénario Équilibré : max F1
thresh_bal, prec_bal, rec_bal = find_threshold_max_f1(
    precision_curve, recall_curve, thresholds)

# Scénario Agressif : Recall ≥ 0.75, Precision maximisée
thresh_agg, prec_agg, rec_agg = find_threshold_for_recall(
    precision_curve, recall_curve, thresholds, target_recall=0.75)

scenarios = {
    "Conservateur": {
        "threshold": thresh_cons, "precision": prec_cons, "recall": rec_cons,
        "color": "#1D9E75", "description": "Peu d'alertes, haute confiance"
    },
    "Équilibré": {
        "threshold": thresh_bal, "precision": prec_bal, "recall": rec_bal,
        "color": "#534AB7", "description": "Compromis Precision/Recall"
    },
    "Agressif": {
        "threshold": thresh_agg, "precision": prec_agg, "recall": rec_agg,
        "color": "#D85A30", "description": "Maximum de détection"
    },
}


# ── Calcul des métriques complètes par scénario ───────────────────────────────

print(f"\n{'='*55}")
print(f"  Résultats par scénario — Val 2024-25 (26 304 matchs)")
print(f"{'='*55}")
print(f"\n  Rappel : {y_val.sum()} vraies blessures dans la période\n")

for name, sc in scenarios.items():
    y_pred      = (y_proba >= sc["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    f1          = f1_score(y_val, y_pred)

    sc["y_pred"] = y_pred
    sc["tp"]     = int(tp)
    sc["fp"]     = int(fp)
    sc["fn"]     = int(fn)
    sc["tn"]     = int(tn)
    sc["f1"]     = f1
    sc["n_alerts"]= int(tp + fp)

    print(f"  ── {name} ({sc['description']}) ──")
    print(f"  Seuil      : {sc['threshold']:.3f}")
    print(f"  Precision  : {sc['precision']:.3f}  "
          f"({sc['tp']} blessures détectées sur {sc['n_alerts']} alertes)")
    print(f"  Recall     : {sc['recall']:.3f}  "
          f"({sc['tp']} détectées sur {y_val.sum()} blessures réelles)")
    print(f"  F1         : {f1:.3f}")
    print(f"  Vrais pos  : {sc['tp']:,}  Faux pos : {sc['fp']:,}  "
          f"Faux nég : {sc['fn']:,}")
    print(f"  Ratio fausses alarmes : "
          f"{sc['fp']/max(sc['tp'],1):.1f} fausses alarmes par vraie blessure")
    print()


# ── Figure principale : courbe PR + 3 scénarios ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Optimisation du seuil — CatBoost réduit (Val 2024-25)",
             fontsize=13, fontweight="bold")

# Courbe PR avec les 3 seuils
axes[0].plot(recall_curve, precision_curve, color="steelblue",
             linewidth=2, label=f"PR Curve (AUC={pr_auc:.4f})")
axes[0].axhline(y=y_val.mean(), color="gray", linestyle="--",
                linewidth=1, label=f"Baseline ({y_val.mean():.2f})")

for name, sc in scenarios.items():
    axes[0].scatter(sc["recall"], sc["precision"], s=120,
                    color=sc["color"], zorder=5,
                    label=f"{name} (seuil={sc['threshold']:.2f})")
    axes[0].annotate(name, (sc["recall"], sc["precision"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9,
                     color=sc["color"], fontweight="bold")

axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Courbe PR — 3 scénarios")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(True, alpha=0.3)


# Precision / Recall / F1 vs Seuil
axes[1].plot(thresholds, precision_curve[:-1], color="#BA7517",
             label="Precision", linewidth=2)
axes[1].plot(thresholds, recall_curve[:-1], color="#D85A30",
             label="Recall", linewidth=2)
axes[1].plot(thresholds, f1_curve[:-1], color="#534AB7",
             label="F1", linewidth=2, linestyle="--")

for name, sc in scenarios.items():
    axes[1].axvline(x=sc["threshold"], color=sc["color"],
                    linestyle=":", linewidth=1.5, label=f"Seuil {name}")

axes[1].set_xlabel("Seuil de décision")
axes[1].set_ylabel("Score")
axes[1].set_title("Precision / Recall / F1 vs Seuil")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)


# Barplot comparatif des 3 scénarios
scenario_names = list(scenarios.keys())
metrics        = ["precision", "recall", "f1"]
labels         = ["Precision", "Recall", "F1"]
colors_bar     = [sc["color"] for sc in scenarios.values()]
x              = np.arange(len(metrics))
width          = 0.25

for i, (name, sc) in enumerate(scenarios.items()):
    vals = [sc["precision"], sc["recall"], sc["f1"]]
    bars = axes[2].bar(x + i * width, vals, width,
                       label=name, color=sc["color"], alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

axes[2].set_xlabel("Métrique")
axes[2].set_ylabel("Score")
axes[2].set_title("Comparaison des 3 scénarios")
axes[2].set_xticks(x + width)
axes[2].set_xticklabels(labels)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis="y")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures_ML/step7_threshold_optimization.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step7_threshold_optimization.png")


# ── Matrices de confusion ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Matrices de confusion — 3 scénarios (Val 2024-25)",
             fontsize=13, fontweight="bold")

for ax, (name, sc) in zip(axes, scenarios.items()):
    cm   = confusion_matrix(y_val, sc["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non blessé", "Blessé"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\n(seuil={sc['threshold']:.3f})", fontweight="bold")

plt.tight_layout()
plt.savefig("figures_ML/step7_confusion_matrices.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  → figures_ML/step7_confusion_matrices.png")


# ── Tableau récapitulatif actionnable ─────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Tableau récapitulatif — décision actionnable")
print(f"{'='*55}")
print(f"\n  {'Scénario':<15} {'Seuil':>7} {'Precision':>10} {'Recall':>8} "
      f"{'Alertes':>8} {'Détectées':>10} {'Manquées':>9} {'FA/Blessure':>12}")
print(f"  {'-'*85}")

for name, sc in scenarios.items():
    print(f"  {name:<15} {sc['threshold']:>7.3f} {sc['precision']:>10.3f} "
          f"{sc['recall']:>8.3f} {sc['n_alerts']:>8,} {sc['tp']:>10,} "
          f"{sc['fn']:>9,} {sc['fp']/max(sc['tp'],1):>12.1f}")

print(f"\n  Blessures totales dans la période : {y_val.sum():,}")
print(f"  Matchs totaux                      : {len(y_val):,}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

results_step7 = {
    "scenarios"       : scenarios,
    "thresholds"      : thresholds,
    "precision_curve" : precision_curve,
    "recall_curve"    : recall_curve,
    "f1_curve"        : f1_curve,
    "pr_auc"          : pr_auc,
    "y_proba"         : y_proba,
}

with open("Models/step7_results.pkl", "wb") as f:
    pickle.dump(results_step7, f)

print(f"\n  Résultats sauvegardés : Models/step7_results.pkl")
print(f"\n{'='*55}")
print(f"  Étape 7 terminée — prêt pour le test final")
print(f"{'='*55}")