"""
Étape 4 — XGBoost + LightGBM + CatBoost
Prédiction de blessures NBA

But : tester les 3 grands modèles de boosting sur notre problème.
      Généralement plus performants que RF sur données tabulaires.

Métriques : PR-AUC, Recall, Precision, F1, ROC-AUC
Évaluation : sur val 2024-25 uniquement
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    roc_curve, recall_score, precision_score, f1_score
)

# ── Chargement ────────────────────────────────────────────────────────────────

print("=" * 55)
print("  Étape 4 — XGBoost + LightGBM + CatBoost")
print("=" * 55)

with open("Models/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train      = data["X_train_tree"]
X_val        = data["X_val_tree"]
y_train      = data["y_train"]
y_val        = data["y_val"]
feature_cols = data["feature_cols"]

# Ratio déséquilibre pour XGBoost et CatBoost
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nRatio négatifs/positifs (train) : {neg_pos_ratio:.2f}")
print(f"Train : {X_train.shape} | Val : {X_val.shape}")

Path("Models").mkdir(exist_ok=True)
Path("figures_ML").mkdir(exist_ok=True)


# ── Utilitaire d'évaluation ───────────────────────────────────────────────────

def evaluate_model(name, y_val, y_pred_proba, prev_results=None):
    """Calcule toutes les métriques et retourne un dict de résultats."""
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc  = average_precision_score(y_val, y_pred_proba)

    # Seuil optimal (max F1)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores   = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-8)
    best_idx    = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    y_pred_opt  = (y_pred_proba >= best_thresh).astype(int)
    rec_opt     = recall_score(y_val, y_pred_opt)
    prec_opt    = precision_score(y_val, y_pred_opt)
    f1_opt      = f1_score(y_val, y_pred_opt)

    print(f"\n  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}  ← métrique principale")
    print(f"\n  Classification report (seuil 0.5) :")
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_val, y_pred_05,
          target_names=["Non blessé", "Blessé"]))
    print(f"  Seuil optimal (max F1) : {best_thresh:.3f}")
    print(f"    Recall    : {rec_opt:.4f}")
    print(f"    Precision : {prec_opt:.4f}")
    print(f"    F1        : {f1_opt:.4f}")

    return {
        "name"        : name,
        "roc_auc"     : roc_auc,
        "pr_auc"      : pr_auc,
        "best_thresh" : best_thresh,
        "recall"      : rec_opt,
        "precision"   : prec_opt,
        "f1"          : f1_opt,
        "proba"       : y_pred_proba,
        "prec_curve"  : prec_curve,
        "rec_curve"   : rec_curve,
        "thresholds"  : thresholds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  XGBoost")
print("="*55)

try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,  # gère le déséquilibre
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print("  Entraînement OK ✓")
    y_proba_xgb = xgb.predict_proba(X_val)[:, 1]
    results_xgb = evaluate_model("XGBoost", y_val, y_proba_xgb)
    results_xgb["model"] = xgb

    with open("Models/step4_xgb.pkl", "wb") as f:
        pickle.dump(results_xgb, f)
    print("  Modèle sauvegardé : Models/step4_xgb.pkl")

except ImportError:
    print("  XGBoost non installé — pip install xgboost")
    results_xgb = None


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  LightGBM")
print("="*55)

try:
    import lightgbm as lgb

    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,       # gère le déséquilibre nativement
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    print("  Entraînement OK ✓")
    y_proba_lgbm = lgbm.predict_proba(X_val)[:, 1]
    results_lgbm = evaluate_model("LightGBM", y_val, y_proba_lgbm)
    results_lgbm["model"] = lgbm

    with open("Models/step4_lgbm.pkl", "wb") as f:
        pickle.dump(results_lgbm, f)
    print("  Modèle sauvegardé : Models/step4_lgbm.pkl")

except ImportError:
    print("  LightGBM non installé — pip install lightgbm")
    results_lgbm = None


# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  CatBoost")
print("="*55)

try:
    from catboost import CatBoostClassifier

    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",  # gère le déséquilibre
        eval_metric="PRAUC",
        random_seed=42,
        verbose=0,
    )
    cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    print("  Entraînement OK ✓")
    y_proba_cat = cat.predict_proba(X_val)[:, 1]
    results_cat = evaluate_model("CatBoost", y_val, y_proba_cat)
    results_cat["model"] = cat

    with open("Models/step4_cat.pkl", "wb") as f:
        pickle.dump(results_cat, f)
    print("  Modèle sauvegardé : Models/step4_cat.pkl")

except ImportError:
    print("  CatBoost non installé — pip install catboost")
    results_cat = None


# ═══════════════════════════════════════════════════════════════════════════════
# Comparaison complète
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print(f"  Comparaison complète — tous les modèles")
print(f"{'='*55}")

all_results = [
    {"name": "Logistic Reg", "pr_auc": 0.3410, "roc_auc": 0.6790,
     "recall": 0.5872, "precision": 0.2781, "f1": 0.3774},
    {"name": "Random Forest", "pr_auc": 0.3458, "roc_auc": 0.6941,
     "recall": 0.5917, "precision": 0.2892, "f1": 0.3885},
]
for r in [results_xgb, results_lgbm, results_cat]:
    if r:
        all_results.append({
            "name"     : r["name"],
            "pr_auc"   : r["pr_auc"],
            "roc_auc"  : r["roc_auc"],
            "recall"   : r["recall"],
            "precision": r["precision"],
            "f1"       : r["f1"],
        })

df_results = pd.DataFrame(all_results).sort_values("pr_auc", ascending=False)

print(f"\n  {'Modèle':<15} {'PR-AUC':>8} {'ROC-AUC':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}")
print(f"  {'-'*60}")
for _, row in df_results.iterrows():
    print(f"  {row['name']:<15} {row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f} "
          f"{row['recall']:>8.4f} {row['precision']:>10.4f} {row['f1']:>8.4f}")


# ── Figure comparaison courbes PR ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Comparaison des modèles — Courbes PR et ROC", fontsize=13, fontweight="bold")

colors = ["#1D9E75", "#534AB7", "#BA7517", "#D85A30", "#378ADD"]
models_to_plot = []
if results_xgb:  models_to_plot.append(results_xgb)
if results_lgbm: models_to_plot.append(results_lgbm)
if results_cat:  models_to_plot.append(results_cat)

for i, r in enumerate(models_to_plot):
    axes[0].plot(r["rec_curve"], r["prec_curve"],
                 label=f"{r['name']} (AUC={r['pr_auc']:.4f})",
                 color=colors[i], linewidth=2)
    fpr, tpr, _ = roc_curve(y_val, r["proba"])
    axes[1].plot(fpr, tpr,
                 label=f"{r['name']} (AUC={r['roc_auc']:.4f})",
                 color=colors[i], linewidth=2)

axes[0].axhline(y=y_val.mean(), color="gray", linestyle="--",
                label=f"Baseline ({y_val.mean():.2f})")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Courbes PR")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Courbes ROC")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_ML/step4_boosting_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → figures_ML/step4_boosting_comparison.png")

# ── Figure barplot comparaison ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparaison des modèles — PR-AUC et ROC-AUC", fontsize=13, fontweight="bold")

palette = ["#378ADD", "#1D9E75", "#BA7517", "#534AB7", "#D85A30"]
names   = df_results["name"].tolist()

axes[0].bar(names, df_results["pr_auc"],   color=palette[:len(names)])
axes[0].set_title("PR-AUC (métrique principale)")
axes[0].set_ylabel("PR-AUC")
axes[0].set_ylim(0.30, max(df_results["pr_auc"]) + 0.02)
for i, v in enumerate(df_results["pr_auc"]):
    axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

axes[1].bar(names, df_results["roc_auc"], color=palette[:len(names)])
axes[1].set_title("ROC-AUC")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_ylim(0.65, max(df_results["roc_auc"]) + 0.02)
for i, v in enumerate(df_results["roc_auc"]):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("figures_ML/step4_models_barplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → figures_ML/step4_models_barplot.png")