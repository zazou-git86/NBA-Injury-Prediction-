"""
Étape 1 — Preprocessing pipeline
Prédiction de blessures NBA

But : préparer X_train, X_val proprement pour la modélisation
      sans leakage entre train et val/test.

Output : affichage des stats + validation du pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ── Chargement ────────────────────────────────────────────────────────────────

DATA_PATH = "../Dataset_ML/nba_features_ml.parquet"

df = pd.read_parquet(DATA_PATH)
print("=" * 55)
print("  Étape 1 — Preprocessing pipeline")
print("=" * 55)
print(f"\nDataset : {len(df):,} lignes | {len(df.columns)} colonnes")


# ── Définition des colonnes ───────────────────────────────────────────────────

ID_COLS     = ["PLAYER_ID", "PLAYER_NAME", "name_key", "GAME_DATE",
               "MATCHUP", "SEASON", "split"]
TARGET_COL  = "injury_next_10d"

# Colonnes à exclure du modèle
EXCLUDE     = ID_COLS + [TARGET_COL]

# Features ML
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

# Features avec NaN intentionnels → imputation médiane
NAN_FEATURES = ["acwr", "days_since_last_inj", "min_avg_vs_season"]

print(f"\nFeatures ML totales : {len(FEATURE_COLS)}")
print(f"Features avec NaN   : {NAN_FEATURES}")


# ── Split train / val / test ──────────────────────────────────────────────────

train = df[df["split"] == "train"].copy()
val   = df[df["split"] == "val"].copy()
test  = df[df["split"] == "test"].copy()

X_train = train[FEATURE_COLS]
y_train = train[TARGET_COL]

X_val   = val[FEATURE_COLS]
y_val   = val[TARGET_COL]

X_test  = test[FEATURE_COLS]
y_test  = test[TARGET_COL]

print(f"\nSplit :")
print(f"  Train : {len(X_train):,} lignes | positifs : {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Val   : {len(X_val):,} lignes | positifs : {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
print(f"  Test  : {len(X_test):,} lignes | positifs : {y_test.sum():,} ({y_test.mean()*100:.1f}%)")


# ── Pipeline pour modèles à base d'arbres (RF, XGBoost, LightGBM) ────────────
# Imputation médiane uniquement — pas de scaling

pipeline_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# ── Pipeline pour Logistic Regression ────────────────────────────────────────
# Imputation médiane + StandardScaler

pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])


# ── Fit sur train uniquement — transform sur train + val ─────────────────────

print("\nFit des pipelines sur train...")

# Pipeline arbres
X_train_tree = pipeline_tree.fit_transform(X_train)
X_val_tree   = pipeline_tree.transform(X_val)
X_test_tree  = pipeline_tree.transform(X_test)

# Pipeline LR
X_train_lr   = pipeline_lr.fit_transform(X_train)
X_val_lr     = pipeline_lr.transform(X_val)
X_test_lr    = pipeline_lr.transform(X_test)

print("  OK — médiane calculée sur train uniquement ✓")
print("  OK — transform appliqué sur val et test sans recalcul ✓")


# ── Vérification anti-leakage ─────────────────────────────────────────────────

# Les médianes du train
imputer_medians = pipeline_tree.named_steps["imputer"].statistics_
median_dict     = dict(zip(FEATURE_COLS, imputer_medians))

print(f"\nMédianes calculées sur train (features avec NaN) :")
for feat in NAN_FEATURES:
    print(f"  {feat:<30} : {median_dict[feat]:.3f}")

# Vérification NaN après imputation
nan_after_train = np.isnan(X_train_tree).sum()
nan_after_val   = np.isnan(X_val_tree).sum()
print(f"\nNaN après imputation :")
print(f"  Train : {nan_after_train} {'✓' if nan_after_train == 0 else '✗ PROBLÈME'}")
print(f"  Val   : {nan_after_val}   {'✓' if nan_after_val == 0 else '✗ PROBLÈME'}")


# ── Stats post-preprocessing ──────────────────────────────────────────────────

print(f"\nDimensions finales :")
print(f"  X_train_tree : {X_train_tree.shape}")
print(f"  X_val_tree   : {X_val_tree.shape}")
print(f"  X_test_tree  : {X_test_tree.shape}")
print(f"  X_train_lr   : {X_train_lr.shape}")

# Scaling check pour LR
scaler      = pipeline_lr.named_steps["scaler"]
means_lr    = scaler.mean_
stds_lr     = scaler.scale_
print(f"\nScaling LR (calculé sur train) :")
print(f"  Moyenne min/max : {means_lr.min():.2f} / {means_lr.max():.2f}")
print(f"  Std min/max     : {stds_lr.min():.2f} / {stds_lr.max():.2f}")


# ── Sauvegarde des arrays pour réutilisation ──────────────────────────────────

import pickle

save_dict = {
    "X_train_tree" : X_train_tree,
    "X_val_tree"   : X_val_tree,
    "X_test_tree"  : X_test_tree,
    "X_train_lr"   : X_train_lr,
    "X_val_lr"     : X_val_lr,
    "X_test_lr"    : X_test_lr,
    "y_train"      : y_train.values,
    "y_val"        : y_val.values,
    "y_test"       : y_test.values,
    "feature_cols" : FEATURE_COLS,
    "pipeline_tree": pipeline_tree,
    "pipeline_lr"  : pipeline_lr,
}

Path("Models").mkdir(exist_ok=True)
with open("Models/preprocessed_data.pkl", "wb") as f:
    pickle.dump(save_dict, f)

print(f"\nDonnées preprocessées sauvegardées : Models/preprocessed_data.pkl")
print(f"\n{'='*55}")
print(f"  Étape 1 terminée — prêt pour la modélisation")
print(f"{'='*55}")