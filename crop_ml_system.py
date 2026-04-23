"""
=============================================================
   CROP RECOMMENDATION SYSTEM — Complete ML Pipeline
   Author  : ML Engineer
   Purpose : Predict best crop based on soil & climate data
=============================================================

APPROACH:
---------
We treat this as a multi-class classification problem.
Given 7 numerical features (N, P, K, temperature, humidity,
pH, rainfall), we predict 1 of 22 crop classes.

Pipeline:
  1. Generate/Load dataset
  2. EDA (distributions, correlations)
  3. Preprocessing (scaling)
  4. Train 4 models, compare accuracy
  5. Best model saved with pickle
  6. Interactive prediction function
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# ─────────────────────────────────────────────
# 1.  DATASET — Synthetic but realistic data
#     (mirrors the public Kaggle "Crop Recommendation" CSV)
# ─────────────────────────────────────────────

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic crop recommendation dataset.

    Each crop has its own typical ranges for the 7 features.
    We sample 100 rows per crop → 2200 total rows.

    Dataset columns:
        N          – Nitrogen content in soil  (kg/ha)
        P          – Phosphorus content        (kg/ha)
        K          – Potassium content         (kg/ha)
        temperature– Air temperature           (°C)
        humidity   – Relative humidity         (%)
        ph         – Soil pH
        rainfall   – Annual rainfall           (mm)
        label      – Crop name (target)
    """
    rng = np.random.default_rng(seed)

    # ── Crop profiles: [N_mean, N_std, P_mean, P_std, K_mean, K_std,
    #                    temp_mean, temp_std, hum_mean, hum_std,
    #                    ph_mean, ph_std, rain_mean, rain_std]
    crops = {
        "rice":        [80, 10, 45, 8,  40, 8,  25, 2,  82, 5,  6.5, 0.4, 200, 30],
        "wheat":       [90, 12, 42, 9,  43, 9,  20, 3,  65, 8,  6.5, 0.5, 75,  15],
        "maize":       [77, 10, 48, 10, 20, 5,  22, 3,  65, 8,  5.9, 0.5, 67,  15],
        "chickpea":    [40, 8,  67, 10, 79, 12, 18, 3,  14, 4,  7.0, 0.4, 72,  15],
        "kidneybeans": [20, 5,  67, 10, 19, 5,  19, 3,  22, 5,  5.8, 0.4, 105, 20],
        "pigeonpeas":  [20, 5,  67, 10, 19, 5,  28, 3,  48, 8,  5.8, 0.5, 149, 25],
        "mothbeans":   [20, 5,  45, 8,  20, 5,  28, 3,  53, 8,  6.9, 0.4, 51,  12],
        "mungbean":    [20, 5,  47, 8,  19, 5,  28, 3,  85, 6,  6.8, 0.4, 48,  12],
        "blackgram":   [40, 8,  67, 10, 19, 5,  30, 3,  65, 8,  7.0, 0.4, 68,  15],
        "lentil":      [18, 4,  68, 10, 19, 5,  24, 3,  64, 8,  6.8, 0.4, 46,  12],
        "pomegranate": [18, 4,  18, 5,  40, 8,  22, 3,  90, 5,  6.4, 0.5, 107, 20],
        "banana":      [100,12, 82, 12, 50, 10, 27, 3,  80, 6,  6.0, 0.5, 104, 20],
        "mango":       [20, 5,  27, 6,  30, 6,  31, 3,  50, 8,  6.0, 0.5, 95,  20],
        "grapes":      [23, 5,  133,15, 200,20, 24, 3,  81, 6,  6.1, 0.5, 70,  15],
        "watermelon":  [99, 12, 17, 5,  50, 10, 25, 3,  85, 6,  6.5, 0.4, 51,  12],
        "muskmelon":   [100,12, 17, 5,  50, 10, 28, 3,  92, 5,  6.4, 0.4, 25,  8 ],
        "apple":       [21, 5,  134,15, 199,20, 22, 3,  92, 5,  5.9, 0.5, 112, 20],
        "orange":      [20, 5,  16, 5,  10, 4,  23, 3,  92, 5,  6.9, 0.4, 110, 20],
        "papaya":      [50, 8,  59, 10, 50, 10, 34, 3,  92, 5,  6.7, 0.4, 143, 25],
        "coconut":     [22, 5,  16, 5,  30, 6,  27, 3,  94, 4,  5.8, 0.4, 176, 28],
        "cotton":      [118,15, 46, 9,  43, 9,  24, 3,  79, 6,  6.9, 0.4, 78,  15],
        "jute":        [78, 10, 46, 9,  40, 8,  25, 3,  80, 6,  6.6, 0.4, 175, 28],
    }

    rows = []
    for crop, p in crops.items():
        N   = rng.normal(p[0],  p[1],  100).clip(0, 200)
        P   = rng.normal(p[2],  p[3],  100).clip(5, 145)
        K   = rng.normal(p[4],  p[5],  100).clip(5, 205)
        T   = rng.normal(p[6],  p[7],  100).clip(8,  43)
        H   = rng.normal(p[8],  p[9],  100).clip(14, 99)
        pH  = rng.normal(p[10], p[11], 100).clip(3.5, 9.5)
        R   = rng.normal(p[12], p[13], 100).clip(20, 300)
        for i in range(100):
            rows.append([N[i], P[i], K[i], T[i], H[i], pH[i], R[i], crop])

    df = pd.DataFrame(rows,
                      columns=["N", "P", "K", "temperature",
                                "humidity", "ph", "rainfall", "label"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2.  EDA  (saves charts to /tmp/)
# ─────────────────────────────────────────────

def perform_eda(df: pd.DataFrame, out_dir: str = "/tmp/"):
    """Exploratory Data Analysis — prints stats and saves figures."""

    print("\n" + "="*55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*55)
    print(f"\nShape          : {df.shape}")
    print(f"Crops (classes): {df['label'].nunique()}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print("\n── First 5 rows ──")
    print(df.head())
    print("\n── Descriptive Statistics ──")
    print(df.describe().round(2))
    print("\n── Crop Distribution ──")
    print(df['label'].value_counts())

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # ── Chart 1: Feature distributions ──────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Feature Distributions by Crop", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    colors = plt.cm.Set2.colors

    for i, feat in enumerate(features):
        axes[i].hist(df[feat], bins=40, color=colors[i % len(colors)],
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{out_dir}01_feature_distributions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] {out_dir}01_feature_distributions.png")

    # ── Chart 2: Correlation heatmap ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{out_dir}02_correlation_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir}02_correlation_heatmap.png")

    # ── Chart 3: Crop count ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df["label"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=plt.cm.tab20.colors[:len(counts)], edgecolor="white")
    ax.set_title("Number of Samples per Crop", fontsize=13, fontweight="bold")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{out_dir}03_crop_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir}03_crop_distribution.png")


# ─────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Steps:
      a. Encode string labels to integers (LabelEncoder)
      b. Split features (X) and target (y)
      c. Train-test split 80/20
      d. Standard-scale features (mean=0, std=1)
    """
    print("\n" + "="*55)
    print("  PREPROCESSING")
    print("="*55)

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features].values
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size : {X_train.shape[0]} rows")
    print(f"Test  size : {X_test.shape[0]} rows")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit on train only!
    X_test_sc  = scaler.transform(X_test)

    print("Feature scaling applied (StandardScaler).")
    return X_train_sc, X_test_sc, y_train, y_test, le, scaler, features


# ─────────────────────────────────────────────
# 4.  MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    """Train 4 classifiers, return results dict."""

    models = {
        "Decision Tree"     : DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest"     : RandomForestClassifier(n_estimators=150, random_state=42),
        "KNN"               : KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    print("\n" + "="*55)
    print("  MODEL TRAINING & ACCURACY")
    print("="*55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "acc": acc, "y_pred": y_pred}
        print(f"  {name:<22} → Accuracy: {acc*100:.2f}%")

    return results


# ─────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────

def evaluate_best(results: dict, y_test, le, out_dir: str = "/tmp/"):
    """Print full evaluation for best model and save charts."""

    best_name = max(results, key=lambda k: results[k]["acc"])
    best = results[best_name]
    print(f"\n✅ Best Model: {best_name}  (Accuracy: {best['acc']*100:.2f}%)")

    print("\n── Classification Report ──")
    print(classification_report(y_test, best["y_pred"],
                                target_names=le.classes_))

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_test, best["y_pred"])
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.4, ax=ax, cbar=False,
                annot_kws={"size": 7})
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    fig.savefig(f"{out_dir}04_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] {out_dir}04_confusion_matrix.png")

    # ── Model comparison bar chart ───────────────────────────────
    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    colors_bar = ["#2ecc71" if n == best_name else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accs, color=colors_bar, edgecolor="white", height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{out_dir}05_model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir}05_model_comparison.png")

    return best_name, best["model"]


# ─────────────────────────────────────────────
# 6.  SAVE MODEL  (pickle)
# ─────────────────────────────────────────────

def save_model(model, scaler, le, features,
               path: str = "output/crop_model.pkl"):
    """Persist model, scaler, encoder, and feature list to disk."""
    bundle = {
        "model"   : model,
        "scaler"  : scaler,
        "encoder" : le,
        "features": features,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[Saved] Model bundle → {path}")


def load_model(path: str = "output/crop_model.pkl"):
    """Load the saved bundle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# 7.  PREDICTION FUNCTION  (interactive CLI)
# ─────────────────────────────────────────────

def predict_crop(model, scaler, le, features,
                 N=None, P=None, K=None,
                 temperature=None, humidity=None,
                 ph=None, rainfall=None):
    """
    Predict the best crop for given soil/climate conditions.

    Parameters can be passed directly OR entered interactively.
    Returns the crop name (string).
    """
    if any(v is None for v in [N, P, K, temperature, humidity, ph, rainfall]):
        print("\n── Enter Soil & Climate Values ──")
        N           = float(input("  Nitrogen   (N)           [kg/ha]: "))
        P           = float(input("  Phosphorus (P)           [kg/ha]: "))
        K           = float(input("  Potassium  (K)           [kg/ha]: "))
        temperature = float(input("  Temperature              [°C]   : "))
        humidity    = float(input("  Humidity                 [%]    : "))
        ph          = float(input("  Soil pH                         : "))
        rainfall    = float(input("  Annual Rainfall          [mm]   : "))

    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_sc = scaler.transform(sample)
    pred_enc  = model.predict(sample_sc)[0]
    crop_name = le.inverse_transform([pred_enc])[0]

    # Top-3 probabilities (if supported)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_sc)[0]
        top3_idx  = np.argsort(proba)[-3:][::-1]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = proba[top3_idx]
        print(f"\n  🌾 Recommended Crop : {crop_name.upper()}")
        print("  Top-3 Candidates   :")
        for c, p in zip(top3_crops, top3_probs):
            print(f"     {c:<15} {p*100:.1f}%")
    else:
        print(f"\n  🌾 Recommended Crop : {crop_name.upper()}")

    return crop_name


# ─────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE  (bonus chart)
# ─────────────────────────────────────────────

def plot_feature_importance(model, features, model_name, out_dir="/tmp/"):
    """Bar chart of Random Forest feature importances."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([features[i] for i in idx],
                  [imp[i] for i in idx],
                  color=plt.cm.viridis(np.linspace(0.2, 0.85, len(features))),
                  edgecolor="white")
    ax.set_title(f"Feature Importance — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, [imp[i] for i in idx]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{out_dir}06_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir}06_feature_importance.png")


# ─────────────────────────────────────────────
# MAIN  —  run the full pipeline
# ─────────────────────────────────────────────

def main():
    print("\n" + "█"*55)
    print("   CROP RECOMMENDATION SYSTEM — ML PIPELINE")
    print("█"*55)

    OUT = "output/"
    import os
    os.makedirs(OUT, exist_ok=True)

    # Step 1 – Dataset
    print("\n[Step 1] Generating dataset …")
    df = generate_dataset()

    # Step 2 – EDA
    perform_eda(df, out_dir=OUT)

    # Step 3 – Preprocessing
    X_tr, X_te, y_tr, y_te, le, scaler, features = preprocess(df)

    # Step 4 – Train models
    results = train_models(X_tr, X_te, y_tr, y_te)

    # Step 5 – Evaluate
    best_name, best_model = evaluate_best(results, y_te, le, out_dir=OUT)

    # Step 6 – Feature importance
    plot_feature_importance(best_model, features, best_name, out_dir=OUT)

    # Step 7 – Save model
    save_model(best_model, scaler, le, features)

    # Step 8 – Demo prediction (no interactive input in batch mode)
    print("\n" + "="*55)
    print("  SAMPLE PREDICTION")
    print("="*55)
    print("  Input: N=90, P=42, K=43, Temp=20°C,")
    print("         Humidity=65%, pH=6.5, Rainfall=75mm")
    predict_crop(best_model, scaler, le, features,
                 N=90, P=42, K=43,
                 temperature=20, humidity=65,
                 ph=6.5, rainfall=75)

    print("\n" + "█"*55)
    print("  PIPELINE COMPLETE ✅")
    print("█"*55 + "\n")
    return best_model, scaler, le, features


if __name__ == "__main__":
    main()
