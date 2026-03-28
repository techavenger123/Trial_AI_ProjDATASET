"""
LightGBM Dense Combination Experiment
======================================
Search space:
  - learning_rate : logspace + linspace merged (dense, ~35 unique values)
  - n_estimators  : range(25, 401, 25)  → 16 values
  - split_ratios  : 4 combinations
  Total           : ~35 × 16 × 4 = ~2,240 runs

Tracks per combination (Train / Val / Test):
  AUC, Log-Loss, Accuracy, Precision, Recall, F1
"""

import os
import time
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# ──────────────────────────────────────────────
# SEARCH SPACE  (matches your spec)
# ──────────────────────────────────────────────
lr_log    = np.logspace(-3, -1, 20)
lr_linear = np.linspace(0.005, 0.1, 20)
LEARNING_RATES = np.unique(np.round(np.concatenate([lr_log, lr_linear]), 4)).tolist()

N_ESTIMATORS = list(range(15, 403, 5))   # [25, 50, 75, ..., 400]

# (train_ratio, val_ratio, test_ratio)
SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),
    (0.70, 0.15, 0.15),
    (0.70, 0.20, 0.10),
    (0.80, 0.10, 0.10),
    (0.60, 0.10, 0.30),
    (0.50, 0.20, 0.30),
    (0.50, 0.20, 0.30)
]

print(f"Learning rates  : {len(LEARNING_RATES)} values → {LEARNING_RATES}")
print(f"n_estimators    : {len(N_ESTIMATORS)} values → {N_ESTIMATORS}")
print(f"Split ratios    : {len(SPLIT_RATIOS)} combinations")
print(f"TOTAL RUNS      : {len(LEARNING_RATES) * len(N_ESTIMATORS) * len(SPLIT_RATIOS)}")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH    = "industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv"
DROP_COLS    = ["location"]
TARGET       = "faulty"
CAT_COLS     = ["equipment"]
NUM_COLS     = ["temperature", "pressure", "vibration", "humidity"]
RANDOM_STATE = 42
THRESHOLD    = 0.5

FIXED_PARAMS = dict(
    max_depth=8,
    num_leaves=50,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    verbose=-1,
)

RESULTS_DIR = "results/new2"
PLOTS_DIR   = "plots/dense_combo3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

METRICS = ["auc", "logloss", "accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "auc"       : "AUC-ROC",
    "logloss"   : "Log-Loss (↓ better)",
    "accuracy"  : "Accuracy",
    "precision" : "Precision",
    "recall"    : "Recall",
    "f1"        : "F1-Score",
}
SPLIT_COLORS = ["#4C9BE8", "#F4845F", "#6BCB77", "#C77DFF"]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])


def compute_metrics(y_true, y_prob, prefix: str) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        f"{prefix}_auc"       : roc_auc_score(y_true, y_prob),
        f"{prefix}_logloss"   : log_loss(y_true, y_prob),
        f"{prefix}_accuracy"  : accuracy_score(y_true, y_pred),
        f"{prefix}_precision" : precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall"    : recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1"        : f1_score(y_true, y_pred, zero_division=0),
    }


def run_single(X, y, n_est, lr, split_ratio) -> dict:
    train_r, val_r, test_r = split_ratio

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_r, stratify=y, random_state=RANDOM_STATE
    )
    val_relative = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    pre = make_preprocessor()
    clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, **FIXED_PARAMS)
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    rec = {
        "n_estimators" : n_est,
        "learning_rate": lr,
        "train_ratio"  : train_r,
        "val_ratio"    : val_r,
        "test_ratio"   : test_r,
    }
    for name, (Xs, ys) in [("train", (X_train, y_train)),
                             ("val",   (X_val,   y_val)),
                             ("test",  (X_test,  y_test))]:
        rec.update(compute_metrics(ys, pipeline.predict_proba(Xs)[:, 1], name))
    return rec


# ──────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────
def run_all(X, y) -> pd.DataFrame:
    combos = list(itertools.product(N_ESTIMATORS, LEARNING_RATES, SPLIT_RATIOS))
    print(f"\n🔬 Starting {len(combos)} runs …\n")

    records = []
    SAVE_EVERY = 100
    checkpoint = f"{RESULTS_DIR}/dense_checkpoint.csv"

    # Resume if interrupted
    done_keys = set()
    if os.path.exists(checkpoint):
        done_df = pd.read_csv(checkpoint)
        records = done_df.to_dict("records")
        done_keys = set(zip(done_df.n_estimators, done_df.learning_rate,
                            done_df.train_ratio, done_df.val_ratio))
        print(f"   ↩️  Resuming: {len(records)} already done.")

    it = tqdm(combos, desc="Experiments") if TQDM else combos
    for n_est, lr, split_ratio in it:
        key = (n_est, lr, split_ratio[0], split_ratio[1])
        if key in done_keys:
            continue
        records.append(run_single(X, y, n_est, lr, split_ratio))
        if len(records) % SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(checkpoint, index=False)

    df = pd.DataFrame(records)
    df["split_label"]  = df.apply(
        lambda r: f"{int(r.train_ratio*100)}/{int(r.val_ratio*100)}/{int(r.test_ratio*100)}", axis=1
    )
    df["overfit_auc"]  = df["train_auc"] - df["val_auc"]
    df = df.sort_values("val_f1", ascending=False).reset_index(drop=True)

    df.to_csv(f"{RESULTS_DIR}/dense_results.csv", index=False)
    try:
        df.to_excel(f"{RESULTS_DIR}/dense_results.xlsx", index=False)
    except Exception:
        pass
    print(f"\n✅ Saved {RESULTS_DIR}/dense_results.csv  ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────
# PLOT 1 — Dense heatmap: val_<metric> over lr × n
# ──────────────────────────────────────────────
def plot_val_heatmaps(df: pd.DataFrame):
    """One heatmap per metric, averaged over split ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle("Validation Metrics  ·  learning_rate × n_estimators\n(averaged over split ratios)",
                 fontsize=14, fontweight="bold")

    cmaps = {"auc": "YlGn", "logloss": "YlOrRd_r", "accuracy": "YlGn",
             "precision": "Blues", "recall": "Oranges", "f1": "Greens"}

    for ax, metric in zip(axes.flat, METRICS):
        pivot = df.pivot_table(
            index="learning_rate", columns="n_estimators",
            values=f"val_{metric}", aggfunc="mean"
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmaps[metric])
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("n_estimators", fontsize=9)
        ax.set_ylabel("learning_rate", fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(np.round(pivot.index, 4), fontsize=6)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_heatmaps.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# PLOT 2 — Metrics vs n_estimators per split ratio
# ──────────────────────────────────────────────
def plot_metrics_vs_n_by_split(df: pd.DataFrame):
    splits = df["split_label"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Val Metrics vs n_estimators  ·  per split ratio\n(averaged over learning_rate)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        for color, split in zip(SPLIT_COLORS, splits):
            sub = df[df["split_label"] == split].groupby("n_estimators")[f"val_{metric}"].mean()
            ax.plot(sub.index, sub.values, marker="o", markersize=3,
                    label=split, color=color, linewidth=1.6)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("n_estimators")
        ax.legend(fontsize=7, title="split", title_fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_metrics_vs_n_by_split.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# PLOT 3 — Metrics vs learning_rate per split ratio
# ──────────────────────────────────────────────
def plot_metrics_vs_lr_by_split(df: pd.DataFrame):
    splits = df["split_label"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Val Metrics vs Learning Rate  ·  per split ratio\n(averaged over n_estimators)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        for color, split in zip(SPLIT_COLORS, splits):
            sub = df[df["split_label"] == split].groupby("learning_rate")[f"val_{metric}"].mean()
            ax.plot(sub.index, sub.values, marker="s", markersize=3,
                    label=split, color=color, linewidth=1.6)
        ax.set_xscale("log")
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("learning_rate (log scale)")
        ax.legend(fontsize=7, title="split", title_fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_metrics_vs_lr_by_split.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# PLOT 4 — Train vs Val for each metric (best split only)
# ──────────────────────────────────────────────
def plot_train_vs_val(df: pd.DataFrame):
    best_split = df.groupby("split_label")["val_f1"].mean().idxmax()
    sub = df[df["split_label"] == best_split].groupby("n_estimators")[
        [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS]
    ].mean()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Train vs Val  ·  split={best_split}  (averaged over lr)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        ax.plot(sub.index, sub[f"train_{metric}"], color="#4C9BE8",
                marker="o", markersize=3, linewidth=1.8, label="Train")
        ax.plot(sub.index, sub[f"val_{metric}"],   color="#F4845F",
                marker="s", markersize=3, linewidth=1.8, label="Val")
        ax.fill_between(sub.index, sub[f"train_{metric}"], sub[f"val_{metric}"],
                        alpha=0.12, color="#888")
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("n_estimators")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/train_vs_val_best_split.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# PLOT 5 — Overfit heatmap (dense)
# ──────────────────────────────────────────────
def plot_overfit_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="learning_rate", columns="n_estimators",
        values="overfit_auc", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=np.percentile(pivot.values, 95))
    plt.colorbar(im, ax=ax, label="Train AUC − Val AUC")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(np.round(pivot.index, 4), fontsize=7)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    ax.set_title("Overfit Heatmap: Train − Val AUC  (red = high overfit, green = low)")
    plt.tight_layout()
    out = f"{PLOTS_DIR}/overfit_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# PLOT 6 — Top-30 configs: val F1 vs val AUC
# ──────────────────────────────────────────────
def plot_top_scatter(df: pd.DataFrame):
    top = df.head(30)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(top["val_f1"], top["val_auc"],
                    c=np.log10(top["learning_rate"]),
                    cmap="plasma", s=100, edgecolors="white", linewidth=0.6, zorder=3)
    plt.colorbar(sc, ax=ax, label="log₁₀(learning_rate)")
    for _, row in top.iterrows():
        ax.annotate(f"n={int(row.n_estimators)}", (row.val_f1, row.val_auc),
                    textcoords="offset points", xytext=(3, 2), fontsize=6, alpha=0.75)
    ax.set_xlabel("Val F1-Score")
    ax.set_ylabel("Val AUC")
    ax.set_title("Top 30 Configs — Val F1 vs Val AUC  (color = log₁₀ lr)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/top30_scatter.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ {out}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("  LightGBM Dense Combination Experiment")
    print("=" * 60)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")
    X = df_raw.drop(columns=[TARGET])
    y = df_raw[TARGET]
    print(f"✅ Data: {df_raw.shape}  |  Classes: {y.value_counts().to_dict()}")

    results = run_all(X, y)

    # ── Summary ──────────────────────────────────
    show = ["n_estimators", "learning_rate", "split_label",
            "train_auc", "val_auc", "test_auc",
            "train_f1",  "val_f1",  "test_f1",
            "val_precision", "val_recall", "val_logloss", "overfit_auc"]
    print("\n🏆 TOP 10 CONFIGS (by Val F1):")
    print(results[show].head(10).to_string(index=False))

    # ── Plots ────────────────────────────────────
    plot_val_heatmaps(results)
    plot_metrics_vs_n_by_split(results)
    plot_metrics_vs_lr_by_split(results)
    plot_train_vs_val(results)
    plot_overfit_heatmap(results)
    plot_top_scatter(results)

    elapsed = (time.time() - t0) / 60
    print(f"\n⏱  Done in {elapsed:.1f} min")
    print(f"📁 Results : {RESULTS_DIR}/dense_results.csv / .xlsx")
    print(f"📊 Plots   : {PLOTS_DIR}/")


if __name__ == "__main__":
    main()