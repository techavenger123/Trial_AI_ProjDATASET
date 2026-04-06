import os
import time
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


MAX_FEATURES = ["sqrt", "log2", 0.3, 0.5]

N_ESTIMATORS = list(range(10, 501, 1))

SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),
    (0.70, 0.15, 0.15),
    (0.70, 0.20, 0.10),
    (0.80, 0.10, 0.10),
    (0.50, 0.20, 0.30),
    (0.50, 0.30, 0.20)
]

print(f"max_features    : {len(MAX_FEATURES)} values -> {MAX_FEATURES}")
print(f"n_estimators    : {len(N_ESTIMATORS)} values -> {N_ESTIMATORS}")
print(f"Split ratios    : {len(SPLIT_RATIOS)} combinations")
print(f"TOTAL RUNS      : {len(MAX_FEATURES) * len(N_ESTIMATORS) * len(SPLIT_RATIOS)}")


DATA_PATH    = "synthetic_nim_parallel_10000.csv"
DROP_COLS    = []
TARGET       = "faulty"
CAT_COLS     = ["equipment"]
NUM_COLS     = ["temperature", "pressure", "vibration", "humidity"]
RANDOM_STATE = 42
THRESHOLD    = 0.5

FIXED_PARAMS = dict(
    max_depth         = None,
    min_samples_leaf  = 5,
    min_samples_split = 10,
    max_samples       = 0.8,
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    oob_score         = False,
)


RESULTS_DIR = "results/synthetic"
PLOTS_DIR   = "Synthetic1/synthetic_plot"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

METRICS = ["auc", "logloss", "accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "auc"       : "AUC-ROC",
    "logloss"   : "Log-Loss (down better)",
    "accuracy"  : "Accuracy",
    "precision" : "Precision",
    "recall"    : "Recall",
    "f1"        : "F1-Score",
}
SPLIT_COLORS = ["#4C9BE8", "#F4845F", "#6BCB77", "#C77DFF"]

ENGINEERED_COLS = [
    "temp_x_pressure", "temp_x_vibration", "press_x_vibration",
    "vibration_x_humid", "temp_over_pressure", "vibration_over_temp",
    "vibration_sq", "temperature_sq",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction and ratio features.
    These often matter more than hyperparameter tuning
    when base features have weak individual signals.
    """
    df = df.copy()
    df["temp_x_pressure"]    = df["temperature"] * df["pressure"]
    df["temp_x_vibration"]   = df["temperature"] * df["vibration"]
    df["press_x_vibration"]  = df["pressure"]    * df["vibration"]
    df["vibration_x_humid"]  = df["vibration"]   * df["humidity"]
    df["temp_over_pressure"]  = df["temperature"] / (df["pressure"]    + 1e-6)
    df["vibration_over_temp"] = df["vibration"]   / (df["temperature"].abs() + 1e-6)
    df["vibration_sq"]        = df["vibration"]   ** 2
    df["temperature_sq"]      = df["temperature"]  ** 2
    return df



def make_preprocessor():
    all_num = NUM_COLS + ENGINEERED_COLS
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), all_num),
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


def run_single(X, y, n_est, max_feat, split_ratio) -> dict:
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
    clf = RandomForestClassifier(
        n_estimators  = n_est,
        max_features  = max_feat,
        **FIXED_PARAMS
    )
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    rec = {
        "n_estimators" : n_est,
        "max_features" : str(max_feat),
        "train_ratio"  : train_r,
        "val_ratio"    : val_r,
        "test_ratio"   : test_r,
    }
    for name, (Xs, ys) in [("train", (X_train, y_train)),
                             ("val",   (X_val,   y_val)),
                             ("test",  (X_test,  y_test))]:
        rec.update(compute_metrics(ys, pipeline.predict_proba(Xs)[:, 1], name))
    return rec



def run_all(X, y) -> pd.DataFrame:
    combos = list(itertools.product(N_ESTIMATORS, MAX_FEATURES, SPLIT_RATIOS))
    print(f"\nStarting {len(combos)} runs ...\n")

    records   = []
    SAVE_EVERY = 20
    checkpoint = f"{RESULTS_DIR}/dense_checkpoint.csv"

    done_keys = set()
    if os.path.exists(checkpoint):
        done_df = pd.read_csv(checkpoint)
        if "max_features" not in done_df.columns:
            print("   WARNING: Stale checkpoint detected -- deleting and starting fresh.")
            os.remove(checkpoint)
        else:
            records   = done_df.to_dict("records")
            done_keys = set(zip(done_df.n_estimators, done_df.max_features,
                                done_df.train_ratio, done_df.val_ratio))
            print(f"   Resuming: {len(records)} already done.")

    it = tqdm(combos, desc="Experiments") if TQDM else combos
    for n_est, max_feat, split_ratio in it:
        key = (n_est, str(max_feat), split_ratio[0], split_ratio[1])
        if key in done_keys:
            continue
        records.append(run_single(X, y, n_est, max_feat, split_ratio))
        if len(records) % SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(checkpoint, index=False)

    df = pd.DataFrame(records)
    df["split_label"] = df.apply(
        lambda r: f"{int(r.train_ratio*100)}/{int(r.val_ratio*100)}/{int(r.test_ratio*100)}", axis=1
    )
    df["overfit_auc"] = df["train_auc"] - df["val_auc"]
    df = df.sort_values("val_f1", ascending=False).reset_index(drop=True)

    df.to_csv(f"{RESULTS_DIR}/dense_results.csv", index=False)
    try:
        df.to_excel(f"{RESULTS_DIR}/dense_results.xlsx", index=False)
    except Exception:
        pass
    print(f"\nSaved {RESULTS_DIR}/dense_results.csv  ({len(df)} rows)")
    return df


def plot_val_heatmaps(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle("Validation Metrics  .  max_features x n_estimators\n(averaged over split ratios)",
                 fontsize=14, fontweight="bold")

    cmaps = {"auc": "YlGn", "logloss": "YlOrRd_r", "accuracy": "YlGn",
             "precision": "Blues", "recall": "Oranges", "f1": "Greens"}

    for ax, metric in zip(axes.flat, METRICS):
        pivot = df.pivot_table(
            index="max_features", columns="n_estimators",
            values=f"val_{metric}", aggfunc="mean"
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmaps[metric])
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("n_estimators", fontsize=9)
        ax.set_ylabel("max_features", fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_heatmaps.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_metrics_vs_n_by_split(df: pd.DataFrame):
    splits = df["split_label"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Val Metrics vs n_estimators  .  per split ratio\n(averaged over max_features)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        for color, split in zip(SPLIT_COLORS, splits):
            sub = df[df["split_label"] == split].groupby("n_estimators")[f"val_{metric}"].mean()
            ax.plot(sub.index, sub.values, marker="o", markersize=4,
                    label=split, color=color, linewidth=1.6)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("n_estimators")
        ax.legend(fontsize=7, title="split", title_fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_metrics_vs_n_by_split.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_metrics_vs_lr_by_split(df: pd.DataFrame):
    splits  = df["split_label"].unique()
    mf_vals = df["max_features"].unique().tolist()
    x_pos   = range(len(mf_vals))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Val Metrics vs max_features  .  per split ratio\n(averaged over n_estimators)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes.flat, METRICS):
        for color, split in zip(SPLIT_COLORS, splits):
            sub = (
                df[df["split_label"] == split]
                .groupby("max_features")[f"val_{metric}"]
                .mean()
                .reindex(mf_vals)
            )
            ax.plot(x_pos, sub.values, marker="s", markersize=5,
                    label=split, color=color, linewidth=1.6)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(mf_vals, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xlabel("max_features")
        ax.legend(fontsize=7, title="split", title_fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/val_metrics_vs_maxfeat_by_split.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_train_vs_val(df: pd.DataFrame):
    best_split = df.groupby("split_label")["val_f1"].mean().idxmax()
    sub = df[df["split_label"] == best_split].groupby("n_estimators")[
        [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS]
    ].mean()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Train vs Val  .  split={best_split}  (averaged over max_features)",
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
    print(f"Saved: {out}")



def plot_overfit_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="max_features", columns="n_estimators",
        values="overfit_auc", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=np.percentile(pivot.values, 95))
    plt.colorbar(im, ax=ax, label="Train AUC - Val AUC")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_features")
    ax.set_title("Overfit Heatmap: Train - Val AUC  (red = high overfit, green = low)")
    plt.tight_layout()
    out = f"{PLOTS_DIR}/overfit_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def plot_top_scatter(df: pd.DataFrame):
    top = df.head(30).copy()
    mf_order = {v: i for i, v in enumerate(df["max_features"].unique())}
    top["mf_num"] = top["max_features"].map(mf_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(top["val_f1"], top["val_auc"],
                    c=top["mf_num"], cmap="plasma",
                    s=100, edgecolors="white", linewidth=0.6, zorder=3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_ticks(list(mf_order.values()))
    cbar.set_ticklabels(list(mf_order.keys()))
    cbar.set_label("max_features")

    for _, row in top.iterrows():
        ax.annotate(f"n={int(row.n_estimators)}", (row.val_f1, row.val_auc),
                    textcoords="offset points", xytext=(3, 2), fontsize=6, alpha=0.75)
    ax.set_xlabel("Val F1-Score")
    ax.set_ylabel("Val AUC")
    ax.set_title("Top 30 Configs -- Val F1 vs Val AUC  (color = max_features)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/top30_scatter.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Random Forest Dense Combination Experiment  v2")
    print("=" * 60)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")

    df_eng = add_features(df_raw)

    X = df_eng.drop(columns=[TARGET])
    y = df_eng[TARGET]
    print(f"Data: {df_eng.shape}  |  Classes: {y.value_counts().to_dict()}")
    print(f"Features: {X.shape[1]} total (original + engineered)")

    results = run_all(X, y)

    show = ["n_estimators", "max_features", "split_label",
            "train_auc", "val_auc", "test_auc",
            "train_f1",  "val_f1",  "test_f1",
            "val_precision", "val_recall", "val_logloss", "overfit_auc"]
    print("\nTOP 10 CONFIGS (by Val F1):")
    print(results[show].head(10).to_string(index=False))

    plot_val_heatmaps(results)
    plot_metrics_vs_n_by_split(results)
    plot_metrics_vs_lr_by_split(results)
    plot_train_vs_val(results)
    plot_overfit_heatmap(results)
    plot_top_scatter(results)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"Results : {RESULTS_DIR}/dense_results.csv / .xlsx")
    print(f"Plots   : {PLOTS_DIR}/")


if __name__ == "__main__":
    main()