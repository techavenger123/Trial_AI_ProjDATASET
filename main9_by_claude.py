"""
LightGBM Hyperparameter Research & Training Pipeline
=====================================================
Improvements over baseline:
  - Optuna Bayesian search (smarter than grid search)
  - Early stopping to prevent overfitting & save time
  - Cross-validation for final best-config evaluation
  - Test-set AUC reported (not just validation)
  - Feature importance & learning curve plots
  - Checkpoint/resume: skips already-run configs
  - ETA display via tqdm
  - Modular, readable structure
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataclasses import dataclass, field
from typing import List, Optional

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  optuna not installed. Falling back to grid search. Run: pip install optuna")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass
class Config:
    data_path: str = "industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv"
    drop_cols: List[str] = field(default_factory=lambda: ["location"])
    target_col: str = "faulty"
    categorical_cols: List[str] = field(default_factory=lambda: ["equipment"])
    numerical_cols: List[str] = field(default_factory=lambda: ["temperature", "pressure", "vibration", "humidity"])

    test_size: float = 0.30
    val_size: float = 0.30
    random_state: int = 42
    cv_folds: int = 5

    # Search mode: "optuna" (recommended) or "grid"
    search_mode: str = "optuna" if OPTUNA_AVAILABLE else "grid"
    n_optuna_trials: int = 100         # Bayesian trials (much more efficient than 18k+ grid runs)
    early_stopping_rounds: int = 30    # Stop if val AUC doesn't improve for N rounds
    save_every_n_runs: int = 50

    results_dir: str = "results"
    plots_dir: str = "plots"
    checkpoint_file: str = "results/checkpoint.csv"


CFG = Config()
os.makedirs(CFG.results_dir, exist_ok=True)
os.makedirs(CFG.plots_dir, exist_ok=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_data(cfg: Config):
    df = pd.read_csv(cfg.data_path)
    df = df.drop(columns=cfg.drop_cols, errors="ignore")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]

    print(f"✅ Loaded data: {df.shape}  |  Class balance: {y.value_counts().to_dict()}")
    return X, y


def make_splits(X, y, cfg: Config):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=cfg.val_size,
        stratify=y_trainval,
        random_state=cfg.random_state
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────
def make_preprocessor(cfg: Config) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.categorical_cols),
        ("num", "passthrough", cfg.numerical_cols)
    ])


# ─────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────
def build_pipeline(preprocessor: ColumnTransformer, params: dict) -> Pipeline:
    clf = LGBMClassifier(
        class_weight="balanced",
        random_state=CFG.random_state,
        verbose=-1,
        **params
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


# ─────────────────────────────────────────────
# TRAIN WITH EARLY STOPPING
# ─────────────────────────────────────────────
def fit_with_early_stopping(pipeline: Pipeline, X_train, y_train, X_val, y_val, cfg: Config):
    """
    Fits the classifier inside the pipeline using LightGBM's native early stopping.
    Returns the fitted pipeline and the best validation AUC.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)

    clf.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
            log_evaluation(period=-1)
        ]
    )

    best_iter = clf.best_iteration_
    y_val_prob = clf.predict_proba(X_val_t)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    return pipeline, val_auc, best_iter


# ─────────────────────────────────────────────
# OPTUNA SEARCH (Bayesian)
# ─────────────────────────────────────────────
def run_optuna_search(preprocessor, X_train, y_train, X_val, y_val, cfg: Config):
    print(f"\n🔬 Bayesian Search with Optuna ({cfg.n_optuna_trials} trials) ...")

    results = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        pipeline = build_pipeline(preprocessor, params)
        _, val_auc, best_iter = fit_with_early_stopping(pipeline, X_train, y_train, X_val, y_val, cfg)

        results.append({**params, "val_auc": val_auc, "best_n_estimators": best_iter})
        return val_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=True)

    results_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    results_df.to_csv(f"{cfg.results_dir}/optuna_results.csv", index=False)

    print(f"\n✅ Optuna search complete. Best val AUC: {study.best_value:.5f}")
    return results_df, study.best_params


# ─────────────────────────────────────────────
# GRID SEARCH (fallback)
# ─────────────────────────────────────────────
def run_grid_search(preprocessor, X_train, y_train, X_val, y_val, cfg: Config):
    print("\n📊 Running Grid Search ...")

    # Load checkpoint if exists
    if os.path.exists(cfg.checkpoint_file):
        done_df = pd.read_csv(cfg.checkpoint_file)
        done_keys = set(zip(done_df.n_estimators, done_df.learning_rate, done_df.max_depth, done_df.num_leaves))
        results = done_df.to_dict("records")
        print(f"   Resuming from checkpoint: {len(results)} runs already done.")
    else:
        done_keys = set()
        results = []

    lr_values = np.unique(np.round(np.concatenate([
        np.logspace(-3, -1, 12),
        np.linspace(0.005, 0.15, 10)
    ]), 4))
    n_values = range(50, 401, 50)
    max_depth_values = [6, 8, 10, 12]
    num_leaves_values = [31, 50, 80]

    grid = [(lr, n, md, nl)
            for lr in lr_values
            for n in n_values
            for md in max_depth_values
            for nl in num_leaves_values]

    total = len(grid)
    iterator = tqdm(grid, desc="Grid Search") if TQDM_AVAILABLE else grid

    for i, (lr, n, md, nl) in enumerate(iterator):
        key = (n, lr, md, nl)
        if key in done_keys:
            continue

        params = {
            "n_estimators": n,
            "learning_rate": lr,
            "max_depth": md,
            "num_leaves": nl,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        pipeline = build_pipeline(preprocessor, params)
        _, val_auc, best_iter = fit_with_early_stopping(pipeline, X_train, y_train, X_val, y_val, cfg)

        results.append({**params, "val_auc": val_auc, "best_n_estimators": best_iter})

        if (i + 1) % cfg.save_every_n_runs == 0:
            pd.DataFrame(results).to_csv(cfg.checkpoint_file, index=False)

    results_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    results_df.to_csv(f"{cfg.results_dir}/grid_results.csv", index=False)
    return results_df, results_df.iloc[0].drop(["val_auc", "best_n_estimators"]).to_dict()


# ─────────────────────────────────────────────
# CROSS-VALIDATION ON BEST CONFIG
# ─────────────────────────────────────────────
def cross_validate_best(preprocessor, X_trainval, y_trainval, best_params: dict, cfg: Config):
    print(f"\n🔁 Cross-validating best config ({cfg.cv_folds}-fold) ...")

    # Use the best_n_estimators found (no early stopping needed during CV)
    params = {k: v for k, v in best_params.items() if k != "best_n_estimators"}
    pipeline = build_pipeline(preprocessor, params)

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scores = cross_val_score(pipeline, X_trainval, y_trainval, cv=skf, scoring="roc_auc", n_jobs=-1)

    print(f"   CV AUC: {scores.mean():.5f} ± {scores.std():.5f}  |  Folds: {np.round(scores, 4)}")
    return scores


# ─────────────────────────────────────────────
# FINAL MODEL TRAINING & TEST EVALUATION
# ─────────────────────────────────────────────
def train_final_model(preprocessor, X_trainval, y_trainval, X_test, y_test, best_params: dict):
    print("\n🚀 Training final model on full train+val set ...")
    params = {k: v for k, v in best_params.items() if k not in ("best_n_estimators",)}
    pipeline = build_pipeline(preprocessor, params)
    pipeline.fit(X_trainval, y_trainval)

    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"   ✅ Test AUC: {test_auc:.5f}")
    return pipeline, test_auc


# ─────────────────────────────────────────────
# LEARNING CURVE
# ─────────────────────────────────────────────
def plot_learning_curve(preprocessor, X_train, y_train, X_val, y_val, best_params: dict, cfg: Config):
    print("\n📈 Generating learning curve ...")
    params = {k: v for k, v in best_params.items() if k not in ("best_n_estimators",)}
    params["n_estimators"] = 500

    pre = preprocessor
    X_tr = pre.fit_transform(X_train, y_train)
    X_v = pre.transform(X_val)

    clf = LGBMClassifier(class_weight="balanced", random_state=cfg.random_state, verbose=-1, **params)
    clf.fit(
        X_tr, y_train,
        eval_set=[(X_tr, y_train), (X_v, y_val)],
        eval_metric="auc",
        callbacks=[log_evaluation(period=-1)]
    )

    evals = clf.evals_result_
    train_auc = evals["training"]["auc"]
    val_auc   = evals["valid_1"]["auc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_auc, label="Train AUC", linewidth=1.5)
    ax.plot(val_auc,   label="Val AUC",   linewidth=1.5)
    ax.axvline(np.argmax(val_auc), color="red", linestyle="--", alpha=0.6, label=f"Best iter ({np.argmax(val_auc)})")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("AUC")
    ax.set_title("Learning Curve (Train vs Validation AUC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/learning_curve.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved: {cfg.plots_dir}/learning_curve.png")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def plot_feature_importance(pipeline: Pipeline, cfg: Config):
    print("\n🌲 Plotting feature importance ...")
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    cat_enc = preprocessor.named_transformers_["cat"]
    cat_features = list(cat_enc.get_feature_names_out(cfg.categorical_cols))
    feature_names = cat_features + cfg.numerical_cols

    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(fi_df) * 0.4)))
    bars = ax.barh(fi_df["feature"], fi_df["importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/feature_importance.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved: {cfg.plots_dir}/feature_importance.png")


# ─────────────────────────────────────────────
# SEARCH RESULTS HEATMAP (val AUC)
# ─────────────────────────────────────────────
def plot_heatmap(results_df: pd.DataFrame, cfg: Config):
    print("\n🌡️  Plotting heatmap ...")
    pivot = results_df.pivot_table(
        index="learning_rate", columns="n_estimators", values="val_auc", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Val AUC")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    ax.set_title("Hyperparameter Search Heatmap (max val AUC)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(np.round(pivot.index, 4), fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/search_heatmap.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Saved: {cfg.plots_dir}/search_heatmap.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 60)
    print("  LightGBM Research & Training Pipeline")
    print("=" * 60)

    # 1. Load & split
    X, y = load_data(CFG)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, CFG)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    preprocessor = make_preprocessor(CFG)

    # 2. Hyperparameter Search
    if CFG.search_mode == "optuna":
        results_df, best_params = run_optuna_search(preprocessor, X_train, y_train, X_val, y_val, CFG)
    else:
        results_df, best_params = run_grid_search(preprocessor, X_train, y_train, X_val, y_val, CFG)

    print(f"\n🏆 Best Config:\n{pd.Series(best_params).to_string()}")
    print(f"\n🏅 Top 10 Configs:")
    print(results_df.head(10).to_string(index=False))
    results_df.head(10).to_csv(f"{CFG.results_dir}/top10_configs.csv", index=False)

    # 3. Cross-validate best config
    cv_scores = cross_validate_best(preprocessor, X_trainval, y_trainval, best_params, CFG)

    # 4. Final model + test AUC
    final_pipeline, test_auc = train_final_model(preprocessor, X_trainval, y_trainval, X_test, y_test, best_params)

    # 5. Plots
    plot_learning_curve(make_preprocessor(CFG), X_train, y_train, X_val, y_val, best_params, CFG)
    plot_feature_importance(final_pipeline, CFG)
    if CFG.search_mode == "grid":
        plot_heatmap(results_df, CFG)

    # 6. Summary
    elapsed = (time.time() - start) / 60
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Search mode    : {CFG.search_mode.upper()}")
    print(f"  Best val AUC   : {results_df.iloc[0]['val_auc']:.5f}")
    print(f"  CV AUC         : {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
    print(f"  Test AUC       : {test_auc:.5f}")
    print(f"  Total time     : {elapsed:.2f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()