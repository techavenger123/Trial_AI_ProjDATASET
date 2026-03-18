import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# --- Setup ---
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

start_time = time.time()

# --- Load Data ---
df = pd.read_csv("industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv")

X = df.drop("faulty", axis=1)
y = df["faulty"]

categorical_cols = ["equipment", "location"]
numerical_cols = ["temperature", "pressure", "vibration", "humidity"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

# --- Split ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=42
)

# =========================
# 🔥 MASSIVE SEARCH SPACE
# =========================

# Learning rates (very dense)
lr_log = np.logspace(-3, -1, 20)
lr_linear = np.linspace(0.005, 0.1, 20)
learning_rates = np.unique(np.round(np.concatenate([lr_log, lr_linear]), 4))

# n_estimators full range
n_values = range(25, 401, 1)

# Add slight variation in model complexity
max_depth_values = [8, 10, 12]
num_leaves_values = [31, 50, 80]

results = []

total_runs = len(learning_rates) * len(n_values) * len(max_depth_values) * len(num_leaves_values)
run_count = 0

print(f"\n🔥 TOTAL RUNS: {total_runs}")

# =========================
# GRID LOOP
# =========================
for lr in learning_rates:
    for n in n_values:
        for md in max_depth_values:
            for nl in num_leaves_values:

                run_count += 1
                print(f"[{run_count}/{total_runs}] n={n}, lr={lr}, depth={md}, leaves={nl}")

                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", LGBMClassifier(
                        n_estimators=n,
                        learning_rate=lr,
                        max_depth=md,
                        num_leaves=nl,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight='balanced',
                        random_state=42,
                        verbose=-1
                    ))
                ])

                model.fit(X_train, y_train)

                # Validation metric
                y_val_prob = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_val_prob)

                results.append({
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": md,
                    "num_leaves": nl,
                    "val_auc": val_auc
                })

                # Save every 200 runs
                if run_count % 200 == 0:
                    pd.DataFrame(results).to_csv("results/LGBM_ULTRA_PROGRESS.csv", index=False)

# =========================
# FINAL SAVE
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("results/LGBM_ULTRA_RESULTS.csv", index=False)

print("\n✅ Saved: results/LGBM_ULTRA_RESULTS.csv")

# =========================
# BEST CONFIG
# =========================
best = results_df.sort_values(by="val_auc", ascending=False).iloc[0]

print("\n🔥 BEST CONFIG FOUND:")
print(best)

# =========================
# TOP 10 CONFIGS
# =========================
print("\n🏆 TOP 10 CONFIGS:")
print(results_df.sort_values(by="val_auc", ascending=False).head(10))

# =========================
# HEATMAP (n vs lr)
# =========================
pivot = results_df.pivot_table(
    index="learning_rate",
    columns="n_estimators",
    values="val_auc",
    aggfunc="max"
)

plt.figure(figsize=(14, 7))
plt.imshow(pivot, aspect='auto')
plt.colorbar(label="Validation AUC")

plt.xlabel("n_estimators")
plt.ylabel("learning_rate")
plt.title("Ultra Search Heatmap")

plt.savefig("plots/LGBM_ULTRA_HEATMAP.png", dpi=300, bbox_inches='tight')
print("✅ Saved: plots/LGBM_ULTRA_HEATMAP.png")

plt.show()

# =========================
# TIME
# =========================
end_time = time.time()
print(f"\n⏱ Total Time: {(end_time - start_time)/60:.2f} minutes")