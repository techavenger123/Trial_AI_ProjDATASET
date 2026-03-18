import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import time

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# --- Hyperparameters ---
n_values = range(25, 401, 1)  # step = 1 (as you requested)
learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1]

results = []

total_runs = len(n_values) * len(learning_rates)
run_count = 0

# --- Grid Loop ---
for lr in learning_rates:
    for n in n_values:
        run_count += 1
        print(f"[{run_count}/{total_runs}] n={n}, lr={lr}")

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(
                n_estimators=n,
                learning_rate=lr,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "n_estimators": n,
            "learning_rate": lr,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        })

        # --- Save intermediate every 50 runs ---
        if run_count % 50 == 0:
            pd.DataFrame(results).to_csv("results/LGBM_GRID_PROGRESS.csv", index=False)

# --- Final Save ---
results_df = pd.DataFrame(results)
results_df.to_csv("results/LGBM_GRID_RESULTS_FINE.csv", index=False)

print("\n✅ Final results saved: results/LGBM_GRID_RESULTS_FINE.csv")

# --- Best Model ---
best = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]

print("\n🔥 BEST CONFIG:")
print(best)

# --- Plot (Line per LR) ---
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    subset = results_df[results_df["learning_rate"] == lr]
    plt.plot(subset["n_estimators"], subset["roc_auc"], label=f"lr={lr}")

plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.title("ROC-AUC vs n_estimators (Fine Search)")
plt.legend()
plt.grid()

plt.savefig("plots/LGBM_FINE_SEARCH.png", dpi=300, bbox_inches='tight')
print("✅ Saved: plots/LGBM_FINE_SEARCH.png")

plt.show()

# --- Time Taken ---
end_time = time.time()
print(f"\n⏱ Total Time: {(end_time - start_time)/60:.2f} minutes")