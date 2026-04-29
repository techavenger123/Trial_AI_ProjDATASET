import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


# --- Create folders ---
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load Data ---
df = pd.read_csv("industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv")

X = df.drop("faulty", axis=1)
y = df["faulty"]

categorical_cols = ["equipment", "location"]
numerical_cols = ["temperature", "pressure", "vibration", "humidity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# --- Store results ---
results = []

# --- Sweep values ---
n_values = range(50, 401, 1)

for n in n_values:
    print(f"\n🔁 Running for n_estimators = {n}")

    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    lgbm = LGBMClassifier(
        n_estimators=n,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    hybrid = VotingClassifier(
        estimators=[('rf', rf), ('lgbm', lgbm)],
        voting='soft',
        weights=[1, 2]
    )

    # Pipelines
    rf_model = Pipeline([("preprocessor", preprocessor), ("clf", rf)])
    lgbm_model = Pipeline([("preprocessor", preprocessor), ("clf", lgbm)])
    hybrid_model = Pipeline([("preprocessor", preprocessor), ("clf", hybrid)])

    # Train
    rf_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    hybrid_model.fit(X_train, y_train)

    # Predict
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
    hyb_prob = hybrid_model.predict_proba(X_test)[:, 1]

    rf_pred = rf_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)
    hyb_pred = hybrid_model.predict(X_test)

    # Store metrics
    results.append({
        "n_estimators": n,

        "rf_accuracy": accuracy_score(y_test, rf_pred),
        "rf_f1": f1_score(y_test, rf_pred),
        "rf_recall": recall_score(y_test, rf_pred),
        "rf_auc": roc_auc_score(y_test, rf_prob),

        "lgbm_accuracy": accuracy_score(y_test, lgbm_pred),
        "lgbm_f1": f1_score(y_test, lgbm_pred),
        "lgbm_recall": recall_score(y_test, lgbm_pred),
        "lgbm_auc": roc_auc_score(y_test, lgbm_prob),

        "hyb_accuracy": accuracy_score(y_test, hyb_pred),
        "hyb_f1": f1_score(y_test, hyb_pred),
        "hyb_recall": recall_score(y_test, hyb_pred),
        "hyb_auc": roc_auc_score(y_test, hyb_prob),
    })

# --- Convert to DataFrame ---
results_df = pd.DataFrame(results)

# --- Save results ---
results_df.to_csv("results/HYBRID_MODEL_RESULTS.csv", index=False)
print("\n✅ Results saved to results/HYBRID_MODEL_RESULTS.csv")

# --- Plot AUC Comparison ---
plt.figure()

plt.plot(results_df["n_estimators"], results_df["rf_auc"], marker='o', label="RF AUC")
plt.plot(results_df["n_estimators"], results_df["lgbm_auc"], marker='o', label="LGBM AUC")
plt.plot(results_df["n_estimators"], results_df["hyb_auc"], marker='o', label="Hybrid AUC")

plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.title("AUC vs n_estimators")
plt.legend()
plt.grid()

plt.savefig("plots/AUC_vs_n_estimators.png", dpi=300, bbox_inches='tight')
print("✅ Saved: plots/AUC_vs_n_estimators.png")

plt.show()

# --- Plot Recall Comparison (IMPORTANT) ---
plt.figure()

plt.plot(results_df["n_estimators"], results_df["rf_recall"], marker='o', label="RF Recall")
plt.plot(results_df["n_estimators"], results_df["lgbm_recall"], marker='o', label="LGBM Recall")
plt.plot(results_df["n_estimators"], results_df["hyb_recall"], marker='o', label="Hybrid Recall")

plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.title("Recall vs n_estimators")
plt.legend()
plt.grid()

plt.savefig("plots/Recall_vs_n_estimators.png", dpi=300, bbox_inches='tight')
print("✅ Saved: plots/Recall_vs_n_estimators.png")

plt.show()

# --- Best Model ---
best = results_df.sort_values(by="hyb_auc", ascending=False).iloc[0]
print("\n🔥 BEST HYBRID CONFIG:")
print(best)