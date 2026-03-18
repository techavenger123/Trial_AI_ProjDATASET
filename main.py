import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# --- Train Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# --- Class imbalance handling ---
fault_ratio = len(y_train) / y_train.sum()

# --- Store results ---
results = []

# --- Loop over n_estimators ---
for n in range(1, 401, 1):
    print("N_Estimators: ", n)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n,
            max_depth=10,
            class_weight={0: 1, 1: fault_ratio},
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results.append({
        "n_estimators": n,
        "accuracy": acc,
        "f1_score": f1,
        "recall": recall,
        "roc_auc": roc
    })
    print("n_estimators", n,
        "accuracy", acc,
        "f1_score", f1,
        "recall", recall,
        "roc_auc", roc)
    print("\n")

# --- Convert to DataFrame ---
results_df = pd.DataFrame(results)

# --- Show Top Results ---
print("\n=== Top 10 Models by ROC-AUC ===")
print(results_df.sort_values(by="roc_auc", ascending=False).head(10))

# --- Plot ---
plt.figure()
plt.plot(results_df["n_estimators"], results_df["accuracy"], label="Accuracy")
plt.plot(results_df["n_estimators"], results_df["f1_score"], label="F1 Score")
plt.plot(results_df["n_estimators"], results_df["recall"], label="Recall")
plt.plot(results_df["n_estimators"], results_df["roc_auc"], label="ROC-AUC")

plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.title("Random Forest Performance vs n_estimators")
plt.legend()
plt.grid()

plt.show()
plt.savefig("Only Random Forest.png")
# --- Convert to DataFrame ---
results_df = pd.DataFrame(results)

# --- Save Results to File ---
results_df.to_csv("RANDOM_FOREST.csv", index=False)

# Optional: Save as Excel
# results_df.to_excel("RANDOM_FOREST.xlsx", index=False)

print("\n✅ Results saved to RANDOM_FOREST.csv")

# --- Show Top Results ---
print("\n=== Top 10 Models by ROC-AUC ===")
print(results_df.sort_values(by="roc_auc", ascending=False).head(10))

# --- Plot ---
plt.figure()
plt.plot(results_df["n_estimators"], results_df["accuracy"], label="Accuracy")
plt.plot(results_df["n_estimators"], results_df["f1_score"], label="F1 Score")
plt.plot(results_df["n_estimators"], results_df["recall"], label="Recall")
plt.plot(results_df["n_estimators"], results_df["roc_auc"], label="ROC-AUC")

plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.title("Random Forest Performance vs n_estimators (1-400,1)")
plt.legend()
plt.grid()

plt.show()

