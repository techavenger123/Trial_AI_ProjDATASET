import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    roc_curve
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# --- Load Data ---
df = pd.read_csv("industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv")

# --- Features & Target ---
X = df.drop("faulty", axis=1)
y = df["faulty"]

categorical_cols = ["equipment", "location"]
numerical_cols = ["temperature", "pressure", "vibration", "humidity"]

# --- Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# --- Models ---
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)

# --- Hybrid Model ---
hybrid = VotingClassifier(
    estimators=[('rf', rf), ('lgbm', lgbm)],
    voting='soft',
    weights=[1, 2]  # give more importance to LightGBM
)

# --- Pipelines ---
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

lgbm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgbm)
])

hybrid_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", hybrid)
])

# --- Train Models ---
rf_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
hybrid_model.fit(X_train, y_train)

# --- Predictions ---
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

lgbm_pred = lgbm_model.predict(X_test)
lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]

hyb_pred = hybrid_model.predict(X_test)
hyb_prob = hybrid_model.predict_proba(X_test)[:, 1]

# --- Evaluation ---
print("\n=== Random Forest ===")
print(classification_report(y_test, rf_pred))

print("\n=== LightGBM ===")
print(classification_report(y_test, lgbm_pred))

print("\n=== Hybrid Model ===")
print(classification_report(y_test, hyb_pred))

# --- Metrics ---
models = ["RandomForest", "LightGBM", "Hybrid"]

accuracy = [
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, lgbm_pred),
    accuracy_score(y_test, hyb_pred)
]

f1 = [
    f1_score(y_test, rf_pred),
    f1_score(y_test, lgbm_pred),
    f1_score(y_test, hyb_pred)
]

recall = [
    recall_score(y_test, rf_pred),
    recall_score(y_test, lgbm_pred),
    recall_score(y_test, hyb_pred)
]

roc_auc = [
    roc_auc_score(y_test, rf_prob),
    roc_auc_score(y_test, lgbm_prob),
    roc_auc_score(y_test, hyb_prob)
]

print("\n=== Summary Metrics ===")
for i, m in enumerate(models):
    print(f"{m}: Acc={accuracy[i]:.3f}, F1={f1[i]:.3f}, Recall={recall[i]:.3f}, AUC={roc_auc[i]:.3f}")

# --- Plot 1: Metric Comparison ---
import os

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# --- Plot 1: Model Comparison ---
title1 = "Model Comparison"
filename1 = title1.replace(" ", "_") + ".png"

plt.figure()
x = range(len(models))

plt.plot(x, accuracy, marker='o', label="Accuracy")
plt.plot(x, f1, marker='o', label="F1 Score")
plt.plot(x, recall, marker='o', label="Recall")
plt.plot(x, roc_auc, marker='o', label="ROC-AUC")

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title(title1)
plt.legend()
plt.grid()

# Save plot
plt.savefig(f"plots/{filename1}", dpi=300, bbox_inches='tight')
print(f"✅ Saved: plots/{filename1}")

plt.show()


# --- Plot 2: ROC Curve ---
title2 = "ROC Curve Comparison"
filename2 = title2.replace(" ", "_") + ".png"

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, lgbm_prob)
fpr_hyb, tpr_hyb, _ = roc_curve(y_test, hyb_prob)

plt.figure()

plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc[0]:.3f})")
plt.plot(fpr_lgbm, tpr_lgbm, label=f"LGBM (AUC={roc_auc[1]:.3f})")
plt.plot(fpr_hyb, tpr_hyb, label=f"Hybrid (AUC={roc_auc[2]:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(title2)
plt.legend()
plt.grid()

# Save plot
plt.savefig(f"plots/{filename2}", dpi=300, bbox_inches='tight')
print(f"✅ Saved: plots/{filename2}")

plt.show()