import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

from lightgbm import LGBMClassifier

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
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 🧠 FINAL MODELS
# =========================

# 🔥 Best LightGBM (from your experiment)
lgbm = LGBMClassifier(
    n_estimators=40,
    learning_rate=0.07,
    max_depth=10,
    num_leaves=50,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    device="gpu",
    verbose=-1
)

# 🌳 Random Forest (stability model)
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 🔥 Hybrid Model
hybrid = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lgbm', lgbm)
    ],
    voting='soft',
    weights=[1, 2]   # LGBM more important
)

# =========================
# 🚀 FINAL PIPELINE
# =========================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", hybrid)
])

pipeline.fit(X_train, y_train)

# =========================
# 🎯 CALIBRATION (CRITICAL)
# =========================

calibrated_model = CalibratedClassifierCV(
    pipeline,
    method='isotonic',
    cv=5
)

calibrated_model.fit(X_train, y_train)

# =========================
# 📊 EVALUATION
# =========================

y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

# =========================
# 🎯 RISK FUNCTION
# =========================

def risk_level(p):
    if p < 0.1:
        return "LOW"
    elif p < 0.4:
        return "MEDIUM"
    else:
        return "HIGH"

# Example
sample_probs = y_prob[:10]
print("\nSample Risk Levels:")
print([risk_level(p) for p in sample_probs])