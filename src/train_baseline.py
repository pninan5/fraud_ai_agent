import os
import joblib
import duckdb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "isFraud"

# -----------------------------
# 1) Load a manageable sample
# -----------------------------
con = duckdb.connect(DB_PATH)

df = con.execute(
    """
    SELECT *
    FROM train_joined
    WHERE isFraud IS NOT NULL
    LIMIT 500000
    """
).df()

con.close()

# -----------------------------
# 2) Split X / y
# -----------------------------
y = df[TARGET_COL].astype(int)

drop_cols = [TARGET_COL]
if "TransactionID" in df.columns:
    drop_cols.append("TransactionID")

X = df.drop(columns=drop_cols)

# -----------------------------
# 3) Type-safe preprocessing
#    - booleans -> Int (0/1/-1)
#    - objects -> category codes
#    - fill remaining missing with -1
# -----------------------------
bool_cols = [c for c in X.columns if str(X[c].dtype) == "boolean" or X[c].dtype == bool]
for c in bool_cols:
    # True/False/NA -> 1/0/<NA>
    # Convert to nullable int so we can fill NA with -1
    X[c] = X[c].astype("Int8").fillna(-1)

obj_cols = [c for c in X.columns if X[c].dtype == "object"]
for c in obj_cols:
    # Missing becomes -1 after cat.codes
    X[c] = X[c].astype("category").cat.codes

# For any remaining nullable numeric types, keep numeric where possible
for c in X.columns:
    if c not in obj_cols and c not in bool_cols:
        # Do not force-convert if it is already numeric; errors='ignore' keeps types stable
        X[c] = pd.to_numeric(X[c], errors="ignore")

X = X.fillna(-1)

# -----------------------------
# 4) Train/validation split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5) Model (CPU friendly)
# -----------------------------
model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 6) Evaluate with PR-AUC
# -----------------------------
proba = model.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, proba)
print("Average Precision (PR-AUC):", round(ap, 6))

# -----------------------------
# 7) Save artifact
# -----------------------------
artifact = {
    "model": model,
    "columns": list(X.columns),
    "sample_rows": len(df)
}

out_path = os.path.join(MODEL_DIR, "lgbm_baseline.joblib")
joblib.dump(artifact, out_path)
print("Saved model artifact:", out_path)
