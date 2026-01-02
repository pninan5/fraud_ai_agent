import os
import joblib
import duckdb
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lgbm_baseline.joblib")


def _encode_like_training(X: pd.DataFrame) -> pd.DataFrame:
    """
    Match the baseline training preprocessing:
      - object -> category codes
      - boolean -> Int8 with -1
      - fillna(-1)
    """
    # Boolean handling (nullable boolean or bool)
    bool_cols = [c for c in X.columns if str(X[c].dtype) == "boolean" or X[c].dtype == bool]
    for c in bool_cols:
        X[c] = X[c].astype("Int8").fillna(-1)

    # Object to category codes
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes

    X = X.fillna(-1)
    return X


def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def fetch_transaction_row(transaction_id: int) -> pd.DataFrame:
    """
    Pull the joined row for one TransactionID from DuckDB.
    Returns a single-row dataframe.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        """
        SELECT *
        FROM train_joined
        WHERE TransactionID = ?
        LIMIT 1
        """,
        [int(transaction_id)],
    ).df()
    con.close()

    if df.empty:
        raise ValueError(f"TransactionID {transaction_id} not found in train_joined")

    return df


def score_transaction(transaction_id: int) -> dict:
    """
    Returns:
      - fraud_proba
      - model_features_used
    """
    artifact = load_model_artifact()
    model = artifact["model"]
    cols = artifact["columns"]

    df = fetch_transaction_row(transaction_id)

    # Drop target + id if present
    drop_cols = []
    if "isFraud" in df.columns:
        drop_cols.append("isFraud")
    if "TransactionID" in df.columns:
        drop_cols.append("TransactionID")

    X = df.drop(columns=drop_cols)

    # Ensure same column order as training
    # If any columns are missing, add them as NaN
    for c in cols:
        if c not in X.columns:
            X[c] = pd.NA
    X = X[cols]

    X = _encode_like_training(X)

    proba = float(model.predict_proba(X)[:, 1][0])

    return {
        "transaction_id": int(transaction_id),
        "fraud_proba": proba,
        "model_features_used": len(cols),
    }
