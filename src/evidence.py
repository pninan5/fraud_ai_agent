import os
import duckdb
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x)


def _to_py_int(x):
    if _is_missing(x):
        return None
    # convert numpy / pandas scalar to Python int
    return int(x)


def _to_py_str(x):
    if _is_missing(x):
        return None
    return str(x)


def get_case_row(transaction_id: int) -> dict:
    con = duckdb.connect(DB_PATH, read_only=True)
    row = con.execute(
        """
        SELECT *
        FROM train_joined
        WHERE TransactionID = ?
        LIMIT 1
        """,
        [int(transaction_id)],
    ).df()
    con.close()

    if row.empty:
        raise ValueError(f"TransactionID {transaction_id} not found in train_joined")

    return row.iloc[0].to_dict()


def compute_identity_presence(case: dict) -> dict:
    identity_cols = [c for c in case.keys() if c.startswith("id_") or c in ["DeviceType", "DeviceInfo"]]
    if not identity_cols:
        return {"identity_present": None, "identity_missing_ratio": None, "identity_cols_count": 0}

    missing = sum(pd.isna(case.get(c)) for c in identity_cols)
    ratio = missing / max(len(identity_cols), 1)
    identity_present = ratio < 0.95

    return {
        "identity_present": bool(identity_present),
        "identity_missing_ratio": float(ratio),
        "identity_cols_count": int(len(identity_cols)),
    }


def compute_velocity_features(transaction_id: int) -> dict:
    """
    Entity-based velocity using proxy key:
      card1 + addr1 + P_emaildomain
    fallback:
      card1 only

    Fix: convert numpy scalar params to Python native types before passing to DuckDB.
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    row = con.execute(
        """
        SELECT TransactionDT, card1, addr1, P_emaildomain
        FROM train_joined
        WHERE TransactionID = ?
        LIMIT 1
        """,
        [int(transaction_id)],
    ).df()

    if row.empty or _is_missing(row.loc[0, "TransactionDT"]):
        con.close()
        return {
            "entity_key_used": None,
            "entity_tx_count_10m": None,
            "entity_tx_count_1h": None,
            "entity_tx_count_24h": None,
            "TransactionDT": None,
        }

    tx_time = _to_py_int(row.loc[0, "TransactionDT"])
    card1 = _to_py_int(row.loc[0, "card1"])
    addr1 = _to_py_int(row.loc[0, "addr1"])
    email = _to_py_str(row.loc[0, "P_emaildomain"])

    use_full_key = (card1 is not None) and (addr1 is not None) and (email is not None)

    if use_full_key:
        counts = con.execute(
            """
            SELECT
              SUM(CASE WHEN TransactionDT BETWEEN ? - 600 AND ? THEN 1 ELSE 0 END) AS c10m,
              SUM(CASE WHEN TransactionDT BETWEEN ? - 3600 AND ? THEN 1 ELSE 0 END) AS c1h,
              SUM(CASE WHEN TransactionDT BETWEEN ? - 86400 AND ? THEN 1 ELSE 0 END) AS c24h
            FROM train_joined
            WHERE card1 = ? AND addr1 = ? AND P_emaildomain = ?
            """,
            [tx_time, tx_time, tx_time, tx_time, tx_time, tx_time, card1, addr1, email],
        ).fetchone()
        key_used = "card1+addr1+P_emaildomain"
    else:
        # If card1 itself is missing, we cannot compute entity velocity
        if card1 is None:
            con.close()
            return {
                "entity_key_used": None,
                "entity_tx_count_10m": None,
                "entity_tx_count_1h": None,
                "entity_tx_count_24h": None,
                "TransactionDT": tx_time,
            }

        counts = con.execute(
            """
            SELECT
              SUM(CASE WHEN TransactionDT BETWEEN ? - 600 AND ? THEN 1 ELSE 0 END) AS c10m,
              SUM(CASE WHEN TransactionDT BETWEEN ? - 3600 AND ? THEN 1 ELSE 0 END) AS c1h,
              SUM(CASE WHEN TransactionDT BETWEEN ? - 86400 AND ? THEN 1 ELSE 0 END) AS c24h
            FROM train_joined
            WHERE card1 = ?
            """,
            [tx_time, tx_time, tx_time, tx_time, tx_time, tx_time, card1],
        ).fetchone()
        key_used = "card1_only"

    con.close()

    return {
        "entity_key_used": key_used,
        "entity_tx_count_10m": int(counts[0]),
        "entity_tx_count_1h": int(counts[1]),
        "entity_tx_count_24h": int(counts[2]),
        "TransactionDT": tx_time,
    }


def compute_amount_features(case: dict) -> dict:
    amt = case.get("TransactionAmt", None)
    if amt is None or pd.isna(amt):
        return {"amount": None, "amount_high": None}

    amt = float(amt)
    return {"amount": amt, "amount_high": bool(amt >= 500)}


def build_evidence(transaction_id: int) -> dict:
    case = get_case_row(transaction_id)

    evidence = {}
    evidence.update(compute_identity_presence(case))
    evidence.update(compute_velocity_features(transaction_id))
    evidence.update(compute_amount_features(case))

    for key in [
        "ProductCD",
        "card1", "card2", "card3", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ]:
        if key in case:
            evidence[key] = case.get(key)

    return evidence
