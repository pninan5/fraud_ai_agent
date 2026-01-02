import os
import duckdb
from evidence import build_evidence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")

# Pick a few fraud and non-fraud cases to test evidence output
con = duckdb.connect(DB_PATH, read_only=True)

fraud_ids = con.execute(
    "SELECT TransactionID FROM train_joined WHERE isFraud=1 LIMIT 5"
).fetchall()

nonfraud_ids = con.execute(
    "SELECT TransactionID FROM train_joined WHERE isFraud=0 LIMIT 5"
).fetchall()

con.close()

test_ids = [x[0] for x in (fraud_ids + nonfraud_ids)]

for tid in test_ids:
    ev = build_evidence(tid)

    print("\nTransactionID:", tid)
    for k in [
        "amount",
        "amount_high",
        "entity_key_used",
        "entity_tx_count_10m",
        "entity_tx_count_1h",
        "entity_tx_count_24h",
        "identity_present",
        "identity_missing_ratio",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
    ]:
        print(f"  {k}: {ev.get(k)}")
