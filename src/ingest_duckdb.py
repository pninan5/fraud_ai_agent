import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")

TRAIN_TXN = os.path.join(RAW_DIR, "train_transaction.csv")
TRAIN_ID = os.path.join(RAW_DIR, "train_identity.csv")

os.makedirs(PARQUET_DIR, exist_ok=True)

if not os.path.exists(TRAIN_TXN):
    raise FileNotFoundError(f"Missing file: {TRAIN_TXN}")
if not os.path.exists(TRAIN_ID):
    raise FileNotFoundError(f"Missing file: {TRAIN_ID}")

con = duckdb.connect(DB_PATH)
con.execute("PRAGMA threads=8;")

con.execute("DROP TABLE IF EXISTS train_transaction;")
con.execute("DROP TABLE IF EXISTS train_identity;")
con.execute("DROP TABLE IF EXISTS train_joined;")

print("Loading train_transaction.csv into DuckDB...")
con.execute(
    f"""
    CREATE TABLE train_transaction AS
    SELECT *
    FROM read_csv_auto('{TRAIN_TXN}');
    """
)

print("Loading train_identity.csv into DuckDB...")
con.execute(
    f"""
    CREATE TABLE train_identity AS
    SELECT *
    FROM read_csv_auto('{TRAIN_ID}');
    """
)

print("Joining tables on TransactionID (LEFT JOIN)...")
con.execute(
    """
    CREATE TABLE train_joined AS
    SELECT
        t.*,
        i.*
    FROM train_transaction t
    LEFT JOIN train_identity i
    USING (TransactionID);
    """
)

out_parquet = os.path.join(PARQUET_DIR, "train_joined.parquet")

print("Exporting joined table to Parquet (fast format for training)...")
con.execute(
    f"""
    COPY (SELECT * FROM train_joined)
    TO '{out_parquet}'
    (FORMAT PARQUET);
    """
)

rows = con.execute("SELECT COUNT(*) FROM train_joined;").fetchone()[0]
print("Saved:", out_parquet)
print("Rows:", rows)

con.close()
