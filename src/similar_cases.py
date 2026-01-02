import os
import json
import argparse
import duckdb
import chromadb
from sentence_transformers import SentenceTransformer

from src.evidence import build_evidence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ieee.duckdb")

# Store Chroma OUTSIDE OneDrive for speed/stability
CHROMA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "C:\\Temp"), "fraud_ai_agent_chroma")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fraud_cases"


def case_text(transaction_id: int, label: int) -> str:
    ev = build_evidence(transaction_id)

    parts = [
        f"TransactionID={transaction_id}",
        f"label={label}",
        f"amount={ev.get('amount')}",
        f"ProductCD={ev.get('ProductCD')}",
        f"card1={ev.get('card1')}",
        f"addr1={ev.get('addr1')}",
        f"P_emaildomain={ev.get('P_emaildomain')}",
        f"R_emaildomain={ev.get('R_emaildomain')}",
        f"DeviceType={ev.get('DeviceType')}",
        f"entity_key={ev.get('entity_key_used')}",
        f"v10={ev.get('entity_tx_count_10m')}",
        f"v1h={ev.get('entity_tx_count_1h')}",
        f"v24h={ev.get('entity_tx_count_24h')}",
        f"identity_missing_ratio={ev.get('identity_missing_ratio')}",
    ]
    return " | ".join([p for p in parts if p is not None])


def get_ids_for_indexing(limit_fraud: int, limit_nonfraud: int):
    """
    NOTE: LIMIT pulls the first rows DuckDB returns (not random).
    For a demo this is fine. If you want random sampling later, we can add it.
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    fraud = con.execute(
        f"SELECT TransactionID, isFraud FROM train_joined WHERE isFraud=1 LIMIT {int(limit_fraud)}"
    ).fetchall()

    nonfraud = con.execute(
        f"SELECT TransactionID, isFraud FROM train_joined WHERE isFraud=0 LIMIT {int(limit_nonfraud)}"
    ).fetchall()

    con.close()
    return fraud + nonfraud


def build_vector_store(limit_fraud: int, limit_nonfraud: int, batch_size: int = 128, rebuild: bool = True):
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(name=COLLECTION_NAME)

    embedder = SentenceTransformer(MODEL_NAME)

    rows = get_ids_for_indexing(limit_fraud, limit_nonfraud)

    ids_batch, docs_batch, meta_batch = [], [], []
    total_added = 0

    for (tid, label) in rows:
        tid_int = int(tid)
        label_int = int(label)

        txt = case_text(tid_int, label_int)

        ids_batch.append(str(tid_int))
        docs_batch.append(txt)
        meta_batch.append({"transaction_id": tid_int, "label": label_int})

        if len(ids_batch) >= batch_size:
            embeddings = embedder.encode(docs_batch, show_progress_bar=False).tolist()
            col.add(ids=ids_batch, documents=docs_batch, metadatas=meta_batch, embeddings=embeddings)
            total_added += len(ids_batch)

            ids_batch, docs_batch, meta_batch = [], [], []

            if total_added % (batch_size * 10) == 0:
                print(f"Indexed {total_added} cases...")

    # flush remainder
    if ids_batch:
        embeddings = embedder.encode(docs_batch, show_progress_bar=False).tolist()
        col.add(ids=ids_batch, documents=docs_batch, metadatas=meta_batch, embeddings=embeddings)
        total_added += len(ids_batch)

    print(f"Built vector store with {total_added} cases at: {CHROMA_DIR}")


def retrieve_similar(transaction_id: int, top_k: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    embedder = SentenceTransformer(MODEL_NAME)

    ev = build_evidence(transaction_id)
    query_parts = [
        f"amount={ev.get('amount')}",
        f"ProductCD={ev.get('ProductCD')}",
        f"card1={ev.get('card1')}",
        f"addr1={ev.get('addr1')}",
        f"P_emaildomain={ev.get('P_emaildomain')}",
        f"R_emaildomain={ev.get('R_emaildomain')}",
        f"DeviceType={ev.get('DeviceType')}",
        f"entity_key={ev.get('entity_key_used')}",
        f"v10={ev.get('entity_tx_count_10m')}",
        f"v1h={ev.get('entity_tx_count_1h')}",
        f"v24h={ev.get('entity_tx_count_24h')}",
        f"identity_missing_ratio={ev.get('identity_missing_ratio')}",
    ]
    query_text = " | ".join([p for p in query_parts if p is not None])

    query_emb = embedder.encode([query_text]).tolist()

    res = col.query(query_embeddings=query_emb, n_results=int(top_k) + 1)

    results = []
    for i in range(len(res["ids"][0])):
        rid = int(res["ids"][0][i])
        if rid == int(transaction_id):
            continue
        results.append(
            {
                "transaction_id": rid,
                "label": res["metadatas"][0][i].get("label"),
                "document": res["documents"][0][i],
                "distance": res["distances"][0][i],
            }
        )
        if len(results) >= top_k:
            break

    return {"query_text": query_text, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud", type=int, default=2000)
    parser.add_argument("--nonfraud", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--test_id", type=int, default=3213699)
    args = parser.parse_args()

    build_vector_store(args.fraud, args.nonfraud, batch_size=args.batch, rebuild=args.rebuild)

    out = retrieve_similar(args.test_id, top_k=5)
    print(json.dumps(out, indent=2))
