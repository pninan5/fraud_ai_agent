import argparse
import os
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

try:
    from .agent_orchestrator import build_case_report
except Exception:
    from src.agent_orchestrator import build_case_report


def resolve_data_path(data_arg: Optional[str]) -> str:
    if data_arg:
        if os.path.isdir(data_arg):
            candidate = os.path.join(data_arg, "train_transaction.csv")
            if os.path.exists(candidate):
                return candidate
            csvs = [x for x in os.listdir(data_arg) if x.lower().endswith(".csv")]
            if csvs:
                return os.path.join(data_arg, csvs[0])
            raise FileNotFoundError(f"No CSV found inside directory: {data_arg}")

        if os.path.exists(data_arg):
            return data_arg

        raise FileNotFoundError(f"--data path not found: {data_arg}")

    candidates = [
        os.environ.get("FRAUD_AGENT_DATA_PATH"),
        "data/raw/train_transaction.csv",
        "data/train_transaction.csv",
        "train_transaction.csv",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find a dataset file. Pass --data folder_or_csv or set FRAUD_AGENT_DATA_PATH."
    )


def compute_bucket_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets = {"allow": [], "review": [], "block": []}
    for r in rows:
        a = str(r["action"]).lower()
        if a not in buckets:
            continue
        buckets[a].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for b, items in buckets.items():
        n = len(items)
        fraud = sum(1 for x in items if int(x["label"]) == 1) if n else 0
        rate = (fraud / n) if n else None
        avg_proba = (sum(float(x["proba"]) for x in items) / n) if n else None
        out[b] = {
            "count": n,
            "fraud_count": fraud,
            "fraud_rate": rate,
            "avg_model_proba": avg_proba,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Folder or CSV containing TransactionID and label")
    parser.add_argument("--id_col", type=str, default="TransactionID")
    parser.add_argument("--label_col", type=str, default="isFraud")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_distance", type=float, default=0.25)
    parser.add_argument("--chroma_path", type=str, default=None)
    parser.add_argument("--chroma_collection", type=str, default=None)

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    df = pd.read_csv(data_path)

    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in columns: {list(df.columns)[:30]} ...")
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in columns: {list(df.columns)[:30]} ...")

    base = df[[args.id_col, args.label_col]].dropna()
    base[args.id_col] = base[args.id_col].astype(int)
    base[args.label_col] = base[args.label_col].astype(int)

    n = min(args.n, len(base))
    sample = base.sample(n=n, random_state=args.seed)

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[int, str]] = []

    for tx_id, lbl in zip(sample[args.id_col].tolist(), sample[args.label_col].tolist()):
        try:
            report = build_case_report(
                transaction_id=int(tx_id),
                top_k=int(args.top_k),
                max_distance=float(args.max_distance),
                chroma_path=args.chroma_path,
                chroma_collection=args.chroma_collection,
            )

            action = str(report.get("recommended_action", "review")).lower()
            proba = float(report.get("fraud_proba", 0.0))

            neighbor_rate = report.get("neighbor_fraud_rate_close")
            close_count = report.get("neighbor_close_count")

            sig_score = report.get("signals", {}).get("signal_score", 0)
            identity_missing_heavy = report.get("signals", {}).get("identity_missing_heavy", False)
            email_mismatch = report.get("signals", {}).get("email_mismatch", False)

            results.append(
                {
                    "tx_id": int(tx_id),
                    "label": int(lbl),
                    "action": action,
                    "proba": proba,
                    "sig_score": int(sig_score or 0),
                    "identity_missing_heavy": bool(identity_missing_heavy),
                    "email_mismatch": bool(email_mismatch),
                    "neighbor_rate": float(neighbor_rate) if neighbor_rate is not None else -1.0,
                    "close_count": int(close_count or 0),
                }
            )

        except Exception as e:
            failures.append((int(tx_id), str(e)))

    stats = compute_bucket_stats(results)

    print("\n=== Batch Policy Evaluation ===")
    print(f"Data: {data_path}")
    print(f"Sample size requested: {args.n} | evaluated: {len(results)} | failures: {len(failures)}")
    print(
        f"top_k={args.top_k} | max_distance={args.max_distance} | "
        f"chroma_path={args.chroma_path} | chroma_collection={args.chroma_collection}\n"
    )

    for b in ["allow", "review", "block"]:
        s = stats[b]
        rate_str = "N/A" if s["fraud_rate"] is None else f"{s['fraud_rate']:.4f}"
        proba_str = "N/A" if s["avg_model_proba"] is None else f"{s['avg_model_proba']:.4f}"
        print(
            f"{b.upper():>5} | "
            f"count={s['count']:>4} | "
            f"fraud_count={s['fraud_count']:>4} | "
            f"fraud_rate={rate_str:<8} | "
            f"avg_model_proba={proba_str}"
        )

    if results:
        overall_rate = sum(r["label"] for r in results) / len(results)
        print(f"\nOverall sample fraud rate: {overall_rate:.4f}")

    if args.debug and results:
        rdf = pd.DataFrame(results)

        print("\n=== DEBUG: Example rows per bucket ===")
        for bucket in ["allow", "review", "block"]:
            sub = rdf[rdf["action"] == bucket].sort_values("proba", ascending=False).head(10)
            print(f"\n-- {bucket.upper()} examples (top 10 by proba) --")
            if len(sub) == 0:
                print("(none)")
            else:
                cols = [
                    "tx_id",
                    "label",
                    "proba",
                    "sig_score",
                    "identity_missing_heavy",
                    "email_mismatch",
                    "neighbor_rate",
                    "close_count",
                ]
                print(sub[cols].to_string(index=False))

        block_only = rdf[rdf["action"] == "block"].copy()
        if len(block_only) > 0:
            print("\n=== DEBUG: BLOCK counts by neighbor_rate bucket ===")
            block_only["neighbor_rate_binned"] = pd.cut(
                block_only["neighbor_rate"],
                bins=[-2.001, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                include_lowest=True,
            )
            print(block_only.groupby("neighbor_rate_binned").size().to_string())

            print("\n=== DEBUG: How many blocked had identity_missing_heavy True? ===")
            print(block_only.groupby("identity_missing_heavy").size().to_string())

    if failures:
        print("\n=== Failures (first 10) ===")
        for tx_id, msg in failures[:10]:
            print(f"tx_id={tx_id} | {msg}")


if __name__ == "__main__":
    main()
