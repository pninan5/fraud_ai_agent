import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from .evidence import build_evidence
    from .scoring import score_transaction
except Exception:
    from src.evidence import build_evidence
    from src.scoring import score_transaction


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_score_to_proba(score_out: Any) -> float:
    """
    score_transaction may return:
      - float probability
      - dict with keys like fraud_proba, proba, probability, score
    """
    if isinstance(score_out, dict):
        for k in ("fraud_proba", "proba", "probability", "score"):
            if k in score_out and score_out[k] is not None:
                return float(score_out[k])
        raise ValueError(
            f"score_transaction returned dict but no usable probability key: {list(score_out.keys())}"
        )
    return float(score_out)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _build_query_text(evidence: Dict[str, Any]) -> str:
    parts = [
        f"amount={evidence.get('amount')}",
        f"ProductCD={evidence.get('ProductCD')}",
        f"card1={evidence.get('card1')}",
        f"addr1={evidence.get('addr1')}",
        f"P_emaildomain={evidence.get('P_emaildomain')}",
        f"R_emaildomain={evidence.get('R_emaildomain')}",
        f"DeviceType={evidence.get('DeviceType')}",
        f"entity_key={evidence.get('entity_key_used')}",
        f"v10={evidence.get('entity_tx_count_10m')}",
        f"v1h={evidence.get('entity_tx_count_1h')}",
        f"v24h={evidence.get('entity_tx_count_24h')}",
        f"identity_missing_ratio={evidence.get('identity_missing_ratio')}",
    ]
    return " | ".join(parts)


@dataclass
class SignalResult:
    high_amount: bool
    high_velocity_10m: bool
    high_velocity_1h: bool
    identity_missing_heavy: bool
    email_mismatch: bool
    signal_score: int
    signal_reasons: List[str]


def _compute_signals(evidence: Dict[str, Any]) -> SignalResult:
    reasons: List[str] = []

    amount = _safe_float(evidence.get("amount"), default=0.0) or 0.0
    v10 = int(evidence.get("entity_tx_count_10m") or 0)
    v1h = int(evidence.get("entity_tx_count_1h") or 0)
    miss_ratio = _safe_float(evidence.get("identity_missing_ratio"), default=0.0) or 0.0

    p_email = evidence.get("P_emaildomain")
    r_email = evidence.get("R_emaildomain")

    high_amount = bool(evidence.get("amount_high")) or amount >= 1000.0
    if high_amount:
        reasons.append("High amount threshold triggered.")

    high_velocity_10m = v10 >= 3
    if high_velocity_10m:
        reasons.append("High velocity in 10m (>= 3).")

    high_velocity_1h = v1h >= 5
    if high_velocity_1h:
        reasons.append("High velocity in 1h (>= 5).")

    identity_missing_heavy = miss_ratio >= 0.95
    if identity_missing_heavy:
        reasons.append("Identity signals mostly missing (>= 95%).")

    email_mismatch = (
        p_email is not None
        and r_email is not None
        and str(p_email) != str(r_email)
    )
    if email_mismatch:
        reasons.append("P and R email domain mismatch.")

    signal_score = 0
    signal_score += 1 if high_amount else 0
    signal_score += 1 if high_velocity_10m else 0
    signal_score += 1 if high_velocity_1h else 0
    signal_score += 1 if identity_missing_heavy else 0
    signal_score += 1 if email_mismatch else 0

    return SignalResult(
        high_amount=high_amount,
        high_velocity_10m=high_velocity_10m,
        high_velocity_1h=high_velocity_1h,
        identity_missing_heavy=identity_missing_heavy,
        email_mismatch=email_mismatch,
        signal_score=signal_score,
        signal_reasons=reasons,
    )


def _get_chroma_path() -> str:
    env_path = os.environ.get("FRAUD_AGENT_CHROMA_PATH")
    if env_path:
        return env_path

    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        local_appdata = os.path.expanduser("~")

    return os.path.join(local_appdata, "fraud_ai_agent_chroma")


def _get_chroma_collection_name() -> str:
    return os.environ.get("FRAUD_AGENT_CHROMA_COLLECTION", "fraud_cases")


def _retrieve_similar_cases(
    query_text: str,
    top_k: int,
    chroma_path: str,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not os.path.exists(chroma_path):
        return []

    try:
        import chromadb
    except Exception:
        return []

    client = chromadb.PersistentClient(path=chroma_path)

    col_name = collection_name or _get_chroma_collection_name()

    try:
        col = client.get_collection(name=col_name)
    except Exception:
        return []

    res = col.query(query_texts=[query_text], n_results=int(top_k))

    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        out.append(
            {
                "transaction_id": meta.get("transaction_id"),
                "label": meta.get("label"),
                "document": docs[i],
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out


def _neighbor_stats(neighbors: List[Dict[str, Any]], max_distance: float) -> Dict[str, Any]:
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        fraud = 0
        count = 0
        for r in rows:
            lbl = r.get("label")
            if lbl is None:
                continue
            try:
                lbl_int = int(lbl)
            except Exception:
                continue
            count += 1
            fraud += 1 if lbl_int == 1 else 0
        rate = (fraud / count) if count > 0 else None
        return {"count": count, "fraud": fraud, "rate": rate}

    all_stats = summarize(neighbors)

    close_neighbors: List[Dict[str, Any]] = []
    for r in neighbors:
        dist = r.get("distance")
        try:
            dist_f = float(dist)
        except Exception:
            continue
        if dist_f <= max_distance:
            close_neighbors.append(r)

    close_stats = summarize(close_neighbors)

    ui_rate = close_stats["rate"]
    if ui_rate is None:
        ui_rate = all_stats["rate"]

    return {
        "max_distance_used": float(max_distance),
        "all_neighbors": all_stats,
        "close_neighbors": close_stats,
        "fraud_rate": ui_rate,
        "close_rate": close_stats["rate"],
        "close_count": close_stats["count"],
    }


def _decide_action(
    proba: float,
    signals: SignalResult,
    close_rate: Optional[float],
    close_count: int,
) -> str:
    """
    POLICY v9 (tighten ALLOW band)

    Goals:
      - Never ALLOW high model probabilities (fix 0.73 to 0.74 slipping into ALLOW)
      - Use neighbor precedent to prevent medium proba fraud leaking into ALLOW
      - Keep BLOCK mostly model driven

    BLOCK:
      - proba >= 0.97
      - OR (proba >= 0.92 AND close_rate >= 0.90 AND close_count >= 5)

    REVIEW:
      - proba >= 0.70  (hard rail)
      - OR (proba >= 0.60 AND close_rate >= 0.90 AND close_count >= 5)  (neighbor override)
      - OR (proba >= 0.60 AND strong_rules)

    ALLOW:
      - everything else

    Notes:
      - identity_missing_heavy alone does not trigger review
      - neighbors alone do not trigger review unless model is at least 0.60
    """

    score = int(getattr(signals, "signal_score", 0))
    identity_missing_heavy = bool(getattr(signals, "identity_missing_heavy", False))
    email_mismatch = bool(getattr(signals, "email_mismatch", False))
    velocity_risk = bool(getattr(signals, "high_velocity_10m", False)) or bool(
        getattr(signals, "high_velocity_1h", False)
    )
    high_amount = bool(getattr(signals, "high_amount", False))

    # Remove identity_missing from "strong rules" unless combined with something else
    score_wo_identity = score - (1 if identity_missing_heavy else 0)

    strong_rules = (score_wo_identity >= 2) or email_mismatch or velocity_risk or high_amount

    neighbor_ok = (close_rate is not None) and (int(close_count) >= 5)

    # BLOCK
    if proba >= 0.97:
        return "block"
    if neighbor_ok and proba >= 0.92 and close_rate >= 0.90:
        return "block"

    # REVIEW hard rail (prevents 0.73 to 0.74 from being allowed)
    if proba >= 0.70:
        return "review"

    # REVIEW neighbor override (prevents cases like proba 0.696 with close_rate 1.0 from being allowed)
    if neighbor_ok and proba >= 0.60 and close_rate >= 0.90:
        return "review"

    # REVIEW mid band plus strong rules
    if proba >= 0.60 and strong_rules:
        return "review"

    return "allow"



def build_case_report(
    transaction_id: int,
    top_k: int = 5,
    max_distance: float = 0.25,
    chroma_path: Optional[str] = None,
    chroma_collection: Optional[str] = None,
) -> Dict[str, Any]:
    tx_id_int = int(transaction_id)

    evidence = build_evidence(tx_id_int)

    score_out = score_transaction(tx_id_int)
    proba = _coerce_score_to_proba(score_out)

    sig = _compute_signals(evidence)
    signals_dict = {
        "high_amount": sig.high_amount,
        "high_velocity_10m": sig.high_velocity_10m,
        "high_velocity_1h": sig.high_velocity_1h,
        "identity_missing_heavy": sig.identity_missing_heavy,
        "email_mismatch": sig.email_mismatch,
        "signal_score": sig.signal_score,
        "signal_reasons": sig.signal_reasons,
    }

    query_text = _build_query_text(evidence)

    chroma_path_final = chroma_path or _get_chroma_path()
    chroma_collection_final = chroma_collection or _get_chroma_collection_name()

    similar_cases = _retrieve_similar_cases(
        query_text=query_text,
        top_k=int(top_k),
        chroma_path=chroma_path_final,
        collection_name=chroma_collection_final,
    )

    nstats = _neighbor_stats(similar_cases, max_distance=float(max_distance))
    close_rate = nstats.get("close_rate")
    close_count = int(nstats.get("close_count") or 0)

    action = _decide_action(proba, sig, close_rate=close_rate, close_count=close_count)

    precedent_summary = ""
    close = nstats.get("close_neighbors", {})
    if close.get("count", 0) > 0 and close.get("rate") is not None:
        precedent_summary = (
            f"Close neighbors (distance <= {float(max_distance)}): "
            f"{close.get('fraud')}/{close.get('count')} fraud (rate {close.get('rate'):.2f})."
        )
    else:
        alln = nstats.get("all_neighbors", {})
        if alln.get("count", 0) > 0 and alln.get("rate") is not None:
            precedent_summary = (
                f"Among the {alln.get('count')} most similar historical cases, "
                f"{alln.get('fraud')} were fraud (fraud rate {alln.get('rate'):.2f})."
            )

    report = {
        "case_id": f"CASE-{tx_id_int}",
        "created_at": utc_now_iso_z(),
        "transaction_id": tx_id_int,
        "fraud_proba": float(proba),
        "recommended_action": action,
        "signals": signals_dict,
        "evidence": evidence,
        "similar_cases": similar_cases,
        "neighbor_stats": nstats,
        "neighbor_fraud_rate_close": nstats.get("close_rate"),
        "neighbor_close_count": close_count,
        "neighbor_fraud_rate_all": nstats.get("all_neighbors", {}).get("rate"),
        "neighbor_all_count": nstats.get("all_neighbors", {}).get("count"),
        "explanation": {
            "decision_logic": "Action is based on model probability, rule signals, and close neighbor precedent.",
            "signal_reasons": signals_dict.get("signal_reasons", []),
            "precedent_summary": precedent_summary,
            "chroma_path": chroma_path_final,
            "chroma_collection": chroma_collection_final,
        },
    }
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=int, default=3213699)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_distance", type=float, default=0.25)
    parser.add_argument("--chroma_path", type=str, default=None)
    parser.add_argument("--chroma_collection", type=str, default=None)
    args = parser.parse_args()

    report = build_case_report(
        transaction_id=args.tx,
        top_k=args.top_k,
        max_distance=args.max_distance,
        chroma_path=args.chroma_path,
        chroma_collection=args.chroma_collection,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
