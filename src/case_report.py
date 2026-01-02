import json
from datetime import datetime, timezone

from src.scoring import score_transaction
from src.evidence import build_evidence


def decide_action(fraud_proba: float) -> str:
    """
    Simple policy thresholds (tune later):
      - >= 0.90: block
      - >= 0.70: review
      - else: allow
    """
    if fraud_proba >= 0.90:
        return "block"
    if fraud_proba >= 0.70:
        return "review"
    return "allow"


def evidence_bullets(ev: dict) -> list:
    bullets = []

    # Velocity
    if ev.get("entity_tx_count_10m") is not None:
        bullets.append(
            f"Entity velocity: {ev.get('entity_tx_count_10m')} tx in 10m, "
            f"{ev.get('entity_tx_count_1h')} tx in 1h, {ev.get('entity_tx_count_24h')} tx in 24h "
            f"(key: {ev.get('entity_key_used')})."
        )

    # Identity presence
    if ev.get("identity_present") is not None:
        bullets.append(
            f"Identity signals present: {ev.get('identity_present')} "
            f"(missing ratio: {round(ev.get('identity_missing_ratio', 0.0), 3)})."
        )

    # Amount
    if ev.get("amount") is not None:
        bullets.append(f"Transaction amount: {ev.get('amount')} (high: {ev.get('amount_high')}).")

    # Context fields
    ctx = []
    for k in ["ProductCD", "card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceType"]:
        v = ev.get(k)
        if v is not None:
            ctx.append(f"{k}={v}")
    if ctx:
        bullets.append("Context: " + ", ".join(ctx) + ".")

    return bullets


def narrative(action: str, fraud_proba: float, ev: dict) -> str:
    """
    Deterministic narrative (no LLM yet).
    """
    amt = ev.get("amount")
    vel10 = ev.get("entity_tx_count_10m")
    id_present = ev.get("identity_present")
    key_used = ev.get("entity_key_used")

    parts = []
    parts.append(f"Model flagged this transaction with fraud probability {fraud_proba:.3f}.")
    if amt is not None:
        parts.append(f"Amount is {amt:.2f}.")
    if vel10 is not None:
        parts.append(f"Recent activity for the entity shows {vel10} transaction(s) in the last 10 minutes (key: {key_used}).")
    if id_present is not None:
        parts.append(f"Identity coverage is {'present' if id_present else 'largely missing'}.")

    if action == "block":
        parts.append("Recommendation: block and initiate investigation due to high predicted risk.")
    elif action == "review":
        parts.append("Recommendation: send to manual review and request additional verification.")
    else:
        parts.append("Recommendation: allow, but monitor for repeated activity or pattern escalation.")

    return " ".join(parts)


def build_case_report(transaction_id: int) -> dict:
    s = score_transaction(transaction_id)
    ev = build_evidence(transaction_id)

    action = decide_action(s["fraud_proba"])

    report = {
        "case_id": f"CASE-{transaction_id}",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transaction_id": transaction_id,
        "fraud_proba": s["fraud_proba"],
        "recommended_action": action,
        "evidence": ev,
        "evidence_bullets": evidence_bullets(ev),
        "investigator_narrative": narrative(action, s["fraud_proba"], ev),
    }
    return report


if __name__ == "__main__":
    # Quick CLI test: change to any TransactionID you want
    test_id = 3213699
    rep = build_case_report(test_id)
    print(json.dumps(rep, indent=2))
