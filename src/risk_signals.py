def compute_risk_signals(evidence: dict) -> dict:
    """
    Human-readable risk signals derived from evidence.
    These are simple and explainable. We will make them richer later.
    """
    signals = {
        "high_amount": False,
        "high_velocity_10m": False,
        "high_velocity_1h": False,
        "identity_missing_heavy": False,
        "email_mismatch": False,
        "signal_score": 0,
        "signal_reasons": [],
    }

    amt = evidence.get("amount")
    if amt is not None and amt >= 500:
        signals["high_amount"] = True
        signals["signal_score"] += 1
        signals["signal_reasons"].append("High transaction amount (>= 500).")

    v10 = evidence.get("entity_tx_count_10m")
    if v10 is not None and v10 >= 5:
        signals["high_velocity_10m"] = True
        signals["signal_score"] += 2
        signals["signal_reasons"].append("High entity velocity in 10 minutes (>= 5).")

    v1h = evidence.get("entity_tx_count_1h")
    if v1h is not None and v1h >= 10:
        signals["high_velocity_1h"] = True
        signals["signal_score"] += 2
        signals["signal_reasons"].append("High entity velocity in 1 hour (>= 10).")

    miss_ratio = evidence.get("identity_missing_ratio")
    if miss_ratio is not None and miss_ratio >= 0.95:
        signals["identity_missing_heavy"] = True
        signals["signal_score"] += 1
        signals["signal_reasons"].append("Identity signals mostly missing (>= 95%).")

    p_email = evidence.get("P_emaildomain")
    r_email = evidence.get("R_emaildomain")
    if p_email is not None and r_email is not None and str(p_email) != str(r_email):
        signals["email_mismatch"] = True
        signals["signal_score"] += 1
        signals["signal_reasons"].append("Purchaser email domain differs from recipient email domain.")

    return signals
