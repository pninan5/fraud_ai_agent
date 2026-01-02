import json
import streamlit as st

from src.agent_orchestrator import build_case_report

st.set_page_config(page_title="Fraud AI Agent", layout="wide")

st.title("Fraud AI Agent Investigator Console")
st.caption("Model and Evidence Tools and Similar Case Retrieval using Chroma")

with st.sidebar:
    st.header("Inputs")
    tx_id = st.number_input("TransactionID", min_value=1, value=3213699, step=1)
    top_k = st.slider("Similar cases top K", min_value=3, max_value=15, value=5, step=1)
    run_btn = st.button("Run Investigation", type="primary")

def _format_rate(x):
    if x is None:
        return "N A"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return  "N A"

if run_btn:
    with st.spinner("Running agent..."):
        report = build_case_report(int(tx_id), top_k=int(top_k))

    stats = report.get("neighbor_stats", {})

    # Support both old and new neighbor_stats structures
    close_rate = None
    all_rate = None

    if isinstance(stats, dict):
        if "fraud_rate" in stats:
            all_rate = stats.get("fraud_rate")
            close_rate = stats.get("fraud_rate")
        else:
            close_neighbors = stats.get("close_neighbors", {})
            all_neighbors = stats.get("all_neighbors", {})
            if isinstance(close_neighbors, dict):
                close_rate = close_neighbors.get("rate")
            if isinstance(all_neighbors, dict):
                all_rate = all_neighbors.get("rate")

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{report['fraud_proba']:.3f}")
    c2.metric("Recommended Action", report["recommended_action"].upper())
    c3.metric("Neighbor Fraud Rate close", _format_rate(close_rate))
    c4.metric("Neighbor Fraud Rate all", _format_rate(all_rate))

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Evidence")
        st.json(report.get("evidence", {}))

        st.subheader("Signals")
        st.json(report.get("signals", {}))

    with right:
        st.subheader("Similar Cases")
        rows = report.get("similar_cases", [])
        if rows:
            table_rows = []
            for r in rows:
                table_rows.append(
                    {
                        "transaction_id": r.get("transaction_id"),
                        "label": r.get("label"),
                        "distance": round(float(r.get("distance", 0.0)), 4) if r.get("distance") is not None else None,
                        "document": r.get("document"),
                    }
                )
            st.dataframe(table_rows, use_container_width=True, height=380)
        else:
            st.info("No similar cases found. Build the vector store first using python -m src.similar_cases")

        st.subheader("Explanation")
        explanation = report.get("explanation", {})
        st.write(explanation.get("precedent_summary", ""))

        signal_reasons = report.get("signals", {}).get("signal_reasons", [])
        if signal_reasons:
            st.write("Rule reasons:")
            for x in signal_reasons:
                st.write("• " + str(x))

    st.divider()

    st.subheader("Download Report")
    report_json = json.dumps(report, indent=2)
    st.download_button(
        label="Download JSON",
        data=report_json,
        file_name=f"case_{tx_id}.json",
        mime="application/json",
    )

else:
    st.info("Enter a TransactionID and click Run Investigation.")
