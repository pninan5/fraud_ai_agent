# Fraud AI Agent (Hybrid ML + Rules + Retrieval)

A practical fraud decisioning agent that **combines a trained ML model with rule based risk signals and precedent from similar historical transactions** (via vector search). The goal is not just a probability score, but an **actionable decision** with **traceable evidence**: `ALLOW`, `REVIEW`, or `BLOCK`.

---

## Why an "agent" instead of only an ML model?

A raw model score is useful, but fraud operations usually need more:

- **Consistent actions** (block / review / allow) rather than a number.
- **Explainability** for analysts and auditability for compliance.
- **Human in the loop** workflows: only send the right cases to review.
- **Precedent awareness**: similar cases may reveal patterns that a single score misses.

This project wraps the model into a policy layer + evidence bundle, so each decision is defensible and easy to debug.

---

## Data source

This project is based on the **IEEE-CIS Fraud Detection** dataset (hosted on Kaggle).

- Kaggle: IEEE-CIS Fraud Detection  
  (Download `train_transaction.csv` and `test_transaction.csv`.)

> The raw dataset files are very large, so they are **not committed to GitHub**. You download them locally and place them under `data/raw/`.

---

## What the system does

### Inputs
For a given `TransactionID`, the agent pulls:

1. **Model probability** from a baseline model (`fraud_proba`)
2. **Rule signals** (high amount, velocity, identity missing, email mismatch, etc.)
3. **Neighbor precedent** by searching similar historical cases in a Chroma vector DB

### Output
A JSON case report (and a Streamlit UI) containing:

- recommended action: `allow | review | block`
- probability + rule signal breakdown
- evidence fields (identity completeness, device info, entity velocity)
- similar historical cases and neighbor fraud rate

---

## How the decision policy works (high level)

The agent uses a guarded policy:

1. **Model first**: very high model score can directly block.
2. **Rules**: can push borderline cases to review.
3. **Neighbors**: can push `ALLOW → REVIEW`, and only push `REVIEW → BLOCK` when the model is already high and neighbor precedent is extremely strong.

This helps reduce false blocks while still catching high confidence fraud.

---

## Project structure

- `app.py` — Streamlit UI to view a case report
- `src/agent_orchestrator.py` — main orchestration: evidence → scoring → neighbors → decision → report
- `src/evidence.py` — extracts evidence features from the dataset/duckdb
- `src/scoring.py` — loads model and returns fraud probability
- `src/similar_cases.py` — builds / queries Chroma vector DB of historical cases
- `src/batch_eval.py` — evaluates the policy on a random batch of transactions

---

## Quickstart (local)

### 1) Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd fraud_ai_agent

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Download data from Kaggle and place files
Create:
```
data/raw/train_transaction.csv
data/raw/test_transaction.csv
```

### 4) Build the local DuckDB (optional but recommended)
```bash
python -m src.ingest_duckdb --raw_dir data/raw --out_db data/ieee.duckdb
```

### 5) Train the baseline model (creates a local model artifact)
```bash
python -m src.train_baseline --data_dir data/raw
```

### 6) Build the similar-cases vector DB (Chroma)
```bash
python -m src.similar_cases --data_dir data/raw --chroma_path <YOUR_CHROMA_PATH>
```

### 7) Run Streamlit
```bash
streamlit run app.py
```

### 8) Evaluate the policy
```bash
python -m src.batch_eval --data data/raw --n 2000 --seed 42 --chroma_path <YOUR_CHROMA_PATH> --debug
```

---

## Environment variables (optional)

- `FRAUD_AGENT_CHROMA_PATH` — where Chroma persists embeddings
- `FRAUD_AGENT_CHROMA_COLLECTION` — default collection name (e.g. `fraud_cases`)

---

## Deploy options

### Option A: GitHub only (recommended)
Yes. A clean GitHub repo with:
- source code
- README with steps
- requirements.txt
- sample commands
is usually enough for reviewers to reproduce the project.

### Option B: Streamlit Community Cloud
Works great for small demos, but this dataset is too large for typical free hosting limits. For Streamlit Cloud, consider:
- shipping a small sample dataset (few thousand rows) for demo mode, or
- using precomputed artifacts hosted in GitHub Releases / external storage.

### Option C: Docker (easy one command run)
Yes. This repo includes a Dockerfile. Users can build and run the app in a container.
They still need to provide the dataset (mounted volume) or use your small demo mode.

---

## Improvements suggestions
- **Probability calibration** (Platt/Isotonic) and threshold tuning by cost
- **Drift monitoring** and data quality checks
- **Active learning**: reviewer feedback loop to improve the model
- **Better retrieval**: smarter embeddings, per-segment indices, temporal weighting
- **Policy optimization**: learn a policy from labels + review outcomes
- **Unit tests + CI** (GitHub Actions) and more robust type checking
- **Security**: auth layer for the UI, PII redaction in reports

---
