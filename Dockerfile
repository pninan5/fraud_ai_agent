# Minimal Streamlit container for this project
# Build:  docker build -t fraud-ai-agent .
# Run:    docker run --rm -p 8501:8501 fraud-ai-agent
#
# Data note:
# This repo intentionally does NOT include the IEEE-CIS CSVs/duckdb/parquet because they are huge.
# To run with real data, mount your local data folder into the container, e.g.:
#   docker run --rm -p 8501:8501 ^
#     -v "%cd%/data:/app/data" ^
#     fraud-ai-agent

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps:
# - libgomp1 is commonly required for LightGBM wheels
# - build-essential helps if a wheel needs compilation
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libgomp1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project files
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
