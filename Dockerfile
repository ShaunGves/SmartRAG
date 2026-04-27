FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Install dependencies ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ─── Copy source ───────────────────────────────────────────────────
COPY . .

# Create artifact dirs
RUN mkdir -p artifacts/finetuned_model artifacts/chroma_db \
             artifacts/raw_data artifacts/processed_data artifacts/eval_results

# ─── Runtime ──────────────────────────────────────────────────────
EXPOSE 7860

CMD ["python", "hf_spaces_app.py"]
