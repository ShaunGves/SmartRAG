# 🧠 SmartRAG — Production AI Assistant for Programmers

> QLoRA Fine-Tuning · Hybrid Search (BM25+Dense) · Cross-Encoder Reranking · ReAct Agent · Embedding Cache · Rate Limiting · Ablation Study · FastAPI · React

---

## 🎯 Use Case: AI Assistant for Programmers

Answers developer questions grounded in real documentation — no hallucinated APIs.

- *"How do I handle exceptions in asyncio?"*
- *"What is the GIL and how does it affect threading?"*
- *"How do I optimize a slow SQL query with N+1 problems?"*

---

## 🏗️ Architecture

```
User Query
    │
    ▼  Rate Limiter (Token Bucket, per-IP)
    │
    ▼  ReAct Agent (Thought → Tool → Observe → Repeat)
       Tools: vector_search | hybrid_search | code_executor | calculator | web_search
    │
    ▼  Hybrid Retrieval
       Dense (ChromaDB) + Sparse (BM25) → RRF Fusion → Cross-Encoder Reranker
    │
    ▼  Embedding Cache (LRU / Redis) — 350× faster on cache hits
    │
    ▼  Fine-Tuned LLM (Mistral-7B + QLoRA, programming domain)
    │
    ▼  RAGAS Evaluation + Ablation Study + MLflow
```

---

## 📊 Ablation Results (Before vs After)

| System | Keyword Coverage | Faithfulness | Failure Rate |
|---|---|---|---|
| A: Base Model + Dense RAG | 0.41 | 0.38 | 67% |
| B: Fine-Tuned + Dense RAG | 0.78 | 0.71 | 17% |
| C: Fine-Tuned + Hybrid | 0.84 | 0.76 | 17% |
| **D: Full Pipeline** | **0.84** | **0.76** | **17%** |

Fine-tuning improved keyword coverage **+90% relative**. Hybrid search captured exact API names that dense-only missed.

---

## ⚙️ System Design

| Decision | Choice | Reason |
|---|---|---|
| Hybrid fusion | RRF (α=0.7) | Robust to different score scales |
| Reranker | ms-marco-MiniLM-L-6-v2 | +27% MRR vs dense-only |
| Cache | LRU → Redis | 350× latency on hits |
| Rate limit | Token bucket | Handles burst traffic |
| HNSW tuning | ef=100, M=16 | Quality/latency tradeoff |
| Quantization | NF4 double-quant | 7B fits in 4GB VRAM |

---

## 📁 Files

```
smartrag/
├── config.py                  ← All configs (model, hybrid, cache, agent, system)
├── data/
│   ├── prepare_dataset.py     ← Generic dataset pipeline
│   └── code_dataset.py        ← Programming domain (18K Python QA pairs)
├── training/finetune.py       ← QLoRA fine-tuning
├── rag/
│   ├── vectorstore.py         ← ChromaDB dense retrieval
│   ├── hybrid_search.py       ← BM25 + dense + RRF fusion
│   ├── reranker.py            ← Cross-encoder reranking
│   ├── cache.py               ← Embedding cache (memory/disk/Redis)
│   ├── pipeline.py            ← Single-pass RAG
│   └── agent.py               ← ReAct multi-step agent
├── evaluation/
│   ├── evaluate.py            ← RAGAS metrics
│   └── ablation.py            ← Before/after comparison + failure analysis
├── api/
│   ├── app.py                 ← FastAPI
│   └── rate_limiter.py        ← Token bucket rate limiting
├── ui/app.py                  ← Streamlit UI
├── frontend/src/App.jsx       ← React frontend
└── tests/test_smartrag.py     ← 18 unit + API tests
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m data.code_dataset          # 1. Prepare programming dataset
python -m training.finetune          # 2. Fine-tune (needs GPU)
python -m evaluation.ablation        # 3. Run ablation study
uvicorn api.app:app --port 8000      # 4. Launch API
streamlit run ui/app.py              # 5. Launch UI
mlflow ui --port 5000                # 6. View experiments
```
