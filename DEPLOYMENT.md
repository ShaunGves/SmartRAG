# 🚀 Deployment & Testing Guide — SmartRAG

## Overview

| Method | Best For | GPU Needed | Cost |
|---|---|---|---|
| Local (dev) | Development & testing | Optional | Free |
| Docker | Production-like local | Optional | Free |
| HuggingFace Spaces | Live demo (portfolio) | Yes (ZeroGPU) | Free |
| AWS / GCP / Azure | Production | Yes | Paid |
| Google Colab | Fine-tuning + quick test | Yes (T4 free) | Free |

---

## 1. 🖥️ Local Development (Fastest to Start)

### Without GPU (CPU only — for testing pipeline logic)
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/smartrag.git
cd smartrag

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Use a smaller model for CPU testing (swap in config.py)
# base_model_id = "microsoft/phi-2"   ← 2.7B, runs on CPU
```

### Step-by-step run order
```bash
# 1. Prepare data
python -m data.prepare_dataset

# 2. (Optional) Fine-tune — skip on CPU, use pre-trained adapter
python -m training.finetune       # Needs GPU with 16GB+ VRAM

# 3. Index your documents
python -c "
from rag.vectorstore import build_vectorstore
build_vectorstore(sources=['your_docs/'])   # point to a folder of .txt/.pdf
"

# 4. Run tests (no GPU needed)
pytest tests/ -v -m 'not slow'

# 5. Start API
uvicorn api.app:app --reload --port 8000

# 6. Start UI (new terminal)
streamlit run ui/app.py

# 7. Start MLflow UI (new terminal)
mlflow ui --port 5000
```

### Access points
| Service | URL |
|---|---|
| API Docs (Swagger) | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |
| Streamlit UI | http://localhost:8501 |
| MLflow Dashboard | http://localhost:5000 |

---

## 2. 🐳 Docker (Production-Like)

### Prerequisites
- Docker + Docker Compose installed
- NVIDIA Docker runtime (for GPU): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Run full stack
```bash
# Build and start all services
docker compose up --build

# Or run in background
docker compose up -d --build

# View logs
docker compose logs -f api
docker compose logs -f ui

# Stop everything
docker compose down
```

### Manual Docker (API only)
```bash
# Build
docker build -t smartrag-api .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  smartrag-api

# Run CPU only
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  smartrag-api
```

### Test the Docker API
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is aspirin?", "top_k": 3}'

# Ingest new document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Ibuprofen is an NSAID."], "metadata": [{"source": "test"}]}'
```

---

## 3. 🤗 HuggingFace Spaces (Free Live Demo — Best for Portfolio)

HuggingFace Spaces gives you a **free public URL** with GPU access (ZeroGPU).
This is the single best way to showcase this project to employers.

### Setup Steps

```bash
# 1. Install HF CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Create a new Space
huggingface-cli repo create smartrag --type space --space_sdk docker

# 4. Add remote and push
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/smartrag
git push space main
```

### Required files for HF Spaces
Create `README.md` at root with this header:
```yaml
---
title: SmartRAG
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---
```

Your Space will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/smartrag`

---

## 4. ☁️ Google Colab (Fine-Tuning + Testing — Free T4 GPU)

Open in Colab and run:

```python
# Cell 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/smartrag.git
%cd smartrag

# Cell 2: Install
!pip install -r requirements.txt

# Cell 3: Prepare data
!python -m data.prepare_dataset

# Cell 4: Fine-tune (T4 GPU = ~2-3 hours for 3 epochs)
!python -m training.finetune

# Cell 5: Build vector store
!python -c "
from rag.vectorstore import build_vectorstore
build_vectorstore(sources=['sample_docs/'])
"

# Cell 6: Quick API test (background)
import subprocess, time
proc = subprocess.Popen(['uvicorn', 'api.app:app', '--port', '8000'])
time.sleep(30)  # Wait for model to load

# Cell 7: Test query
import requests
r = requests.post('http://localhost:8000/query',
                  json={'question': 'What is aspirin?'})
print(r.json())
```

---

## 5. 🧪 Testing Guide

### Run all unit tests (no GPU)
```bash
pytest tests/ -v -m "not slow"
```

### Run with coverage report
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html    # View coverage report
```

### Run specific test class
```bash
pytest tests/test_smartrag.py::TestAPI -v           # API tests only
pytest tests/test_smartrag.py::TestConfig -v        # Config tests
pytest tests/test_smartrag.py::TestVectorStore -v   # RAG tests
```

### Manual API testing (HTTPie or curl)

```bash
# Install HTTPie (nicer than curl)
pip install httpie

# Health check
http GET localhost:8000/health

# Query the RAG system
http POST localhost:8000/query question="What is the mechanism of aspirin?"

# Query with custom top-k
http POST localhost:8000/query question="What is aspirin?" top_k=5

# Ingest a new document
http POST localhost:8000/ingest \
  texts:='["Paracetamol reduces fever by acting on the hypothalamus."]' \
  metadata:='[{"source": "pharmacology.txt"}]'
```

### Load testing with Locust
```bash
pip install locust

# Create locustfile.py:
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class SmartRAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def query(self):
        self.client.post("/query", json={"question": "What is aspirin?"})

    @task(1)
    def health(self):
        self.client.get("/health")
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 → set users=10, spawn rate=2
```

### Run RAGAS evaluation
```bash
python -m evaluation.evaluate
# View results at http://localhost:5000 (MLflow)
```

---

## 6. 🔧 Common Issues & Fixes

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 in config.py |
| `bitsandbytes not found` | `pip install bitsandbytes --upgrade` |
| `No module named rag` | Run from project root: `cd smartrag && python -m ...` |
| `ChromaDB collection not found` | Run `build_vectorstore()` first |
| `Model not found at adapter_path` | Run `python -m training.finetune` first |
| `Port 8000 already in use` | `lsof -ti:8000 \| xargs kill -9` |

---

## 7. 📊 Expected Test Results

```
tests/test_smartrag.py::TestConfig::test_config_loads              PASSED
tests/test_smartrag.py::TestConfig::test_lora_config               PASSED
tests/test_smartrag.py::TestConfig::test_training_config           PASSED
tests/test_smartrag.py::TestDataPreparation::test_format_example_with_context    PASSED
tests/test_smartrag.py::TestDataPreparation::test_format_example_without_context PASSED
tests/test_smartrag.py::TestDataPreparation::test_format_example_skips_empty     PASSED
tests/test_smartrag.py::TestDataPreparation::test_clean_text       PASSED
tests/test_smartrag.py::TestVectorStore::test_chunk_documents      PASSED
tests/test_smartrag.py::TestVectorStore::test_chunk_preserves_metadata  PASSED
tests/test_smartrag.py::TestVectorStore::test_build_vectorstore    PASSED
tests/test_smartrag.py::TestVectorStore::test_retrieve_returns_documents PASSED
tests/test_smartrag.py::TestRAGPipeline::test_pipeline_query       PASSED
tests/test_smartrag.py::TestRAGPipeline::test_pipeline_no_results  PASSED
tests/test_smartrag.py::TestAPI::test_health_endpoint              PASSED
tests/test_smartrag.py::TestAPI::test_root_endpoint                PASSED
tests/test_smartrag.py::TestAPI::test_query_endpoint_valid         PASSED
tests/test_smartrag.py::TestAPI::test_query_endpoint_too_short     PASSED
tests/test_smartrag.py::TestAPI::test_ingest_endpoint              PASSED

18 passed in 4.32s
```
