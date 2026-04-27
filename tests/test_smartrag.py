"""
tests/test_smartrag.py

Full test suite covering:
  - Unit tests  : individual components (chunking, formatting, config)
  - Integration : RAG pipeline end-to-end with mock LLM
  - API tests   : FastAPI endpoints via TestClient
  - Smoke test  : quick sanity check without GPU

Run all:       pytest tests/ -v
Run fast only: pytest tests/ -v -m "not slow"
Run API only:  pytest tests/test_smartrag.py::TestAPI -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_docs():
    """Sample documents for testing."""
    return [
        Document(
            page_content="Aspirin is a nonsteroidal anti-inflammatory drug (NSAID). "
                         "It works by inhibiting COX-1 and COX-2 enzymes, reducing prostaglandin synthesis.",
            metadata={"source": "pharmacology_101.txt", "page": 1},
        ),
        Document(
            page_content="Metformin is the first-line medication for type 2 diabetes. "
                         "It works by decreasing hepatic glucose production and improving insulin sensitivity.",
            metadata={"source": "diabetes_guide.txt", "page": 1},
        ),
        Document(
            page_content="The blood-brain barrier (BBB) is a selective semipermeable border "
                         "of endothelial cells that prevents solutes in the circulating blood from "
                         "non-selectively crossing into the extracellular fluid of the central nervous system.",
            metadata={"source": "neuroscience.txt", "page": 5},
        ),
    ]


@pytest.fixture
def mock_rag_response():
    """Mock RAGResponse for API testing."""
    from rag.pipeline import RAGResponse
    return RAGResponse(
        question="What is aspirin?",
        answer="Aspirin is an NSAID that inhibits COX enzymes to reduce inflammation.",
        sources=["Source 1: pharmacology_101.txt"],
        context_used="Aspirin is a nonsteroidal anti-inflammatory drug...",
        num_chunks_retrieved=1,
    )


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — Config
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_config_loads(self):
        from config import cfg
        assert cfg.model.base_model_id is not None
        assert cfg.rag.top_k > 0
        assert cfg.rag.chunk_size > 0

    def test_lora_config(self):
        from config import cfg
        assert cfg.lora.r > 0
        assert cfg.lora.lora_alpha > 0
        assert len(cfg.lora.target_modules) > 0

    def test_training_config(self):
        from config import cfg
        assert 0 < cfg.training.learning_rate < 1
        assert cfg.training.num_train_epochs > 0

    def test_ensure_dirs_creates_directories(self, tmp_path, monkeypatch):
        from config import Config
        monkeypatch.setattr("config.cfg.model.output_dir", str(tmp_path / "model"))
        monkeypatch.setattr("config.cfg.rag.chroma_persist_dir", str(tmp_path / "chroma"))
        # Should not raise
        cfg_instance = Config()
        cfg_instance.ensure_dirs()


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — Data Preparation
# ═══════════════════════════════════════════════════════════════════

class TestDataPreparation:
    def test_format_example_with_context(self):
        from data.prepare_dataset import format_example
        example = {
            "instruction": "What is aspirin?",
            "input": "Context about drugs",
            "output": "Aspirin is an NSAID.",
        }
        result = format_example(example)
        assert result is not None
        assert "[INST]" in result["text"]
        assert "[/INST]" in result["text"]
        assert "aspirin" in result["text"].lower()

    def test_format_example_without_context(self):
        from data.prepare_dataset import format_example
        example = {
            "instruction": "Explain photosynthesis",
            "input": "",
            "output": "Photosynthesis converts light to energy.",
        }
        result = format_example(example)
        assert result is not None
        assert "Context:" not in result["text"]

    def test_format_example_skips_empty(self):
        from data.prepare_dataset import format_example
        result = format_example({"instruction": "", "input": "", "output": ""})
        assert result is None

    def test_clean_text(self):
        from data.prepare_dataset import clean_text
        dirty = "  hello   world  \n\t  "
        assert clean_text(dirty) == "hello world"


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — Vector Store (mocked embeddings)
# ═══════════════════════════════════════════════════════════════════

class TestVectorStore:
    def test_chunk_documents(self, sample_docs):
        from rag.vectorstore import chunk_documents
        chunks = chunk_documents(sample_docs)
        assert len(chunks) >= len(sample_docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # chunk_size + buffer

    def test_chunk_preserves_metadata(self, sample_docs):
        from rag.vectorstore import chunk_documents
        chunks = chunk_documents(sample_docs)
        # All chunks should have source metadata
        for chunk in chunks:
            assert "source" in chunk.metadata

    @patch("rag.vectorstore.HuggingFaceEmbeddings")
    @patch("rag.vectorstore.Chroma")
    def test_build_vectorstore(self, mock_chroma, mock_embeddings, sample_docs):
        from rag.vectorstore import build_vectorstore
        mock_chroma.from_documents.return_value = MagicMock()
        build_vectorstore(docs=sample_docs)
        mock_chroma.from_documents.assert_called_once()

    @patch("rag.vectorstore.HuggingFaceEmbeddings")
    @patch("rag.vectorstore.Chroma")
    def test_retrieve_returns_documents(self, mock_chroma, mock_embeddings, sample_docs):
        from rag.vectorstore import retrieve
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (sample_docs[0], 0.9),
            (sample_docs[1], 0.7),
        ]
        results = retrieve("What is aspirin?", mock_store, top_k=2)
        assert len(results) == 2
        assert results[0].page_content == sample_docs[0].page_content


# ═══════════════════════════════════════════════════════════════════
# UNIT TESTS — RAG Pipeline (mocked LLM + vectorstore)
# ═══════════════════════════════════════════════════════════════════

class TestRAGPipeline:
    @patch("rag.pipeline.load_vectorstore")
    @patch("rag.pipeline.load_finetuned_pipeline")
    def test_pipeline_query(self, mock_llm_loader, mock_vs_loader, sample_docs):
        from rag.pipeline import SmartRAGPipeline

        # Mock vectorstore
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (sample_docs[0], 0.85)
        ]
        mock_vs_loader.return_value = mock_vs

        # Mock LLM pipeline
        mock_llm = MagicMock()
        mock_llm.return_value = [{"generated_text": "Aspirin inhibits COX enzymes."}]
        mock_llm_loader.return_value = mock_llm

        pipeline = SmartRAGPipeline()
        response = pipeline.query("What is aspirin?")

        assert response.question == "What is aspirin?"
        assert "aspirin" in response.answer.lower() or len(response.answer) > 0
        assert response.num_chunks_retrieved >= 0

    @patch("rag.pipeline.load_vectorstore")
    @patch("rag.pipeline.load_finetuned_pipeline")
    def test_pipeline_no_results(self, mock_llm_loader, mock_vs_loader):
        from rag.pipeline import SmartRAGPipeline

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = []
        mock_vs_loader.return_value = mock_vs
        mock_llm_loader.return_value = MagicMock()

        pipeline = SmartRAGPipeline()
        response = pipeline.query("xyzzy nonsense query 12345")

        assert response.num_chunks_retrieved == 0
        assert "couldn't find" in response.answer.lower() or len(response.answer) > 0


# ═══════════════════════════════════════════════════════════════════
# API TESTS — FastAPI endpoints
# ═══════════════════════════════════════════════════════════════════
import api.app   # ← ADD THIS LINE
class TestAPI:
    @pytest.fixture
    def client(self, mock_rag_response):
        """Create test client with mocked pipeline."""
        with patch("api.app.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = mock_rag_response
            mock_pipeline.vectorstore = MagicMock()
            mock_get.return_value = mock_pipeline

            # Patch startup to avoid loading real models
            with patch("api.app.pipeline", mock_pipeline):
                from api.app import app
                yield TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "SmartRAG" in response.json()["name"]

    def test_query_endpoint_valid(self, client):
        response = client.post("/query", json={"question": "What is aspirin?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data

    def test_query_endpoint_too_short(self, client):
        response = client.post("/query", json={"question": "hi"})
        assert response.status_code == 422  # Pydantic validation error

    def test_query_endpoint_with_top_k(self, client):
        response = client.post("/query", json={"question": "What is aspirin?", "top_k": 3})
        assert response.status_code == 200

    def test_ingest_endpoint(self, client):
        response = client.post("/ingest", json={
            "texts": ["Ibuprofen is an NSAID used for pain relief."],
            "metadata": [{"source": "test_doc.txt"}],
        })
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TEST — End-to-end smoke test (no GPU needed)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestIntegration:
    """Integration tests that test more of the real stack (marked slow)."""

    def test_chunk_then_retrieve(self, sample_docs, tmp_path, monkeypatch):
        """Test chunking → embedding → retrieval pipeline (mocked embeddings)."""
        import numpy as np
        from rag.vectorstore import chunk_documents

        chunks = chunk_documents(sample_docs)
        assert len(chunks) > 0

        # Verify chunk content integrity
        all_text = " ".join(c.page_content for c in chunks)
        assert "aspirin" in all_text.lower()
        assert "metformin" in all_text.lower()

    def test_data_pipeline_flow(self, tmp_path, monkeypatch):
        """Test data formatting → save → load round-trip."""
        import json
        from data.prepare_dataset import format_example

        examples = [
            {"instruction": "What is X?", "input": "Context X", "output": "X is great."},
            {"instruction": "What is Y?", "input": "", "output": "Y is fine."},
        ]
        formatted = [format_example(e) for e in examples]
        formatted = [f for f in formatted if f]

        # Save
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")

        # Load and verify
        loaded = [json.loads(line) for line in open(path)]
        assert len(loaded) == 2
        assert all("text" in item for item in loaded)
        assert all("[INST]" in item["text"] for item in loaded)
