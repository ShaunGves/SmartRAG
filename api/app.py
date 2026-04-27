"""
api/app.py

Production-ready FastAPI REST API for the SmartRAG pipeline.

Features:
  - /query endpoint with structured JSON response
  - /health endpoint for monitoring
  - /ingest endpoint to add new documents
  - Request validation with Pydantic
  - Async support + error handling
  - CORS middleware for frontend integration

Run: uvicorn api.app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).parent.parent))
from rag.pipeline import get_pipeline, RAGResponse
from rag.vectorstore import build_vectorstore
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="SmartRAG API",
    description="Production LLM RAG system with fine-tuned Mistral-7B",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The question to answer")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    num_chunks_retrieved: int
    latency_ms: float


class IngestRequest(BaseModel):
    texts: List[str] = Field(..., description="List of document texts to add to the knowledge base")
    metadata: Optional[List[dict]] = Field(None, description="Metadata for each document")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    version: str


# ─── State ────────────────────────────────────────────────────────
pipeline = None
startup_error = None


# ─── Lifecycle ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global pipeline, startup_error
    try:
        log.info("Loading SmartRAG pipeline...")
        pipeline = get_pipeline()
        log.info("✅ Pipeline ready.")
    except Exception as e:
        startup_error = str(e)
        log.error(f"Failed to load pipeline: {e}")


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring/load balancers."""
    return HealthResponse(
        status="healthy" if pipeline else "degraded",
        model_loaded=pipeline is not None,
        vectorstore_loaded=pipeline is not None and pipeline.vectorstore is not None,
        version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.

    Returns the generated answer along with source documents
    and performance metrics.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not loaded. Error: {startup_error}"
        )

    start = time.perf_counter()

    try:
        response: RAGResponse = pipeline.query(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as e:
        log.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        num_chunks_retrieved=response.num_chunks_retrieved,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Add new documents to the vector store knowledge base.
    Ingestion runs in the background to avoid blocking.
    """
    docs = []
    for i, text in enumerate(request.texts):
        meta = request.metadata[i] if request.metadata else {}
        docs.append(Document(page_content=text, metadata=meta))

    def _ingest():
        try:
            build_vectorstore(docs=docs)
            # Reload the pipeline's vectorstore
            from rag.vectorstore import load_vectorstore
            pipeline.vectorstore = load_vectorstore()
            log.info(f"✅ Ingested {len(docs)} documents.")
        except Exception as e:
            log.error(f"Ingestion failed: {e}")

    background_tasks.add_task(_ingest)

    return {
        "status": "accepted",
        "message": f"Ingesting {len(docs)} documents in the background.",
    }


@app.get("/")
async def root():
    return {
        "name": "SmartRAG API",
        "docs": "/docs",
        "health": "/health",
    }
