"""
rag/reranker.py

Cross-encoder reranking on top of hybrid retrieval results.

Pipeline:
  1. Hybrid search retrieves top-20 candidate chunks  (fast, approximate)
  2. Cross-encoder scores ALL 20 candidates vs query  (slow, accurate)
  3. Return top-4 by reranker score                   (precision maximized)

Why reranking works:
  - Bi-encoders (embeddings) encode query and doc SEPARATELY → fast but imprecise
  - Cross-encoders encode query+doc TOGETHER → sees full interaction → much more accurate
  - Classic precision/recall tradeoff: retrieve broadly, rerank precisely

Performance gains (empirically measured on MSMARCO):
  - Dense-only retrieval:         MRR@10 ≈ 0.33
  - Dense + cross-encoder rerank: MRR@10 ≈ 0.39  (+18% relative)
  - Hybrid + cross-encoder rerank: MRR@10 ≈ 0.42 (+27% relative)

Use case (AI Programmer Assistant):
  Without reranking: top result might be a vaguely related doc about async
  With reranking:    top result is the exact asyncio error handling section
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg
from rag.hybrid_search import ScoredChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using a fine-tuned MS MARCO model.

    The cross-encoder takes a (query, passage) pair and outputs a relevance
    score in [-inf, +inf]. Higher = more relevant to the query.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - 22M parameters (tiny — runs on CPU in <100ms for 20 candidates)
      - Trained on MS MARCO passage ranking (500K query-passage pairs)
      - Excellent general-purpose reranker for any domain

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query="asyncio error handling", chunks=candidates)
    """

    def __init__(self, model_id: Optional[str] = None):
        model_id = model_id or cfg.reranker.model_id
        log.info(f"Loading cross-encoder: {model_id}")
        self.model = CrossEncoder(
            model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512,
        )
        log.info("✅ Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        chunks: List[ScoredChunk],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredChunk]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query           : The user's question
            chunks          : Candidates from hybrid retrieval
            top_k           : How many to return after reranking
            score_threshold : Discard chunks below this score

        Returns:
            Reranked ScoredChunk list with .rerank_score populated
        """
        top_k           = top_k           or cfg.reranker.top_k_final
        score_threshold = score_threshold or cfg.reranker.score_threshold

        if not chunks:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.document.page_content) for chunk in chunks]

        # Score all pairs in one batched forward pass
        t0 = time.perf_counter()
        scores = self.model.predict(
            pairs,
            batch_size=cfg.reranker.batch_size,
            show_progress_bar=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug(f"Cross-encoder scored {len(pairs)} pairs in {latency_ms:.1f}ms")

        # Attach scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        # Filter by threshold then sort by reranker score
        filtered = [c for c in chunks if c.rerank_score >= score_threshold]
        reranked = sorted(filtered, key=lambda c: c.rerank_score, reverse=True)

        log.debug(
            f"Reranked: {len(chunks)} → {len(filtered)} above threshold "
            f"→ returning top-{top_k}"
        )
        return reranked[:top_k]

    def explain(self, query: str, chunks: List[ScoredChunk]) -> List[dict]:
        """
        Show score breakdown: hybrid score vs reranker score.
        Great for debugging why a particular chunk was promoted/demoted.
        """
        reranked = self.rerank(query, chunks, top_k=len(chunks))
        return [
            {
                "final_rank":     i + 1,
                "rerank_score":   round(c.rerank_score, 3),
                "hybrid_score":   round(c.hybrid_score, 4),
                "dense_score":    round(c.dense_score, 4),
                "bm25_score":     round(c.bm25_score, 4),
                "snippet":        c.document.page_content[:100] + "...",
                "source":         c.document.metadata.get("source", "unknown"),
            }
            for i, c in enumerate(reranked)
        ]


# ─── Convenience: rerank from plain Document list ─────────────────
def rerank_documents(
    query: str,
    documents: List[Document],
    reranker: Optional[CrossEncoderReranker] = None,
    top_k: int = 4,
) -> List[Document]:
    """
    Convenience wrapper — rerank plain Documents (not ScoredChunks).
    Used when you want reranking but not the full hybrid pipeline.
    """
    if reranker is None:
        reranker = CrossEncoderReranker()

    # Wrap in ScoredChunk
    chunks = [ScoredChunk(document=doc) for doc in documents]
    reranked = reranker.rerank(query, chunks, top_k=top_k)
    return [c.document for c in reranked]


# ─── Singleton ────────────────────────────────────────────────────
_reranker_instance: Optional[CrossEncoderReranker] = None

def get_reranker() -> CrossEncoderReranker:
    """Singleton — load cross-encoder once, reuse across requests."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
