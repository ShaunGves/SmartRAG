"""
rag/hybrid_search.py

Hybrid retrieval combining:
  - Dense search  : semantic similarity via embedding vectors (ChromaDB)
  - Sparse search : keyword relevance via BM25 (rank_bm25)

Results are fused using Reciprocal Rank Fusion (RRF) weighted by alpha.

Why hybrid beats dense-only:
  - Dense search handles paraphrasing ("what does X do" → finds "X is used for")
  - BM25 handles exact keyword matches ("TypeError: NoneType", function names, version numbers)
  - Together they cover both semantic AND lexical retrieval

Research backing: BEIR benchmark shows hybrid consistently outperforms
either method alone by 3–8% on NDCG@10.

Use case (AI Programmer Assistant):
  Dense: "how do I handle async errors" → finds conceptually related docs
  BM25:  "asyncio.TimeoutError"         → finds exact error name instantly
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from langchain.docstore.document import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────
@dataclass
class ScoredChunk:
    """A document chunk with its retrieval scores from each system."""
    document: Document
    dense_score: float = 0.0      # Cosine similarity from embedding search
    bm25_score: float = 0.0       # BM25 relevance score
    hybrid_score: float = 0.0     # Final blended score
    rerank_score: float = 0.0     # Cross-encoder score (added by reranker)
    dense_rank: int = 9999
    bm25_rank: int = 9999


# ─── BM25 Index ──────────────────────────────────────────────────
class BM25Index:
    """
    In-memory BM25 index over document chunks.

    BM25 (Best Match 25) is the gold standard sparse retrieval algorithm.
    Used by Elasticsearch, Solr, and Lucene under the hood.

    Parameters:
        k1 (float): Term frequency saturation — how quickly TF saturates.
                    Typical range: 1.2–2.0
        b  (float): Document length normalization.
                    0 = no normalization, 1 = full normalization
    """

    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or cfg.hybrid.bm25_k1
        self.b  = b  or cfg.hybrid.bm25_b
        self.docs: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.idf: dict = {}
        self.avgdl: float = 0.0
        self._built = False

    def build(self, documents: List[Document]) -> None:
        """Tokenize corpus and compute IDF scores."""
        self.docs = documents
        self.tokenized_corpus = [self._tokenize(d.page_content) for d in documents]

        # Compute average document length
        doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]
        self.avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

        # Compute IDF for each term
        N = len(self.tokenized_corpus)
        df: dict = {}
        for tokens in self.tokenized_corpus:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        self._built = True
        log.info(f"BM25 index built: {N} docs, {len(self.idf)} unique terms, avgdl={self.avgdl:.1f}")

    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Return top-k documents with BM25 scores for the query."""
        if not self._built:
            raise RuntimeError("Call build() before search()")

        top_k = top_k or cfg.hybrid.bm25_candidates
        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            dl = len(doc_tokens)
            tf_map: dict = {}
            for token in doc_tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                idf = self.idf[term]
                # BM25 term score formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], score) for i, score in scores[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        # Keep code tokens like "asyncio.TimeoutError", "np.array", etc.
        tokens = re.findall(r"[a-zA-Z0-9_.]+", text.lower())
        return [t for t in tokens if len(t) > 1]


# ─── Reciprocal Rank Fusion ───────────────────────────────────────
def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    alpha: float = None,
    k: int = 60,   # RRF constant — 60 is standard
) -> List[ScoredChunk]:
    """
    Fuse dense and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score(d) = α * Σ(1/(k+rank_dense)) + (1-α) * Σ(1/(k+rank_bm25))

    Why RRF over score normalization:
      - Score distributions differ between models (cosine vs BM25 magnitude)
      - RRF is robust to score scale differences
      - RRF consistently outperforms score-based fusion in benchmarks

    Args:
        dense_results : List of (doc, cosine_score) from embedding search
        bm25_results  : List of (doc, bm25_score) from BM25
        alpha         : Weight for dense search (0.0–1.0)
        k             : RRF smoothing constant
    """
    alpha = alpha if alpha is not None else cfg.hybrid.alpha
    fused: dict = {}   # doc_id → ScoredChunk

    def doc_id(doc: Document) -> str:
        return doc.page_content[:100]  # Use content prefix as ID

    # ── Process dense results ────────────────────────────────────
    for rank, (doc, score) in enumerate(dense_results):
        did = doc_id(doc)
        if did not in fused:
            fused[did] = ScoredChunk(document=doc)
        fused[did].dense_score = score
        fused[did].dense_rank = rank
        fused[did].hybrid_score += alpha * (1.0 / (k + rank + 1))

    # ── Process BM25 results ─────────────────────────────────────
    for rank, (doc, score) in enumerate(bm25_results):
        did = doc_id(doc)
        if did not in fused:
            fused[did] = ScoredChunk(document=doc)
        fused[did].bm25_score = score
        fused[did].bm25_rank = rank
        fused[did].hybrid_score += (1 - alpha) * (1.0 / (k + rank + 1))

    # Sort by hybrid score descending
    ranked = sorted(fused.values(), key=lambda x: x.hybrid_score, reverse=True)
    return ranked


# ─── Main Hybrid Retriever ───────────────────────────────────────
class HybridRetriever:
    """
    Production hybrid retriever combining ChromaDB dense search + BM25.

    Architecture:
        Query → [Dense retriever (ChromaDB)]  ──┐
             → [Sparse retriever (BM25)]       ──┤→ RRF fusion → top-k chunks
                                                 │   (→ cross-encoder reranker)

    Usage:
        retriever = HybridRetriever(vectorstore, documents)
        chunks = retriever.retrieve("how to handle asyncio errors")
    """

    def __init__(self, vectorstore, documents: List[Document]):
        from rag.vectorstore import retrieve as dense_retrieve
        self._dense_retrieve = dense_retrieve
        self.vectorstore = vectorstore
        self.bm25 = BM25Index()
        self.bm25.build(documents)
        log.info("HybridRetriever initialized (dense + BM25)")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[ScoredChunk]:
        """
        Retrieve and fuse results from both retrieval systems.

        Returns ScoredChunk objects with individual scores for analysis.
        """
        top_k_blend  = top_k or cfg.hybrid.top_k_after_blend
        dense_k      = cfg.hybrid.dense_candidates
        bm25_k       = cfg.hybrid.bm25_candidates

        # ── Dense retrieval ──────────────────────────────────────
        dense_raw = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=dense_k
        )
        dense_results = [(doc, score) for doc, score in dense_raw]

        # ── BM25 retrieval ───────────────────────────────────────
        bm25_results = self.bm25.search(query, top_k=bm25_k)

        # ── Fuse via RRF ─────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_results, bm25_results, alpha=alpha)

        log.debug(
            f"Hybrid retrieval: dense={len(dense_results)}, "
            f"bm25={len(bm25_results)}, fused={len(fused)}, "
            f"returning top-{top_k_blend}"
        )
        return fused[:top_k_blend]

    def explain(self, query: str) -> dict:
        """
        Debug helper: show how dense vs BM25 contribute to the final ranking.
        Useful for understanding retrieval failures.
        """
        chunks = self.retrieve(query)
        return {
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "snippet": c.document.page_content[:120] + "...",
                    "hybrid_score": round(c.hybrid_score, 4),
                    "dense_score":  round(c.dense_score, 4),
                    "dense_rank":   c.dense_rank,
                    "bm25_score":   round(c.bm25_score, 4),
                    "bm25_rank":    c.bm25_rank,
                }
                for i, c in enumerate(chunks)
            ]
        }
