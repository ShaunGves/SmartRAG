"""
config.py — Centralized configuration for SmartRAG.
All hyperparameters, paths, and model choices live here.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────
ROOT = Path(__file__).parent


# ─── Model Configuration ─────────────────────────────────────────
@dataclass
class ModelConfig:
    # Base model to fine-tune (swap to any HuggingFace model ID)
    base_model_id: str = "microsoft/phi-2"

    # Where to save the fine-tuned adapter
    output_dir: str = str(ROOT / "artifacts" / "finetuned_model")

    # Embedding model for RAG retrieval
    embedding_model_id: str = "BAAI/bge-base-en-v1.5"

    # Max token lengths
    max_seq_length: int = 2048
    max_new_tokens: int = 512


# ─── QLoRA Configuration ─────────────────────────────────────────
@dataclass
class LoRAConfig:
    r: int = 16                          # LoRA rank (higher = more params)
    lora_alpha: int = 32                 # Scaling factor
    target_modules: list = field(        # Which layers to apply LoRA to
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ─── Training Configuration ──────────────────────────────────────
@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    load_best_model_at_end: bool = True
    report_to: str = "mlflow"            # Experiment tracking


# ─── RAG Configuration ───────────────────────────────────────────
@dataclass
class RAGConfig:
    # ChromaDB persistence path
    chroma_persist_dir: str = str(ROOT / "artifacts" / "chroma_db")
    collection_name: str = "smartrag_docs"

    # Retrieval settings
    top_k: int = 4                       # Number of chunks to retrieve
    chunk_size: int = 512                # Characters per chunk
    chunk_overlap: int = 64             # Overlap between chunks

    # Similarity threshold (0.0–1.0)
    similarity_threshold: float = 0.3


# ─── Data Configuration ──────────────────────────────────────────
@dataclass
class DataConfig:
    # HuggingFace dataset for fine-tuning
    # Using medical QA as domain example — swap for any domain
    dataset_name: str = "medalpaca/medical_meadow_wikidoc"
    dataset_split: str = "train"

    # Local paths
    raw_data_dir: str = str(ROOT / "artifacts" / "raw_data")
    processed_data_dir: str = str(ROOT / "artifacts" / "processed_data")

    # Train/val split ratio
    val_size: float = 0.1
    seed: int = 42


# ─── Evaluation Configuration ────────────────────────────────────
@dataclass
class EvalConfig:
    mlflow_experiment_name: str = "smartrag-evaluation"
    results_dir: str = str(ROOT / "artifacts" / "eval_results")

    # RAGAS metrics to compute
    metrics: list = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )


# ─── Use Case Configuration ──────────────────────────────────────
@dataclass
class UseCaseConfig:
    """
    SmartRAG is focused on: AI Assistant for Programmers.
    Helps developers query codebases, docs, Stack Overflow Q&A,
    debug errors, and understand APIs — all grounded in real sources.
    """
    name: str = "AI Assistant for Programmers"
    domain: str = "software_engineering"

    # Fine-tuning dataset — code-focused instruction pairs
    finetune_dataset: str = "iamtarun/python_code_instructions_18k_alpaca"

    # Documents to index (paths or URLs)
    default_doc_sources: list = field(default_factory=lambda: [
        "https://docs.python.org/3/",
        "https://fastapi.tiangolo.com/",
    ])

    # System prompt for this domain
    system_prompt: str = (
        "You are an expert programming assistant. "
        "Answer questions about code, APIs, debugging, and software architecture "
        "using ONLY the provided context. Show code examples where helpful. "
        "If unsure, say so — never hallucinate function names or APIs."
    )


# ─── Hybrid Search Configuration ─────────────────────────────────
@dataclass
class HybridSearchConfig:
    """BM25 (keyword) + Dense (embedding) hybrid retrieval."""
    enabled: bool = True

    # Weight blending: final_score = α*dense + (1-α)*bm25
    alpha: float = 0.7          # 0.0 = pure BM25, 1.0 = pure dense

    # BM25 parameters
    bm25_k1: float = 1.5        # Term frequency saturation
    bm25_b: float = 0.75        # Document length normalization

    # How many candidates each retriever fetches before merging
    dense_candidates: int = 20
    bm25_candidates: int = 20

    # Final top-k after blending
    top_k_after_blend: int = 10


# ─── Reranker Configuration ──────────────────────────────────────
@dataclass
class RerankerConfig:
    """Cross-encoder reranking on top of hybrid retrieval."""
    enabled: bool = True

    # Cross-encoder model (much more accurate than bi-encoder for ranking)
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Take top-k from hybrid search → rerank → return top_k_final
    top_k_final: int = 4

    # Score threshold: discard chunks below this reranker score
    score_threshold: float = -5.0

    # Batch size for cross-encoder inference
    batch_size: int = 16


# ─── Embedding Cache Configuration ───────────────────────────────
@dataclass
class CacheConfig:
    """
    Embedding cache to avoid re-computing vectors for repeated queries.
    Cuts latency by 80–95% on cache hits.
    """
    enabled: bool = True
    backend: str = "memory"     # "memory" | "redis" | "disk"

    # Redis settings (only used if backend="redis")
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600   # 1 hour TTL

    # In-memory LRU cache size (number of embeddings)
    max_memory_entries: int = 10_000

    # Disk cache path (only used if backend="disk")
    disk_cache_dir: str = str(ROOT / "artifacts" / "embedding_cache")


# ─── Rate Limiter Configuration ──────────────────────────────────
@dataclass
class RateLimitConfig:
    """Per-IP API rate limiting to prevent abuse and manage GPU cost."""
    enabled: bool = True

    # Requests per window
    requests_per_minute: int = 20
    requests_per_hour: int = 200
    requests_per_day: int = 1000

    # Burst allowance (allows short bursts above per-minute limit)
    burst_multiplier: float = 1.5

    # Storage backend for rate limit counters
    backend: str = "memory"     # "memory" | "redis"


# ─── Agent Configuration ─────────────────────────────────────────
@dataclass
class AgentConfig:
    """
    Multi-step reasoning agent with tool calling.
    Agent decides WHEN to retrieve, WHEN to search the web,
    WHEN to execute code — producing richer answers than single-pass RAG.
    """
    enabled: bool = True
    max_iterations: int = 5     # Max reasoning steps before forcing an answer
    max_tokens_per_step: int = 256

    # Tools available to the agent
    tools: list = field(default_factory=lambda: [
        "vector_search",        # Search the ChromaDB vector store
        "hybrid_search",        # BM25 + dense hybrid search
        "web_search",           # Real-time web search fallback
        "code_executor",        # Safe Python code execution (sandbox)
        "calculator",           # Math evaluation
    ])

    # Temperature for agent reasoning steps (lower = more deterministic)
    temperature: float = 0.05


# ─── System Design Configuration ─────────────────────────────────
@dataclass
class SystemConfig:
    """
    Production system design parameters.
    These control latency, throughput, and cost.
    """
    # Latency targets (milliseconds)
    target_p50_latency_ms: int = 500
    target_p95_latency_ms: int = 2000
    target_p99_latency_ms: int = 5000

    # Concurrency
    api_workers: int = 1        # Increase for multi-GPU setups
    max_concurrent_requests: int = 10

    # Vector DB tuning
    chroma_hnsw_ef: int = 100           # Higher = better recall, slower
    chroma_hnsw_m: int = 16             # Connections per node (16–64)
    chroma_batch_size: int = 512        # Ingestion batch size

    # Embedding optimization
    embedding_batch_size: int = 32      # Batch queries for GPU efficiency
    embedding_normalize: bool = True    # L2 normalize for cosine similarity

    # API gateway settings
    request_timeout_seconds: int = 60
    max_payload_size_mb: int = 10


# ─── Global Config Object ─────────────────────────────────────────
class Config:
    model    = ModelConfig()
    lora     = LoRAConfig()
    training = TrainingConfig()
    rag      = RAGConfig()
    data     = DataConfig()
    eval     = EvalConfig()
    usecase  = UseCaseConfig()
    hybrid   = HybridSearchConfig()
    reranker = RerankerConfig()
    cache    = CacheConfig()
    ratelimit = RateLimitConfig()
    agent    = AgentConfig()
    system   = SystemConfig()

    @staticmethod
    def ensure_dirs():
        """Create all artifact directories if they don't exist."""
        dirs = [
            Path(Config.model.output_dir),
            Path(Config.rag.chroma_persist_dir),
            Path(Config.data.raw_data_dir),
            Path(Config.data.processed_data_dir),
            Path(Config.eval.results_dir),
            Path(Config.cache.disk_cache_dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
