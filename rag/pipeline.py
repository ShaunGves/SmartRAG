"""
rag/pipeline.py

End-to-end RAG pipeline combining:
  - Vector retrieval (ChromaDB)
  - Prompt construction with retrieved context
  - Generation with fine-tuned LLM
  - Source attribution
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from langchain_core.documents import Document
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg
from rag.vectorstore import load_vectorstore, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────
@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    question: str
    answer: str
    sources: List[str]
    context_used: str
    num_chunks_retrieved: int


# ─── Prompt Template ─────────────────────────────────────────────
RAG_PROMPT = """<s>[INST] You are a helpful, precise assistant. Answer the question below using ONLY the provided context.
If the context doesn't contain enough information to answer confidently, say "I don't have enough context to answer this."

### Context:
{context}

### Question:
{question} [/INST]"""


# ─── Model Loading ────────────────────────────────────────────────
def load_finetuned_pipeline(
    adapter_path: Optional[str] = None,
    use_base_only: bool = False,
):
    """
    Load the fine-tuned model as a HuggingFace text-generation pipeline.

    Args:
        adapter_path: Path to LoRA adapter. Defaults to cfg.model.output_dir
        use_base_only: If True, load base model without LoRA (for testing)
    """
    adapter_path = adapter_path or cfg.model.output_dir
    model_id = cfg.model.base_model_id

    # Check if adapter folder exists AND has files in it
    adapter_path_obj = Path(adapter_path)
    adapter_exists = (
        adapter_path_obj.exists()
        and adapter_path_obj.is_dir()
        and any(adapter_path_obj.iterdir())
    )

    # Load tokenizer — use base model if adapter folder is empty
    tokenizer_source = adapter_path if adapter_exists else model_id
    log.info(f"Loading tokenizer from: {tokenizer_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",           # CPU mode — no GPU required
        torch_dtype=torch.float32,  # float32 for CPU stability
        trust_remote_code=True,
    )

    if not use_base_only and adapter_exists:
        log.info(f"Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
    else:
        log.warning("Using base model without fine-tuned adapter (no GPU / adapter not found).")
        model = base_model

    model.eval()

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.model.max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.15,
        return_full_text=False,
    )

    log.info("✅ Generation pipeline ready.")
    return gen_pipeline


# ─── RAG Pipeline Class ──────────────────────────────────────────
class SmartRAGPipeline:
    """
    Full RAG pipeline: retrieve relevant context → generate grounded answer.

    Usage:
        rag = SmartRAGPipeline()
        response = rag.query("How do I handle asyncio exceptions?")
        print(response.answer)
    """

    def __init__(self, adapter_path: Optional[str] = None):
        log.info("Initializing SmartRAG Pipeline...")
        self.vectorstore = load_vectorstore()
        self.llm = load_finetuned_pipeline(adapter_path, use_base_only=True)
        log.info("✅ SmartRAG Pipeline ready!")

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Run the full RAG pipeline for a question.

        Steps:
          1. Embed the question
          2. Retrieve top-k relevant chunks from ChromaDB
          3. Build prompt with context
          4. Generate answer with fine-tuned LLM
          5. Return structured response with sources
        """
        # ── Step 1: Retrieve ──────────────────────────────────────
        chunks: List[Document] = retrieve(question, self.vectorstore, top_k)

        if not chunks:
            return RAGResponse(
                question=question,
                answer="I couldn't find relevant information in my knowledge base to answer this question.",
                sources=[],
                context_used="",
                num_chunks_retrieved=0,
            )

        # ── Step 2: Build Context ─────────────────────────────────
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {chunk.page_content.strip()}")
            source = chunk.metadata.get("source", "Unknown")
            sources.append(f"Source {i}: {source}")

        context = "\n\n".join(context_parts)

        # ── Step 3: Generate ──────────────────────────────────────
        prompt = RAG_PROMPT.format(context=context, question=question)
        output = self.llm(prompt)[0]["generated_text"]
        answer = output.strip()

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context_used=context,
            num_chunks_retrieved=len(chunks),
        )

    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Process multiple questions."""
        return [self.query(q) for q in questions]


# ─── Convenience function ─────────────────────────────────────────
_pipeline_instance = None


def get_pipeline() -> SmartRAGPipeline:
    """Singleton pattern — load once, reuse across API calls."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SmartRAGPipeline()
    return _pipeline_instance