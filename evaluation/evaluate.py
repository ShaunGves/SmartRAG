"""
evaluation/evaluate.py

Evaluates the full RAG pipeline using RAGAS metrics + custom metrics.
All results are tracked in MLflow for experiment comparison.

Metrics:
  - Faithfulness       : Is the answer grounded in retrieved context?
  - Answer Relevancy   : Does the answer address the question?
  - Context Precision  : Are retrieved chunks actually useful?
  - Context Recall     : Were all relevant chunks retrieved?
  - Latency            : End-to-end response time
  - Token Efficiency   : Answer length vs context length ratio

Run: python -m evaluation.evaluate
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg
from rag.pipeline import SmartRAGPipeline, RAGResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Test Set ────────────────────────────────────────────────────
# Replace these with domain-specific evaluation questions
EVAL_QUESTIONS = [
    {
        "question": "What is the mechanism of action of aspirin?",
        "ground_truth": "Aspirin irreversibly inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis, which decreases inflammation, pain, and fever.",
    },
    {
        "question": "What are the common side effects of metformin?",
        "ground_truth": "Common side effects of metformin include gastrointestinal issues such as nausea, diarrhea, and abdominal discomfort. Rarely, it can cause lactic acidosis.",
    },
    {
        "question": "How does the blood-brain barrier work?",
        "ground_truth": "The blood-brain barrier is formed by tight junctions between endothelial cells lining brain capillaries, restricting passage of most substances except small lipophilic molecules and nutrients with specific transporters.",
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "ground_truth": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. Type 2 diabetes involves insulin resistance and relative insulin deficiency, typically associated with lifestyle factors.",
    },
    {
        "question": "What is CRISPR-Cas9 used for?",
        "ground_truth": "CRISPR-Cas9 is a gene editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it makes a cut, allowing for gene knockout, correction, or insertion.",
    },
]


# ─── Evaluation Functions ─────────────────────────────────────────
def run_pipeline_on_testset(
    pipeline: SmartRAGPipeline,
    questions: List[Dict],
) -> List[Dict[str, Any]]:
    """Run the RAG pipeline on all evaluation questions and collect results."""
    results = []

    for item in questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        start = time.perf_counter()
        response: RAGResponse = pipeline.query(question)
        latency = time.perf_counter() - start

        results.append({
            "question": question,
            "answer": response.answer,
            "contexts": [response.context_used] if response.context_used else ["No context retrieved"],
            "ground_truth": ground_truth,
            "latency_s": round(latency, 3),
            "num_chunks": response.num_chunks_retrieved,
        })

        log.info(f"Q: {question[:60]}... | Latency: {latency:.2f}s | Chunks: {response.num_chunks_retrieved}")

    return results


def compute_ragas_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute RAGAS evaluation metrics on pipeline results."""
    dataset = Dataset.from_dict({
        "question":    [r["question"] for r in results],
        "answer":      [r["answer"] for r in results],
        "contexts":    [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    log.info("Computing RAGAS metrics...")
    score = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return dict(score)


def compute_custom_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute additional custom metrics beyond RAGAS."""
    latencies = [r["latency_s"] for r in results]
    chunk_counts = [r["num_chunks"] for r in results]
    answer_lengths = [len(r["answer"].split()) for r in results]
    context_lengths = [len(r["contexts"][0].split()) if r["contexts"] else 0 for r in results]

    token_efficiency = [
        a / max(c, 1) for a, c in zip(answer_lengths, context_lengths)
    ]

    return {
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "avg_chunks_retrieved": round(sum(chunk_counts) / len(chunk_counts), 2),
        "avg_answer_words": round(sum(answer_lengths) / len(answer_lengths), 1),
        "avg_token_efficiency": round(sum(token_efficiency) / len(token_efficiency), 3),
    }


def evaluate_pipeline(
    pipeline: SmartRAGPipeline,
    eval_questions: List[Dict] = None,
    run_name: str = "rag-evaluation",
) -> pd.DataFrame:
    """
    Full evaluation pipeline with MLflow tracking.

    Returns:
        DataFrame with per-question results and aggregate metrics
    """
    cfg.ensure_dirs()
    eval_questions = eval_questions or EVAL_QUESTIONS

    mlflow.set_experiment(cfg.eval.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name):
        # ── Log config ────────────────────────────────────────────
        mlflow.log_params({
            "model": cfg.model.base_model_id,
            "embedding_model": cfg.model.embedding_model_id,
            "top_k": cfg.rag.top_k,
            "chunk_size": cfg.rag.chunk_size,
            "num_eval_questions": len(eval_questions),
        })

        # ── Run pipeline ──────────────────────────────────────────
        log.info(f"Evaluating on {len(eval_questions)} questions...")
        results = run_pipeline_on_testset(pipeline, eval_questions)

        # ── RAGAS metrics ─────────────────────────────────────────
        ragas_metrics = compute_ragas_metrics(results)
        log.info(f"RAGAS metrics: {ragas_metrics}")

        # ── Custom metrics ────────────────────────────────────────
        custom_metrics = compute_custom_metrics(results)
        log.info(f"Custom metrics: {custom_metrics}")

        # ── Log to MLflow ─────────────────────────────────────────
        all_metrics = {**ragas_metrics, **custom_metrics}
        mlflow.log_metrics(all_metrics)

        # ── Save results ──────────────────────────────────────────
        results_df = pd.DataFrame(results)
        out_path = Path(cfg.eval.results_dir) / "eval_results.csv"
        results_df.to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path))

        metrics_path = Path(cfg.eval.results_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        # ── Print summary ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        for metric, value in all_metrics.items():
            bar = "█" * int(value * 20) if value <= 1.0 else ""
            print(f"  {metric:<30} {value:.4f}  {bar}")
        print("=" * 60)
        print(f"\n✅ Results saved → {out_path}")
        print(f"📈 MLflow UI: http://localhost:5000")

    return results_df


if __name__ == "__main__":
    from rag.pipeline import SmartRAGPipeline
    pipeline = SmartRAGPipeline()
    evaluate_pipeline(pipeline)
