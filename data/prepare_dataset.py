"""
data/prepare_dataset.py

Downloads a HuggingFace dataset, cleans it, formats it into
instruction-tuning format, and saves train/val splits.

Run: python -m data.prepare_dataset
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Prompt Template (Mistral Instruct Format) ───────────────────
PROMPT_TEMPLATE = """<s>[INST] {instruction}

Context: {input} [/INST] {output} </s>"""

PROMPT_TEMPLATE_NO_INPUT = """<s>[INST] {instruction} [/INST] {output} </s>"""


def format_example(example: dict) -> dict:
    """Convert a raw dataset example to instruction-tuning format."""
    instruction = example.get("instruction", "").strip()
    context = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if not instruction or not output:
        return None

    if context:
        text = PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=context,
            output=output,
        )
    else:
        text = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output,
        )

    return {"text": text, "instruction": instruction, "context": context, "output": output}


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace
    return text


def prepare_dataset():
    """Main pipeline: download → clean → format → split → save."""
    cfg.ensure_dirs()
    log.info(f"Loading dataset: {cfg.data.dataset_name}")

    # ── 1. Load from HuggingFace Hub ─────────────────────────────
    raw = load_dataset(
        cfg.data.dataset_name,
        split=cfg.data.dataset_split,
        trust_remote_code=True,
    )
    log.info(f"Raw dataset size: {len(raw):,} examples")

    # ── 2. Clean & Format ─────────────────────────────────────────
    formatted = []
    skipped = 0
    for example in raw:
        # Normalize field names (datasets vary)
        normalized = {
            "instruction": clean_text(example.get("instruction", example.get("question", ""))),
            "input":       clean_text(example.get("input", example.get("context", ""))),
            "output":      clean_text(example.get("output", example.get("answer", ""))),
        }
        result = format_example(normalized)
        if result:
            formatted.append(result)
        else:
            skipped += 1

    log.info(f"Formatted: {len(formatted):,} | Skipped (empty): {skipped:,}")

    # ── 3. Train / Val Split ──────────────────────────────────────
    train_data, val_data = train_test_split(
        formatted,
        test_size=cfg.data.val_size,
        random_state=cfg.data.seed,
    )
    log.info(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ── 4. Save as JSONL ──────────────────────────────────────────
    out_dir = Path(cfg.data.processed_data_dir)

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        log.info(f"Saved {split_name} → {path}")

    # ── 5. Save metadata ─────────────────────────────────────────
    meta = {
        "dataset": cfg.data.dataset_name,
        "total_examples": len(formatted),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "prompt_format": "mistral-instruct",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("✅ Dataset preparation complete!")
    return train_data, val_data


def load_processed_dataset() -> DatasetDict:
    """Load already-processed JSONL files as a HuggingFace DatasetDict."""
    from datasets import Dataset
    out_dir = Path(cfg.data.processed_data_dir)

    splits = {}
    for split in ["train", "val"]:
        path = out_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Run prepare_dataset() first. Missing: {path}")
        data = [json.loads(line) for line in open(path)]
        splits[split] = Dataset.from_list(data)

    return DatasetDict(splits)


if __name__ == "__main__":
    prepare_dataset()
