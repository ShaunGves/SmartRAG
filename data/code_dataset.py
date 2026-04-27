"""
data/code_dataset.py

Dataset preparation for the "AI Assistant for Programmers" use case.

Uses: iamtarun/python_code_instructions_18k_alpaca
  - 18K Python instruction-answer pairs
  - Covers algorithms, data structures, debugging, APIs, best practices
  - Each example: instruction + optional context + code answer

Why this dataset for fine-tuning:
  - Domain-specific data → model learns programmer vocabulary
  - Code-heavy outputs → model learns to format code blocks properly
  - QA format matches how developers actually ask questions
  - 18K examples → enough for meaningful domain adaptation via LoRA

Run: python -m data.code_dataset
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Mistral instruct format with code-aware system prompt
CODE_PROMPT_TEMPLATE = """<s>[INST] {system}

### Question:
{instruction}

{context_block}[/INST]
### Answer:
{output} </s>"""

SYSTEM_PROMPT = cfg.usecase.system_prompt


def format_code_example(example: dict) -> dict:
    """Format a code instruction example into Mistral chat format."""
    instruction = example.get("instruction", "").strip()
    context     = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if not instruction or not output:
        return None

    # Ensure code blocks are properly fenced
    if "def " in output or "import " in output or "class " in output:
        if "```" not in output:
            output = f"```python\n{output}\n```"

    context_block = f"### Context:\n{context}\n\n" if context else ""

    text = CODE_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        instruction=instruction,
        context_block=context_block,
        output=output,
    )

    return {
        "text":        text,
        "instruction": instruction,
        "context":     context,
        "output":      output,
        "domain":      "programming",
    }


def prepare_code_dataset():
    """Download and format the programming dataset."""
    cfg.ensure_dirs()
    out_dir = Path(cfg.data.processed_data_dir)

    log.info(f"Loading dataset: {cfg.usecase.finetune_dataset}")
    raw = load_dataset(cfg.usecase.finetune_dataset, split="train", trust_remote_code=True)
    log.info(f"Raw size: {len(raw):,}")

    formatted = []
    skipped = 0
    for ex in raw:
        result = format_code_example(ex)
        if result:
            formatted.append(result)
        else:
            skipped += 1

    log.info(f"Formatted: {len(formatted):,} | Skipped: {skipped:,}")

    train_data, val_data = train_test_split(
        formatted, test_size=cfg.data.val_size, random_state=cfg.data.seed
    )

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        path = out_dir / f"{split_name}_code.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        log.info(f"Saved {split_name} → {path} ({len(split_data):,} examples)")

    log.info("✅ Code dataset ready for fine-tuning!")
    return train_data, val_data


if __name__ == "__main__":
    prepare_code_dataset()
