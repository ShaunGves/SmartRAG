"""
training/finetune.py

QLoRA fine-tuning pipeline using HuggingFace PEFT + TRL SFTTrainer.

Key techniques:
  - 4-bit NF4 quantization (bitsandbytes) → fits 7B model on single GPU
  - LoRA adapters on attention + MLP layers
  - Gradient checkpointing + Flash Attention 2 (if available)
  - MLflow experiment tracking

Run: python -m training.finetune
"""

import logging
import os
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg
from data.prepare_dataset import load_processed_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_quantized_model(model_id: str):
    """Load base model with 4-bit NF4 quantization (QLoRA)."""
    log.info(f"Loading model: {model_id} (4-bit quantized)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLMs
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,       # Nested quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",                    # Automatically place layers on GPU/CPU
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
    )

    # Required for gradient computation with quantized weights
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Disable KV cache during training

    return model


def load_tokenizer(model_id: str):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"  # Required for SFTTrainer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def apply_lora(model):
    """Wrap model with LoRA adapters using PEFT."""
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_trainer(model, tokenizer, dataset):
    """Configure and return SFTTrainer."""
    output_dir = cfg.model.output_dir

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        report_to=cfg.training.report_to,   # MLflow
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",            # Memory-efficient optimizer
        group_by_length=True,                # Speed up training
    )

    # Response template for completion-only training
    # (Only compute loss on the answer, not the instruction)
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=False,
    )

    return trainer


def finetune():
    """Full fine-tuning pipeline with MLflow tracking."""
    cfg.ensure_dirs()

    mlflow.set_experiment("smartrag-finetuning")

    with mlflow.start_run(run_name="qlora-mistral-7b"):
        # Log hyperparameters
        mlflow.log_params({
            "base_model": cfg.model.base_model_id,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "learning_rate": cfg.training.learning_rate,
            "epochs": cfg.training.num_train_epochs,
            "batch_size": cfg.training.per_device_train_batch_size,
            "grad_accum_steps": cfg.training.gradient_accumulation_steps,
            "quantization": "4-bit NF4",
        })

        # ── Load components ───────────────────────────────────────
        dataset = load_processed_dataset()
        log.info(f"Dataset loaded: {dataset}")

        model = load_quantized_model(cfg.model.base_model_id)
        tokenizer = load_tokenizer(cfg.model.base_model_id)
        model = apply_lora(model)

        # ── Train ─────────────────────────────────────────────────
        trainer = build_trainer(model, tokenizer, dataset)
        log.info("🚀 Starting QLoRA fine-tuning...")
        trainer.train()

        # ── Save ──────────────────────────────────────────────────
        log.info(f"Saving adapter to: {cfg.model.output_dir}")
        trainer.model.save_pretrained(cfg.model.output_dir)
        tokenizer.save_pretrained(cfg.model.output_dir)

        # Log final metrics
        final_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in final_metrics.items() if isinstance(v, float)})
        mlflow.log_artifact(cfg.model.output_dir)

        log.info("✅ Fine-tuning complete!")
        log.info(f"Model saved to: {cfg.model.output_dir}")

    return cfg.model.output_dir


def _has_flash_attn() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    finetune()
