"""
scripts/train_sft.py
SFT fine-tuning script for Qwen3-VL using TRL + PEFT (QLoRA).

Usage:
    python scripts/train_sft.py --config configs/qlora_7b.yaml
"""

import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


# ── Argument parsing ───────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_id = cfg["model_name_or_path"]
    print(f"Loading model: {model_id}")

    # BitsAndBytes config for 4-bit QLoRA
    bnb_config = None
    if cfg.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves extra VRAM
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # spreads across all GPUs automatically
        trust_remote_code=True,     # needed for Qwen models
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # Qwen models use a pad token — make sure it's set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


# ── LoRA config ────────────────────────────────────────────────────────────────

def make_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.get("lora_target_modules", "all-linear"),
    )


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    dataset_name = cfg["dataset_name"]
    print(f"Loading dataset: {dataset_name}")

    # Supports HF Hub ids or local paths
    dataset = load_dataset(dataset_name)

    # Expected format: each row has a "text" field (or adjust dataset_text_field)
    # For instruction tuning, format should be:
    #   <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    return dataset


# ── Training ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # GPU check
    if not torch.cuda.is_available():
        print("⚠️  Warning: No GPU detected. Training will be very slow.")
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")

    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_data(cfg)
    lora_config = make_lora_config(cfg) if cfg.get("use_peft") else None

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        max_seq_length=cfg.get("max_seq_length", 2048),
        dataset_text_field=cfg.get("dataset_text_field", "text"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") or dataset.get("test"),
        peft_config=lora_config,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    print(f"\n✅ Training complete. Saving to {cfg['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
