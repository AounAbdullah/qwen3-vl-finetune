# Qwen3-VL Fine-Tuning Repo

SFT fine-tuning for Qwen3-VL using TRL + PEFT (QLoRA). Built for day 1 of the training lifecycle.

---

## Repo Structure

```
qwen3-vl-finetune/
├── setup.sh                  ← Run this FIRST — installs everything
├── verify_env.py             ← Run this SECOND — confirms GPU + packages
├── requirements.txt          ← All dependencies listed
│
├── configs/
│   └── qlora_7b.yaml         ← Training config (edit model + dataset here)
│
├── scripts/
│   └── train_sft.py          ← Main training script
│
├── data/                     ← Put local datasets here
├── notebooks/                ← For exploration
├── checkpoints/              ← Model checkpoints saved here
└── outputs/                  ← Inference outputs
```

---

## Step-by-Step Setup

### Step 1 — Clone / enter the repo
```bash
cd qwen3-vl-finetune
```

### Step 2 — Run setup (creates venv, installs everything)
```bash
bash setup.sh
```
This installs: `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `qwen-vl-utils`

### Step 3 — Activate the environment
```bash
source venv/bin/activate
```

### Step 4 — Verify GPU + packages
```bash
python verify_env.py
```
Expected output:
```
✅ PyTorch          2.3.x
✅ Transformers     4.57.x
✅ PEFT             0.11.x
✅ TRL              0.9.x
✅ BitsAndBytes     0.43.x
GPU: NVIDIA A100 / RTX 3090 / etc  (XX.X GB VRAM)
→ Recommended: Qwen3-VL-7B with QLoRA
```

---

## Picking Your Model Size

| Your VRAM | Recommended model          | Mode        |
|-----------|----------------------------|-------------|
| 8–12 GB   | Qwen3-VL-2B-Instruct       | QLoRA       |
| 16–24 GB  | Qwen3-VL-7B-Instruct       | QLoRA       |
| 40 GB     | Qwen3-VL-7B-Instruct       | Full FT     |
| 40 GB     | Qwen3-VL-32B-Instruct      | QLoRA       |
| 80 GB+    | Qwen3-VL-32B-Instruct      | Full FT     |

Change `model_name_or_path` in `configs/qlora_7b.yaml` accordingly.

---

## Running Training

### 1. Edit the config
Open `configs/qlora_7b.yaml` and set:
```yaml
model_name_or_path: Qwen/Qwen3-VL-7B-Instruct   # or 2B / 32B
dataset_name: your_hf_dataset_or_local_path
```

### 2. Run
```bash
python scripts/train_sft.py --config configs/qlora_7b.yaml
```

Checkpoints save to `./checkpoints/qwen3-vl-7b-qlora/`

---

## What is SFT and where does it fit?

```
Pre-training          →  SFT (you are here)       →  Alignment (RLHF / DPO)
(learn language)         (learn to follow             (prefer good answers
  done by Qwen             instructions)                over bad ones)
```

**SFT (Supervised Fine-Tuning)** = take a pre-trained base model + train it on
(instruction, response) pairs so it learns to behave like an assistant.

**Full FT** = update all model weights. Expensive. Needs lots of VRAM.

**PEFT / LoRA** = freeze base weights, inject small trainable matrices into
attention layers. ~90% fewer trainable params. Fits on consumer GPUs.

**QLoRA** = LoRA + 4-bit quantization of the base model. Even less VRAM.

---

## Dataset Format Expected

Each training example should be a conversation in Qwen's chat format:

```python
{
  "text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>"
}
```

For vision tasks, images are passed separately via the Qwen3-VL processor.
See `notebooks/` for vision dataset preparation examples (coming next).

---

## Key Libraries

| Library          | What it does                              |
|------------------|-------------------------------------------|
| `transformers`   | Load Qwen3-VL model + tokenizer           |
| `peft`           | LoRA / QLoRA adapter logic                |
| `trl`            | SFTTrainer — wraps the training loop      |
| `bitsandbytes`   | 4-bit / 8-bit quantization                |
| `accelerate`     | Multi-GPU / mixed precision               |
| `qwen-vl-utils`  | Image/video preprocessing for Qwen3-VL   |
