#!/bin/bash
# ============================================================
# Bootstrap script for Qwen3-VL fine-tuning environment
# Run once: bash setup.sh
# ============================================================

set -e  # Exit on any error

echo "========================================="
echo " Qwen3-VL Fine-Tune Environment Setup"
echo "========================================="

# --- 1. Python version check ---
echo ""
echo "[1/6] Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "✅ Python $PYTHON_VERSION OK"
else
    echo "❌ Need Python 3.10+. Current: $PYTHON_VERSION"
    exit 1
fi

# --- 2. Create virtual environment ---
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ venv created"
else
    echo "⚠️  venv already exists, skipping"
fi

source venv/bin/activate
pip install --upgrade pip -q

# --- 3. Install PyTorch (CUDA 12.1 build) ---
echo ""
echo "[3/6] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo "✅ PyTorch installed"

# --- 4. Install remaining requirements ---
echo ""
echo "[4/6] Installing HuggingFace stack + training libraries..."
pip install \
    "transformers>=4.57.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "peft>=0.11.0" \
    "trl>=0.9.0" \
    "bitsandbytes>=0.43.0" \
    "huggingface_hub>=0.23.0" \
    qwen-vl-utils \
    einops sentencepiece scipy pandas pillow numpy -q
echo "✅ Libraries installed"

# --- 5. Optional: flash-attention (skip if it fails, needs CUDA toolkit) ---
echo ""
echo "[5/6] Attempting flash-attention install (optional, may take a while)..."
pip install flash-attn --no-build-isolation -q && echo "✅ flash-attn installed" || echo "⚠️  flash-attn skipped (ok to continue without it)"

# --- 6. Verify GPU ---
echo ""
echo "[6/6] Verifying GPU / CUDA..."
python3 - <<'EOF'
import torch

print(f"  PyTorch version  : {torch.__version__}")
print(f"  CUDA available   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  GPU count        : {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}           : {props.name}  ({mem_gb:.1f} GB VRAM)")

    # Recommend model size
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print()
    if vram >= 40:
        rec = "Qwen3-VL-7B (full FT) or Qwen3-VL-32B (QLoRA)"
    elif vram >= 20:
        rec = "Qwen3-VL-7B (QLoRA) — recommended"
    elif vram >= 10:
        rec = "Qwen3-VL-2B or Qwen3-VL-7B with aggressive QLoRA (r=8)"
    else:
        rec = "Qwen3-VL-2B with QLoRA"
    print(f"  Recommended size : {rec}")
else:
    print("  ⚠️  No GPU detected — training will be very slow on CPU")
EOF

echo ""
echo "========================================="
echo " Setup complete! Activate env with:"
echo "   source venv/bin/activate"
echo "========================================="
