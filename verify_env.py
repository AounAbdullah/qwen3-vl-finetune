"""
verify_env.py
Run after setup.sh to confirm everything is installed and CUDA is detected.
Usage: python verify_env.py
"""

import sys
import importlib

# ── 1. Python ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Qwen3-VL Environment Verification")
print("=" * 55)
print(f"\n[Python]  {sys.version}\n")

# ── 2. Package versions ────────────────────────────────────────────────────────
packages = {
    "torch":           "PyTorch",
    "transformers":    "Transformers",
    "datasets":        "Datasets",
    "accelerate":      "Accelerate",
    "peft":            "PEFT",
    "trl":             "TRL",
    "bitsandbytes":    "BitsAndBytes",
    "huggingface_hub": "HF Hub",
    "qwen_vl_utils":   "Qwen-VL-Utils",
}

print("[Packages]")
all_ok = True
for pkg, label in packages.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✅  {label:<18} {ver}")
    except ImportError:
        print(f"  ❌  {label:<18} NOT FOUND")
        all_ok = False

# ── 3. CUDA / GPU ──────────────────────────────────────────────────────────────
print("\n[GPU / CUDA]")
import torch

print(f"  CUDA available   : {torch.cuda.is_available()}")
print(f"  CUDA version     : {torch.version.cuda}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  Device count     : {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        vram_gb = p.total_memory / 1024 ** 3
        print(f"  GPU {i}  {p.name}  —  {vram_gb:.1f} GB VRAM")

    # Simple tensor op to confirm compute works
    x = torch.tensor([1.0, 2.0]).cuda()
    assert x.sum().item() == 3.0
    print("  Tensor compute   : ✅ OK")

    # Model size recommendation
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("\n[Recommended Qwen3-VL size for your GPU]")
    if vram >= 40:
        print(f"  {vram:.0f} GB → Qwen3-VL-7B full FT  OR  Qwen3-VL-32B with QLoRA")
    elif vram >= 20:
        print(f"  {vram:.0f} GB → Qwen3-VL-7B with QLoRA (sweet spot)")
    elif vram >= 10:
        print(f"  {vram:.0f} GB → Qwen3-VL-2B full FT  OR  Qwen3-VL-7B with QLoRA r=8")
    else:
        print(f"  {vram:.0f} GB → Qwen3-VL-2B with QLoRA (only option)")
else:
    print("  ⚠️  No GPU — CPU only. Fine-tuning not practical.")

# ── 4. Flash-Attention (optional) ─────────────────────────────────────────────
print("\n[Flash-Attention (optional)]")
try:
    import flash_attn
    print(f"  ✅  flash-attn {flash_attn.__version__}")
except ImportError:
    print("  ⚠️  Not installed — will use standard attention (slower, more VRAM)")

# ── 5. Quick Transformers smoke test ──────────────────────────────────────────
print("\n[Transformers smoke test]")
try:
    from transformers import AutoTokenizer
    print("  ✅  AutoTokenizer importable")
except Exception as e:
    print(f"  ❌  {e}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if all_ok and torch.cuda.is_available():
    print("  ✅  All good — ready to fine-tune!")
elif all_ok:
    print("  ⚠️  Packages OK but no GPU detected")
else:
    print("  ❌  Some packages missing — re-run setup.sh")
print("=" * 55)
