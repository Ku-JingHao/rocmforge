#!/bin/bash
set -e

echo "============================================="
echo "  ROCmForge — MI300X Instance Setup"
echo "============================================="
echo ""

# --- Verify GPU is available ---
echo "[1/5] Verifying AMD GPU..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi
    echo ""
    echo "[OK] ROCm detected."
else
    echo "[ERROR] rocm-smi not found. Make sure you're on an AMD Developer Cloud MI300X instance."
    echo "        ROCm should be pre-installed. If not, see: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
    exit 1
fi

echo ""
echo "[2/5] Checking GPU architecture..."
rocminfo | grep -i "gfx" | head -5
echo ""

# --- Create virtual environment ---
echo "[3/5] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[SKIP] venv already exists."
fi
source venv/bin/activate

# --- Install PyTorch with ROCm ---
echo ""
echo "[4/5] Installing PyTorch (ROCm) + training dependencies..."
echo "      This may take a few minutes..."

pip install --upgrade pip setuptools wheel

# PyTorch ROCm build (the ROCm Software image usually has this preinstalled,
# but we re-install to ensure version match)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Config & data parsing
pip install pyyaml

# Training stack (TRL pinned to a stable API version)
pip install "transformers>=4.44.0,<5.0.0"
pip install "peft>=0.12.0"
pip install "trl>=0.9.0,<0.12.0"
pip install "datasets>=2.20.0"
pip install "accelerate>=0.33.0"

# Monitoring (optional — needed only if config.yaml has report_to: "wandb")
pip install wandb

# NOTE: bitsandbytes is intentionally NOT installed.
#   The default bitsandbytes build is CUDA-only and would fail on ROCm.
#   Our config uses optim: "adamw_torch" which works natively on ROCm.
#
# NOTE: vLLM is intentionally NOT installed here.
#   It pulls heavy CUDA-related deps. Install it separately for Phase 3 (serving):
#     pip install vllm
#   Or use the AMD ROCm vLLM Docker image: rocm/vllm:latest

echo ""
echo "[OK] All packages installed."

# --- Verify PyTorch sees the GPU ---
echo ""
echo "[5/5] Verifying PyTorch GPU access..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f'         Memory: {mem:.1f} GB')
    print()
    print('[OK] GPU ready for training!')
else:
    print('[ERROR] PyTorch cannot see the GPU. Check ROCm installation.')
    exit(1)
"

echo ""
echo "============================================="
echo "  Setup Complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Upload your data:  scp data/processed/train.jsonl <instance>:~/rocmforge/data/processed/"
echo "  2. Start training:    bash training/run_training.sh"
echo ""
