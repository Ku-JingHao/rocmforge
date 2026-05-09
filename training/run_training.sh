#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================="
echo "  ROCmForge — Training Pipeline"
echo "  Working directory: $PROJECT_DIR"
echo "============================================="
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated."
fi

# Verify data exists
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "[ERROR] Training data not found at data/processed/train.jsonl"
    echo "        Run the data pipeline first: bash scripts/07_run_pipeline.sh"
    exit 1
fi

TRAIN_LINES=$(wc -l < data/processed/train.jsonl)
VAL_LINES=$(wc -l < data/processed/val.jsonl)
echo "[OK] Training data: $TRAIN_LINES examples"
echo "[OK] Validation data: $VAL_LINES examples"
echo ""

# Verify GPU
echo "[CHECK] GPU status:"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram 2>/dev/null | head -20 || true
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo ""

# =============================================
# PHASE 1: Quick sanity check (200 examples, ~3-5 min)
# Verifies model loads, data flows, gradients update, no crashes.
# We use 200 samples (not 50) so eval/save can trigger at the new lower steps.
# =============================================
echo "============================================="
echo "[PHASE 1] Quick sanity check (200 samples)..."
echo "============================================="

python training/train.py \
    --max_samples 200 \
    --config training/config.yaml \
    --sanity_check

if [ $? -ne 0 ]; then
    echo "[ERROR] Sanity check failed. Fix errors above before full training."
    exit 1
fi
echo "[OK] Sanity check passed."
echo ""

# Clean up sanity check artifacts
rm -rf checkpoints/

# =============================================
# PHASE 2: Full training run
# =============================================
echo "============================================="
echo "[PHASE 2] Starting full training run..."
echo "          This will take ~2.5-3 hours on MI300X"
echo "============================================="
echo ""

python training/train.py --config training/config.yaml

echo ""
echo "[OK] Training complete."
echo ""

# =============================================
# PHASE 3: Merge LoRA into base model
# =============================================
echo "============================================="
echo "[PHASE 3] Merging LoRA adapter into base model..."
echo "============================================="
echo ""

python training/merge_model.py \
    --adapter_path ./checkpoints/final-adapter \
    --output_dir ./rocmforge-7b-merged

echo ""
echo "[OK] Model merged."
echo ""

# =============================================
# PHASE 4: Quick inference test
# =============================================
echo "============================================="
echo "[PHASE 4] Running inference test..."
echo "============================================="
echo ""

python training/test_inference.py --model_path ./rocmforge-7b-merged

echo ""
echo "============================================="
echo "  ALL DONE!"
echo "============================================="
echo ""
echo "  Merged model:  ./rocmforge-7b-merged/"
echo "  LoRA adapter:  ./checkpoints/final-adapter/"
echo ""
echo "  Next steps:"
echo "  1. Start vLLM server:"
echo "     python -m vllm.entrypoints.openai.api_server \\"
echo "       --model ./rocmforge-7b-merged \\"
echo "       --dtype bfloat16 \\"
echo "       --max-model-len 8192 \\"
echo "       --gpu-memory-utilization 0.85"
echo ""
echo "  2. Push to HuggingFace:"
echo "     python training/merge_model.py --push_to_hub your-name/rocmforge-7b"
echo ""
