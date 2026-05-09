#!/bin/bash
# Start the ROCmForge inference server.
#
# Run this AFTER training is complete and the merged model exists at
# ./rocmforge-7b-merged/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# === Configuration via env vars ===
export ROCMFORGE_MODEL_PATH="${ROCMFORGE_MODEL_PATH:-./rocmforge-7b-merged}"
export ROCMFORGE_MODEL_NAME="${ROCMFORGE_MODEL_NAME:-rocmforge-7b}"
export ROCMFORGE_USE_VLLM="${ROCMFORGE_USE_VLLM:-1}"
export ROCMFORGE_TARGET_ARCH="${ROCMFORGE_TARGET_ARCH:-gfx942}"

# Install serving dependencies if missing
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[INFO] Installing serving dependencies..."
    pip install fastapi uvicorn pydantic
fi

if [ "$ROCMFORGE_USE_VLLM" = "1" ] && ! python -c "import vllm" 2>/dev/null; then
    echo "[INFO] Installing vLLM (ROCm build)..."
    pip install vllm
fi

# === Verify model exists ===
if [ ! -d "$ROCMFORGE_MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $ROCMFORGE_MODEL_PATH"
    echo "        Run training/run_training.sh first."
    exit 1
fi

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"

echo "============================================="
echo "  ROCmForge — Inference Server"
echo "============================================="
echo "  Model:     $ROCMFORGE_MODEL_PATH"
echo "  Engine:    $([ "$ROCMFORGE_USE_VLLM" = "1" ] && echo "vLLM" || echo "HuggingFace")"
echo "  Target:    $ROCMFORGE_TARGET_ARCH"
echo "  Endpoint:  http://${HOST}:${PORT}"
echo ""

# Use module path so relative imports work
exec python -m uvicorn inference.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info
