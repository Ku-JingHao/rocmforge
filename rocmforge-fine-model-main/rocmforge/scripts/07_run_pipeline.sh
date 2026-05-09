#!/bin/bash
set -e

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

echo "============================================="
echo "  ROCmForge Data Pipeline"
echo "  Project: $PROJECT_DIR"
echo "============================================="
echo ""

# Step 1: Clone repos
echo "[STEP 1/5] Cloning source repositories..."
bash "$SCRIPTS_DIR/01_clone_repos.sh"
echo ""

# Step 2: Process CUDA samples → HIP pairs
echo "[STEP 2/5] Processing CUDA samples (CUDA → HIP pairs)..."
python3 "$SCRIPTS_DIR/02_process_cuda_samples.py"
echo ""

# Step 3: Extract optimized kernels from ROCm repos
echo "[STEP 3/5] Extracting optimized kernels (composable_kernel, rocBLAS, MIOpen)..."
python3 "$SCRIPTS_DIR/03_extract_rocm_optimized.py"
echo ""

# Step 4: Extract Triton examples
echo "[STEP 4/5] Extracting Triton kernel examples..."
python3 "$SCRIPTS_DIR/04_extract_triton.py"
echo ""

# Step 5: Generate instruction data (golden examples)
echo "[STEP 5/5] Generating instruction training data..."
python3 "$SCRIPTS_DIR/05_generate_instruction_data.py"
echo ""

# Step 6: Augment and create final train/val split
echo "[STEP 6/5] Augmenting data and creating train/val split..."
python3 "$SCRIPTS_DIR/06_augment_data.py"
echo ""

echo "============================================="
echo "  Pipeline Complete!"
echo "============================================="
echo ""
echo "Output files:"
ls -lh "$PROJECT_DIR/data/processed/"
echo ""
echo "Final training data:"
echo "  Train: $PROJECT_DIR/data/processed/train.jsonl"
echo "  Val:   $PROJECT_DIR/data/processed/val.jsonl"
echo ""
wc -l "$PROJECT_DIR/data/processed/train.jsonl" "$PROJECT_DIR/data/processed/val.jsonl" 2>/dev/null || true
echo ""
echo "Next step: Run training with rocmforge/training/train.py"
