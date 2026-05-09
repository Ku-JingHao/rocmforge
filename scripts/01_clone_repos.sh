#!/bin/bash
set -e

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "=== Cloning data source repositories ==="

repos=(
    "https://github.com/NVIDIA/cuda-samples.git"
    "https://github.com/ROCm/composable_kernel.git"
    "https://github.com/ROCm/rocBLAS.git"
    "https://github.com/ROCm/MIOpen.git"
    "https://github.com/openai/triton.git"
    "https://github.com/NVIDIA/CUDALibrarySamples.git"
)

for repo in "${repos[@]}"; do
    dir_name=$(basename "$repo" .git)
    if [ -d "$dir_name" ]; then
        echo "[SKIP] $dir_name already exists"
    else
        echo "[CLONE] $repo"
        git clone --depth 1 "$repo"
    fi
done

echo ""
echo "=== All repos cloned to $RAW_DIR ==="
ls -la
