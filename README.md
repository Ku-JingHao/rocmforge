# ROCmForge — The AMD Performance Compiler

> **"Any GPU code in. Hand-tuned ROCm out. AMD goes faster."**

ROCmForge is a fine-tuned **Qwen2.5-Coder-7B-Instruct** model that converts PyTorch / CUDA / Triton / natural-language descriptions into hand-tuned **HIP / ROCm kernels optimized for AMD Instinct MI300X (gfx942)**. The model is trained to emit wavefront-64 patterns, MFMA intrinsics, LDS staging, and HBM3-aware vectorization — optimizations that frontier LLMs hallucinate because almost no MI300X-specific training data exists on the public internet.

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Phase 1        │ →  │  Phase 2     │ →  │  Phase 3     │ →  │  Phase 4     │
│  Data Pipeline  │    │  Fine-Tuning │    │  Backend     │    │  Frontend    │
│  scripts/       │    │  training/   │    │  inference/  │    │  frontend/   │
└─────────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Table of Contents

1. [Why Fine-Tuning](#why-fine-tuning)
2. [System Architecture](#system-architecture)
3. [Project Layout](#project-layout)
4. [Quick Start](#quick-start)
5. [Phase 1 — Data Pipeline](#phase-1--data-pipeline)
6. [Phase 2 — Fine-Tuning](#phase-2--fine-tuning)
7. [Phase 3 — Inference Backend](#phase-3--inference-backend)
8. [Phase 4 — Frontend](#phase-4--frontend)
9. [Evaluation](#evaluation)
10. [API Reference](#api-reference)
11. [Environment Variables](#environment-variables)
12. [AMD Resource Utilization](#amd-resource-utilization)
13. [Dependencies](#dependencies)
14. [Key MI300X Implementation Notes](#key-mi300x-implementation-notes)

---

## Why Fine-Tuning

Frontier LLMs train on the public internet. The internet has roughly:

- ~10 million Python files on GitHub
- ~3 million CUDA files
- **~50,000 HIP files** (1,000× less than CUDA)
- **Almost zero MI300X-specific MFMA examples** (chip released late 2023)

**Target headline numbers:**

| Comparison | Target |
|---|---|
| ROCmForge vs naive HIPify | 1.5–3× speedup |
| ROCmForge vs `torch.eager` on ROCm | 1.2–2× speedup |
| ROCmForge vs `torch.compile` on ROCm | 0.9–1.3× (parity = win) |
| ROCmForge vs rocBLAS (hand-tuned) | 0.7–0.95× (close = win for AI-generated) |

---

## System Architecture

```
                ┌─────────────────────────────────────────────┐
                │             User's browser                  │
                │     (Vite + React + Monaco + Recharts)      │
                └──────────────────────┬──────────────────────┘
                              SSE / JSON over HTTP
                ┌──────────────────────▼──────────────────────┐
                │       FastAPI server  (inference/)          │
                │  /api/compile  /api/benchmark  /api/health  │
                │  /api/compile/stream  /api/full_pipeline    │
                │  /api/compare                               │
                └─────┬────────────────┬─────────────────┬────┘
                      │                │                 │
              ┌───────▼────┐    ┌──────▼──────┐    ┌─────▼──────┐
              │  vLLM      │    │  hipcc      │    │ PyTorch    │
              │  ROCmForge │    │  sandbox    │    │ baselines  │
              │  -7B       │    │  + rocprof  │    │ + rocBLAS  │
              └────────────┘    └─────────────┘    └────────────┘
                              all on AMD MI300X
```

### Pipeline Stages

| Stage | Component | Typical time |
|---|---|---|
| 1. Parse intent | Prompt template | < 50 ms |
| 2. Generate HIP kernel | ROCmForge-7B via vLLM | 1–3 s |
| 3. Compile | `hipcc` sandbox | 2–5 s |
| 4. Run + profile | `rocprof` CSV parsing | 1–2 s |
| 5. Compare to baseline | PyTorch eager + `torch.compile` | 1–2 s |
| 6. *(Optional)* Refine | Loop back to step 2 with perf data | +3 s/iter |

---

## Project Layout

```
rocmforge/
├── scripts/                    # Phase 1: data pipeline
│   ├── 01_clone_repos.sh       # clone CUDA/ROCm/Triton source repos
│   ├── 02_process_cuda_samples.py   # CUDA → HIP pairs via hipify + regex
│   ├── 03_extract_rocm_optimized.py # mine hand-tuned kernels from AMD repos
│   ├── 04_extract_triton.py    # @triton.jit kernels + test files
│   ├── 05_generate_instruction_data.py  # golden examples + synthetic pairs
│   ├── 06_augment_data.py      # dtype/size/rephrase augmentation; train/val split
│   └── 07_run_pipeline.sh      # orchestrate all 6 stages
│
├── data/
│   └── processed/              # JSONL intermediates + train.jsonl / val.jsonl
│                               
│
├── training/                   # Phase 2: LoRA fine-tuning
│   ├── config.yaml             # all hyperparameters (single source of truth)
│   ├── train.py                # SFTTrainer loop with NaN/collapse detection
│   ├── data_loader.py          # JSONL → Qwen chat template → SFT text field
│   ├── merge_model.py          # merge LoRA adapter into base for vLLM
│   ├── test_inference.py       # smoke-test merged model
│   ├── setup_instance.sh       # provision ROCm venv + PyTorch wheels
│   └── run_training.sh         # sanity check → full train → merge
│
├── eval/
│   └── eval_holdout.py         # heuristic scoring on val split
│
├── inference/                  # Phase 3: FastAPI + tooling
│   ├── server.py               # FastAPI app (vLLM or HF backend)
│   ├── models.py               # Pydantic request/response types
│   ├── sandbox.py              # hipcc compile sandbox → .o syntax check
│   ├── benchmark.py            # rocprof CSV parsing + TFLOPS helper
│   ├── baselines.py            # PyTorch eager / torch.compile / rocBLAS proxy
│   └── start_server.sh         # dependency check + uvicorn launcher
│
├── frontend/                   # Phase 4: Vite + React demo UI
│   ├── src/
│   │   ├── App.tsx             # 4 tabs: Generate, Compare, Live Compare, Eval
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── CodeEditor.tsx
│   │   │   ├── ExamplesGallery.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   └── PerformanceChart.tsx
│   │   └── lib/
│   │       ├── api.ts
│   │       └── examples.ts
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts          # /api/* proxy → localhost:8001
│
├── requirements.txt
├── vercel.json                 # static frontend deploy
└── README.md
```

---

## Quick Start

### Prerequisites

- AMD Instinct MI300X instance with **ROCm 6.2** pre-installed (phases 2–3)
- Python 3.10+, Node.js 18+ (phase 4 can run anywhere)

### 1 — Build the dataset (Phase 1, runs anywhere)

```bash
bash scripts/07_run_pipeline.sh
# Output: data/processed/train.jsonl  (~45 K examples)
#         data/processed/val.jsonl
```

### 2 — Fine-tune on MI300X (Phase 2)

```bash
bash training/setup_instance.sh    # create venv, install PyTorch ROCm + stack
bash training/run_training.sh      # sanity check → full 2-epoch LoRA run → merge
# Output: ./rocmforge-7b-merged/   (drop-in for vLLM)
```

### 3 — Start the backend (Phase 3, on MI300X)

```bash
bash inference/start_server.sh
# FastAPI + vLLM at http://0.0.0.0:8001
```

### 4 — Run the demo UI (Phase 4, anywhere)

```bash
cd frontend
npm install
npm run dev
# Vite dev server at http://localhost:5173
```

---

## Phase 1 — Data Pipeline

The pipeline runs entirely on CPU (no GPU required) and produces instruction-tuned JSONL pairs in `{instruction, input, output}` format.

### Scripts

| Script | Input | Output |
|---|---|---|
| `01_clone_repos.sh` | GitHub URLs | `data/raw/` (cuda-samples, composable_kernel, rocBLAS, MIOpen, triton, CUDALibrarySamples) |
| `02_process_cuda_samples.py` | `.cu` files | `01_cuda_to_hip_pairs.jsonl` — CUDA→HIP via `hipify-perl` + regex fallback |
| `03_extract_rocm_optimized.py` | AMD repos | `02_rocm_optimized_kernels.jsonl` — kernel functions + descriptions |
| `04_extract_triton.py` | Triton source | `03_triton_kernels.jsonl` — `@triton.jit` functions + tests |
| `05_generate_instruction_data.py` | Golden examples (inline) | `04_instruction_data.jsonl` — synthetic PyTorch→HIP + doc-style pairs |
| `06_augment_data.py` | All JSONLs above | `train.jsonl` + `val.jsonl` — dtype/size/rephrase augmentations + 90/10 split |
| `07_run_pipeline.sh` | — | Orchestrates all six stages end-to-end |

### Data row format

```json
{
  "instruction": "Convert this PyTorch matmul to an optimized HIP kernel for MI300X",
  "input": "import torch\nC = torch.mm(A, B)",
  "output": "__global__ void matmul_kernel(...) { ... }",
  "metadata": {
    "source": "ROCm/composable_kernel",
    "type": "optimized_kernel",
    "augmented": true,
    "augment_type": "rephrased"
  }
}
```

### Target dataset composition

| Type | Examples |
|---|---|
| CUDA → naive HIP | ~30,000 |
| ROCm-optimized kernels | ~5,000 |
| Triton → HIP | ~2,000 |
| Synthetic instruction pairs | ~5,000 |
| Augmented variants | remainder |
| **Total** | **~45,000** |

---

## Phase 2 — Fine-Tuning

### Model & method

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Fine-tuning method | LoRA (Rank 64, alpha 128, dropout 0.05) |
| LoRA target modules | q/k/v/o_proj, gate/up/down_proj |
| Precision | bfloat16 weights |
| Attention implementation | `eager` (avoids bf16 SDPA backward NaNs on ROCm) |
| Optimizer | `adamw_torch` (no bitsandbytes — not ROCm-compatible) |
| Effective batch size | 16 (per_device=1, grad_accum=16) |
| Learning rate | 2e-4, cosine scheduler, 3% warmup |
| Max sequence length | 4096 |
| Epochs | 2 |
| Eval strategy | every 500 steps, `load_best_model_at_end=True` |
| NaN/collapse guard | `train.py` exits non-zero on loss collapse |

All hyperparameters live in `training/config.yaml` and can be overridden on the command line.

### System prompt (baked into every training example)

```
You are ROCmForge, an expert AMD GPU kernel optimizer. You specialize in:
- Converting PyTorch, CUDA, and Triton code to optimized HIP kernels
- Targeting AMD Instinct MI300X (gfx942) architecture
- Using wavefront-64 operations, MFMA intrinsics, and LDS optimization
- Maximizing TFLOPS and memory bandwidth utilization
Generate compilable, correct, high-performance HIP/ROCm code.
```

### Training commands

```bash
# Sanity check on 200 samples
python training/train.py --config training/config.yaml --max_samples 200 --sanity_check

# Full training run
python training/train.py --config training/config.yaml

# Merge LoRA adapter into base model for vLLM deployment
python training/merge_model.py \
  --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter_path checkpoints/final-adapter \
  --output_dir ./rocmforge-7b-merged

# Smoke test
python training/test_inference.py --model_path ./rocmforge-7b-merged
```

### Note on PyTorch installation (ROCm)

`requirements.txt` intentionally excludes PyTorch; install it separately before training:

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2
```

---

## Phase 3 — Inference Backend

### Server

`inference/server.py` — async FastAPI app with lifespan model loading.

**Default backend**: vLLM (`ROCMFORGE_USE_VLLM=1`)  
**Fallback backend**: Hugging Face `pipeline` (set `ROCMFORGE_USE_VLLM=0`)  
**vLLM config**: `max_model_len=8192`, `gpu_memory_utilization=0.85`, `dtype=bfloat16`

The `/api/compare` endpoint lazy-loads the **base** Qwen2.5-Coder-7B model for side-by-side comparison in the Live Compare UI tab.

### Sandbox (`inference/sandbox.py`)

- Receives generated HIP code from the model
- Writes to a temp file and invokes `hipcc --offload-arch=gfx942 -c` (syntax-only compile to `.o`)
- Returns `{success, errors, warnings}` — does **not** link or execute the binary
- `hipcc` path configurable via `HIPCC` env var

### Benchmark (`inference/benchmark.py`)

- Invokes `rocprof --stats` on a compiled executable
- Parses the CSV output for `DurationNs`, computes TFLOPS from problem size
- Returns `{kernel_time_ns, tflops, occupancy, memory_bandwidth_gb_s}`
- Available only when the sandbox produces a fully executable binary (currently `.o` only, so baseline TFLOPS are computed via PyTorch)

### Baselines (`inference/baselines.py`)

- **PyTorch eager**: plain `torch.matmul` on `cuda` device
- **torch.compile**: `torch.compile(fn)` with default backend
- **rocBLAS proxy**: estimated TFLOPS from hardware peak for the given problem size

### Start the server

```bash
# Via wrapper script (recommended)
bash inference/start_server.sh

# Or directly
uvicorn inference.server:app --host 0.0.0.0 --port 8001

# Custom model path
ROCMFORGE_MODEL_PATH=/path/to/merged-model \
ROCMFORGE_USE_VLLM=1 \
uvicorn inference.server:app --host 0.0.0.0 --port 8001
```

---

## Phase 4 — Frontend

Built with **Vite + React 18 + TypeScript + Tailwind CSS**.

### Tabs

| Tab | Description |
|---|---|
| **Generate** | Two-pane Monaco editor: input (PyTorch/CUDA/Triton) → HIP output with SSE streaming |
| **Compare Models** | Side-by-side diff of ROCmForge output vs base Qwen |
| **Live Compare** | Real-time streaming comparison via `/api/compare` |
| **Eval Results** | Displays `eval/eval_results.json` from the holdout eval run |

### Components

| Component | Role |
|---|---|
| `Header.tsx` | AMD-branded header; polls `/api/health` for GPU/model status |
| `CodeEditor.tsx` | Monaco editor wrapper (C++ output, Python/CUDA input) |
| `ExamplesGallery.tsx` | Pre-baked demo prompts (matmul, flash-attn, LayerNorm, softmax) |
| `PerformanceChart.tsx` | Recharts bar chart — PyTorch eager / torch.compile / ROCmForge / rocBLAS |
| `StatsPanel.tsx` | TFLOPS, occupancy, kernel time, memory bandwidth tiles |

### Run the frontend

```bash
cd frontend
npm install
npm run dev          # dev server at http://localhost:5173

# Point to a remote backend
VITE_API_URL=https://your-mi300x-host:8001 npm run dev

# Production build
npm run build        # outputs to frontend/dist/
```

The dev server proxies `/api/*` to `http://localhost:8001` via `vite.config.ts`.

---

## Evaluation

`eval/eval_holdout.py` loads the merged model and runs heuristic scoring on `val.jsonl`.

### Scoring criteria

Each generated output is scored for the presence of:

- HIP API markers (`hipLaunchKernelGGL`, `__global__`, `__device__`)
- MFMA intrinsics (`__builtin_amdgcn_mfma_*`)
- LDS usage (`__shared__`)
- gfx942-specific patterns
- Absence of CUDA-only APIs

### Usage

```bash
python eval/eval_holdout.py \
  --model_path ./rocmforge-7b-merged \
  --val_file data/processed/val.jsonl \
  --max_samples 200 \
  --output_file eval/eval_results.json
```

Results are written to `eval/eval_results.json` and displayed in the **Eval Results** tab of the frontend.

---

## API Reference

All endpoints are served by `inference/server.py` on port **8001** by default.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service status, GPU info, model name, vLLM/HF backend |
| `/api/compile` | POST | Generate HIP code from input; returns JSON |
| `/api/compile/stream` | POST | Same generation, chunked via **Server-Sent Events** |
| `/api/benchmark` | POST | Compile + profile with `rocprof` + compare to PyTorch baselines |
| `/api/full_pipeline` | POST | Generate + compile + benchmark in one call (demo "Run" button) |
| `/api/compare` | POST | Run same prompt through ROCmForge *and* base Qwen; return both |

### `POST /api/compile` — request body

```json
{
  "input_code": "import torch\nC = torch.mm(A, B)",
  "input_lang": "pytorch",
  "target": "mi300x",
  "temperature": 0.2,
  "max_tokens": 2048
}
```

`input_lang` values: `"pytorch"`, `"cuda"`, `"triton"`, `"english"`

### `GET /api/health` — example response

```json
{
  "status": "ok",
  "model": "rocmforge-7b",
  "backend": "vllm",
  "gpu_count": 1,
  "gpu_name": "AMD Instinct MI300X",
  "rocm_version": "6.2"
}
```

---

## Environment Variables

### Backend (`inference/server.py`, `inference/start_server.sh`)

| Variable | Default | Description |
|---|---|---|
| `ROCMFORGE_MODEL_PATH` | `./rocmforge-7b-merged` | Path to merged model |
| `ROCMFORGE_MODEL_NAME` | `rocmforge-7b` | Display name in `/api/health` |
| `ROCMFORGE_USE_VLLM` | `1` | `1` = vLLM backend, `0` = HF pipeline |
| `ROCMFORGE_BASE_MODEL_PATH` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Base model for `/api/compare` |
| `PORT` | `8001` | Server port |
| `HIPCC` | `hipcc` | Path to `hipcc` binary |
| `ROCMFORGE_TARGET_ARCH` | `gfx942` | `--offload-arch` passed to `hipcc` |
| `ROCPROF` | `rocprof` | Path to `rocprof` binary |

### Frontend (`frontend/vite.config.ts`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8001` | Backend URL for dev proxy |

---

## AMD Resource Utilization

| Resource | Where it is used |
|---|---|
| **AMD Instinct MI300X** | LoRA fine-tuning (192 GB HBM3 fits full bf16 weights + 32 K context), vLLM serving, rocprof profiling |
| **ROCm 6.2** | PyTorch ROCm wheels; `torch.bfloat16` training without `bitsandbytes` |
| **HIP / hipcc** | Compilation sandbox in `inference/sandbox.py` (`--offload-arch=gfx942`) |
| **rocprof** | Kernel profiling in `inference/benchmark.py` |
| **rocBLAS / MIOpen** | Baseline TFLOPS reference in the demo bar chart |
| **HIPify** | CUDA→HIP translation in `scripts/02_process_cuda_samples.py` |
| **Composable Kernel** | Gold-standard training examples in Phase 1 data pipeline |

### Why MI300X specifically enables this project

- **192 GB HBM3**: holds Qwen-7B in bf16 with room for 4 K sequence batches — a 24 GB or 80 GB GPU cannot.
- **MFMA instruction set (gfx942)**: the target architecture the model is trained to emit; the only way to verify correctness is to compile on the actual hardware.
- **Higher HBM bandwidth than H100**: faster token generation during the live demo.
- **AMD flywheel narrative**: trained on MI300X *to make MI300X faster* — "We used AMD to make AMD better."

---

## Dependencies

### Python (`requirements.txt`)

```
pyyaml>=6.0
transformers>=4.44.0,<5.0.0
peft>=0.12.0
trl>=0.9.0,<0.12.0
datasets>=2.20.0
accelerate>=0.33.0
wandb>=0.17.0

fastapi>=0.111.0
uvicorn>=0.30.0

pandas>=2.2.0
numpy>=1.26.0
```

Install separately (ROCm-specific, not on PyPI):

```bash
# PyTorch ROCm
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2

# vLLM (use ROCm Docker image or build from source)
pip install vllm   # or: docker pull rocm/vllm:latest
```

> **Note**: `bitsandbytes` is intentionally excluded — the default CUDA-only build crashes on ROCm. The training script uses `optim: "adamw_torch"` which works natively and is unproblematic given MI300X's 192 GB HBM3.

### Frontend (`frontend/package.json`)

```
react, react-dom (^18)
@monaco-editor/react
recharts
zustand
tailwindcss, postcss, autoprefixer
typescript, vite
```

---

## Key MI300X Implementation Notes

### Attention implementation

`training/config.yaml` sets `attn_implementation: "eager"`. This uses the manual attention path which upcasts softmax to fp32, avoiding NaN gradients that occur with SDPA's bf16-only backward pass on ROCm. Switch to `"flash_attention_2"` if you install the ROCm-built `flash-attn` package for a 2–3× training speedup.

### Always compile for the target arch

All `hipcc` invocations pass `--offload-arch=gfx942`. The default arch may produce incorrect or slow code on MI300X.

### Critical MFMA intrinsics (gfx942)

```c
__builtin_amdgcn_mfma_f32_16x16x16f16   // primary FP16 matmul
__builtin_amdgcn_mfma_f32_32x32x8f16    // larger tile FP16
__builtin_amdgcn_mfma_f32_16x16x16bf16_1k  // BF16 variant
__builtin_amdgcn_mfma_i32_16x16x16i8    // INT8 quantized
```

### Wavefront 64 vs warp 32

AMD uses **wavefront-64** (64 threads); NVIDIA uses warp-32. Naive CUDA-ported kernels that assume 32-thread warps will silently halve utilization on AMD. The training data hammers this distinction throughout.

### Graceful degradation

If `rocprof` is unavailable or the generated kernel fails to compile to an executable, the server returns `compile_success: false` with the `hipcc` stderr rather than crashing. Baseline TFLOPS are still computed via PyTorch.
