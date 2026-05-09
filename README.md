# ROCmForge — The AMD Performance Compiler

ROCmForge is a fine-tuned LLM (Qwen2.5-Coder-7B) that converts PyTorch / CUDA /
Triton / natural-language descriptions into hand-tuned **HIP / ROCm kernels
optimized for AMD Instinct MI300X (gfx942)**. The model is trained to use
wavefront-64 ops, MFMA intrinsics, LDS staging, and HBM3-aware vectorization.

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Phase 1        │ →  │  Phase 2     │ →  │  Phase 3     │ →  │  Phase 4     │
│  Data Pipeline  │    │  Fine-Tuning │    │  Backend     │    │  Frontend    │
│  scripts/       │    │  training/   │    │  inference/  │    │  frontend/   │
└─────────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Quick start

### 1. Clone repos and prepare data (Phase 1 — anywhere)

```bash
bash scripts/07_run_pipeline.sh
# Output: data/processed/train.jsonl + val.jsonl
```

### 2. Set up the MI300X instance and fine-tune (Phase 2)

```bash
bash training/setup_instance.sh
bash training/run_training.sh
# Output: ./rocmforge-7b-merged/   (deployable model)
```

### 3. Start the backend (Phase 3 — on MI300X)

```bash
bash inference/start_server.sh
# FastAPI + vLLM at http://0.0.0.0:8000
```

### 4. Run the demo UI (Phase 4 — anywhere)

```bash
cd frontend
npm install
npm run dev
# Vite dev server at http://localhost:5173
```

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │             User's browser                  │
                │     (Vite + React + Monaco + Recharts)      │
                └──────────────────────┬──────────────────────┘
                              SSE / JSON over HTTP
                ┌──────────────────────▼──────────────────────┐
                │       FastAPI server  (inference/)          │
                │   /api/compile  /api/benchmark  /api/health │
                └─────┬────────────────┬─────────────────┬────┘
                      │                │                 │
              ┌───────▼────┐    ┌──────▼──────┐    ┌─────▼──────┐
              │  vLLM      │    │  hipcc      │    │ PyTorch    │
              │  rocmforge │    │  sandbox    │    │ baselines  │
              │  -7b       │    │  + rocprof  │    │ + rocBLAS  │
              └────────────┘    └─────────────┘    └────────────┘
                              all on AMD MI300X
```

## Project layout

```
rocmforge/
├── scripts/             # Phase 1: data pipeline (clone, hipify, augment)
├── data/
│   ├── raw/             # cloned GitHub repos        (.gitignored)
│   └── processed/       # train.jsonl, val.jsonl     (.gitignored)
├── training/            # Phase 2: QLoRA fine-tuning
│   ├── setup_instance.sh
│   ├── config.yaml
│   ├── train.py
│   ├── data_loader.py
│   ├── merge_model.py
│   ├── test_inference.py
│   └── run_training.sh
├── eval/                # Holdout-set quality eval
│   └── eval_holdout.py
├── inference/           # Phase 3: FastAPI + sandbox + benchmark
│   ├── server.py
│   ├── models.py
│   ├── sandbox.py       # hipcc compilation sandbox
│   ├── benchmark.py     # rocprof harness
│   ├── baselines.py     # PyTorch eager / torch.compile / rocBLAS
│   └── start_server.sh
├── frontend/            # Phase 4: Vite + React UI
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── README.md
├── rocmforge-7b-merged/ # Trained model output (.gitignored)
└── checkpoints/         # LoRA adapters         (.gitignored)
```

## How AMD resources are used

| Resource          | Where it shows up                                           |
| ----------------- | ----------------------------------------------------------- |
| AMD MI300X        | Data filtering, full QLoRA fine-tuning, vLLM serving        |
| ROCm 6.2          | PyTorch ROCm wheels (`rocm6.2` index)                       |
| HIP / hipcc       | Compilation sandbox in `inference/sandbox.py`               |
| rocprof           | Performance profiling in `inference/benchmark.py`           |
| rocBLAS / MIOpen  | "Oracle" baseline in the demo bar chart                     |
| Hugging Face Optimum-AMD | (optional) Quantized inference path                  |

## API reference

| Endpoint                  | Method | Use                                   |
| ------------------------- | ------ | ------------------------------------- |
| `/api/health`             | GET    | Service / GPU / model status          |
| `/api/compile`            | POST   | Generate HIP code (returns JSON)      |
| `/api/compile/stream`     | POST   | Stream tokens via Server-Sent Events  |
| `/api/benchmark`          | POST   | Compile + run + profile + compare     |
| `/api/full_pipeline`      | POST   | One-shot for the demo "Run" button    |

See [`ROCmForge-Implementation-Plan.md`](../ROCmForge-Implementation-Plan.md) at
the repo root for the full hackathon plan, timeline, and pitch outline.
