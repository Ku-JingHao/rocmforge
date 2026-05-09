# ROCmForge — The AMD Performance Compiler

> **Tagline**: *"Any GPU code in. Hand-tuned ROCm out. AMD goes faster."*
>
> A fine-tuned LLM that takes ANY GPU compute description (PyTorch, CUDA, Triton, OpenCL, pseudocode) and produces hand-tuned HIP/ROCm kernels optimized for AMD MI300X — often outperforming the original.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Killer Defense: Why Fine-Tuning > Frontier LLM](#2-the-killer-defense)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack & Rationale](#4-tech-stack--rationale)
5. [AMD Resource Utilization Map](#5-amd-resource-utilization-map)
6. [Data Sources (Direct Links + APIs)](#6-data-sources)
7. [Fine-Tuning Implementation](#7-fine-tuning-implementation)
8. [Evaluation & Benchmarking](#8-evaluation--benchmarking)
9. [Backend Implementation (FastAPI + vLLM)](#9-backend-implementation)
10. [Frontend Implementation (Vite + React)](#10-frontend-implementation)
11. [30-Hour Hour-by-Hour Timeline](#11-30-hour-timeline)
12. [Demo Script (The 5-Minute Pitch)](#12-demo-script)
13. [Submission Checklist](#13-submission-checklist)
14. [Stretch Goals](#14-stretch-goals)

---

## 1. Executive Summary

### The Problem
99% of "GPU developers" today write **PyTorch**, not raw CUDA. They want speed, not migration. Meanwhile:
- AMD MI300X is **2-3x cheaper** per FLOP than NVIDIA H100
- But naive PyTorch on ROCm leaves **20-50% performance** on the table
- Hand-tuning HIP kernels for MI300X is **PhD-level work** (chiplet topology, MFMA instructions, wavefront scheduling, LDS bank conflicts)

### The Solution
**ROCmForge** — a fine-tuned 7B-14B code LLM that:
1. Accepts input in PyTorch / CUDA / Triton / OpenCL / English description
2. Generates AMD-native HIP kernels with MI300X-specific optimizations
3. Compiles, profiles, and self-refines using `rocprof` feedback
4. Outputs production-ready code + performance report

### What Makes This Win the Hackathon
- **Pull-based demand**: every ML team wants speed, not migration
- **Bulletproof "why fine-tune?" defense**: GPT-4 has near-zero data on MI300X intrinsics
- **AMD strategic alignment**: makes AMD GPUs *faster*, not just compatible
- **Killer demo**: live perf benchmark showing fine-tuned model beats torch.compile baseline
- **Cloud provider angle**: CoreWeave, Lambda, Crusoe immediately want this

---

## 2. The Killer Defense (Why Fine-Tuning Beats GPT-4)

When a judge asks *"why not just use GPT-4?"*, you have 30 seconds to win:

### Live Demo Beat
1. Open ChatGPT in browser
2. Prompt: *"Write a HIP kernel for matmul that uses MI300X MFMA intrinsics with wavefront-64 optimization."*
3. GPT-4 hallucinates `__builtin_amdgcn_mfma_f32_16x16x16f16` with wrong signatures, invents non-existent intrinsics, uses CUDA-style `__shared__` instead of `__shared__` HIP attribute, ignores LDS bank conflicts entirely.
4. Now run yours. Show working, compilable code that benchmarks at 1.4x faster.

### The Data Argument
Frontier LLMs train on the public internet. The internet has:
- ~10 million Python files on GitHub
- ~3 million CUDA files
- **~50,000 HIP files** (1000x less)
- **Almost zero MI300X-specific MFMA examples** (chip released late 2023)

Fine-tuning is the *only* way to make a model good at MI300X-targeted code.

---

## 3. System Architecture

```
                          ROCmForge System
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│   │  Vite + React   │───▶│   FastAPI        │───▶│  vLLM ROCm    │  │
│   │  Monaco Editor  │    │   /api/compile   │    │  Backend      │  │
│   │  Recharts       │◀───│   /api/benchmark │◀───│  ROCmForge-7B │  │
│   └─────────────────┘    └─────────┬────────┘    │  (LoRA tuned) │  │
│                                    │             └───────────────┘  │
│                                    ▼                                 │
│                          ┌──────────────────┐                        │
│                          │  Sandbox Runner  │                        │
│                          │  ┌────────────┐  │                        │
│                          │  │  hipcc     │  │  Compile             │
│                          │  ├────────────┤  │                        │
│                          │  │  rocprof   │  │  Profile             │
│                          │  ├────────────┤  │                        │
│                          │  │  PyTorch   │  │  Reference baseline  │
│                          │  └────────────┘  │                        │
│                          └──────────────────┘                        │
│                                    │                                 │
│                                    ▼                                 │
│                       ┌────────────────────────┐                     │
│                       │  Refinement Loop       │                     │
│                       │  (if perf < target)    │                     │
│                       │  Feeds rocprof JSON    │                     │
│                       │  back to model         │                     │
│                       └────────────────────────┘                     │
│                                                                      │
│   AMD Instinct MI300X (192GB HBM3) — Training + Serving + Profiling  │
└──────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages
| Stage | Component | Time |
|-------|-----------|------|
| 1. Parse intent | Lightweight prompt template | <50ms |
| 2. Generate HIP kernel | ROCmForge-7B (vLLM) | 1-3s |
| 3. Compile | `hipcc` in sandbox | 2-5s |
| 4. Run + profile | `rocprof` JSON output | 1-2s |
| 5. Compare to baseline | PyTorch eager + torch.compile | 1-2s |
| 6. (Optional) Refine | Loop back to (2) with perf data | +3s/iter |

---

## 4. Tech Stack & Rationale

### Models
| Component | Choice | Why |
|-----------|--------|-----|
| Base model | **Qwen2.5-Coder-7B-Instruct** | Best open code model at 7B; comfortably fits MI300X for full fine-tune; fast inference for live demo |
| Backup base | DeepSeek-Coder-6.7B | If Qwen has issues |
| Stretch base | Qwen2.5-Coder-32B (LoRA) | If time permits, scale up; fits MI300X with 192GB HBM3 |

### Training
| Component | Choice | Why |
|-----------|--------|-----|
| Framework | **HuggingFace `transformers` + `peft` + `trl`** | Best ROCm support, well-tested |
| Method | **QLoRA (rank 64, alpha 128)** | Fast iteration in 30h; quality close to full fine-tune |
| Data format | **ChatML / instruction format** | Compatible with Qwen's native template |
| Optimizer | **AdamW 8-bit (bitsandbytes ROCm)** | Memory efficient |
| Acceleration | **Flash Attention 2 (ROCm port)** | 2-3x speedup |
| Scheduler | **Cosine with 3% warmup** | Standard, robust |

### Serving
| Component | Choice | Why |
|-----------|--------|-----|
| Inference engine | **vLLM (ROCm build)** | Best throughput, paged attention, ROCm-supported |
| API server | **FastAPI** | Async, fast, easy WebSocket support for streaming |
| Sandbox | **Docker container with ROCm + hipcc** | Isolation for arbitrary kernel compilation |

### Frontend
| Component | Choice | Why |
|-----------|--------|-----|
| Build tool | **Vite** | (Per requirement) Fast HMR, great DX |
| Framework | **React 18 + TypeScript** | Industry standard |
| Code editor | **Monaco Editor** (`@monaco-editor/react`) | VS Code-grade, syntax highlighting for C/C++ |
| Charts | **Recharts** | Beautiful perf comparison bar charts |
| UI library | **Tailwind CSS + shadcn/ui** | Modern, fast to build |
| State | **Zustand** | Lighter than Redux, perfect for hackathon |
| Streaming | **EventSource (SSE)** | Stream model output to UI |

### Compilation & Profiling (AMD Stack)
| Tool | Purpose |
|------|---------|
| `hipcc` | HIP → MI300X binary compilation |
| `rocprof` | Kernel profiling (TFLOPS, latency, occupancy) |
| `rocBLAS` | Reference baseline for matmul, GEMM |
| `MIOpen` | Reference baseline for conv, reduction |
| `Composable Kernel` | Hand-tuned kernel templates (training data + reference) |
| `HIPify` | CUDA→HIP translator (data generation) |

---

## 5. AMD Resource Utilization Map

### Where AMD MI300X Specifically Enables This Project

| Phase | AMD Resource | Why MI300X Specifically |
|-------|--------------|-------------------------|
| **Data Generation** | MI300X compute | Run HIPify on 10K+ CUDA repos in parallel; profile every generated kernel with `rocprof` |
| **Fine-Tuning** | 192GB HBM3 | Fits Qwen-7B fully + 32K context for whole-kernel + perf-trace context. A 24GB or 80GB GPU literally cannot hold this. |
| **Profiling Loop** | ROCm + `rocprof` | Ground truth performance data to train against (DPO with fast vs slow kernel pairs) |
| **Serving** | vLLM ROCm | Fast inference for live demo; MI300X has higher HBM bandwidth than H100 → faster token generation |
| **Demo Benchmarks** | MI300X + rocBLAS | Live "your kernel vs hand-tuned" comparison |
| **Marketing Story** | Entire AMD AI stack | "We used MI300X to make MI300X faster" — the AMD flywheel narrative |

### AMD Developer Cloud Setup
```bash
# After provisioning MI300X instance from AMD Developer Cloud
# The instance comes with ROCm pre-installed

# Verify
rocm-smi
rocminfo

# Python environment
python -m venv venv && source venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers peft trl bitsandbytes accelerate
pip install vllm  # ROCm-built version
```

### Cost Estimate (Within $100 Credit)
| Activity | Hours | Approx Cost |
|----------|-------|-------------|
| Data generation pipeline | 4h on 1x MI300X | ~$8 |
| Fine-tuning runs (3 iterations) | 12h on 1x MI300X | ~$24 |
| Eval + benchmarking | 4h on 1x MI300X | ~$8 |
| Demo serving | 8h on 1x MI300X | ~$16 |
| Buffer | 6h | ~$12 |
| **TOTAL** | ~34h | **~$68** |

Comfortably within $100 credit, leaves room for re-runs.

---

## 6. Data Sources

### A. CUDA → HIP Pairs (Primary)

#### Source 1: HIPify on GitHub CUDA Code
```bash
# Auto-translate CUDA → HIP
git clone https://github.com/ROCm/HIPIFY
hipify-clang input.cu -o output.hip

# Scrape CUDA repos via GitHub API
curl -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/search/code?q=extension:cu+language:cuda&per_page=100"
```
**Direct repos to mine:**
- `NVIDIA/cuda-samples` → all samples have ground-truth pairs
- `NVIDIA/CUDALibrarySamples`
- Top 1000 CUDA repos by stars

#### Source 2: AMD's Composable Kernel (Gold Standard)
- Repo: https://github.com/ROCm/composable_kernel
- **Hand-tuned MI300X kernels** with extensive comments — perfect training data for "optimized" examples
- Use `git log` to find optimization-improving commits → DPO pairs

#### Source 3: rocBLAS / MIOpen Source
- https://github.com/ROCm/rocBLAS
- https://github.com/ROCm/MIOpen
- Reference implementations of every standard linear algebra operation

### B. PyTorch → HIP Kernel Pairs

#### Source 4: torch.compile Trace Data
```python
import torch
import torch._inductor.config as config
config.trace.enabled = True
config.trace.debug_log = True

@torch.compile
def my_func(x): return x.relu().sum()

my_func(torch.randn(1024, device='cuda'))
# Extracts the generated kernels for training pairs
```

#### Source 5: Triton → HIP
- Triton repo: https://github.com/openai/triton
- `python/test/unit/language/test_*.py` has hundreds of kernel examples
- Triton compiles to both CUDA and HIP via different backends → free pairs

### C. Synthetic Data Generation (Critical for Volume)

#### Pipeline Code (run on MI300X)
```python
import subprocess, json
from pathlib import Path

def generate_pair(cuda_file: Path):
    # Step 1: HIPify (naive translation)
    naive_hip = subprocess.check_output(
        ['hipify-clang', str(cuda_file), '--print-stats'],
        text=True
    )
    
    # Step 2: Compile both
    cuda_perf = compile_and_profile_cuda(cuda_file)  # via NVCC
    naive_perf = compile_and_profile_hip(naive_hip)  # via hipcc + rocprof
    
    # Step 3: Use a teacher model to optimize, score with rocprof
    optimized_hip = teacher_optimize(naive_hip, naive_perf)
    optimized_perf = compile_and_profile_hip(optimized_hip)
    
    return {
        'input_cuda': cuda_file.read_text(),
        'naive_hip': naive_hip,
        'naive_tflops': naive_perf['tflops'],
        'optimized_hip': optimized_hip,
        'optimized_tflops': optimized_perf['tflops'],
        'speedup': optimized_perf['tflops'] / naive_perf['tflops'],
    }
```

### D. AMD Architecture Documentation (Knowledge Base)
| Document | URL | Use |
|----------|-----|-----|
| MI300X ISA Reference | https://www.amd.com/system/files/TechDocs/cdna3-shader-instruction-set-architecture-feb-2024.pdf | Train model on MFMA intrinsics |
| ROCm HIP Programming Guide | https://rocm.docs.amd.com/projects/HIP/ | Best-practices instructions |
| ROCm Performance Guide | https://rocm.docs.amd.com/en/latest/conceptual/optimization-guide.html | Optimization patterns |
| Composable Kernel docs | https://github.com/ROCm/composable_kernel/wiki | Template patterns |

Convert all docs to instruction-following format using a teacher model:
```python
# Convert "MFMA instructions provide..." into:
# {"instruction": "How do I use MFMA on MI300X?", "output": "..."}
```

### E. Target Dataset Size

| Type | Examples | How |
|------|----------|-----|
| CUDA→naive HIP | 30,000 | Auto-HIPify on GitHub scrape |
| naive HIP→optimized HIP | 5,000 | Profile + teacher optimization |
| PyTorch→fused HIP | 3,000 | torch.compile traces |
| Triton→HIP | 2,000 | Triton dual-backend |
| Doc instruction pairs | 5,000 | Doc parsing + teacher conversion |
| Hand-curated golden examples | 100 | You + manual tuning |
| **TOTAL** | **~45,000** | Doable in 4-6 hours on MI300X |

---

## 7. Fine-Tuning Implementation

### Directory Structure
```
rocmforge/
├── data/
│   ├── raw/              # Scraped repos
│   ├── processed/        # JSONL training data
│   └── golden/           # Hand-curated test set
├── training/
│   ├── train.py          # Main training script
│   ├── config.yaml       # Hyperparameters
│   └── data_loader.py
├── inference/
│   ├── server.py         # vLLM + FastAPI
│   └── sandbox.py        # Compilation runner
├── eval/
│   ├── benchmark.py      # rocprof harness
│   └── correctness.py    # Output verification
├── frontend/             # Vite + React
└── README.md
```

### Training Script (`training/train.py`)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="data/processed/train.jsonl")

def format_example(ex):
    messages = [
        {"role": "system", "content": "You are ROCmForge, an expert AMD GPU kernel optimizer."},
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_example)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    max_seq_length=8192,             # MI300X enables long context
    optim="adamw_8bit",
    gradient_checkpointing=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model, args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
)

trainer.train()
trainer.save_model("./rocmforge-7b-lora-final")
```

### Stretch: DPO for Performance Pairs
```python
# After SFT, train preference model using profiled (slow, fast) pairs
from trl import DPOTrainer

# Each example: {"prompt": ..., "chosen": fast_kernel, "rejected": slow_kernel}
dpo_trainer = DPOTrainer(model=model, beta=0.1, ...)
dpo_trainer.train()
```

---

## 8. Evaluation & Benchmarking

### Three Metrics That Matter

#### Metric 1: Functional Correctness
```python
def check_correctness(generated_hip: str, reference_cuda: str) -> bool:
    inputs = generate_random_inputs()
    cuda_output = run_cuda(reference_cuda, inputs)
    hip_output = run_hip(generated_hip, inputs)
    return torch.allclose(cuda_output, hip_output, rtol=1e-3, atol=1e-3)
```
**Target**: >90% on golden test set

#### Metric 2: Compilation Success Rate
```python
def check_compiles(generated_hip: str) -> bool:
    result = subprocess.run(
        ['hipcc', '-O3', '--offload-arch=gfx942', '-c', '-'],
        input=generated_hip, capture_output=True, text=True
    )
    return result.returncode == 0
```
**Target**: >95%

#### Metric 3: Performance vs Baseline
```python
def benchmark(kernel_path: str, problem_size: tuple) -> dict:
    cmd = ['rocprof', '--stats', '--basenames', 'on',
           '-o', 'profile.csv', kernel_path]
    subprocess.run(cmd, check=True)
    
    df = pd.read_csv('profile.csv')
    return {
        'kernel_time_ns': df['DurationNs'].iloc[0],
        'tflops': compute_tflops(problem_size, df['DurationNs'].iloc[0]),
        'occupancy': df['OCCUPANCY'].iloc[0],
        'memory_bandwidth_gb_s': df['MemBandwidth'].iloc[0],
    }
```

### Benchmark Suite
| Operation | Reference | Sizes |
|-----------|-----------|-------|
| GEMM (FP16) | rocBLAS hgemm | 1024², 4096², 16384² |
| GEMM (BF16) | rocBLAS gemm_ex | 1024², 4096² |
| Conv2D | MIOpen | ResNet-50 layers |
| Attention | FlashAttention-2 ROCm | seq_len 2K, 8K, 32K |
| LayerNorm | torch.nn.LayerNorm | 4096 dim |
| Reduction | torch.sum | 100M elements |

### Headline Numbers To Aim For
| Comparison | Target |
|------------|--------|
| ROCmForge vs naive HIPify | **1.5-3x speedup** |
| ROCmForge vs torch eager on ROCm | **1.2-2x speedup** |
| ROCmForge vs torch.compile on ROCm | **0.9-1.3x** (parity = win, since torch.compile is mature) |
| ROCmForge vs rocBLAS (hand-tuned) | **0.7-0.95x** (close = win for AI-generated) |

---

## 9. Backend Implementation

### `inference/server.py`
```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import asyncio
from sandbox import compile_and_benchmark

app = FastAPI()

llm = LLM(
    model="./rocmforge-7b-lora-final-merged",
    dtype="bfloat16",
    max_model_len=16384,
    gpu_memory_utilization=0.85,
)

class CompileRequest(BaseModel):
    input_code: str
    input_lang: str  # "cuda" | "pytorch" | "triton" | "english"
    target: str = "mi300x"

@app.post("/api/compile")
async def compile_endpoint(req: CompileRequest):
    prompt = build_prompt(req)
    sampling = SamplingParams(temperature=0.2, max_tokens=4096)
    
    outputs = llm.generate([prompt], sampling)
    hip_code = extract_code_block(outputs[0].outputs[0].text)
    
    bench = await compile_and_benchmark(hip_code)
    
    return {
        "hip_code": hip_code,
        "compile_success": bench["compiled"],
        "tflops": bench["tflops"],
        "speedup_vs_baseline": bench["speedup"],
        "occupancy": bench["occupancy"],
    }

@app.post("/api/compile/stream")
async def compile_stream(req: CompileRequest):
    """Server-Sent Events for live token streaming to UI."""
    async def event_generator():
        prompt = build_prompt(req)
        async for token in llm.generate_async(prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### `inference/sandbox.py`
```python
import subprocess, tempfile, json, os
from pathlib import Path

ROCPROF_TEMPLATE = """
#include <hip/hip_runtime.h>
{kernel_code}

int main() {{
  // Auto-generated benchmark harness
  {benchmark_harness}
  return 0;
}}
"""

async def compile_and_benchmark(hip_code: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "kernel.hip"
        bin_path = Path(tmp) / "kernel.bin"
        src.write_text(ROCPROF_TEMPLATE.format(kernel_code=hip_code, benchmark_harness=...))
        
        compile_result = subprocess.run(
            ['hipcc', '-O3', '--offload-arch=gfx942', str(src), '-o', str(bin_path)],
            capture_output=True, text=True, timeout=60,
        )
        if compile_result.returncode != 0:
            return {"compiled": False, "error": compile_result.stderr}
        
        prof = subprocess.run(
            ['rocprof', '--stats', '-o', f'{tmp}/prof.csv', str(bin_path)],
            capture_output=True, text=True, timeout=120,
        )
        
        return parse_rocprof_output(f'{tmp}/prof.csv')
```

---

## 10. Frontend Implementation

### Setup
```bash
npm create vite@latest rocmforge-ui -- --template react-ts
cd rocmforge-ui
npm install @monaco-editor/react recharts zustand
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npx shadcn-ui@latest init
```

### Page Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  ROCmForge — AMD Performance Compiler          [Powered by MI300X]  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │  INPUT (Monaco Editor)   │    │  OUTPUT (Monaco Editor)  │      │
│  │  Language: [PyTorch ▼]   │ ⮕ │  HIP / ROCm Kernel        │      │
│  │                          │    │  (streaming as generated) │      │
│  │  import torch            │    │                           │      │
│  │  def matmul(a, b):       │    │  __global__ void mat_     │      │
│  │      return a @ b        │    │  mul_kernel(...) {        │      │
│  │                          │    │    using namespace ck;    │      │
│  └──────────────────────────┘    └──────────────────────────┘      │
│                                                                     │
│   [Compile & Benchmark]   [Try Examples ▼]    [Download .hip]      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  PERFORMANCE COMPARISON (Recharts)                                  │
│  ┌─────────────────────────────────────────────────┐                │
│  │  ████████████████████  PyTorch eager: 142 TFLOPS │                │
│  │  ███████████████████████████  torch.compile: 198 │                │
│  │  ███████████████████████████████████  ROCmForge: 287 ⭐  │       │
│  │  ████████████████████████████████████████  rocBLAS: 312 │        │
│  └─────────────────────────────────────────────────┘                │
│                                                                     │
│  Occupancy: 88%  |  Memory BW: 5.2 TB/s  |  Compile: ✓ 2.3s        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Component: `App.tsx`
```tsx
import { useState } from 'react';
import Editor from '@monaco-editor/react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function App() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [perf, setPerf] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleCompile = async () => {
    setLoading(true);
    setOutput('');
    
    const eventSource = new EventSource(
      `/api/compile/stream?input=${encodeURIComponent(input)}&lang=pytorch`
    );
    eventSource.onmessage = (e) => setOutput(prev => prev + e.data);
    eventSource.addEventListener('done', async (e: any) => {
      const { hip_code } = JSON.parse(e.data);
      const bench = await fetch('/api/benchmark', {
        method: 'POST', body: JSON.stringify({ hip_code })
      }).then(r => r.json());
      setPerf(bench);
      setLoading(false);
      eventSource.close();
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <header className="flex justify-between mb-6">
        <h1 className="text-3xl font-bold">
          ROCm<span className="text-red-500">Forge</span>
        </h1>
        <span className="text-sm text-slate-400">Powered by AMD MI300X</span>
      </header>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <h2 className="mb-2">Input (PyTorch / CUDA / Triton)</h2>
          <Editor
            height="400px" theme="vs-dark" language="python"
            value={input} onChange={(v) => setInput(v ?? '')}
          />
        </div>
        <div>
          <h2 className="mb-2">Output (HIP / ROCm)</h2>
          <Editor
            height="400px" theme="vs-dark" language="cpp"
            value={output} options={{ readOnly: true }}
          />
        </div>
      </div>

      <button
        onClick={handleCompile} disabled={loading}
        className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-bold"
      >
        {loading ? 'Compiling...' : 'Compile & Benchmark'}
      </button>

      {perf && (
        <div className="mt-6 bg-slate-900 p-6 rounded-lg">
          <h2 className="text-2xl mb-4">Performance Comparison</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={[
              { name: 'PyTorch eager',  TFLOPS: perf.baselines.eager },
              { name: 'torch.compile',  TFLOPS: perf.baselines.compiled },
              { name: 'ROCmForge',      TFLOPS: perf.tflops },
              { name: 'rocBLAS (oracle)', TFLOPS: perf.baselines.rocblas },
            ]}>
              <XAxis dataKey="name" /><YAxis /><Tooltip />
              <Bar dataKey="TFLOPS" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-3 gap-4 mt-4">
            <Stat label="Speedup vs eager" value={`${perf.speedup}x`} />
            <Stat label="Occupancy" value={`${perf.occupancy}%`} />
            <Stat label="Memory BW" value={`${perf.bw_gb_s} GB/s`} />
          </div>
        </div>
      )}
    </div>
  );
}
```

### Examples Gallery (Pre-loaded)
Pre-bake 5-10 impressive examples that always work:
1. `torch.matmul(a, b)` → fused MFMA kernel
2. CUDA flash attention v1 → ROCm flash attention v2
3. PyTorch LayerNorm → fused LDS-optimized kernel
4. Triton softmax → HIP wavefront reduction
5. English: *"compute the column-wise max of an FP16 matrix"* → kernel

These are your **demo safety net** — guaranteed to look great on stage.

---

## 11. 30-Hour Timeline

### Phase 1: Setup & Data (Hours 0-8)

| Hour | Task | Output |
|------|------|--------|
| 0-1 | Provision MI300X instance on AMD Developer Cloud; verify ROCm | Working environment |
| 1-2 | Install PyTorch ROCm, transformers, peft, vLLM, hipify-clang | Dependencies green |
| 2-4 | Write & launch data scraping pipeline (GitHub API + HIPify) | 10K raw CUDA→HIP pairs |
| 4-6 | Build synthetic optimizer (HIPify→teacher LLM→profile loop) | 5K (slow, fast) HIP pairs |
| 6-7 | Process Composable Kernel + rocBLAS as gold examples | 2K hand-tuned examples |
| 7-8 | Format all data into JSONL, train/val/test split | `train.jsonl` (45K examples) |

### Phase 2: Fine-Tuning (Hours 8-18)

| Hour | Task | Output |
|------|------|--------|
| 8-9 | Set up training script, sanity-check on 100 examples | Loss decreasing |
| 9-15 | Full training run (Qwen2.5-Coder-7B + LoRA, 2 epochs) | `rocmforge-7b-v1` |
| 15-16 | Merge LoRA, quantize for serving | Production weights |
| 16-17 | Stand up vLLM ROCm server, smoke test inference | Working API |
| 17-18 | Build eval harness, run on holdout | Initial metrics |

### Phase 3: Backend & Benchmarking (Hours 18-22)

| Hour | Task | Output |
|------|------|--------|
| 18-19 | FastAPI server with `/api/compile` and `/api/benchmark` | Backend API |
| 19-20 | Sandbox compilation (Docker + hipcc) | Safe compile pipeline |
| 20-21 | rocprof integration, parse JSON, compute speedups | Benchmark JSON output |
| 21-22 | Reference baselines (PyTorch eager, torch.compile, rocBLAS) | Comparison data |

### Phase 4: Frontend (Hours 22-26)

| Hour | Task | Output |
|------|------|--------|
| 22-23 | Vite + React + TS + Tailwind + Monaco scaffolding | Skeleton UI |
| 23-24 | Two-pane editor with SSE streaming | Live code generation |
| 24-25 | Recharts perf comparison + stats panel | Beautiful results display |
| 25-26 | Examples gallery + "Try this!" buttons | Demo safety net |

### Phase 5: Polish & Submission (Hours 26-30)

| Hour | Task | Output |
|------|------|--------|
| 26-27 | End-to-end integration test, fix bugs | Working demo |
| 27-28 | Record demo video (3 min); prepare slide deck | Video + slides |
| 28-29 | Push to GitHub (open source); deploy HF Space | Public links |
| 29-30 | Submit to lablab; final sanity check | Submission complete |

### Critical Path & Risks
| Risk | Mitigation |
|------|-----------|
| ROCm vLLM build issues | Pre-test in hour 0; fallback to native HF generate |
| Training divergence | Start with smaller LR; use gradient clipping |
| Insufficient data | Start scraping in hour 0 in parallel; have synthetic fallback |
| Demo failure | Pre-bake 5 examples that ALWAYS work |
| Time crunch | Skip DPO stretch; ship SFT-only |

---

## 12. Demo Script (The 5-Minute Pitch)

### Slide 1 (15s): The Problem
> "AMD MI300X is 2-3x cheaper than NVIDIA H100. But your PyTorch code leaves 50% of its performance on the table. Hand-tuning HIP kernels for MI300X requires PhD-level expertise. *Until now.*"

### Slide 2 (15s): The Solution
> "ROCmForge — a fine-tuned 7B code LLM. Drop in PyTorch, get back hand-tuned ROCm. We trained it on AMD MI300X, for AMD MI300X."

### Live Demo (3 min)

**Beat 1 (45s) — The "Why Fine-Tune?" Killer**
- "First, let's see what GPT-4 does." [Show GPT-4 hallucinating MFMA intrinsics]
- "Now ROCmForge." [Live generation, perfect MFMA usage]
- **Kill shot**: "The data doesn't exist on the public internet. Only fine-tuning solves this."

**Beat 2 (90s) — The Performance Demo**
- Paste in `torch.matmul(a, b)` for a 4096x4096 FP16 problem
- Hit "Compile & Benchmark"
- Live streaming: model generates fused MFMA kernel
- Bar chart appears:
  - PyTorch eager: 142 TFLOPS
  - torch.compile: 198 TFLOPS
  - **ROCmForge: 287 TFLOPS** (highlighted)
  - rocBLAS oracle: 312 TFLOPS
- **Kill shot**: "That's 92% of hand-tuned rocBLAS, generated by AI in 3 seconds."

**Beat 3 (45s) — The CUDA Migration Bonus**
- Paste in real CUDA flash attention kernel
- ROCmForge translates AND optimizes
- Show speedup vs naive HIPify
- **Kill shot**: "Now your existing CUDA workloads run faster on AMD."

### Slide 3 (15s): The Business Case
> - Every PyTorch shop wants this
> - Every cloud provider with AMD GPUs (Crusoe, CoreWeave, Lambda) immediately deploys
> - AMD itself uses this to demonstrate MI300X superiority
> - Open-sourced on Hugging Face today

### Slide 4 (15s): The Meta-Story
> "We used AMD MI300X — its 192GB HBM3, its ROCm stack, its rocprof — to build a tool that makes AMD MI300X faster for everyone else. **The flywheel starts here.**"

---

## 13. Submission Checklist

### Required Artifacts
- [ ] **Project Title**: ROCmForge — The AMD Performance Compiler
- [ ] **Short description** (140 chars): "Fine-tuned LLM that turns PyTorch/CUDA into hand-tuned MI300X kernels — outperforming torch.compile, approaching rocBLAS."
- [ ] **Long description**: 1-2 pages describing problem, solution, technical approach, results
- [ ] **Tags**: `Fine-Tuning`, `LLM`, `ROCm`, `MI300X`, `PyTorch`, `Compiler`, `GPU`, `Qwen`
- [ ] **Cover image**: Screenshot of UI with bar chart showing ROCmForge winning
- [ ] **Video presentation**: 3-min demo (record with OBS)
- [ ] **Slide deck**: 6-8 slides
- [ ] **Public GitHub**: `github.com/<you>/rocmforge`
- [ ] **Demo URL**: HuggingFace Space hosting frontend + model
- [ ] **HuggingFace model card**: `huggingface.co/<you>/rocmforge-7b-mi300x`

### Hugging Face Space Setup
```bash
huggingface-cli login
huggingface-cli repo create rocmforge --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-org>/rocmforge
cp -r frontend/dist/* rocmforge/
git push
```

Join the AMD hackathon HF org first: https://huggingface.co/AMDDevHack (or actual link from event page)

### Build-in-Public Bonus Track
- [ ] Tweet 1: Day 0 setup with MI300X — tag @AIatAMD @lablab
- [ ] Tweet 2: First successful kernel generation — show the bar chart
- [ ] LinkedIn post: full technical writeup
- [ ] Feedback to AMD: open GitHub issue documenting any ROCm pain points encountered

---

## 14. Stretch Goals (If Time Permits)

| Goal | Difficulty | Impact |
|------|-----------|--------|
| **DPO training** with profile-pair preferences | Medium | +10% perf on outputs |
| **Iterative refinement loop** (model gets rocprof feedback, retries) | Medium | +20% perf on hard cases |
| **VS Code extension** for in-IDE use | Easy | Judge wow factor |
| **Auto-PR bot**: scans repo, opens PRs with optimized kernels | Hard | Real-world utility demo |
| **Multimodal**: input architecture diagrams → kernel skeleton | Hard | Innovation points |
| **Benchmark dashboard**: track ROCmForge perf across operations vs SOTA | Easy | Credibility |
| **Scale to 32B model** with LoRA | Easy if time | Better outputs |
| **Comparison page**: side-by-side with GPT-4/Claude generated kernels | Easy | "Why fine-tune?" proof |

---

## 15. Critical Notes & Common Pitfalls

### ROCm vLLM Build
vLLM ROCm builds change frequently. Pre-build a working Docker image in hour 0:
```dockerfile
FROM rocm/vllm:latest
COPY rocmforge-7b-merged /model
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/model"]
```

### MFMA Intrinsics Reference
Most important MI300X intrinsics for the model to know:
- `__builtin_amdgcn_mfma_f32_16x16x16f16` — primary FP16 matmul
- `__builtin_amdgcn_mfma_f32_32x32x8f16` — larger tile FP16
- `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` — BF16 variant
- `__builtin_amdgcn_mfma_i32_16x16x16i8` — INT8 quantized

### Wavefront vs Warp
Hammer this in training data: AMD = **wavefront 64**, NVIDIA = warp 32. Many "ported" CUDA kernels break performance because they assume warp 32.

### Test on the ACTUAL Target Arch
Always compile with `--offload-arch=gfx942` (MI300X). Default arch may produce incorrect or slow code.

### Don't Forget the Graceful Fallback
If rocprof fails on a generated kernel, return *correctness verified, perf TBD* rather than crashing the demo.

---

## TL;DR — What You're Building

**A fine-tuned 7B code LLM, served on MI300X via vLLM, behind a beautiful Vite/React UI, that takes PyTorch code as input and emits hand-tuned ROCm/HIP kernels that outperform torch.compile and approach rocBLAS performance, with a live benchmark visualization powered by rocprof on the same MI300X that trained the model.**

It's the AMD performance flywheel in a single, demonstrable artifact. Ship it in 30 hours. Win the hackathon.

---

## Appendix A: Useful Links

- AMD Developer Cloud: https://www.amd.com/en/developer/resources/cloud-access.html
- ROCm Documentation: https://rocm.docs.amd.com/
- HIPify: https://github.com/ROCm/HIPIFY
- Composable Kernel: https://github.com/ROCm/composable_kernel
- rocBLAS: https://github.com/ROCm/rocBLAS
- MIOpen: https://github.com/ROCm/MIOpen
- vLLM ROCm: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- HuggingFace PEFT: https://huggingface.co/docs/peft
- TRL (SFT/DPO): https://huggingface.co/docs/trl
- AMD MI300X ISA: https://www.amd.com/system/files/TechDocs/cdna3-shader-instruction-set-architecture-feb-2024.pdf
- Triton: https://github.com/openai/triton
- AMD Hackathon HF Org: https://huggingface.co/amd-hackathon (verify exact URL on event page)

## Appendix B: Slack/Discord Help Channels During Hackathon

- AMD Developer Discord: ask in `#rocm-developer-cloud` for instance issues
- Hugging Face Discord: ask in `#training` for fine-tuning tips
- vLLM GitHub Discussions: for ROCm serving issues
