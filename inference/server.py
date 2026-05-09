"""
ROCmForge FastAPI server.

Endpoints:
  GET  /api/health            — service status, GPU/model info
  POST /api/compile           — generate HIP code from input (PyTorch/CUDA/Triton/English)
  POST /api/compile/stream    — same, but streams tokens via Server-Sent Events
  POST /api/benchmark         — compile + run + profile + compare to baselines
  POST /api/full_pipeline     — compile + benchmark in one call (for demo)

Run with:
  uvicorn inference.server:app --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    CompileRequest,
    CompileResponse,
    BenchmarkRequest,
    BenchmarkResult,
    HealthResponse,
)
from .sandbox import compile_hip, extract_hip_code, is_hipcc_available
from .benchmark import benchmark_binary, is_rocprof_available
from .baselines import get_all_baselines

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("ROCMFORGE_MODEL_PATH", "./rocmforge-7b-merged")
MODEL_NAME = os.environ.get("ROCMFORGE_MODEL_NAME", "rocmforge-7b")
USE_VLLM = os.environ.get("ROCMFORGE_USE_VLLM", "1") == "1"
SYSTEM_PROMPT = """You are ROCmForge, an expert AMD GPU kernel optimizer. You specialize in:
- Converting PyTorch, CUDA, and Triton code to optimized HIP kernels
- Targeting AMD Instinct MI300X (gfx942) architecture
- Using wavefront-64 operations, MFMA intrinsics, and LDS optimization
- Maximizing TFLOPS and memory bandwidth utilization
Generate compilable, correct, high-performance HIP/ROCm code. Wrap output in ```cpp fences."""


_state = {
    "llm": None,
    "tokenizer": None,
    "engine_type": None,
}


def build_user_prompt(req: CompileRequest) -> str:
    """Construct the user-facing prompt based on input language."""
    lang_intro = {
        "pytorch": "Convert this PyTorch code to an optimized HIP kernel for AMD MI300X (gfx942):",
        "cuda":    "Convert this CUDA kernel to an optimized HIP kernel for AMD MI300X (gfx942). "
                   "Fix any NVIDIA-specific assumptions (warp-32 → wavefront-64):",
        "triton":  "Convert this Triton kernel to an equivalent optimized HIP kernel for AMD MI300X (gfx942):",
        "english": "Write an optimized HIP kernel for AMD MI300X (gfx942) based on this description:",
    }
    intro = lang_intro.get(req.input_lang, lang_intro["pytorch"])
    return f"{intro}\n\n{req.input_code}"


def load_vllm():
    """Load model with vLLM for fast inference."""
    from vllm import LLM, SamplingParams
    logger.info("Loading model with vLLM from %s", MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    _state["llm"] = llm
    _state["engine_type"] = "vllm"
    logger.info("vLLM model loaded.")


def load_hf():
    """Fallback: load model with HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model with HuggingFace transformers from %s", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _state["llm"] = model
    _state["tokenizer"] = tokenizer
    _state["engine_type"] = "hf"
    logger.info("HuggingFace model loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    if not Path(MODEL_PATH).exists():
        logger.warning(
            "Model path %s not found. Server will start, but /api/compile will fail. "
            "Train and merge a model first.",
            MODEL_PATH,
        )
    else:
        try:
            if USE_VLLM:
                load_vllm()
            else:
                load_hf()
        except Exception as e:
            logger.error("Failed to load model: %s. Falling back to HF.", e)
            try:
                load_hf()
            except Exception as e2:
                logger.error("HF load also failed: %s", e2)

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ROCmForge",
    description="The AMD Performance Compiler — fine-tuned LLM for MI300X kernel generation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Service health and capability check."""
    gpu_name = None
    gpu_memory_gb = None
    rocm_version = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            rocm_version = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
    except Exception:
        pass

    status = "ok" if _state["llm"] is not None else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=_state["llm"] is not None,
        model_name=MODEL_NAME,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        rocm_version=rocm_version,
    )


def _generate_vllm(prompt: str, temperature: float, max_tokens: int) -> str:
    from vllm import SamplingParams
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    outputs = _state["llm"].chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        sampling,
    )
    return outputs[0].outputs[0].text


def _generate_hf(prompt: str, temperature: float, max_tokens: int) -> str:
    import torch
    tokenizer = _state["tokenizer"]
    model = _state["llm"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def generate(prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Dispatch to vLLM or HF based on which engine is loaded."""
    if _state["llm"] is None:
        raise HTTPException(503, "Model not loaded")
    if _state["engine_type"] == "vllm":
        return _generate_vllm(prompt, temperature, max_tokens)
    return _generate_hf(prompt, temperature, max_tokens)


@app.post("/api/compile", response_model=CompileResponse)
async def compile_endpoint(req: CompileRequest):
    """Generate HIP code from the input."""
    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] /api/compile lang=%s len=%d", request_id, req.input_lang, len(req.input_code))

    user_prompt = build_user_prompt(req)
    raw_output = await asyncio.to_thread(generate, user_prompt, req.temperature, req.max_tokens)
    hip_code, warnings = extract_hip_code(raw_output)

    return CompileResponse(
        request_id=request_id,
        hip_code=hip_code,
        raw_output=raw_output,
        extraction_warnings=warnings,
    )


@app.post("/api/compile/stream")
async def compile_stream(req: CompileRequest):
    """Stream tokens via Server-Sent Events for live UI updates."""
    if _state["llm"] is None:
        raise HTTPException(503, "Model not loaded")

    async def event_generator() -> AsyncIterator[bytes]:
        request_id = str(uuid.uuid4())[:8]
        yield f"event: start\ndata: {{\"request_id\": \"{request_id}\"}}\n\n".encode()

        user_prompt = build_user_prompt(req)
        full_output = await asyncio.to_thread(generate, user_prompt, req.temperature, req.max_tokens)

        chunk_size = 32
        for i in range(0, len(full_output), chunk_size):
            chunk = full_output[i:i + chunk_size]
            chunk_escaped = chunk.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
            yield f"data: {{\"chunk\": \"{chunk_escaped}\"}}\n\n".encode()
            await asyncio.sleep(0.01)

        hip_code, warnings = extract_hip_code(full_output)
        hip_escaped = hip_code.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
        yield (
            f"event: done\ndata: {{\"hip_code\": \"{hip_escaped}\", "
            f"\"warnings\": {warnings}}}\n\n"
        ).encode()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/benchmark", response_model=BenchmarkResult)
async def benchmark_endpoint(req: BenchmarkRequest):
    """Compile, run, profile, and compare to baselines."""
    if not is_hipcc_available():
        raise HTTPException(
            500,
            "hipcc not available — must run on a ROCm-enabled machine.",
        )

    work_dir = Path(tempfile.mkdtemp(prefix="rocmforge_bench_"))
    try:
        compile_result = await compile_hip(req.hip_code, work_dir)
        if not compile_result["success"]:
            return BenchmarkResult(
                compile_success=False,
                compile_error=compile_result["error"],
            )

        operation = req.operation if req.operation != "auto" else _detect_operation(req.hip_code)
        bench = await benchmark_binary(
            compile_result["binary_path"],
            work_dir,
            operation,
            req.problem_size,
        )

        baselines = await asyncio.to_thread(
            get_all_baselines, operation, req.problem_size, req.dtype
        )

        speedup_eager = None
        speedup_compile = None
        pct_rocblas = None
        if bench.get("tflops") and baselines.get("eager"):
            speedup_eager = bench["tflops"] / baselines["eager"]
        if bench.get("tflops") and baselines.get("compiled"):
            speedup_compile = bench["tflops"] / baselines["compiled"]
        if bench.get("tflops") and baselines.get("rocblas"):
            pct_rocblas = 100 * bench["tflops"] / baselines["rocblas"]

        return BenchmarkResult(
            compile_success=True,
            tflops=bench.get("tflops"),
            kernel_time_ms=bench.get("kernel_time_ms"),
            occupancy_pct=bench.get("occupancy_pct"),
            memory_bw_gb_s=bench.get("memory_bw_gb_s"),
            baselines=baselines,
            speedup_vs_eager=speedup_eager,
            speedup_vs_compile=speedup_compile,
            pct_of_rocblas=pct_rocblas,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/full_pipeline")
async def full_pipeline(req: CompileRequest):
    """One-shot: compile + benchmark + return everything (powers the demo button)."""
    compile_resp = await compile_endpoint(req)

    bench_req = BenchmarkRequest(
        hip_code=compile_resp.hip_code,
        operation="auto",
        problem_size={"M": 4096, "N": 4096, "K": 4096},
        dtype="fp16",
    )

    try:
        bench_resp = await benchmark_endpoint(bench_req)
    except HTTPException as e:
        return {
            "compile": compile_resp.dict(),
            "benchmark": None,
            "benchmark_error": e.detail,
        }

    return {
        "compile": compile_resp.dict(),
        "benchmark": bench_resp.dict(),
    }


def _detect_operation(hip_code: str) -> str:
    """Heuristically detect what operation this kernel implements."""
    code = hip_code.lower()
    if "mfma" in code or "matmul" in code or "gemm" in code or "matrix multiply" in code:
        return "gemm"
    if "softmax" in code or ("exp" in code and "max" in code and "sum" in code):
        return "softmax"
    if "layernorm" in code or "layer_norm" in code:
        return "layernorm"
    if "reduce" in code or "reduction" in code or "sum_kernel" in code:
        return "reduction"
    if "vector_add" in code or ("c[i] = a[i] + b[i]" in code):
        return "vector_add"
    return "gemm"
