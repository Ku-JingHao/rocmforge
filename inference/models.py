"""Pydantic request/response models for the ROCmForge API."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


InputLanguage = Literal["pytorch", "cuda", "triton", "english"]


class CompileRequest(BaseModel):
    """Request to convert input code into HIP/ROCm kernel."""
    input_code: str = Field(..., description="The user's input code or description")
    input_lang: InputLanguage = Field("pytorch", description="Source language")
    target: Literal["mi300x"] = Field("mi300x", description="Target architecture")
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=64, le=8192)


class CompileResponse(BaseModel):
    """Response with the generated HIP code."""
    request_id: str
    hip_code: str
    raw_output: str
    extraction_warnings: list[str] = []


class BenchmarkRequest(BaseModel):
    """Request to compile and benchmark a HIP kernel."""
    hip_code: str
    operation: Literal["gemm", "softmax", "vector_add", "reduction", "layernorm", "auto"] = "auto"
    problem_size: dict = Field(default_factory=lambda: {"M": 4096, "N": 4096, "K": 4096})
    dtype: Literal["fp32", "fp16", "bf16"] = "fp16"


class BenchmarkResult(BaseModel):
    """Performance results from compiling and running the kernel."""
    compile_success: bool
    compile_error: Optional[str] = None

    # Performance numbers (None if compilation failed)
    tflops: Optional[float] = None
    kernel_time_ms: Optional[float] = None
    occupancy_pct: Optional[float] = None
    memory_bw_gb_s: Optional[float] = None

    # Comparisons
    baselines: dict = Field(default_factory=dict)
    speedup_vs_eager: Optional[float] = None
    speedup_vs_compile: Optional[float] = None
    pct_of_rocblas: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    model_name: str
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    rocm_version: Optional[str] = None
