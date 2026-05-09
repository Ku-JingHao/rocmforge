"""
Benchmark harness: runs a compiled HIP binary and parses rocprof output.

Provides performance metrics for the demo bar chart:
- TFLOPS (operation-specific)
- Kernel time (ms)
- Occupancy
- Memory bandwidth (GB/s)
"""

import asyncio
import csv
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROCPROF_PATH = os.environ.get("ROCPROF", "rocprof")


def is_rocprof_available() -> bool:
    """Check if rocprof is installed."""
    try:
        result = subprocess.run(
            [ROCPROF_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def compute_tflops(operation: str, problem_size: dict, kernel_time_s: float) -> Optional[float]:
    """
    Compute theoretical TFLOPS based on operation and problem size.

    GEMM: 2*M*N*K floating-point operations
    Vector add: N operations
    Reduction: ~N operations
    LayerNorm: ~5*N operations (mean, var, normalize, scale, bias)
    Softmax: ~5*N operations
    """
    if kernel_time_s <= 0:
        return None

    M = problem_size.get("M", 0)
    N = problem_size.get("N", 0)
    K = problem_size.get("K", 0)

    if operation == "gemm":
        flops = 2 * M * N * K
    elif operation == "vector_add":
        flops = max(M, N) if M or N else 0
    elif operation == "reduction":
        flops = max(M, N) if M or N else 0
    elif operation == "layernorm":
        flops = 5 * M * N
    elif operation == "softmax":
        flops = 5 * M * N
    else:
        flops = M * N if M and N else 0

    if flops == 0:
        return None
    return (flops / kernel_time_s) / 1e12


async def run_rocprof(binary_path: str, output_dir: Path) -> dict:
    """
    Run rocprof on the binary and parse the resulting CSV.
    Returns: { kernel_time_ns, occupancy, memory_bw, error }
    """
    if not is_rocprof_available():
        return {
            "error": "rocprof not available. Run on a ROCm-enabled machine.",
        }

    output_csv = output_dir / "rocprof_results.csv"
    cmd = [
        ROCPROF_PATH,
        "--stats",
        "--basenames", "on",
        "-o", str(output_csv),
        binary_path,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(output_dir),
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=60,
        )
    except asyncio.TimeoutError:
        return {"error": "rocprof execution timed out (60s)"}

    if process.returncode != 0:
        return {
            "error": f"rocprof failed: {stderr.decode('utf-8', errors='replace')[:1000]}"
        }

    stats_csv = output_dir / "rocprof_results.stats.csv"
    if not stats_csv.exists():
        stats_csv = output_csv

    if not stats_csv.exists():
        return {"error": "rocprof output CSV not found"}

    return parse_rocprof_csv(stats_csv)


def parse_rocprof_csv(csv_path: Path) -> dict:
    """Parse rocprof CSV output. Falls back gracefully on missing fields."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return {"error": f"Failed to parse {csv_path}: {e}"}

    if not rows:
        return {"error": "No kernel data in rocprof output"}

    row = rows[0]

    def get_float(*keys):
        for k in keys:
            if k in row and row[k]:
                try:
                    return float(row[k])
                except ValueError:
                    continue
        return None

    duration_ns = get_float("DurationNs", "Duration_ns", "AverageDurationNs")
    occupancy = get_float("OCCUPANCY", "Occupancy")
    bandwidth = get_float("MemBandwidth", "MemoryBandwidth")

    return {
        "kernel_time_ns": duration_ns,
        "kernel_time_ms": (duration_ns / 1e6) if duration_ns else None,
        "kernel_time_s": (duration_ns / 1e9) if duration_ns else None,
        "occupancy": occupancy,
        "memory_bw_gb_s": bandwidth,
        "error": None,
    }


async def benchmark_binary(
    binary_path: str,
    work_dir: Path,
    operation: str,
    problem_size: dict,
) -> dict:
    """End-to-end: run the binary with rocprof, compute TFLOPS, return metrics."""
    rocprof_result = await run_rocprof(binary_path, work_dir)
    if rocprof_result.get("error"):
        return rocprof_result

    kernel_time_s = rocprof_result.get("kernel_time_s")
    if kernel_time_s is None:
        return {"error": "Could not parse kernel time from rocprof output"}

    tflops = compute_tflops(operation, problem_size, kernel_time_s)

    return {
        "tflops": tflops,
        "kernel_time_ms": rocprof_result["kernel_time_ms"],
        "occupancy_pct": rocprof_result.get("occupancy"),
        "memory_bw_gb_s": rocprof_result.get("memory_bw_gb_s"),
        "error": None,
    }
