"""
HIP kernel compilation sandbox.

Takes generated HIP code, wraps it in a benchmark harness, compiles with hipcc,
and reports compilation success / errors.

For safety:
- Each compilation runs in an isolated temporary directory
- Time-limited (60s compile, 30s execute)
- Output binary is deleted after benchmarking
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HIPCC_PATH = os.environ.get("HIPCC", "hipcc")
TARGET_ARCH = os.environ.get("ROCMFORGE_TARGET_ARCH", "gfx942")
COMPILE_TIMEOUT_SEC = 60
EXECUTE_TIMEOUT_SEC = 30


def extract_hip_code(raw_output: str) -> tuple[str, list[str]]:
    """
    Pull the HIP/C++ code block out of the model's raw output.
    Models often wrap output in ```cpp or ```hip fences.
    """
    warnings = []

    fence_pattern = r"```(?:cpp|c\+\+|hip|c|cuda)?\s*\n(.*?)```"
    matches = re.findall(fence_pattern, raw_output, re.DOTALL)
    if matches:
        code = max(matches, key=len)
        return code.strip(), warnings

    if "__global__" in raw_output or "#include" in raw_output:
        warnings.append("No fenced code block found; using raw output as code.")
        return raw_output.strip(), warnings

    warnings.append("Output does not look like HIP code.")
    return raw_output.strip(), warnings


def is_hipcc_available() -> bool:
    """Check if hipcc is installed (only true on a ROCm machine)."""
    try:
        result = subprocess.run(
            [HIPCC_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


async def compile_hip(hip_code: str, work_dir: Optional[Path] = None) -> dict:
    """
    Compile the given HIP code with hipcc.
    Returns: { success: bool, error: Optional[str], binary_path: Optional[str] }
    """
    if not is_hipcc_available():
        return {
            "success": False,
            "error": "hipcc not available on this machine. "
                     "Run on an AMD ROCm instance for compilation.",
            "binary_path": None,
        }

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="rocmforge_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    src_path = work_dir / "kernel.hip"
    bin_path = work_dir / "kernel.bin"

    src_path.write_text(hip_code)

    cmd = [
        HIPCC_PATH,
        f"--offload-arch={TARGET_ARCH}",
        "-O3",
        "-std=c++17",
        str(src_path),
        "-o", str(bin_path),
    ]

    logger.info("Running hipcc: %s", " ".join(cmd))

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=COMPILE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Compilation timed out after {COMPILE_TIMEOUT_SEC}s",
            "binary_path": None,
        }

    if process.returncode != 0:
        return {
            "success": False,
            "error": stderr.decode("utf-8", errors="replace")[:4000],
            "binary_path": None,
        }

    return {
        "success": True,
        "error": None,
        "binary_path": str(bin_path),
        "work_dir": str(work_dir),
        "_cleanup": cleanup,
    }
