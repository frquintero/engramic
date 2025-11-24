import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


DEFAULT_WORKSPACE = Path(os.environ.get("CODE_AGENT_WORKSPACE", "code_agent_workspace"))


@dataclass
class PipelineResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    truncated: bool
    duration_ms: float
    cwd: str
    error: Optional[str] = None


def ensure_workspace(base: Path = DEFAULT_WORKSPACE) -> Path:
    """
    Create the workspace directory if it does not exist.

    The workspace is the only location we intend to write to. Call this at app startup.
    """
    base.mkdir(parents=True, exist_ok=True)
    return base


def _truncate(text: str, limit: int) -> Tuple[str, bool]:
    if limit <= 0:
        return text, False
    if len(text) > limit:
        return text[:limit], True
    return text, False


def run_shell_pipeline(
    pipeline: str,
    timeout_secs: Optional[int] = None,
    max_output_chars: int = 65536,
    env: Optional[Dict[str, str]] = None,
    mode: str = "full",
) -> PipelineResult:
    """
    Execute a shell pipeline in the workspace context.

    mode:
      - full: run as-is
      - constrained: reject a few obviously dangerous patterns
    """
    workspace = ensure_workspace()
    exec_cwd = workspace

    # Basic sanity checks
    if not pipeline.strip():
        return PipelineResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            truncated=False,
            duration_ms=0.0,
            cwd=str(exec_cwd),
            error="Empty pipeline",
        )

    if mode == "constrained":
        lowered = pipeline.lower()
        if "rm -rf /" in lowered or "chmod /" in lowered or "chown /" in lowered:
            return PipelineResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                truncated=False,
                duration_ms=0.0,
                cwd=str(exec_cwd),
                error="Pipeline rejected in constrained mode",
            )

    env_vars = os.environ.copy()
    if env:
        env_vars.update({k: str(v) for k, v in env.items()})

    start = time.time()
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", pipeline],
            cwd=str(exec_cwd),
            env=env_vars,
            capture_output=True,
            text=True,
            timeout=timeout_secs if timeout_secs else None,
        )
        duration_ms = (time.time() - start) * 1000.0

        stdout, t1 = _truncate(result.stdout or "", max_output_chars)
        stderr, t2 = _truncate(result.stderr or "", max_output_chars)

        return PipelineResult(
            success=result.returncode == 0,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=t1 or t2,
            duration_ms=duration_ms,
            cwd=str(exec_cwd),
        )
    except subprocess.TimeoutExpired as e:
        duration_ms = (time.time() - start) * 1000.0
        return PipelineResult(
            success=False,
            returncode=-1,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + "\n[timeout]",
            truncated=False,
            duration_ms=duration_ms,
            cwd=str(exec_cwd),
            error=f"Timed out after {timeout_secs}s" if timeout_secs else "Timed out",
        )
    except Exception as e:
        duration_ms = (time.time() - start) * 1000.0
        return PipelineResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            truncated=False,
            duration_ms=duration_ms,
            cwd=str(exec_cwd),
            error=str(e),
        )
