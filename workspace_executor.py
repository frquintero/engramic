import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import shlex


DEFAULT_WORKSPACE = Path(os.environ.get("CODE_AGENT_WORKSPACE", "code_agent_workspace"))
DEFAULT_TIMEOUT_SECS = int(os.environ.get("CODE_AGENT_PIPELINE_TIMEOUT_SECS", "30"))
DEFAULT_MAX_OUTPUT_CHARS = int(os.environ.get("CODE_AGENT_MAX_OUTPUT_CHARS", "65536"))


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


def _resolve_workspace_path(raw_path: str, base: Path = DEFAULT_WORKSPACE) -> Path:
    """
    Resolve a user-supplied path inside the workspace. Reject traversal or
    absolute paths that escape the workspace.
    """
    workspace = ensure_workspace(base).resolve()
    candidate = Path(raw_path)
    resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError("Path is outside the workspace")

    return resolved


def _truncate(text: str, limit: int) -> Tuple[str, bool]:
    if limit <= 0:
        return text, False
    if len(text) > limit:
        return text[:limit], True
    return text, False


def run_shell_pipeline(
    pipeline: str,
    timeout_secs: Optional[int] = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    env: Optional[Dict[str, str]] = None,
    mode: str = "constrained",
) -> PipelineResult:
    """
    Execute a shell pipeline in the workspace context.

    mode:
      - full: run as-is
      - constrained: reject a few obviously dangerous patterns
    """
    workspace = ensure_workspace()
    exec_cwd = workspace
    if timeout_secs is None:
        timeout_secs = DEFAULT_TIMEOUT_SECS

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
        interactive_flags = {"-i", "--interactive", "--login", "--shell"}
        interactive_commands = {
            "bash",
            "zsh",
            "sh",
            "fish",
            "mysql",
            "psql",
            "sqlite3",
            "redis-cli",
            "mongo",
            "mongosh",
            "vim",
            "nano",
            "emacs",
            "less",
            "more",
            "top",
            "htop",
            "screen",
            "tmux",
        }
        try:
            tokens = shlex.split(pipeline)
        except ValueError as e:
            return PipelineResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                truncated=False,
                duration_ms=0.0,
                cwd=str(exec_cwd),
                error=f"Failed to parse pipeline: {e}",
            )

        # Reject obvious interactive or destructive patterns
        for tok in tokens:
            if tok in interactive_commands or tok in interactive_flags:
                return PipelineResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    truncated=False,
                    duration_ms=0.0,
                    cwd=str(exec_cwd),
                    error="Pipeline rejected: interactive commands are not allowed in constrained mode",
                )
            if tok == "rm":
                if any(flag in tokens for flag in ("-rf", "-fr", "-r", "-f")):
                    return PipelineResult(
                        success=False,
                        returncode=-1,
                        stdout="",
                        stderr="",
                        truncated=False,
                        duration_ms=0.0,
                        cwd=str(exec_cwd),
                        error="Pipeline rejected: destructive rm usage blocked",
                    )
            if tok in {"chmod", "chown"}:
                return PipelineResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    truncated=False,
                    duration_ms=0.0,
                    cwd=str(exec_cwd),
                    error="Pipeline rejected: permission-changing commands are not allowed",
                )
        if "rm -rf /" in lowered:
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

    # Validate that tokens referencing filesystem paths stay within workspace
    try:
        for token in shlex.split(pipeline):
            # Skip obvious operators
            if token in {"|", ">", ">>", "<", "<<", "2>", "2>>"}:
                continue
            # If token looks like a path (contains a slash or starts with dot), ensure it resolves inside the workspace
            if "/" in token or token.startswith("."):
                _resolve_workspace_path(token, base=workspace)
    except ValueError as e:
        return PipelineResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            truncated=False,
            duration_ms=0.0,
            cwd=str(exec_cwd),
            error=str(e),
        )
    except Exception as e:
        return PipelineResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="",
            truncated=False,
            duration_ms=0.0,
            cwd=str(exec_cwd),
            error=f"Failed to validate pipeline: {e}",
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
