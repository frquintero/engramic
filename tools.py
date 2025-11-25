import json
import os
import platform
from datetime import datetime
from typing import Any, Dict, Optional, List

from workspace_executor import run_shell_pipeline, ensure_workspace, _resolve_workspace_path


DEFAULT_PIPELINE_TIMEOUT = int(os.environ.get("CODE_AGENT_PIPELINE_TIMEOUT_SECS", "30"))
DEFAULT_MAX_OUTPUT_CHARS = int(os.environ.get("CODE_AGENT_MAX_OUTPUT_CHARS", "65536"))


def build_response(
    *,
    tool: str,
    success: bool,
    result: Optional[Any] = None,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    stderr: Optional[str] = None,
    truncated: bool = False,
    timeout: bool = False,
    duration_ms: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Standard envelope for tool responses.
    """
    payload: Dict[str, Any] = {
        "success": success,
        "tool": tool,
        "result": result if success else None,
        "error": None,
        "stderr": stderr,
        "truncated": truncated,
        "timeout": timeout,
        "duration_ms": duration_ms,
        "meta": meta,
    }
    if not success:
        payload["error"] = {
            "type": error_type or "execution_error",
            "message": message or "Tool execution failed",
            "details": details,
        }
    return json.dumps(payload)


def get_system_info() -> dict:
    workspace = str(ensure_workspace())
    return {
        "os": platform.system(),
        "date_time": datetime.now().isoformat(),
        "workspace": workspace,
    }


def list_files() -> str:
    try:
        command = "ls -1"
        result = run_shell_pipeline(
            pipeline=command,
            timeout_secs=DEFAULT_PIPELINE_TIMEOUT,
            max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            mode="constrained",
        )

        if result.success:
            lines = result.stdout.strip().split("\n") if result.stdout else []
            return build_response(tool="list_files", success=True, result={"files": lines}, truncated=result.truncated, duration_ms=result.duration_ms)
        return build_response(
            tool="list_files",
            success=False,
            error_type="execution_error",
            message=result.error or result.stderr or "Command failed",
            stderr=result.stderr,
            truncated=result.truncated,
            timeout=bool(result.error and "Timed out" in result.error),
            duration_ms=result.duration_ms,
        )
    except Exception as e:
        return build_response(tool="list_files", success=False, error_type="execution_error", message=str(e))


def read_file(file_path: str) -> str:
    try:
        file_path = _resolve_workspace_path(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        return build_response(tool="read_file", success=True, result={"content": content})
    except Exception as e:
        return build_response(
            tool="read_file",
            success=False,
            error_type="not_found" if "No such file" in str(e) else "execution_error",
            message=str(e),
        )


def write_file(file_path: str, content: str) -> str:
    """Write the specified content to the specified file, overwriting any existing content"""
    try:
        file_path = _resolve_workspace_path(file_path)
        with open(file_path, "w") as f:
            f.write(content)
        return build_response(
            tool="write_file",
            success=True,
            result={"message": f"Content written to {file_path}"},
        )
    except Exception as e:
        return build_response(tool="write_file", success=False, error_type="execution_error", message=str(e))


def run_pipeline_tool(pipeline: str = "", pipeline_lines: Optional[List[str]] = None) -> str:
    """Run a shell pipeline inside the workspace."""
    try:
        pipeline_str = pipeline or ""
        if pipeline_lines:
            if not isinstance(pipeline_lines, list):
                return build_response(
                    tool="run_shell_pipeline",
                    success=False,
                    error_type="validation_error",
                    message="pipeline_lines must be a list of strings",
                )
            pipeline_str = "\n".join(str(line) for line in pipeline_lines)

        if not pipeline_str:
            return build_response(
                tool="run_shell_pipeline",
                success=False,
                error_type="validation_error",
                message="pipeline or pipeline_lines is required",
            )

        result = run_shell_pipeline(
            pipeline=pipeline_str,
            timeout_secs=DEFAULT_PIPELINE_TIMEOUT,
            max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            mode="constrained",
        )
        if result.success:
            return build_response(
                tool="run_shell_pipeline",
                success=True,
                result={"stdout": result.stdout, "stderr": result.stderr},
                truncated=result.truncated,
                timeout=False,
                duration_ms=result.duration_ms,
            )
        return build_response(
            tool="run_shell_pipeline",
            success=False,
            error_type="execution_error" if not result.error else "rejected_command" if "outside the workspace" in result.error else "execution_error",
            message=result.error or result.stderr or "Pipeline failed",
            stderr=result.stderr,
            truncated=result.truncated,
            timeout=bool(result.error and "Timed out" in result.error),
            duration_ms=result.duration_ms,
        )
    except Exception as e:
        return build_response(tool="run_shell_pipeline", success=False, error_type="execution_error", message=str(e))


available_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_pipeline": run_pipeline_tool,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace root (code_agent_workspace) using 'ls -1'. No other paths are allowed. Returns JSON with success/error envelope.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the entire contents of the specified workspace file. Use workspace-relative paths (e.g., 'file.txt'). Returns JSON envelope with content or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Workspace-relative path to the file to read"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the specified content string to a workspace file (overwrites). Use workspace-relative paths. Returns JSON envelope with success or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Workspace-relative path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_pipeline",
            "description": "Execute a non-interactive shell pipeline in the workspace shell (cwd: code_agent_workspace). Constrained mode: blocks interactive/destructive commands, enforces timeout/output caps. Returns JSON envelope with stdout/stderr or error.\n"
            "Usage:\n"
            "1. Single commands: Use 'pipeline' for simple operations (e.g., 'pwd', 'date', 'cal', 'python -c \"print(1+1)\"', 'awk \"{print $1}\" file.txt'). Choose when the task is standalone and doesn't need piping.\n"
            "2. Pipeline commands: Use 'pipeline' for chained operations with | (e.g., 'cat file.txt | grep pattern | sort -n'). Choose for efficient data streaming, filtering, and transformation.\n"
            "3. Multiline scripts: Use 'pipeline_lines' as an array for complex logic (e.g., [\"for i in 1 2 3; do echo $i; done\", \"if [ -f file.txt ]; then cat file.txt; fi\"]). Choose for loops, conditionals, or multi-step scripts.\n"
            "Provide either a single-line, JSON-safe pipeline string or pipeline_lines as an array of command lines joined with newlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "description": "Complete shell pipeline string (e.g., \"ls -1 | head -n 5\", \"cat file.txt | sed 's/foo/bar/' | awk '{print $1}'\", or \"python -c \\\"print('hi')\\\" | sed 's/hi/hello/' > out.txt\"). Must be JSON-safe (no raw newlines)."},
                    "pipeline_lines": {"type": "array", "description": "Array of command lines to join with newlines for execution (use for multiline scripts instead of heredocs).", "items": {"type": "string"}}
                },
                "required": [],
            },
        },
    },
]

tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
