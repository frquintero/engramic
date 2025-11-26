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


def get_system_info_inxi() -> str:
    """Get full system information using inxi -F. Checks if inxi is installed first."""
    try:
        # Check if inxi is installed
        check_result = run_shell_pipeline(
            pipeline="which inxi",
            timeout_secs=5,
            max_output_chars=1024,
            mode="unconstrained",  # Allow system-wide check
        )
        if not check_result.success or not check_result.stdout.strip():
            return build_response(
                tool="get_system_info_inxi",
                success=False,
                error_type="tool_not_available",
                message="inxi is not installed on this system. Use another tool for system information.",
            )

        # Run inxi -F for full system information
        command = "inxi -F"

        # Run the command
        result = run_shell_pipeline(
            pipeline=command,
            timeout_secs=DEFAULT_PIPELINE_TIMEOUT,
            max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
            mode="unconstrained",  # Allow access to system info
        )
        if result.success:
            return build_response(
                tool="get_system_info_inxi",
                success=True,
                result={"system_info": result.stdout},
                truncated=result.truncated,
                duration_ms=result.duration_ms,
            )
        return build_response(
            tool="get_system_info_inxi",
            success=False,
            error_type="execution_error",
            message=result.error or result.stderr or "Failed to retrieve system info",
            stderr=result.stderr,
            truncated=result.truncated,
            timeout=bool(result.error and "Timed out" in result.error),
            duration_ms=result.duration_ms,
        )
    except Exception as e:
        return build_response(tool="get_system_info_inxi", success=False, error_type="execution_error", message=str(e))


available_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_pipeline": run_pipeline_tool,
    "get_system_info_inxi": get_system_info_inxi,
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
            "description": "Execute non-interactive shell commands or pipelines in the workspace (constrained mode blocks destructive operations). Use 'pipeline' for single commands/pipes, 'pipeline_lines' for multiline scripts. Returns JSON with stdout/stderr or error.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "description": "Shell command or pipeline string (e.g., 'ls -1 | head -n 5'). Must be JSON-safe."},
                    "pipeline_lines": {"type": "array", "description": "Array of command lines for multiline scripts.", "items": {"type": "string"}}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info_inxi",
            "description": "Retrieve full system information using 'inxi -F'. Returns JSON envelope with system info or error if inxi is not installed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
