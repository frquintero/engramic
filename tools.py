import json
import os
import platform
from datetime import datetime

from workspace_executor import run_shell_pipeline, ensure_workspace, _resolve_workspace_path


def get_system_info() -> dict:
    return {
        "os": platform.system(),
        "date_time": datetime.now().isoformat(),
        "path": os.environ.get("PATH", ""),
        "pwd": os.getcwd(),
        "workspace": str(ensure_workspace()),
    }


def list_files() -> str:
    try:
        command = "ls -1"
        result = run_shell_pipeline(pipeline=command, timeout_secs=10)

        if result.success:
            lines = result.stdout.strip().split("\n") if result.stdout else []
            return json.dumps({"files": lines})
        return json.dumps({"error": result.error or result.stderr or "Command failed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def read_file(file_path: str) -> str:
    try:
        file_path = _resolve_workspace_path(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        return json.dumps({"content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


def write_file(file_path: str, content: str) -> str:
    """Write the specified content to the specified file, overwriting any existing content"""
    try:
        file_path = _resolve_workspace_path(file_path)
        with open(file_path, "w") as f:
            f.write(content)
        return json.dumps({"success": True, "message": f"Content written to {file_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_pipeline_tool(pipeline: str) -> str:
    """Run a shell pipeline inside the workspace."""
    try:
        result = run_shell_pipeline(
            pipeline=pipeline,
        )
        return json.dumps(result.__dict__)
    except Exception as e:
        return json.dumps({"error": str(e)})


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
            "description": "List files in the workspace root (code_agent_workspace) using 'ls -1'. No other paths are allowed.",
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
            "description": "Read and return the entire contents of the specified file as a string. Useful for examining file contents or passing data to other tools. Use relative paths (e.g., 'file.txt') for files in the current directory, or absolute paths for files elsewhere.",
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
            "description": "Write the specified content string (provided as a parameter) to the specified file, overwriting any existing content. Use this to create or update files with new data.",
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
            "description": "Execute a shell pipeline in the workspace shell (cwd: code_agent_workspace). Supports pipes and redirects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "description": "Complete shell pipeline string (e.g., \"ls -1 | head -n 5\", \"cat file.txt | sed 's/foo/bar/' | awk '{print $1}'\", or \"python -c \\\"print('hi')\\\" | sed 's/hi/hello/' > out.txt\")"}
                },
                "required": ["pipeline"],
            },
        },
    },
]

tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
