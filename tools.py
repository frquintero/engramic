import json
import os
import platform
import shlex
from datetime import datetime

from secure_executor import execute_secure_command, analyze_command
from workspace_executor import run_shell_pipeline, ensure_workspace


def get_system_info() -> dict:
    return {
        "os": platform.system(),
        "date_time": datetime.now().isoformat(),
        "path": os.environ.get("PATH", ""),
        "pwd": os.getcwd(),
        "workspace": str(ensure_workspace()),
    }


def list_files(path: str, detailed: bool = False, show_hidden: bool = False) -> str:
    try:
        path = os.path.abspath(path)
        flags = ["-1"]
        if detailed:
            flags.append("-l")
        if show_hidden:
            flags.append("-a")

        command = f"ls {' '.join(flags)} {shlex.quote(path)}"
        result = execute_secure_command(command, timeout=10)

        if result.success:
            lines = result.stdout.strip().split("\n") if result.stdout else []
            return json.dumps({"files": lines})
        return json.dumps({"error": result.error_message or "Command failed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def awk_process(pattern: str, file: str, fs: str = ",") -> str:
    try:
        command = f"awk -F{shlex.quote(fs)} {shlex.quote(pattern)} {shlex.quote(file)}"
        result = execute_secure_command(command, timeout=20)

        if result.success:
            return json.dumps({"output": result.stdout.strip().split("\n")})
        return json.dumps({"error": result.error_message or "Command failed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def read_file(file_path: str) -> str:
    try:
        file_path = os.path.abspath(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        return json.dumps({"content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


def write_file(file_path: str, content: str) -> str:
    """Write the specified content to the specified file, overwriting any existing content"""
    try:
        file_path = os.path.abspath(file_path)
        with open(file_path, "w") as f:
            f.write(content)
        return json.dumps({"success": True, "message": f"Content written to {file_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_shell_command(command: str) -> str:
    """Analyze a shell command to understand its structure and safety before execution"""
    try:
        analysis = analyze_command(command)
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_pipeline_tool(
    pipeline: str, max_output_chars: int = 65536, env: dict = None, mode: str = "full"
) -> str:
    """Run a shell pipeline inside the workspace. Intentionally powerful; guard with mode and output limits."""
    try:
        result = run_shell_pipeline(
            pipeline=pipeline,
            max_output_chars=max_output_chars,
            env=env,
            mode=mode,
        )
        return json.dumps(result.__dict__)
    except Exception as e:
        return json.dumps({"error": str(e)})


available_functions = {
    "list_files": list_files,
    "awk_process": awk_process,
    "read_file": read_file,
    "write_file": write_file,
    "analyze_shell_command": analyze_shell_command,
    "run_shell_pipeline": run_pipeline_tool,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files/directories in the specified path. Defaults to a simple one-per-line listing. Set detailed=true for -l output and show_hidden=true to include dotfiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the directory to list (relative or absolute)"},
                    "detailed": {"type": "boolean", "description": "Include long (-l) details like permissions and sizes"},
                    "show_hidden": {"type": "boolean", "description": "Include hidden files (-a)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "awk_process",
            "description": "Execute the awk shell command for advanced text processing. Common uses include processing columns (e.g., extracting specific fields), filtering lines based on patterns, and performing calculations on structured text data like CSV or log files. Basic syntax: 'pattern { action }' where pattern matches lines and action processes them. Example: '{print $1, $3}' to print the first and third columns of each line. Specify field separator if not comma.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The awk pattern and action, e.g., '{print $1}'"},
                    "file": {"type": "string", "description": "The file path to process"},
                    "fs": {"type": "string", "description": "Field separator (default ',') for CSV; use '\\t' for tab, ' ' for space, etc."},
                },
                "required": ["pattern", "file"],
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
                    "file_path": {"type": "string", "description": "The path to the file to read (relative or absolute)"},
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
                    "file_path": {"type": "string", "description": "The path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_shell_command",
            "description": "Analyze a shell command to understand its structure, safety assessment, and execution requirements before running it. Use this to check if a command is safe and what it will do.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to analyze"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_pipeline",
            "description": "Execute a full shell pipeline in the code_agent workspace. Supports pipes, redirects, and env overrides. Defaults to full-power mode inside the workspace with timeouts and output truncation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "description": "Complete shell pipeline string (e.g., \"ls -1 | head -n 5\")"},
                    "max_output_chars": {"type": "integer", "description": "Truncate stdout/stderr beyond this length (default 65536)"},
                    "env": {"type": "object", "description": "Optional environment variable overrides", "additionalProperties": {"type": "string"}},
                    "mode": {"type": "string", "description": "Execution mode: 'full' (default) or 'constrained' for minimal guardrails"},
                },
                "required": ["pipeline"],
            },
        },
    },
]

tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
