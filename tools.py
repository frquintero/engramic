import json
import subprocess
import os
import platform
from datetime import datetime

# Tool implementations
def get_system_info() -> dict:
    return {
        "os": platform.system(),
        "date_time": datetime.now().isoformat(),
        "path": os.environ.get('PATH', ''),
        "pwd": os.getcwd()
    }

def get_cwd() -> str:
    return json.dumps({"cwd": os.getcwd()})

def list_files(path: str) -> str:
    try:
        # Resolve relative paths to absolute paths
        path = os.path.abspath(path)
        result = subprocess.run(['ls', '-1', path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {str(e)}"

def git_status(repo_path: str) -> str:
    try:
        result = subprocess.run(['git', 'status'], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"status": result.stdout})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def git_add(repo_path: str, file_path: str) -> str:
    try:
        result = subprocess.run(['git', 'add', file_path], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"status": "File(s) added to staging area"})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def git_add_all(repo_path: str) -> str:
    try:
        result = subprocess.run(['git', 'add', '.'], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"status": "All files added to staging area"})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def git_commit(repo_path: str, message: str) -> str:
    try:
        result = subprocess.run(['git', 'commit', '-m', message], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"status": "Commit successful", "output": result.stdout.strip()})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def git_log(repo_path: str, limit: int = 5) -> str:
    try:
        result = subprocess.run(['git', 'log', f'--oneline', f'-{limit}'], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"log": result.stdout.strip().split('\n')})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def awk_process(pattern: str, file: str, fs: str = ',') -> str:
    try:
        result = subprocess.run(['awk', f'-F{fs}', pattern, file], capture_output=True, text=True, check=True)
        return json.dumps({"output": result.stdout.strip().split('\n')})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def read_file(file_path: str) -> str:
    try:
        # Resolve relative paths to absolute paths based on current working directory
        file_path = os.path.abspath(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
        return json.dumps({"content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})

def write_file(file_path: str, content: str) -> str:
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return json.dumps({"status": "File written successfully"})
    except Exception as e:
        return json.dumps({"error": str(e)})

available_functions = {
    "list_files": list_files,
    "git_status": git_status,
    "git_add": git_add,
    "git_add_all": git_add_all,
    "git_commit": git_commit,
    "git_log": git_log,
    "awk_process": awk_process,
    "read_file": read_file,
    "write_file": write_file,
    "get_cwd": get_cwd
}

# Tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files and directories in the specified path, including hidden files and detailed information like permissions, sizes, and timestamps. Use relative paths (e.g., '.') for the current directory, or absolute paths.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The path to the directory to list (relative or absolute)"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Get the current status of the Git repository at the specified path, showing staged changes, unstaged changes, untracked files, and any merge conflicts.",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string", "description": "The absolute path to the Git repository directory"}},
                "required": ["repo_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage a file for commit in the Git repository at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "The absolute path to the Git repository directory"},
                    "file_path": {"type": "string", "description": "The path to the file to add (relative to repo)"}
                },
                "required": ["repo_path", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add_all",
            "description": "Stage all changes in the Git repository (equivalent to git add .).",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string", "description": "The absolute path to the Git repository directory"}},
                "required": ["repo_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit staged changes in the Git repository with a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "The absolute path to the Git repository directory"},
                    "message": {"type": "string", "description": "The commit message"}
                },
                "required": ["repo_path", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent commit history in the Git repository, limited to a specified number of entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "The absolute path to the Git repository directory"},
                    "limit": {"type": "integer", "description": "Number of commits to show (default 5)"}
                },
                "required": ["repo_path"]
            }
        }
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
                    "fs": {"type": "string", "description": "Field separator (default ',') for CSV; use '\\t' for tab, ' ' for space, etc."}
                },
                "required": ["pattern", "file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the entire contents of the specified file as a string. Useful for examining file contents or passing data to other tools. Use relative paths (e.g., 'file.txt') for files in the current directory, or absolute paths for files elsewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to read (relative or absolute)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the specified content string (provided as a parameter) to the specified file, overwriting any existing content. Use this to create or update files with new data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cwd",
            "description": "Get the current working directory path. Useful to know the base directory for relative paths.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Build tools descriptions dict for auto-registration
tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}