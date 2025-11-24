import json
import subprocess
import os
from groq import Groq
import pytest

# Tool implementations (from the example)
def list_files(path: str) -> str:
    try:
        result = subprocess.run(['ls', '-la', path], capture_output=True, text=True, check=True)
        return json.dumps({"files": result.stdout.strip().split('\n')})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def git_status(repo_path: str) -> str:
    try:
        result = subprocess.run(['git', 'status'], cwd=repo_path, capture_output=True, text=True, check=True)
        return json.dumps({"status": result.stdout})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def awk_process(pattern: str, file: str) -> str:
    try:
        result = subprocess.run(['awk', pattern, file], capture_output=True, text=True, check=True)
        return json.dumps({"output": result.stdout.strip().split('\n')})
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": str(e)})

def read_file(file_path: str) -> str:
    try:
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
    "awk_process": awk_process,
    "read_file": read_file,
    "write_file": write_file
}

# Tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lista archivos en un directorio",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Verifica el estado de un repositorio Git",
            "parameters": {
                "type": "object",
                "properties": {"repo_path": {"type": "string"}},
                "required": ["repo_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "awk_process",
            "description": "Use awk for text processing, pattern scanning, and data manipulation. Provide a pattern (e.g., '{print $1}') and a file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The awk pattern and action, e.g., '{print $1}'"},
                    "file": {"type": "string", "description": "The file path to process"}
                },
                "required": ["pattern", "file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to read"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"}
                },
                "required": ["file_path", "content"]
            }
        }
    }
]

# Orchestration function with real Groq API calls
def run_multi_tool_agent(user_query, max_iterations=5):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    model = 'openai/gpt-oss-120b'  # Or any available model

    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools for listing files, checking git status, processing text with awk, reading files, and writing files."},
        {"role": "user", "content": user_query}
    ]

    iteration = 0
    while iteration < max_iterations:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            return response_message.content

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response
            })

        iteration += 1

    return "Max iterations reached"

# Pytest tests
class TestMultiTool:
    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_list_files_success(self):
        result = list_files(".")
        data = json.loads(result)
        assert "files" in data
        assert isinstance(data["files"], list)
        assert len(data["files"]) > 0  # Should list files in current dir

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_git_status_success(self):
        result = git_status(".")
        data = json.loads(result)
        assert "status" in data
        assert "On branch" in data["status"] or "not a git repository" in data["status"]

    def test_list_files_error(self):
        result = list_files("/nonexistent")
        data = json.loads(result)
        assert "error" in data

    def test_git_status_error(self):
        result = git_status("/nonexistent")
        data = json.loads(result)
        assert "error" in data

    def test_awk_success(self):
        # Create a test file
        with open("test_file.txt", "w") as f:
            f.write("line1 col2\nline2 col2\n")
        result = awk_process("{print $1}", "test_file.txt")
        data = json.loads(result)
        assert "output" in data
        assert data["output"] == ["line1", "line2"]
        os.remove("test_file.txt")

    def test_awk_error(self):
        result = awk_process("{print $1}", "/nonexistent")
        data = json.loads(result)
        assert "error" in data

    def test_read_file_success(self):
        # Create a test file
        with open("test_read.txt", "w") as f:
            f.write("Hello World")
        result = read_file("test_read.txt")
        data = json.loads(result)
        assert "content" in data
        assert data["content"] == "Hello World"
        os.remove("test_read.txt")

    def test_read_file_error(self):
        result = read_file("/nonexistent")
        data = json.loads(result)
        assert "error" in data

    def test_write_file_success(self):
        result = write_file("test_write.txt", "Test content")
        data = json.loads(result)
        assert "status" in data
        assert data["status"] == "File written successfully"
        # Verify file exists
        assert os.path.exists("test_write.txt")
        with open("test_write.txt", "r") as f:
            assert f.read() == "Test content"
        os.remove("test_write.txt")

    def test_write_file_error(self):
        result = write_file("/invalid/path/test.txt", "content")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_single_tool_list_files(self):
        result = run_multi_tool_agent("List files in current directory")
        assert isinstance(result, str)  # Should return a string response

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_single_tool_git_status(self):
        result = run_multi_tool_agent("Check git status")
        assert isinstance(result, str)

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_no_tools(self):
        result = run_multi_tool_agent("Hello world")
        assert isinstance(result, str)

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_chaining(self):
        result = run_multi_tool_agent("List files and check git status")
        assert isinstance(result, str)

    @pytest.mark.skipif(not os.environ.get('GROQ_API_KEY'), reason="GROQ_API_KEY not set")
    def test_max_iterations(self):
        result = run_multi_tool_agent("Keep calling tools endlessly", max_iterations=2)
        assert result == "Max iterations reached"