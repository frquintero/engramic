import json
import os
import platform
from datetime import datetime
from typing import Any, Dict, Optional, List

import requests

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
        "os": platform.platform(),
        "system_local_time": datetime.now().isoformat() + " (UTC-5)",
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


def get_user_location() -> str:
    """Get the user's current location based on IP address (single-provider lookup)."""
    try:
        # ipinfo.io allows anonymous lookups with reasonable accuracy; add UA for courtesy
        response = requests.get("https://ipinfo.io/json", timeout=10, headers={"User-Agent": "code-agent/1.0"})
        response.raise_for_status()
        data = response.json()

        loc = data.get("loc", "")
        if loc and "," in loc:
            try:
                lat_str, lng_str = loc.split(",", 1)
                lat = float(lat_str)
                lng = float(lng_str)
            except Exception:
                lat = lng = None
        else:
            lat = lng = None

        if lat is None or lng is None:
            return build_response(
                tool="get_user_location",
                success=False,
                error_type="location_not_found",
                message="Unable to detect location. Please provide a location manually.",
            )

        return build_response(
            tool="get_user_location",
            success=True,
            result={
                "city": data.get("city", "Unknown"),
                "country": data.get("country", "Unknown"),
                "latitude": lat,
                "longitude": lng,
            },
        )
    except requests.RequestException as e:
        return build_response(
            tool="get_user_location",
            success=False,
            error_type="network_error",
            message=f"Failed to fetch location: {str(e)}. Please provide a location manually.",
        )
    except Exception as e:
        return build_response(tool="get_user_location", success=False, error_type="execution_error", message=str(e))


def _geocode_name(location: str) -> Optional[Dict[str, Any]]:
    """Geocode a place name via Open-Meteo's geocoding API."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(
        geo_url,
        params={"name": location, "count": 1},
        timeout=10,
        headers={"User-Agent": "code-agent/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    return results[0] if results else None


def get_weather(location: str, days: int = 1) -> str:
    """Get weather forecast for a location using Open-Meteo API."""
    try:
        if not location or not isinstance(location, str):
            return build_response(
                tool="get_weather",
                success=False,
                error_type="validation_error",
                message="location must be a non-empty string",
            )

        try:
            days_int = int(days)
        except Exception:
            days_int = 1
        days_clamped = max(1, min(days_int, 3))

        lat = lng = None
        resolved_name = None

        # Prefer explicit lat,lng inputs (comma separated floats)
        if "," in location:
            parts = [p.strip() for p in location.split(",")]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lng = float(parts[1])
                    resolved_name = location
                except ValueError:
                    lat = lng = None

        # Fallback: geocode place name
        if lat is None or lng is None:
            geocode = _geocode_name(location)
            if not geocode:
                return build_response(
                    tool="get_weather",
                    success=False,
                    error_type="invalid_location",
                    message=f"Location '{location}' not found. Please check spelling or provide lat,lng.",
                )
            lat = float(geocode["latitude"])
            lng = float(geocode["longitude"])
            resolved_name = geocode.get("name")

        # Fetch weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lng,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode",
                "current": "temperature_2m,weathercode",
                "forecast_days": days_clamped,
                "timezone": "auto",
            },
            timeout=10,
            headers={"User-Agent": "code-agent/1.0"},
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current = weather_data.get("current", {}) or {}

        daily = weather_data.get("daily", {})
        if not daily.get("time"):
            return build_response(
                tool="get_weather",
                success=False,
                error_type="no_data",
                message="No weather data available for this location.",
            )

        # Map weather code to description
        weather_codes = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "fog", 48: "depositing rime fog",
            51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
            56: "light freezing drizzle", 57: "dense freezing drizzle",
            61: "slight rain", 63: "moderate rain", 65: "heavy rain",
            66: "light freezing rain", 67: "heavy freezing rain",
            71: "slight snow", 73: "moderate snow", 75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
            85: "slight snow showers", 86: "heavy snow showers",
            95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail"
        }

        days_list = []
        for i, date in enumerate(daily["time"]):
            if i >= days_clamped:
                break
            days_list.append(
                {
                    "date": date,
                    "max_temp_c": daily["temperature_2m_max"][i],
                    "min_temp_c": daily["temperature_2m_min"][i],
                    "precip_chance": daily["precipitation_probability_max"][i],
                    "precip_mm": daily["precipitation_sum"][i],
                    "condition": weather_codes.get(daily["weathercode"][i], "unknown"),
                }
            )

        return build_response(
            tool="get_weather",
            success=True,
            result={
                "location_query": location,
                "resolved_location": resolved_name or location,
                "latitude": lat,
                "longitude": lng,
                "current_temp_c": current.get("temperature_2m"),
                "current_condition": weather_codes.get(current.get("weathercode"), "unknown")
                if current.get("weathercode") is not None
                else None,
                "days": days_list,
            },
        )
    except requests.RequestException as e:
        return build_response(
            tool="get_weather",
            success=False,
            error_type="network_error",
            message=f"Failed to fetch weather: {str(e)}",
        )
    except Exception as e:
        return build_response(tool="get_weather", success=False, error_type="execution_error", message=str(e))


available_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_pipeline": run_pipeline_tool,
    "get_system_info_inxi": get_system_info_inxi,
    "get_user_location": get_user_location,
    "get_weather": get_weather,
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
            "description": "Execute non-interactive shell commands or pipelines in the workspace (awk, sed, bc, sort, grep, 'python -c', etc.) Use 'pipeline' for single commands/pipes, 'pipeline_lines' for multiline scripts. Arguments must be valid JSON without comments or invalid syntax.",
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
            "description": "Retrieve full system information like OS, CPU, GPU, RAM, etc.",
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
            "name": "get_user_location",
            "description": "Get the user's current geographical location based on IP address'",
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
            "name": "get_weather",
            "description": "Get weather forecast for a location. Preferred: pass lat,lng from get_user_location (e.g., '4.6097,-74.0817'). If using names, use city or 'City Country'; avoid commas with 2-letter codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location as city name (e.g., 'Miami') or 'lat,lng'."},
                    "days": {"type": "integer", "description": "Number of days to forecast (1-3, default 1).", "default": 1}
                },
                "required": ["location"],
            },
        },
    },
]

tools_descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
