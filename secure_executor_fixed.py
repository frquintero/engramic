"""
Secure Executor for Sandboxed Command Execution

Integrates command parsing, policy evaluation, and containerized execution
to provide safe shell command execution with granular control.
"""

import os
import tempfile
import time
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from command_parser import CommandParser, CommandContext
from policy_engine import PolicyEngine, PolicyResult, PolicyDecision
from container_manager import ContainerManager, create_default_config


@dataclass
class ExecutionResult:
    """Result of secure command execution"""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    execution_time: float
    policy_decision: PolicyDecision
    policy_reason: str
    container_used: Optional[str] = None
    error_message: Optional[str] = None


class SecureExecutor:
    """Main secure execution engine"""

    def __init__(self, workspace_dir: str = "/tmp/container_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.command_parser = CommandParser()
        self.policy_engine = PolicyEngine()
        self.container_manager = ContainerManager()
        self.container_enabled = True

        # Create default container if it doesn't exist
        self._ensure_default_container()

    def execute_command(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a command securely through the full pipeline"""
        start_time = time.time()

        try:
            # Step 1: Parse the command
            context = self.command_parser.parse_command(command)

            # Step 2: Evaluate against security policies
            policy_result = self.policy_engine.evaluate(context)

            if policy_result.decision == PolicyDecision.BLOCK:
                return ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    execution_time=time.time() - start_time,
                    policy_decision=policy_result.decision,
                    policy_reason=policy_result.reason,
                    error_message=f"Command blocked by policy: {policy_result.reason}"
                )

            if policy_result.decision == PolicyDecision.REQUIRE_APPROVAL:
                return ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    execution_time=time.time() - start_time,
                    policy_decision=policy_result.decision,
                    policy_reason=policy_result.reason,
                    error_message=f"Command requires approval: {policy_result.reason}"
                )

            # Step 3: Execute in container (if available)
            if not self.container_enabled:
                return ExecutionResult(
                    success=False,
                    returncode=-1,
                    stdout="",
                    stderr="",
                    execution_time=time.time() - start_time,
                    policy_decision=policy_result.decision,
                    policy_reason="Container runtime unavailable",
                    error_message="systemd-nspawn/machinectl not available"
                )

            container_name = "secure-executor-default"
            exec_result = self.container_manager.execute_in_container(
                container_name, command, timeout
            )

            return ExecutionResult(
                success=exec_result["success"],
                returncode=exec_result.get("returncode", -1),
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", ""),
                execution_time=time.time() - start_time,
                policy_decision=policy_result.decision,
                policy_reason=policy_result.reason,
                container_used=container_name,
                error_message=exec_result.get("error")
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                execution_time=time.time() - start_time,
                policy_decision=PolicyDecision.BLOCK,
                policy_reason="Execution error",
                error_message=f"Failed to execute command: {str(e)}"
            )

    def execute_with_files(self, command: str, input_files: Dict[str, str] = None,
                          output_files: List[str] = None, timeout: int = 30) -> ExecutionResult:
        """Execute command with file handling"""
        try:
            # Prepare input files in workspace
            file_mappings = {}
            if input_files:
                for filename, content in input_files.items():
                    file_path = self.workspace_dir / filename
                    file_path.write_text(content)
                    file_mappings[filename] = str(file_path)

            # Execute command
            result = self.execute_command(command, timeout)

            # Collect output files if requested
            if output_files and result.success:
                for filename in output_files:
                    file_path = self.workspace_dir / filename
                    if file_path.exists():
                        # Could store output files for retrieval
                        pass

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="",
                execution_time=0.0,
                policy_decision=PolicyDecision.BLOCK,
                policy_reason="File handling error",
                error_message=str(e)
            )

    def get_command_info(self, command: str) -> Dict[str, Any]:
        """Get detailed information about a command without executing it"""
        try:
            context = self.command_parser.parse_command(command)
            policy_result = self.policy_engine.evaluate(context)

            return {
                "command": command,
                "parsed": {
                    "executable": context.executable,
                    "arguments": context.arguments,
                    "environment": context.environment,
                    "interactive_status": context.interactive_status.value,
                    "command_type": context.command_type.value
                },
                "policy": {
                    "decision": policy_result.decision.value,
                    "reason": policy_result.reason,
                    "risk_level": policy_result.risk_level,
                    "alternatives": policy_result.suggested_alternatives
                }
            }
        except Exception as e:
            return {
                "command": command,
                "error": str(e)
            }

    def _ensure_default_container(self):
        """Ensure the default secure execution container exists"""
        container_name = "secure-executor-default"

        # If required binaries are missing, skip container setup and warn
        if not shutil.which("systemd-nspawn") or not shutil.which("machinectl"):
            print("Warning: systemd-nspawn/machinectl not available; containerized execution disabled")
            self.container_enabled = False
            return

        # Check if container already exists
        status = self.container_manager.get_container(container_name)
        if status and status.is_running:
            return

        # Create default container configuration
        config = create_default_config(container_name)

        # Ensure workspace directory exists and is writable
        workspace_path = Path("/tmp/container_workspace")
        workspace_path.mkdir(exist_ok=True)

        # Create the container
        success = self.container_manager.create_container(config)
        if not success:
            print(f"Warning: Failed to create default container {container_name}")

    def cleanup(self):
        """Clean up resources"""
        # Stop idle containers
        stopped = self.container_manager.cleanup_idle_containers()
        if stopped > 0:
            print(f"Cleaned up {stopped} idle containers")


# Convenience functions for easy integration
def execute_secure_command(command: str, timeout: int = 30) -> ExecutionResult:
    """Execute a command securely"""
    executor = SecureExecutor()
    return executor.execute_command(command, timeout)


def analyze_command(command: str) -> Dict[str, Any]:
    """Analyze a command without executing it"""
    executor = SecureExecutor()
    return executor.get_command_info(command)


if __name__ == "__main__":
    # Test the secure executor
    executor = SecureExecutor()

    test_commands = [
        "ls -la",
        "echo 'Hello World'",
        "python3 -c 'print(42)'",
        "python3 -i",  # Should be blocked
        "cat /etc/passwd",  # Should be blocked or restricted
    ]

    for cmd in test_commands:
        print(f"\nTesting command: {cmd}")

        # First analyze
        info = executor.get_command_info(cmd)
        print(f"Analysis: {info}")

        # Then execute if allowed
        if info.get("policy", {}).get("decision") == "allow":
            result = executor.execute_command(cmd, timeout=10)
            print(f"Execution result: success={result.success}, returncode={result.returncode}")
            if result.stdout:
                print(f"Output: {result.stdout[:100]}...")
            if result.error_message:
                print(f"Error: {result.error_message}")
        else:
            print("Command blocked by policy")

    # Clean up
    executor.cleanup()
