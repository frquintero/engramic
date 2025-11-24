"""
Container Manager for Persistent systemd-nspawn Containers

Manages the lifecycle of sandboxed execution environments using systemd-nspawn
for secure command execution with controlled resource limits and filesystem access.
"""

import subprocess
import os
import tempfile
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContainerConfig:
    """Configuration for a container instance"""
    name: str
    root_directory: str
    bind_mounts: List[str]
    read_only_mounts: List[str]
    environment: Dict[str, str]
    capabilities: List[str]
    resource_limits: Dict[str, Any]


@dataclass
class ContainerStatus:
    """Status of a container"""
    name: str
    is_running: bool
    pid: Optional[int]
    start_time: Optional[float]
    last_used: Optional[float]


class ContainerManager:
    """Manages persistent systemd-nspawn containers"""

    def __init__(self, base_dir: str = "/tmp/containers"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.containers: Dict[str, ContainerStatus] = {}
        self._load_container_states()

    def create_container(self, config: ContainerConfig) -> bool:
        """Create a new persistent container"""
        try:
            container_dir = self.base_dir / config.name
            container_dir.mkdir(exist_ok=True)

            # Create basic root filesystem structure
            self._setup_container_root(container_dir, config)

            # Create systemd-nspawn configuration
            self._create_nspawn_config(config)

            # Start the container
            success = self._start_container(config.name)
            if success:
                self.containers[config.name] = ContainerStatus(
                    name=config.name,
                    is_running=True,
                    pid=self._get_container_pid(config.name),
                    start_time=time.time(),
                    last_used=time.time()
                )
                self._save_container_states()

            return success

        except Exception as e:
            print(f"Failed to create container {config.name}: {e}")
            return False

    def get_container(self, name: str) -> Optional[ContainerStatus]:
        """Get container status"""
        return self.containers.get(name)

    def execute_in_container(self, container_name: str, command: str,
                           timeout: int = 30) -> Dict[str, Any]:
        """Execute command in specified container"""
        try:
            if not self._ensure_container_running(container_name):
                return {
                    "success": False,
                    "error": f"Container {container_name} is not running",
                    "stdout": "",
                    "stderr": ""
                }

            # Execute command via machinectl
            result = subprocess.run(
                ["machinectl", "shell", container_name, "/bin/bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Update last used time
            if container_name in self.containers:
                self.containers[container_name].last_used = time.time()
                self._save_container_states()

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }

    def stop_container(self, name: str) -> bool:
        """Stop a container"""
        try:
            subprocess.run(["machinectl", "stop", name], check=True)
            if name in self.containers:
                self.containers[name].is_running = False
                self._save_container_states()
            return True
        except subprocess.CalledProcessError:
            return False

    def cleanup_idle_containers(self, max_idle_time: int = 300) -> int:
        """Stop containers that have been idle for too long"""
        current_time = time.time()
        stopped_count = 0

        for name, status in list(self.containers.items()):
            if status.is_running and status.last_used:
                idle_time = current_time - status.last_used
                if idle_time > max_idle_time:
                    if self.stop_container(name):
                        stopped_count += 1

        return stopped_count

    def _setup_container_root(self, container_dir: Path, config: ContainerConfig):
        """Set up basic container root filesystem"""
        # Create basic directories
        dirs = ["bin", "lib", "lib64", "usr", "etc", "tmp", "var", "home"]
        for dir_name in dirs:
            (container_dir / dir_name).mkdir(exist_ok=True)

        # Copy essential binaries and libraries
        self._copy_essentials(container_dir, config)

        # Create minimal passwd/group files
        self._create_minimal_passwd(container_dir)

    def _copy_essentials(self, container_dir: Path, config: ContainerConfig):
        """Copy essential binaries and libraries to container"""
        essentials = [
            "/bin/bash",
            "/bin/cat",
            "/bin/ls",
            "/bin/grep",
            "/bin/awk",
            "/bin/sed",
            "/usr/bin/git",
            "/usr/bin/python3",
            "/usr/bin/node"
        ]

        for essential in essentials:
            if os.path.exists(essential):
                try:
                    # Copy binary
                    subprocess.run(["cp", essential, str(container_dir / "bin" / Path(essential).name)], check=True)

                    # Copy libraries (simplified - in production, use ldd to find all dependencies)
                    self._copy_libraries(essential, container_dir)
                except subprocess.CalledProcessError:
                    continue  # Skip if copy fails

    def _copy_libraries(self, binary: str, container_dir: Path):
        """Copy required libraries for a binary"""
        try:
            # Use ldd to find library dependencies
            result = subprocess.run(["ldd", binary], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '=>' in line and not line.strip().startswith('linux-vdso'):
                        parts = line.split('=>')
                        if len(parts) >= 2:
                            lib_path = parts[1].split('(')[0].strip()
                            if lib_path and os.path.exists(lib_path):
                                lib_dest = container_dir / "lib" / Path(lib_path).name
                                try:
                                    subprocess.run(["cp", lib_path, str(lib_dest)], check=True)
                                except subprocess.CalledProcessError:
                                    continue
        except subprocess.CalledProcessError:
            pass

    def _create_minimal_passwd(self, container_dir: Path):
        """Create minimal passwd and group files"""
        passwd_content = """root:x:0:0:root:/root:/bin/bash
user:x:1000:1000:user:/home/user:/bin/bash
"""
        group_content = """root:x:0:
user:x:1000:
"""

        (container_dir / "etc" / "passwd").write_text(passwd_content)
        (container_dir / "etc" / "group").write_text(group_content)

    def _create_nspawn_config(self, config: ContainerConfig):
        """Create systemd-nspawn configuration file"""
        config_content = f"""# systemd-nspawn configuration for {config.name}
[Exec]
Boot=no
Ephemeral=no

[Files]
"""

        # Add bind mounts
        for mount in config.bind_mounts:
            config_content += f"Bind={mount}\n"

        for mount in config.read_only_mounts:
            config_content += f"BindReadOnly={mount}\n"

        # Add environment variables
        if config.environment:
            config_content += "\n[Environment]\n"
            for key, value in config.environment.items():
                config_content += f"{key}={value}\n"

        config_file = Path(f"/etc/systemd/nspawn/{config.name}.nspawn")
        config_file.parent.mkdir(exist_ok=True)
        config_file.write_text(config_content)

    def _start_container(self, name: str) -> bool:
        """Start a container using systemd-nspawn"""
        try:
            container_dir = self.base_dir / name
            subprocess.run([
                "systemd-nspawn",
                "--machine", name,
                "--directory", str(container_dir),
                "--persistent",
                "--quiet",
                "--boot"  # This might need adjustment based on container setup
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _ensure_container_running(self, name: str) -> bool:
        """Ensure container is running, start if necessary"""
        status = self.containers.get(name)
        if not status or not status.is_running:
            return self._start_container(name)

        # Check if container is actually running
        try:
            result = subprocess.run(["machinectl", "show", name],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def _get_container_pid(self, name: str) -> Optional[int]:
        """Get PID of container leader process"""
        try:
            result = subprocess.run(["machinectl", "show", name, "--property=Leader"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Leader='):
                        return int(line.split('=')[1])
        except (subprocess.CalledProcessError, ValueError):
            pass
        return None

    def _load_container_states(self):
        """Load container states from persistent storage"""
        state_file = self.base_dir / "container_states.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    for name, status_data in data.items():
                        self.containers[name] = ContainerStatus(**status_data)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_container_states(self):
        """Save container states to persistent storage"""
        state_file = self.base_dir / "container_states.json"
        data = {}
        for name, status in self.containers.items():
            data[name] = {
                "name": status.name,
                "is_running": status.is_running,
                "pid": status.pid,
                "start_time": status.start_time,
                "last_used": status.last_used
            }

        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)


# Default container configuration
def create_default_config(name: str) -> ContainerConfig:
    """Create a default secure container configuration"""
    return ContainerConfig(
        name=name,
        root_directory=f"/tmp/containers/{name}",
        bind_mounts=[
            "/tmp/container_workspace:/workspace"
        ],
        read_only_mounts=[
            "/usr:/usr",
            "/bin:/bin",
            "/lib:/lib",
            "/lib64:/lib64"
        ],
        environment={
            "PATH": "/bin:/usr/bin",
            "HOME": "/home/user",
            "USER": "user"
        },
        capabilities=[],  # No special capabilities
        resource_limits={
            "cpu_quota": "50%",  # Limit to 50% of CPU
            "memory_limit": "512M",  # Limit to 512MB RAM
            "max_processes": 100
        }
    )


if __name__ == "__main__":
    # Test the container manager
    manager = ContainerManager()

    # Create a test container
    config = create_default_config("test-container")
    success = manager.create_container(config)

    if success:
        print("Container created successfully")

        # Test command execution
        result = manager.execute_in_container("test-container", "echo 'Hello from container'")
        print(f"Command result: {result}")

        # Clean up
        manager.stop_container("test-container")
        print("Container stopped")
    else:
        print("Failed to create container")