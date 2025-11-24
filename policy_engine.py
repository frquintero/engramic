"""
Policy Engine for Secure Command Execution

Implements PolicyEngine.evaluate() that applies explicit allow/block rules
for interpreters, database CLIs, and other tools based on required flags,
script extensions, and stdin requirements.
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass
from command_parser import CommandContext, InteractiveStatus


class PolicyDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class PolicyCategory(Enum):
    INTERPRETER = "interpreter"
    DATABASE = "database"
    VERSION_CONTROL = "version_control"
    TEXT_PROCESSING = "text_processing"
    SYSTEM = "system"
    UTILITY = "utility"


@dataclass
class PolicyRule:
    """Security policy rule for command evaluation"""
    executable: str
    category: PolicyCategory
    allowed_flags: Set[str]
    blocked_flags: Set[str]
    required_flags: Set[str]  # Flags that must be present
    environment_restrictions: Dict[str, str]  # Allowed env vars
    path_restrictions: Optional[List[str]] = None  # Allowed path patterns
    max_arguments: Optional[int] = None
    requires_stdin: bool = False
    allows_stdout_redirect: bool = True
    allows_stderr_redirect: bool = True


@dataclass
class PolicyResult:
    """Result of policy evaluation"""
    decision: PolicyDecision
    reason: str
    risk_level: str  # "low", "medium", "high"
    suggested_alternatives: Optional[List[str]] = None


class PolicyEngine:
    """Main policy evaluation engine"""

    def __init__(self):
        self.policies = self._load_policies()

    def evaluate(self, context: CommandContext) -> PolicyResult:
        """Evaluate command context against security policies"""
        try:
            # Get applicable policy
            policy = self._get_policy(context.executable)
            if not policy:
                return PolicyResult(
                    PolicyDecision.REQUIRE_APPROVAL,
                    f"No policy defined for executable: {context.executable}",
                    "high",
                    ["Contact administrator to add policy"]
                )

            # Check basic requirements
            basic_check = self._check_basic_requirements(context, policy)
            if basic_check:
                return basic_check

            # Check flags and arguments
            flag_check = self._check_flags_and_arguments(context, policy)
            if flag_check:
                return flag_check

            # Check environment variables
            env_check = self._check_environment(context, policy)
            if env_check:
                return env_check

            # Check paths
            path_check = self._check_paths(context, policy)
            if path_check:
                return path_check

            # Check I/O redirections
            io_check = self._check_io_redirections(context, policy)
            if io_check:
                return io_check

            # All checks passed
            return PolicyResult(
                PolicyDecision.ALLOW,
                "Command passed all security checks",
                "low"
            )

        except Exception as e:
            return PolicyResult(
                PolicyDecision.BLOCK,
                f"Policy evaluation error: {str(e)}",
                "high"
            )

    def _load_policies(self) -> Dict[str, PolicyRule]:
        """Load security policies for various executables"""
        return {
            # Shell interpreters
            "bash": PolicyRule(
                executable="bash",
                category=PolicyCategory.INTERPRETER,
                allowed_flags={"-c", "--help", "--version"},
                blocked_flags={"-i", "--interactive", "--login", "-l"},
                required_flags=set(),
                environment_restrictions={},
                max_arguments=100
            ),
            "sh": PolicyRule(
                executable="sh",
                category=PolicyCategory.INTERPRETER,
                allowed_flags={"-c", "--help", "--version"},
                blocked_flags={"-i", "--interactive"},
                required_flags=set(),
                environment_restrictions={}
            ),

            # Python interpreters
            "python": PolicyRule(
                executable="python",
                category=PolicyCategory.INTERPRETER,
                allowed_flags={"-c", "-m", "--help", "--version", "-B", "-u"},
                blocked_flags={"-i", "--interactive"},
                required_flags=set(),
                environment_restrictions={"PYTHONPATH": "*", "PYTHONHOME": "*"},
                max_arguments=50
            ),
            "python3": PolicyRule(
                executable="python3",
                category=PolicyCategory.INTERPRETER,
                allowed_flags={"-c", "-m", "--help", "--version", "-B", "-u"},
                blocked_flags={"-i", "--interactive"},
                required_flags=set(),
                environment_restrictions={"PYTHONPATH": "*", "PYTHONHOME": "*"},
                max_arguments=50
            ),

            # Node.js
            "node": PolicyRule(
                executable="node",
                category=PolicyCategory.INTERPRETER,
                allowed_flags={"-e", "--eval", "--help", "--version", "-p", "--print"},
                blocked_flags={"-i", "--interactive"},
                required_flags=set(),
                environment_restrictions={"NODE_PATH": "*", "NODE_ENV": "*"},
                max_arguments=50
            ),

            # Database CLIs
            "mysql": PolicyRule(
                executable="mysql",
                category=PolicyCategory.DATABASE,
                allowed_flags={"-u", "-p", "-h", "-P", "-D", "-e", "--help", "--version"},
                blocked_flags={"-A", "--auto-rehash"},
                required_flags={"-u"},  # Must specify user
                environment_restrictions={},
                max_arguments=20
            ),
            "psql": PolicyRule(
                executable="psql",
                category=PolicyCategory.DATABASE,
                allowed_flags={"-U", "-d", "-h", "-p", "-c", "--help", "--version"},
                blocked_flags={"-f"},  # Block file execution for security
                required_flags=set(),
                environment_restrictions={"PGUSER": "*", "PGDATABASE": "*"},
                max_arguments=20
            ),
            "sqlite3": PolicyRule(
                executable="sqlite3",
                category=PolicyCategory.DATABASE,
                allowed_flags={"-batch", "-header", "-csv", "-separator", "--help", "--version"},
                blocked_flags={"-interactive", "-column", "-box"},
                required_flags=set(),  # -batch is preferred but not required
                environment_restrictions={},
                max_arguments=10,
                requires_stdin=False
            ),

            # Version control
            "git": PolicyRule(
                executable="git",
                category=PolicyCategory.VERSION_CONTROL,
                allowed_flags={"-C", "-m", "status", "log", "diff", "show", "branch", "tag", "config", "add", "commit",
                             "--help", "--version", "--oneline", "--name-only", "--stat"},
                blocked_flags={"--interactive", "--force", "--hard"},
                required_flags=set(),
                environment_restrictions={"GIT_DIR": "*", "GIT_WORK_TREE": "*"},
                path_restrictions=None,  # Git can access any path it needs
                max_arguments=20
            ),

            # Text processing
            "awk": PolicyRule(
                executable="awk",
                category=PolicyCategory.TEXT_PROCESSING,
                allowed_flags={"-F", "-f", "-v", "--help", "--version"},
                blocked_flags={"-i"},  # No in-place editing for security
                required_flags=set(),
                environment_restrictions={},
                max_arguments=10
            ),
            "sed": PolicyRule(
                executable="sed",
                category=PolicyCategory.TEXT_PROCESSING,
                allowed_flags={"-e", "-f", "-n", "-i", "--help", "--version"},
                blocked_flags={"-i.bak"},  # Block backup in-place edits
                required_flags=set(),
                environment_restrictions={},
                max_arguments=10
            ),
            "grep": PolicyRule(
                executable="grep",
                category=PolicyCategory.TEXT_PROCESSING,
                allowed_flags={"-i", "-v", "-n", "-l", "-r", "-E", "-F", "--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=20
            ),

            # Shell builtins
            "cd": PolicyRule(
                executable="cd",
                category=PolicyCategory.SYSTEM,
                allowed_flags=set(),
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=1  # cd takes at most one argument (directory)
            ),
            "ls": PolicyRule(
                executable="ls",
                category=PolicyCategory.SYSTEM,
                allowed_flags={"-l", "-a", "-h", "-1", "--help", "--version"},
                blocked_flags={"-R"},  # Block recursive to prevent excessive output
                required_flags=set(),
                environment_restrictions={},
                max_arguments=10
            ),
            "cat": PolicyRule(
                executable="cat",
                category=PolicyCategory.SYSTEM,
                allowed_flags={"--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=10
            ),
            "head": PolicyRule(
                executable="head",
                category=PolicyCategory.SYSTEM,
                allowed_flags={"-n", "-c", "--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=5
            ),
            "tail": PolicyRule(
                executable="tail",
                category=PolicyCategory.SYSTEM,
                allowed_flags={"-n", "-c", "-f", "--help", "--version"},
                blocked_flags={"-f"},  # Block follow mode to prevent hanging
                required_flags=set(),
                environment_restrictions={},
                max_arguments=5
            ),

            # Utility tools
            "wc": PolicyRule(
                executable="wc",
                category=PolicyCategory.UTILITY,
                allowed_flags={"-l", "-w", "-c", "-m", "--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=5
            ),
            "sort": PolicyRule(
                executable="sort",
                category=PolicyCategory.UTILITY,
                allowed_flags={"-n", "-r", "-k", "-t", "-u", "--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=10
            ),
            "uniq": PolicyRule(
                executable="uniq",
                category=PolicyCategory.UTILITY,
                allowed_flags={"-c", "-d", "-u", "--help", "--version"},
                blocked_flags=set(),
                required_flags=set(),
                environment_restrictions={},
                max_arguments=5
            )
        }

    def _get_policy(self, executable: str) -> Optional[PolicyRule]:
        """Get policy for executable"""
        return self.policies.get(executable)

    def _check_basic_requirements(self, context: CommandContext, policy: PolicyRule) -> Optional[PolicyResult]:
        """Check basic command requirements"""

        # Check if interactive commands are blocked
        if context.interactive_status == InteractiveStatus.INTERACTIVE:
            return PolicyResult(
                PolicyDecision.BLOCK,
                f"Interactive command '{context.executable}' is not allowed",
                "high",
                ["Use non-interactive alternatives", "Provide script content directly"]
            )

        # Check maximum arguments
        if policy.max_arguments and len(context.arguments) > policy.max_arguments:
            return PolicyResult(
                PolicyDecision.BLOCK,
                f"Too many arguments ({len(context.arguments)}) for {context.executable} (max: {policy.max_arguments})",
                "medium"
            )

        return None

    def _check_flags_and_arguments(self, context: CommandContext, policy: PolicyRule) -> Optional[PolicyResult]:
        """Check command flags and arguments"""

        # Check required flags
        if policy.required_flags:
            has_required = any(flag in context.arguments for flag in policy.required_flags)
            if not has_required:
                return PolicyResult(
                    PolicyDecision.BLOCK,
                    f"Command '{context.executable}' requires one of: {', '.join(policy.required_flags)}",
                    "medium",
                    [f"Add one of: {', '.join(policy.required_flags)}"]
                )

        # Check blocked flags
        for arg in context.arguments:
            if arg in policy.blocked_flags:
                return PolicyResult(
                    PolicyDecision.BLOCK,
                    f"Flag '{arg}' is not allowed for {context.executable}",
                    "high",
                    ["Remove the blocked flag", "Use alternative approach"]
                )

        # Check allowed flags (if whitelist is enabled)
        if policy.allowed_flags:
            for arg in context.arguments:
                if arg.startswith('-') and arg not in policy.allowed_flags:
                    return PolicyResult(
                        PolicyDecision.REQUIRE_APPROVAL,
                        f"Unknown flag '{arg}' for {context.executable} - requires approval",
                        "medium"
                    )

        return None

    def _check_environment(self, context: CommandContext, policy: PolicyRule) -> Optional[PolicyResult]:
        """Check environment variable restrictions"""

        for env_var, env_value in context.environment.items():
            if env_var not in policy.environment_restrictions:
                if policy.environment_restrictions:  # If restrictions are defined, block unknown vars
                    return PolicyResult(
                        PolicyDecision.BLOCK,
                        f"Environment variable '{env_var}' is not allowed for {context.executable}",
                        "medium"
                    )

        return None

    def _check_paths(self, context: CommandContext, policy: PolicyRule) -> Optional[PolicyResult]:
        """Check path restrictions in arguments"""

        if not policy.path_restrictions:
            return None

        # Check arguments for paths
        for arg in context.arguments:
            if '/' in arg or '\\' in arg:  # Likely a path
                # Simple check - in production, use proper path validation
                allowed = any(pattern in arg for pattern in policy.path_restrictions)
                if not allowed:
                    return PolicyResult(
                        PolicyDecision.BLOCK,
                        f"Path '{arg}' is not in allowed directories for {context.executable}",
                        "high"
                    )

        return None

    def _check_io_redirections(self, context: CommandContext, policy: PolicyRule) -> Optional[PolicyResult]:
        """Check I/O redirection permissions"""

        if context.stdout_dest and not policy.allows_stdout_redirect:
            return PolicyResult(
                PolicyDecision.BLOCK,
                f"Output redirection not allowed for {context.executable}",
                "medium"
            )

        if context.stderr_dest and not policy.allows_stderr_redirect:
            return PolicyResult(
                PolicyDecision.BLOCK,
                f"Error redirection not allowed for {context.executable}",
                "medium"
            )

        return None


# Convenience function
def evaluate_command(command: str) -> PolicyResult:
    """Parse and evaluate a command string"""
    from command_parser import parse_command

    try:
        context = parse_command(command)
        engine = PolicyEngine()
        return engine.evaluate(context)
    except Exception as e:
        return PolicyResult(
            PolicyDecision.BLOCK,
            f"Failed to evaluate command: {str(e)}",
            "high"
        )


if __name__ == "__main__":
    # Test the policy engine
    test_commands = [
        "ls -la",
        "python -c 'print(hello)'",
        "python -i",  # Should be blocked
        "git status",
        "mysql -u user -p db",  # Should require approval
        "sqlite3 --interactive",  # Should be blocked
        "awk '{print $1}' file.txt"
    ]

    for cmd in test_commands:
        result = evaluate_command(cmd)
        print(f"Command: {cmd}")
        print(f"  Decision: {result.decision.value}")
        print(f"  Reason: {result.reason}")
        print(f"  Risk: {result.risk_level}")
        if result.suggested_alternatives:
            print(f"  Alternatives: {', '.join(result.suggested_alternatives)}")
        print()
