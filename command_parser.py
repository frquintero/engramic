"""
Advanced Command Parser for Secure Shell Execution

This module provides a robust shell command analysis system that safely determines
whether commands are interactive or non-interactive using a deterministic,
policy-driven approach.
"""

import shlex
import re
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    SIMPLE = "simple"
    PIPELINE = "pipeline"
    REDIRECTION = "redirection"
    COMPOUND = "compound"
    BACKGROUND = "background"


class InteractiveStatus(Enum):
    NON_INTERACTIVE = "non_interactive"
    INTERACTIVE = "interactive"
    AMBIGUOUS = "ambiguous"


@dataclass
class CommandContext:
    """Context extracted from parsed command"""
    executable: str
    arguments: List[str]
    environment: Dict[str, str]
    stdin_source: Optional[Union[str, List[str]]]  # File path or heredoc content
    stdout_dest: Optional[str]  # File path or None
    stderr_dest: Optional[str]  # File path or None
    is_pipeline: bool
    pipeline_position: int
    background: bool
    command_type: CommandType
    interactive_status: InteractiveStatus


class ShellLexer:
    """Enhanced shell lexer with advanced tokenization"""

    def __init__(self):
        self.tokens = []
        self.position = 0

    def tokenize(self, command: str) -> List[str]:
        """Tokenize shell command with advanced parsing"""
        try:
            # Use shlex for basic tokenization
            tokens = shlex.split(command, comments=True)

            # Post-process tokens for advanced features
            processed_tokens = []
            i = 0
            while i < len(tokens):
                token = tokens[i]

                # Handle environment variables (VAR=value)
                if '=' in token and i < len(tokens) - 1:
                    # Check if this looks like an environment assignment
                    if not token.startswith('-') and tokens[i + 1] not in ['=', '>', '<', '|', '&']:
                        processed_tokens.append(token)
                        i += 1
                        continue

                # Handle redirections and operators
                if token in ['>', '>>', '<', '<<', '|', '&', '&&', '||', ';']:
                    processed_tokens.append(token)
                else:
                    processed_tokens.append(token)

                i += 1

            self.tokens = processed_tokens
            return self.tokens

        except ValueError as e:
            raise ValueError(f"Failed to parse command: {e}")


class ASTNode:
    """Base class for AST nodes"""
    pass


@dataclass
class SimpleCommand(ASTNode):
    """Simple command node"""
    executable: str
    arguments: List[str]
    environment: Dict[str, str] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = {}


@dataclass
class Pipeline(ASTNode):
    """Pipeline of commands"""
    commands: List[ASTNode]


@dataclass
class Redirection(ASTNode):
    """Command with redirections"""
    command: ASTNode
    stdin: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    heredoc: Optional[List[str]] = None


@dataclass
class CompoundCommand(ASTNode):
    """Compound command with operators like && and ||"""
    left: ASTNode
    operator: str  # '&&' or '||'
    right: ASTNode


@dataclass
class Background(ASTNode):
    """Backgrounded command (trailing &)"""
    command: ASTNode


class ShellParser:
    """AST parser for shell commands"""

    def __init__(self):
        self.lexer = ShellLexer()

    def parse(self, command: str) -> ASTNode:
        """Parse command string into AST"""
        tokens = self.lexer.tokenize(command)
        return self._parse_tokens(tokens)

    def _parse_tokens(self, tokens: List[str]) -> ASTNode:
        """Parse token list into AST nodes"""
        if not tokens:
            raise ValueError("Empty command")

        # Check for background execution
        if tokens[-1] == '&':
            # Background nodes need to wrap the preceding command
            return Background(self._parse_tokens(tokens[:-1]))

        # Check for compound commands (&& and ||)
        if '&&' in tokens or '||' in tokens:
            return self._parse_compound(tokens)

        # Check for pipeline
        if '|' in tokens:
            return self._parse_pipeline(tokens)

        # Check for redirections
        if any(op in tokens for op in ['>', '>>', '<', '<<']):
            return self._parse_redirection(tokens)

        # Check for command separators like ';' (treat right side as main action)
        if ';' in tokens:
            return self._parse_sequence(tokens)

        # Simple command
        return self._parse_simple_command(tokens)

    def _parse_pipeline(self, tokens: List[str]) -> Pipeline:
        """Parse pipeline commands"""
        pipeline_parts = []
        current_part = []

        for token in tokens:
            if token == '|':
                if current_part:
                    pipeline_parts.append(current_part)
                    current_part = []
            else:
                current_part.append(token)

        if current_part:
            pipeline_parts.append(current_part)

        commands = [self._parse_simple_command(part) for part in pipeline_parts]
        return Pipeline(commands)

    def _parse_compound(self, tokens: List[str]) -> CompoundCommand:
        """Parse compound commands with && and ||"""
        # Find the rightmost && or || (respect operator precedence)
        # For simplicity, we'll find the first one from the right
        operators = ['&&', '||']

        # Look for operators from right to left
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in operators:
                left_tokens = tokens[:i]
                operator = tokens[i]
                right_tokens = tokens[i + 1:]

                if left_tokens and right_tokens:
                    left_cmd = self._parse_tokens(left_tokens)
                    right_cmd = self._parse_tokens(right_tokens)
                    return CompoundCommand(left_cmd, operator, right_cmd)

        # If no valid compound command found, treat as simple command
        return self._parse_simple_command(tokens)

    def _parse_redirection(self, tokens: List[str]) -> Redirection:
        """Parse command with redirections"""
        command_tokens = []
        stdin = None
        stdout = None
        stderr = None
        heredoc = None

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '<':
                if i + 1 < len(tokens):
                    stdin = tokens[i + 1]
                    i += 2
                    continue
            elif token == '<<':
                # Heredoc - collect until matching delimiter
                if i + 1 < len(tokens):
                    delimiter = tokens[i + 1]
                    heredoc = []
                    i += 2
                    # In a real implementation, we'd collect heredoc content
                    # For now, just mark that heredoc exists
                    continue
            elif token in ['>', '>>']:
                if i + 1 < len(tokens):
                    stdout = tokens[i + 1]
                    i += 2
                    continue
            else:
                command_tokens.append(token)

            i += 1

        command = self._parse_simple_command(command_tokens)
        return Redirection(command, stdin, stdout, stderr, heredoc)

    def _parse_sequence(self, tokens: List[str]) -> ASTNode:
        """Parse command sequences split by ';'. Return the rightmost command context."""
        # Split on the last ';' to keep right-hand side as main action
        last_sep = len(tokens) - 1 - tokens[::-1].index(';')
        left_tokens = tokens[:last_sep]
        right_tokens = tokens[last_sep + 1:]

        if right_tokens:
            return self._parse_tokens(right_tokens)

        # If separator at end, fall back to left side
        return self._parse_tokens(left_tokens)

    def _parse_simple_command(self, tokens: List[str]) -> SimpleCommand:
        """Parse simple command with environment variables"""
        executable = None
        arguments = []
        environment = {}

        for token in tokens:
            if executable is None and '=' in token and not token.startswith('-') and '/' not in token:
                # Environment variable assignment precedes executable (basic heuristic)
                key, value = token.split('=', 1)
                environment[key] = value
            elif executable is None:
                executable = token
            else:
                arguments.append(token)

        if not executable:
            raise ValueError("No executable found in command")

        return SimpleCommand(executable, arguments, environment)


class CommandParser:
    """Main command parser class"""

    def __init__(self):
        self.shell_parser = ShellParser()

    def parse_command(self, command: str) -> CommandContext:
        """Parse command and extract context"""
        try:
            ast = self.shell_parser.parse(command)
            return self._extract_context(ast, command)
        except Exception as e:
            raise ValueError(f"Failed to parse command '{command}': {e}")

    def _extract_context(self, ast: ASTNode, original_command: str) -> CommandContext:
        """Extract CommandContext from AST"""

        if isinstance(ast, SimpleCommand):
            return CommandContext(
                executable=ast.executable,
                arguments=ast.arguments,
                environment=ast.environment,
                stdin_source=None,
                stdout_dest=None,
                stderr_dest=None,
                is_pipeline=False,
                pipeline_position=0,
                background=False,
                command_type=CommandType.SIMPLE,
                interactive_status=self._determine_interactive(ast.executable, ast.arguments)
            )

        elif isinstance(ast, Pipeline):
            # For pipelines, return context for the first command
            # In a full implementation, we'd handle each command separately
            first_cmd = ast.commands[0]
            if isinstance(first_cmd, SimpleCommand):
                return CommandContext(
                    executable=first_cmd.executable,
                    arguments=first_cmd.arguments,
                    environment=first_cmd.environment,
                    stdin_source=None,
                    stdout_dest=None,
                    stderr_dest=None,
                    is_pipeline=True,
                    pipeline_position=0,
                    background=False,
                    command_type=CommandType.PIPELINE,
                    interactive_status=self._determine_interactive(first_cmd.executable, first_cmd.arguments)
                )

        elif isinstance(ast, Redirection):
            base_context = self._extract_context(ast.command, original_command)
            base_context.stdin_source = ast.stdin
            base_context.stdout_dest = ast.stdout
            base_context.stderr_dest = ast.stderr
            base_context.command_type = CommandType.REDIRECTION
            return base_context

        elif isinstance(ast, Background):
            base_context = self._extract_context(ast.command, original_command)
            base_context.background = True
            base_context.command_type = CommandType.BACKGROUND
            return base_context

        elif isinstance(ast, CompoundCommand):
            # For compound commands, return context of the right command (main action)
            base_context = self._extract_context(ast.right, original_command)
            base_context.command_type = CommandType.COMPOUND
            return base_context
        return CommandContext(
            executable="unknown",
            arguments=[],
            environment={},
            stdin_source=None,
            stdout_dest=None,
            stderr_dest=None,
            is_pipeline=False,
            pipeline_position=0,
            background=False,
            command_type=CommandType.SIMPLE,
            interactive_status=InteractiveStatus.AMBIGUOUS
        )

    def _determine_interactive(self, executable: str, arguments: List[str]) -> InteractiveStatus:
        """Determine if command is interactive based on executable and flags"""

        # Common interactive commands
        interactive_commands = {
            'bash', 'zsh', 'sh', 'fish', 'csh', 'tcsh',
            'python', 'python3', 'ipython',
            'node', 'nodejs',
            'mysql', 'psql', 'sqlite3',
            'redis-cli', 'mongo', 'mongosh',
            'vim', 'nano', 'emacs', 'less', 'more',
            'top', 'htop', 'screen', 'tmux'
        }

        # Interactive flags that indicate interactive mode
        interactive_flags = {
            '-i', '--interactive', '--shell',
            '--login', '-l'
        }

        # Non-interactive flags that override interactive behavior
        non_interactive_flags = {
            '-c', '--command', '-e', '--execute',
            '-m', '--module', '--batch'
        }

        # Check for non-interactive flags first
        for arg in arguments:
            if arg in non_interactive_flags:
                return InteractiveStatus.NON_INTERACTIVE

        # Check for interactive flags
        for arg in arguments:
            if arg in interactive_flags:
                return InteractiveStatus.INTERACTIVE

        # Check if executable is inherently interactive
        if executable in interactive_commands:
            return InteractiveStatus.INTERACTIVE

        # Default to non-interactive for safety
        return InteractiveStatus.NON_INTERACTIVE


# Convenience function for easy usage
def parse_command(command: str) -> CommandContext:
    """Parse a shell command and return its context"""
    parser = CommandParser()
    return parser.parse_command(command)


if __name__ == "__main__":
    # Test the parser
    test_commands = [
        "ls -la",
        "git status",
        "python -c 'print(hello)'",
        "cat file.txt | grep pattern",
        "echo 'hello' > output.txt",
        "python -i",
        "mysql -u user -p database"
    ]

    parser = CommandParser()
    for cmd in test_commands:
        try:
            context = parser.parse_command(cmd)
            print(f"Command: {cmd}")
            print(f"  Executable: {context.executable}")
            print(f"  Interactive: {context.interactive_status.value}")
            print(f"  Type: {context.command_type.value}")
            print()
        except Exception as e:
            print(f"Failed to parse '{cmd}': {e}")
            print()
