Groq Multi-Tool Code Agent
This repository contains a lightweight command-line agent that talks to Groq’s chat completions API and safely drives a small set of tools against a sandboxed workspace. It is intended as a reference for building iterative tool-calling loops that enforce workspace boundaries and return structured envelopes to the model.

How it works
- Start the orchestrator with code_agent.py; it loads config.json, ensures the workspace directory exists, prints the available tools, and waits for user queries.
- Each query is sent to Groq with a concise system prompt that describes the workspace rules and tool schemas; the agent keeps a short rolling interaction history to give context across turns.
- When the model requests tool calls, the loop validates arguments, executes the requested functions from tools.py, wraps results or errors in a consistent JSON envelope, and feeds them back to the model.
- Shell pipelines are executed inside the dedicated workspace via workspace_executor.py with guardrails that block interactive or destructive commands and restrict paths to the workspace.
- The loop continues until the model stops asking for tools or a maximum iteration limit is hit, then prints the model’s final response.

Architecture notes
- Orchestration: code_agent.py controls the chat loop, tool invocation, validation, and logging. It reads the Groq API key from the GROQ_API_KEY environment variable and toggles verbose logging with the debug flag in config.json.
- Tool layer: tools.py implements list_files, read_file, write_file, and run_shell_pipeline, each returning a standard response envelope via build_response.
- Workspace safety: workspace_executor.py enforces workspace-relative paths, command token validation, timeouts, and output truncation for shell pipelines.
- Parsing and sandbox utilities: command_parser.py provides a richer AST-driven parser for future command analysis, and container_manager.py sketches a systemd-nspawn-based executor for isolated runs.
- Demo and tests: demo.py is a standalone multi-tool loop for financial calculations; tests/test_multitool.py exercises the workspace executor and pipeline guardrails.

Main files
- code_agent.py: interactive Groq agent loop and tool-call orchestration.
- tools.py: tool implementations, schemas, and response envelope helper.
- workspace_executor.py: workspace setup and guarded shell pipeline runner.
- command_parser.py: advanced shell command lexer/parser and metadata helpers.
- container_manager.py: prototype container lifecycle manager for sandboxed execution.
- demo.py: example of a minimal Groq multi-tool chat loop with calculator tools.
- config.json: configuration for debug logging and model selection.
- history/optimizing-code-agent-plan.md: notes from the optimization effort.
