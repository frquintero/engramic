Groq Multi-Tool Code Agent

A lightweight CLI agent that talks to Groq’s chat completions API and safely drives a curated set of tools against a guarded workspace. It demonstrates iterative tool-calling loops, argument validation, and structured envelopes for tool responses.

How it works
- Run `python code_agent.py` to start the orchestrator; it loads `config.json`, ensures the workspace exists, prints the available tools, and waits for user queries.
- Each query is sent to Groq with a concise system prompt that includes tool schemas and recent interaction summaries.
- A memory layer embeds the user query and the agent’s final response, stores them in SQLite, retrieves top-k similar past interactions with recency decay, and injects the compacted results into the prompt.
- Tool calls from the model are validated, executed from `tools.py`, wrapped in a standard JSON envelope, and fed back to the model until a final answer or the iteration limit is reached.
- Shell pipelines are executed through `workspace_executor.py`, which enforces workspace scoping, rejects interactive/destructive commands in constrained mode, validates paths, and truncates output.
- The loop tracks a short rolling history so the model can summarize prior exchanges within the same session.

Tool catalog (tools.py)
- `list_files`: `ls -1` in the workspace root.
- `read_file` / `write_file`: workspace-scoped file access with path validation.
- `run_shell_pipeline`: constrained pipeline runner (rejects interactive shells, destructive `rm`, permission changes; enforces workspace-bound paths). Multiline scripts can be supplied via `pipeline_lines`.
- `get_system_info_inxi`: optional system snapshot using `inxi -F` (runs in unconstrained mode only if `inxi` is present).
- `get_user_location`: single-provider IP-based geolocation (ipinfo.io).
- `get_weather`: weather forecast via Open-Meteo; accepts city names or `lat,lng` (pairs well with `get_user_location`).
- All tools return structured envelopes via `build_response` with success, error type, stderr, truncation, timeout, and duration metadata.

Key files
- `code_agent.py`: interactive agent loop, argument validation, and tool-call orchestration (model default: `openai/gpt-oss-120b`).
- `memory_store.py`: SQLite-backed memory store with cosine search, recency-aware scoring, and formatted memory blocks.
- `workspace_executor.py`: workspace setup (`CODE_AGENT_WORKSPACE`), guarded pipeline execution, timeouts, and output truncation.
- `command_parser.py`: AST-driven shell parser and interactive detection (currently exploratory).
- `container_manager.py`: prototype systemd-nspawn container lifecycle manager.
- `demo.py`: standalone multi-tool loop for financial calculations (calculator/interest/percentage).
- `tests/test_multitool.py`: exercises workspace guardrails and the demo multi-tool flow; some tests skip unless `GROQ_API_KEY` and groq-sdk are present.
- `config.json`: toggles debug logging.
- `history/optimizing-code-agent-plan.md`: notes from the optimization effort.

Configuration and defaults
- Environment: `GROQ_API_KEY` must be set; `CODE_AGENT_PIPELINE_TIMEOUT_SECS` (default 30) and `CODE_AGENT_MAX_OUTPUT_CHARS` (default 65536) tune pipeline execution; `CODE_AGENT_WORKSPACE` sets the workspace directory (`code_agent_workspace` by default).
- Memory: configured in `config.json` (`memory.enabled`, `top_k`, `similarity_threshold`, `half_life_hours`, `max_memory_chars`, `embedding_model` = `models/gemini-embedding-001`, `embedding_provider` = `gemini`, `db_path`). Requires `GEMINI_API_KEY` in `.env` for embeddings.
- Dependencies: install with `pip install -r requirements.txt`.
- Optional: `inxi` is needed for `get_system_info_inxi`.

Running and testing
- Start the agent: `python code_agent.py` (see console for available tools).
- Demo: `python demo.py` to run the calculator sample loop.
- Tests: `pytest tests/test_multitool.py` (Groq-dependent tests are skipped when credentials or SDK are absent).
