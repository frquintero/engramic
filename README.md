Groq Multi-Tool Code Agent

Status and scope
- Single-user, low-traffic prototype.
- Memory design lives in `history/memory_consolidation.md` — a vision doc and source of truth, not a rigid spec; pragmatism and shipping-first choices win when they diverge.
- High level conceptualization of memory design lives in `readme-glossary.md`
- Default behavior mirrors the current prototype notes in those docs.

How it works
- Run `python engramic_orchestrator.py` to start the orchestrator; it loads `config.json`, ensures the workspace exists, prints available tools, and waits for user queries.
- Each query goes to Groq with tool schemas and retrieved memory context; the agent loops over tool calls until a final response or iteration limit.
- Memory layer: embeds the query/answer pair, stores to SQLite, retrieves beacon/keyword/ANN matches, and injects consolidated cards plus recent turns into the prompt.
- Tool calls are validated, executed from `tools.py`, wrapped in a standard envelope, and fed back to the model.
- Shell pipelines run through `workspace_executor.py`, which scopes paths, rejects destructive/interactive commands in constrained mode, and truncates output.

Memory (prototype defaults)
- Storage/index: SQLite + FAISS inner-product ANN; index fully rebuilds on each card write (acceptable at small scale). Configurable via `config.json` (`memory.*`).
- Embeddings: default `embedding_choice: "bge"` → BAAI/bge-m3 via FlagEmbedding when available; falls back to Hugging Face Inference (`HUGGINGFACE_API_TOKEN`). Card embeddings re-embed summaries on every merge/new card by default.
- Merging: adaptive threshold `max(0.68, best_entity_sim - 0.05, 0.75)`; duplicates gated by similarity and entity match; merges log `card_events` and bump access counts.
- Contradictions: cheap negation + token-overlap trigger; when hit, summaries are reconciled via the LLM and re-embedded.
- Beacons: primordial registry seeded; cards always anchor `__user_self__` and `__agent_self__` (full seven-beacon anchoring is aspirational). Beacon discovery/prune jobs are available per config.
- Retrieval: beacon-first (global top-2 + primordial), then keyword, then ANN fallback; access counts increment on hits. Recent raw turns are included.
- Auditing: `card_events` table logs new cards, merges, contradictions, beacon assignments, and pruning. Splits are not implemented in this prototype.

Tool catalog (tools.py)
- `list_files`: `ls -1` in the workspace root.
- `read_file` / `write_file`: workspace-scoped file access with path validation.
- `run_shell_pipeline`: constrained pipeline runner (rejects interactive shells, destructive `rm`, permission changes; enforces workspace-bound paths). Multiline scripts can be supplied via `pipeline_lines`.
- `get_system_info_inxi`: optional system snapshot using `inxi -F` (runs unconstrained only if `inxi` is present).
- `get_user_location`: IP-based geolocation (ipinfo.io).
- `get_weather`: weather via Open-Meteo; accepts city names or `lat,lng` (pairs well with `get_user_location`).
- All tools return structured envelopes via `build_response` with success/error metadata, stderr, truncation, timeout, and duration.

Key files
- `engramic_orchestrator.py`: interactive agent code loop, argument validation, and tool-call orchestration (model default: `openai/gpt-oss-120b`).
- `memory_store.py`: SQLite/FAISS memory store, adaptive merge logic, beacon discovery/pruning, and context formatter.
- `workspace_executor.py`: workspace setup (`CODE_AGENT_WORKSPACE`), guarded pipeline execution, timeouts, and truncation.
- `command_parser.py`: AST-driven shell parser for interactive/dangerous detection (experimental).
- `container_manager.py`: prototype systemd-nspawn container lifecycle manager.
- `demo.py`: standalone multi-tool loop for basic calculator/interest/percentage calls.
- `tests/test_multitool.py`: exercises workspace guardrails and demo flow; Groq-dependent tests skip when creds/SDK are absent.
- `history/memory_consolidation.md`: memory vision doc + prototype defaults (single source of truth for the design).
- `config.json`: runtime defaults (memory toggles, thresholds, embedding choice).

Configuration and defaults
- Environment: `GROQ_API_KEY` must be set. `CODE_AGENT_PIPELINE_TIMEOUT_SECS` (default 30) and `CODE_AGENT_MAX_OUTPUT_CHARS` (default 65536) tune pipeline execution. `CODE_AGENT_WORKSPACE` sets the workspace directory (`code_agent_workspace` by default).
- Memory: see `config.json` → `memory.*` (card_k, thresholds, beacon intervals, pruning). Embeddings default to BGE; if using Hugging Face Inference, set `HUGGINGFACE_API_TOKEN`. You can override embedding provider/model via config.
- Optional: `inxi` is needed for `get_system_info_inxi`.
- Dependencies: `pip install -r requirements.txt`.

## Requirements

- **Python**: Version 3.8 or higher.
- **Environment Variables**:
  - `GROQ_API_KEY`: Required for API access.
  - `HUGGINGFACE_API_TOKEN`: Optional, for Hugging Face Inference embeddings.
- **Optional Tools**: `inxi` for system information tool.
- **Dependencies**: Install via `pip install -r requirements.txt`.

## Running and testing
- Start the agent: `python engramic_orchestrator.py` (console lists available tools).
- Demo: `python demo.py` for the calculator sample loop.
- Tests: `pytest tests/test_multitool.py` (Groq-dependent tests are skipped when credentials or SDK are absent).
