# Plan: Harden and Optimize Code Agent

Context: Branch `optimizing-code-agent`. Address gaps noted in tool orchestration, shell safety, outputs, prompting, context handling, and tests.

## Goals
- Guardrail tool execution without blocking power-users (principles: trust the AI but with safe defaults; single shared shell; power-user audience).
- Normalize tool schemas/outputs and prompting for reliability.
- Improve context handling and add safety-focused tests.

## Workstream 1: Tool Call Validation & Error Shaping
- Add validation layer before executing tool calls (unknown tool, JSON parse errors, missing/extra args). Return structured tool error payloads instead of raising.
- Centralize a standard response schema: `{success: bool, result?: any, error?: {message, type?}, stderr?, truncated?}`.
- Update orchestrator to catch tool execution errors and append structured tool results so the model can recover.

## Workstream 2: Shell Execution Hardening
- Default `run_shell_pipeline` to sane limits (timeout, max output). Add constrained mode as default for orchestrator tool.
- Enforce command safety: reject interactive shells and destructive patterns (rm -rf, chmod/chown on root, fork bombs); integrate command parser or lightweight checks before execution.
- Preserve power-user flexibility: allow read/write within workspace; support pipes/redirects; document allowed/blocked patterns.

## Workstream 3: Tool Schemas & Prompts
- Refresh tool descriptions with precise scope (workspace-only paths, defaults, limits).
- Hide host details in `get_system_info` (omit PATH/pwd or replace with workspace info).
- Expand system prompt with guidance: when to use each tool, avoid interactive commands, expect structured errors, keep shell in workspace.

## Workstream 4: Context Handling
- Increase retained message window or add summarization of prior tool interactions so multi-step shell work doesn’t lose state.
- Keep system prompt stable; trim only excess user/assistant turns.

## Workstream 5: Testing & Safety Coverage
- Add offline tests for `run_shell_pipeline` (timeouts, truncation, path escape rejection, blocked commands) and for orchestrator error handling (unknown tool, bad args).
- If feasible, add a smoke test for the command parser classification of interactive vs non-interactive commands.

## Sequencing
1) Schema/response normalization + tool-call validation (Workstream 1).  
2) Shell hardening defaults + validation (Workstream 2).  
3) Prompt and system info tightening (Workstream 3).  
4) Context handling adjustments (Workstream 4).  
5) Tests to lock behaviors (Workstream 5).  

## Risks/Notes
- Keep timeouts/output caps reasonable for power-users; make configurable via env.
- Ensure error payloads stay small to avoid context bloat.
- Validate changes against existing examples to avoid breaking demos.***
