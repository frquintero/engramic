# Memory Upgrade Plan: Hybrid Entity-Centric Consolidation

Target: implement the “Ultimate version: Hybrid + Entity-Centric Memory Consolidation” requested by the user. No code changes yet—this is the execution plan.

## Current State (baseline)
- Single-table memory (`turns` + `turn_embeddings`) in `memory_store.py` storing raw user/assistant pairs with combined embeddings; retrieval mixes cosine + recency decay + topic expansion.
- No entity extraction, no consolidation; recency and similarity thresholds gate results; prompt injects formatted turns only.
- Config in `config.json` controls k/threshold/half-life/max chars and embedding choice (Gemini/BGE/groq).

## Goals
- Persistent append-only `raw_turns` while introducing `consolidated_memories` (“cards”) keyed by entities/topics.
- Entities JSON column on both tables; consolidate on write with merge/insert/discard logic using entity overlap + cosine thresholds.
- Retrieval: keyword-first on entities, then low-threshold vector fallback, plus a small recency slice of raw turns.
- Remove recency decay from consolidated cards; keep simple recency window for raw turns only.
- Keep prompt context small: 1–4 consolidated cards + last 3–8 raw turns.
- Preserve reliability with graceful degradation and traceability; avoid blocking on model hiccups.

## Schema and Storage Plan
- `raw_turns` (new, append-only): `id`, `user_content`, `assistant_content`, `combined_text` (cached), `timestamp_ms`, `conversation_id`, `entities_json` (array of strings), `combined_embedding_json`.
- `consolidated_memories`: `id`, `title` (short handle/slug), `summary` (card text), `entities_json`, `embedding_json`, `created_ms`, `updated_ms`, `source_turn_ids` (JSON array) for traceability.
- Indexes: entity GIN-like via plain index on `entities_json` (with LIKE/JSON_SEARCH), and standard index on `updated_ms`. Keep embeddings as JSON; reuse Python cosine for now (no sqlite-vec dependency yet).
- Migration/backfill: add new tables without altering existing `turns`; optionally copy existing `turns` into `raw_turns` and seed `consolidated_memories` via batch consolidation routine.

## Entity Extraction Strategy
- Use spaCy `en_core_web_sm` (or similar small model) to extract nouns/proper nouns/entities; fallback regex/keyword splitter if spaCy unavailable.
- Normalize entities: lowercase, strip punctuation, de-dup; persist as JSON array on both tables.
- Extraction input: combined user+assistant turn text when storing; user query text when retrieving.

## Write Path (on each completed turn)
1) Build `combined_text` = `<user_query>…</user_query>\n<agent_response>…</agent_response>`.
2) Embed `combined_text` with existing `_embed_text` (provider/config driven). Abort store if embedding missing.
3) Extract entities from `combined_text`.
4) Insert into `raw_turns` with embedding + entities.
5) Consolidation step:
   - Find candidate cards sharing ≥1 entity (entity overlap filter).
   - For candidates, compute cosine between new turn embedding and card embedding.
   - Branches:
     - Near-duplicate: cosine ≥ 0.92 AND same entity set → discard (do nothing).
     - Strong match: cosine ≥ 0.75 AND shares entity → merge. Use an LLM one-liner prompt to update card summary with new facts (keep concise, idempotent). Update `summary`, `embedding` (re-embed new summary), `entities` (union), `updated_ms`, append turn_id to `source_turn_ids`.
     - Weak/no match: create new card with summary generated from new turn (LLM short abstractive pass), initial entities/embedding.
   - Embedding update strategies (choose via config toggle):
     - Default: re-embed the updated summary (highest fidelity).
     - Fast path: weighted blend of prior card embedding with new turn embedding (e.g., 70/30) to avoid LLM re-embed cost.
   - Title generation: at card creation, add a tiny prompt: “Give this memory a short 3–6 word title.”
   - Conflict tolerance: if merge detects conflicting statements, append both with timestamps; do not block or discard—let downstream reasoning handle it.
   - Graceful degradation: if the consolidation LLM call fails, fall back to bullet-append (raw text addition) and embed/merge using the raw concatenation instead of failing the write.

## Retrieval Flow (per user query)
1) Extract entities from current user query.
2) **Keyword pass:** exact/in set lookup against `consolidated_memories.entities_json`; return hits (no similarity gating).
3) If no keyword hits: **Vector pass** over consolidated embeddings with low threshold (0.38–0.45) and top_k ~4; keep cosine only (no recency decay).
4) Working memory: fetch last N raw turns (e.g., 3–8) by `timestamp_ms` without similarity filter; skip if already represented in consolidated hits for same entities.
5) Assemble prompt context:
   - Up to 1–4 consolidated cards (highest score first; clip summaries to budget).
   - Plus recent raw turns block.
   - Total memory chars/tokens budgeted (~4–6k tokens) before user query.

## Prompt/Orchestrator Integration
- Update `code_agent.py` to build memory block from consolidated cards + recency turns; remove topic expansion/recency scoring logic.
- Keep tool list/system prompt unchanged except to explain the new memory semantics (“cards” trusted, working-memory turns for immediacy).

## Configuration Additions
- `memory.consolidation_enabled` (default true).
- Thresholds: `duplicate_threshold` (0.92), `merge_threshold` (0.75), `vector_threshold` (0.38–0.45), `recent_turn_limit` (3–8), `card_k` (1–4).
- Entity extractor choice (`spacy_model`, `entity_fallback`), and flag to allow non-English if needed.
- Optional `consolidation_model` for the merge/summary LLM (defaults to main chat model).
- Embedding update mode: `card_embedding_strategy` ∈ {`reembed_summary` (default), `blend_turn` (70/30 turn-card mix)}.

## Testing Plan
- Unit: entity extraction normalization; schema creation/migration; add_turn stores embeddings/entities; consolidation branch outcomes for near-duplicate/merge/new-card; vector + keyword retrieval ordering; raw-turn recency window behavior.
- Integration: orchestrator call that stores a turn and retrieves matching card via keyword (e.g., “Arlington”); ensure cosine threshold lowered doesn’t return unrelated cards when entities mismatch.
- Regression: ensure trivial queries are still skipped; ensure failure to embed skips storage without raising.
- Degradation path: when consolidation LLM is unavailable, verify bullet-append merge still creates/updates cards.

## Rollout/Operational Notes
- Logging: emit consolidation decisions (discard/merge/new), thresholds used, entity lists, and card IDs for observability.
- Maintenance: periodic re-embedding of cards optional but not required; vacuum old raw turns only if size becomes an issue (cards remain timeless). Consider future entity index upgrade (FTS5 or `memory_entities` junction table) if card count >10k.
- Backward compatibility: leave old `turns` table untouched; new code reads/writes only new tables but can optionally expose a migration flag to import historical turns. Provide a one-click backfill script to seed `consolidated_memories` from historical `raw_turns`.
- Traceability: keep `source_turn_ids` wired for future UI hover/jump-back to original turns.
