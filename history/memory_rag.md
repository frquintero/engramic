# Dynamic Memory Selection for the Orchestrator (RAG)

## Objective
Give the orchestrator an efficient, semantics-aware memory layer that retrieves only the most relevant prior interactions and injects them into the model prompt without exceeding context limits.

## Core approach
- Encode the current user input (and optional system hints) into embeddings.
- Query a local vector store (SQLite + sqlite-vec) for the top-k most similar past interactions.
- Apply a similarity threshold to drop low-relevance memories.
- Concatenate the surviving memories into a compact prompt segment that precedes the user query.

## Architecture sketch
- **Memory store**: SQLite database with sqlite-vec for similarity search. All data stays local.
- **Indexer**: writes embeddings + metadata for each interaction.
- **Retriever**: fast top-k cosine search with optional min similarity.
- **Orchestrator integration**: injects retrieved memories into the system/user context before sending to Groq.

## Data model (proposed)
- Table `interactions`:
  - `id` (INTEGER PRIMARY KEY)
  - `role` (TEXT) — user|assistant|system
  - `content` (TEXT)
  - `timestamp` (INTEGER, ms)
  - `conversation_id` (TEXT) — to group sessions
- Table `interaction_embeddings`:
  - `interaction_id` (INTEGER REFERENCES interactions(id) ON DELETE CASCADE)
  - `embedding` (VECTOR) — sqlite-vec column
  - Composite index on `(interaction_id)`
- Optional: `tags` (TEXT) for lightweight filtering (e.g., domain, language).

## Embeddings
- Source: call embeddings endpoint via configured provider; default model `models/gemini-embedding-001` (Gemini) for speed/size balance.
- Shape: 768–1536 dims depending on model; store as `FLOAT32`.
- Caching: deduplicate identical content hashes to avoid repeated embedding calls.

## Ingestion flow
1) After each turn, embed only the user query and the agent’s final response (no intermediate/tool text).  
2) Compute embeddings for each stored item and insert into `interaction_embeddings`.  
3) Persist interaction metadata alongside each embedding in the same transaction.

## Retrieval flow
1) Encode the current user query into an embedding.  
2) Run `SELECT ... ORDER BY cosine_similarity(...) DESC LIMIT k` with `WHERE similarity >= threshold` if provided.  
3) Blend similarity with recency by adding a time-decay score (e.g., `score = sim * exp(-age_hours / half_life_hours)` or a linear decay cap) before ranking; keep pure similarity as a fallback if timestamps are missing.  
4) Return the top-k memories with metadata.  
5) Compact them (strip excess whitespace, optionally summarize long items).  
6) Inject into the prompt before the latest user message. Example block:
```
Relevant memories:
- [ts 2024-06-01] User asked about deploying with systemd; assistant recommended nspawn.
- [ts 2024-06-10] User shared API key handling constraints.
```

## Parameters
- `k` (default 5, clamp e.g. 1–15).
- `similarity_threshold` (default 0.2–0.3; disable if None).
- `max_memory_chars` to cap injected text length (fallback to trimming).
- `max_token_budget` for the whole prompt (stop adding memories when near limit).
- Recency tuning: `half_life_hours` (e.g., 24–168) for exponential decay, or a `recency_bonus` for items within a sliding window (e.g., last 24h) applied before final ranking.

## Prompt assembly rules
- Always include system prompt and current user query.
- Add retrieved memories in chronological order (oldest to newest) after labeling them.
- If no memory passes threshold, inject a short note: “No relevant past interactions retrieved.”
- Avoid duplicating recent turns already present in the live context.

## Validation and guardrails
- Truncate overly long contents before embedding and before injection.
- Reject storage of empty/whitespace-only content.
- Ensure embeddings and interactions share a transaction for consistency.
- Provide feature flag to disable retrieval for debugging.

## Tests (proposed)
- Unit: schema creation/migration, insert + retrieve roundtrip, threshold filtering, k-bounds, dedup hash cache.
- Integration: orchestrator builds prompts with/without memories; budget guard prevents overflow; empty retrieval path noted.
- Regression: multilingual inputs retrieve matching multilingual memories (e.g., “hola, ¿cómo estás?” recalls prior Spanish greetings).

## Metrics and observability
- Log retrieval latency, k, threshold, and final injected memory count.
- Surface embedding call failures and fallback behavior (run without memory).
- Optional: histogram of similarity scores to tune defaults.

## Rollout plan
1) Add SQLite/sqlite-vec dependency and migration to create tables.  
2) Implement embedding client + cache.  
3) Add ingestion hook after each turn in `code_agent.py`.  
4) Add retrieval hook before request construction; wire parameters via config.  
5) Add prompt assembly helper and tests.  
6) Ship behind a config flag; enable by default after bake-in.  
