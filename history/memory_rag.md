# Hybrid Entity-Centric Memory (Consolidated Cards + Raw Turns)

## Objective
Maintain compact, high-recall long-term memory via consolidated, entity-grouped cards, while keeping a short window of raw turns for immediate context.

## Core approach
- Persist every user+assistant turn in `raw_turns` with embeddings, entities (raw + canonical), and an optional observation fingerprint.
- Maintain `consolidated_memories` (“cards”) keyed by entities; each card holds a concise summary, embeddings, canonical entities, and the last observation fingerprint.
- On each new turn: extract entities, compute a fingerprint, find overlapping cards (canonical entity overlap), then merge, discard, or create based on similarity and fingerprints.
- Retrieval is keyword-first on canonical entities, then low-threshold vector fallback, plus a small recency slice of raw turns.
- Prompt injection stays small: a few cards (1–4) plus last 3–8 raw turns.

## Data model (current)
- `raw_turns`: id, user_content, assistant_content, combined_text, entities_json, entities_canonical_json, combined_embedding_json, observation_fingerprint, timestamp_ms, conversation_id.
- `consolidated_memories`: id, title, summary, entities_json, entities_canonical_json, embedding_json, observation_fingerprint, created_ms, updated_ms, source_turn_ids_json.
- Legacy `turns`/`turn_embeddings` remain for backward compatibility (unused by consolidation).

## Entities and canonical IDs
- Extract entities from natural combined text (`User: …\nAssistant: …`) using spaCy when available, else a nounish fallback.
- Canonicalize entities (lowercase, normalize slashes, strip leading `./`, collapse whitespace/punct) and store both raw and canonical lists.
- Matching/duplicate logic uses canonical sets; display uses raw.

## Observation fingerprints (deterministic, no ML)
- Compute a stable fingerprint for the observation (hash of normalized combined text) and store it on both raw turns and cards.
- If a new turn shares canonical entities with a card and the fingerprint matches, we update timestamps/source ids and skip summary rewrites (no duplicate card churn).

## Ingestion (per completed turn)
1) Build clean combined text: `User: …\nAssistant: …`.
2) Embed combined text.
3) Extract entities → canonicalize → store both.
4) Compute observation fingerprint (hash of normalized combined text).
5) Insert into `raw_turns`.
6) Consolidate:
   - Find cards with overlapping canonical entities.
   - Compute cosine between turn embedding and card embedding.
   - Branches:
     - Near-duplicate discard: canonical sets identical AND (cosine ≥ duplicate_threshold or fingerprint match) → update timestamp/source list and return (no new summary).
     - Merge: cosine ≥ merge_threshold → update summary (LLM one-liner, fallback to simple merge), union entities (raw+canonical), update embedding (re-embed summary or blend), update fingerprint, timestamps, and source ids.
     - New card: create with summary/title, entities (raw+canonical), embedding (re-embed summary or use turn embedding), fingerprint.

## Retrieval
1) Extract entities from user query; canonicalize.
2) Keyword pass: match consolidated cards whose canonical entities overlap; return up to card_k, freshest first.
3) If none, vector search over card embeddings with low threshold (~0.38–0.45).
4) Add last N raw turns (recency only) as short-term memory.
5) Format block for the prompt:
   - Long-term: card list with titles, timestamps, entities, summaries.
   - Short-term: recent raw turns with timestamps and entities.

## Prompt assembly rules
- System prompt explains long-term cards (trust), short-term turns (recency), and behavior when memory is empty.
- Inject formatted memory block into the user message before the current query.
- Keep under memory_max_chars (~4–6k tokens total with user input).

## Parameters
- `card_k` (default 4), `vector_threshold` (~0.38–0.45), `recent_turn_limit` (3–8).
- Merge/duplicate thresholds: `merge_threshold` (0.75), `duplicate_threshold` (0.92).
- Embedding strategy: re-embed summary (default) or blend turn/card.

## Guardrails and observability
- Stop storing trivial queries.
- Log consolidation decisions (discard/merge/new) with entities/fingerprint and card ids for traceability.
- Keep `source_turn_ids` to trace back to raw observations.

## Maintenance/migration
- New columns are added lazily via schema checks; legacy tables left intact.
- If legacy data exists, run a backfill through `store_and_consolidate_turn` to seed cards.
- DB is local SQLite; safe to regenerate if needed.
