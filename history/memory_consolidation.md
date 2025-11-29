FINAL CONSOLIDATED MEMORY ARCHITECTURE — ONE-TURN MASTER PLAN

Feasibility
- Viable locally: SQLite base retained; ANN via sqlite-vector or FAISS fits 100k cards with sub-millisecond top-k if indexed and normalized. No new cloud dependencies. Main risks are ANN rebuild costs, embedding dimension drift, and beacon strength drift; mitigations are scheduled rebuilds, vector length checks, and strength decay rules.

Vision
- A cognitive memory that starts with seven primordial beacons and then autonomously discovers, strengthens, mutates, merges, splits, and forgets concepts like a lightweight neocortex. Fully local, no cloud adds.

Prototype defaults (current implementation)
- Embeddings: re-embed summaries on every merge/new card; blending is an optional optimization for later.
- Contradiction trigger: cheap text-only negation + overlap; cosine-gated version is a future upgrade.
- Beacon anchoring: always attach __user_self__ and __agent_self__; self beacons (__self_identity__, __self_location__, __self_work__, __self_health__, __self_relationships__, __self_preferences__, __self_projects__) are injected from the canonicalization helper’s self_* types (no regex fallback).
- Retrieval: uses top-2 strongest global beacons plus primordial; query-specific beacon ranking is a later improvement.
- ANN updates: FAISS index rebuilds fully on every write; fine at prototype scale, revisit >10k cards or higher churn.
- Splits: not implemented in prototype; remains a v2+ capability.

Critical Fixes and Immediate Upgrades
- ANN candidate search: replace Pass B full-scan with ANN index (sqlite-vector or FAISS). Keep top-5 under sub-ms at 100k cards; update index on card writes or via short batch jobs; prototype uses full rebuild per write (acceptable while small).
- Adaptive merge threshold: full_scan_threshold = max(0.68, best_entity_sim - 0.05, 0.75).
- Hybrid embedding strategy: target is blend with new_weight 0.4 for normal merges; contradiction or correction paths re-embed reconciled summary. Prototype default re-embeds all merges for simplicity.
- Cheap contradiction trigger: target is regex negation plus high-cosine-with-negation heuristic; prototype uses the text-only trigger and can add cosine gating later.
- engram_events mandatory: log merge, contradiction, prune with before/after payloads for audit and rollback; splits stay planned.

Cognitive Architecture
- Phase 0 — Primordial Beacons (seeded at birth): __user_self__, __agent_self__, __environment__, __people__, __projects__, __knowledge__, __interests__. Inject on weak first-person, context, or interest patterns to anchor structure; prototype always anchors user/agent self and keeps the rest in the roadmap.
- Phase 1 — Emergent Beacon Discovery (every 100 turns or nightly): co-occurrence clustering (PMI or Jaccard) over consolidated cards. When 4+ entities co-occur in 8+ cards with strength >= 0.75, promote synthetic beacon __beacon_{hash}__ (e.g., __topic_common_lisp__, __project_quantum_os__, __person_elon_musk__, __concept_category_theory__).
- Phase 2 — Beacon Lifecycle and Strength: beacon_registry tracks beacon_id, strength (0–1), card_count, generation, parent_beacon, last_touched_ms. Demote when strength < 0.5 and untouched 180 days; immortalize when strength > 0.98 and card_count > 100.
- Phase 3 — Hierarchical Retrieval: retrieve_context extracts query canonical entities, finds beacons containing any, returns cards from top-2 strongest beacons plus primordial fallback and recency slice; prototype uses global top beacons and can refine to query-specific ranking later.
- Multi-faceted self beacons: canonicalization helper emits self_* types for first-person cues; these map directly to self beacons and are forced onto cards during consolidation (new/merge/duplicate/fingerprint paths).
- Beacon mutability: primordial beacons are seeded and never pruned; discovered beacons gain/lose strength, can be pruned when weak/stale, and can be added/removed from cards by discovery/pruning jobs.

Data Model
- raw_turns: stores spans, structured_entities_json (typed canonical entries), entities_canonical_json, embeddings, fingerprint, timestamps.
- consolidated_cards: add access_count, merge_version, beacon_list (JSON of beacon ids), structured_entities_json (typed canonical entries).
- beacon_registry: id, strength, card_count, generation, parent_beacon, last_touched_ms.
- engram_events: append-only permanent log with card_id, event_type, payload_json, before_json, after_json, source_turn_id, timestamp_ms (audit trail for merges, contradictions, prunes, canonicalization, etc.).

Processing and Wiring
- Store and consolidate turn: build combined text and embedding; extract entities; canonicalize via structured helper (JSON schema, max 7, confidence filter, falls back to legacy spans if the helper fails) to derive canonical names and structured payload; canonicalization may emit self_* types which map directly to self beacons. Fingerprint; insert raw_turn (including structured_entities_json). Candidates: entity overlap first, then ANN index. Apply adaptive threshold. Paths: fingerprint shortcut, duplicate, merge, contradiction reconcile, or new card. Update embeddings per hybrid strategy, bump access_count, emit engram_events for canonicalization outcomes and every mutation.
- Retrieval: keyword on canonical entities; beacon-aware hierarchical selection; ANN fallback; recency slice; increment access_count on returned cards.
- Contradiction handling: run cheap trigger; if hit, reconcile summary, re-embed, log contradiction event.
- Beacon discovery job: compute co-occurrence, promote/demote per lifecycle rules, update beacon_list on cards.
- Maintenance: pruning uses access_count and strength; optional compression; ANN rebuild cadence tied to card churn.
- Summaries and titles: after each non-trivial turn, the orchestrator calls store_and_consolidate_turn with helper LLMs. New cards get a summary from the combined turn text via a concise summarizer and a short title generator; merges use a merge-updater, and contradictions use a reconciliation summarizer. All helper calls use Groq chat model openai/gpt-oss-20b (main chat stays on openai/gpt-oss-120b). Summaries are re-embedded when configured (default) or blended per strategy.
- Beacon assignment (LLM, structured, hard-capped): during consolidation, after summary generation, a dedicated helper (Groq structured output) receives the summary, canonical entities, current card beacons (for merges), and the entire beacon_registry (primordials + all added beacons). It must return exactly one beacon via JSON schema {existing: [id?], proposed_new: [id?]} with a mutual exclusivity rule: either one existing registry id or one new id, never both, never more than one total. If a new id is proposed, it is inserted immediately into beacon_registry (generation=1, strength seed=0.6) and attached to the card; no promotion thresholds. Self beacons from canonicalization are forced into the beacon list before this helper runs; primordial anchors are added separately after the helper selection.

Rollout Order
- Step 1: ANN index, adaptive threshold, engram_events baseline.
- Step 2: Primordial seven plus multi-faceted self beacons.
- Step 3: Contradiction reconciliation with hybrid embedding strategy.
- Step 4: Beacon discovery job and hierarchical retrieval.

Outcome
- A local, auditable cognitive layer: anchored by primordial beacons, expanded by discovered topics, contradiction-aware, fast at scale, and traceable via engram_events.
