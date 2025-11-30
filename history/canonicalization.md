# Canonicalization Upgrade Plan

## Current Issue
- Structured canonicalization is live: the helper returns typed entities (span, type, canonical_name, optional confidence) with self_* taxonomy support.
- Canonical names are derived from those structured entities and drive merge/duplicate/beacon logic; the deterministic lowercase spans only appear as a fallback when the helper or schema fails.
- Raw turns and cards now store `structured_entities_json` alongside the derived `entities_canonical_json`, and engram_events capture helper success/failure for telemetry.

## Goal
Establish a reliable canonicalization layer that maps spans to typed, disambiguated canonical names with graceful fallbacks and telemetry, without breaking existing storage and retrieval paths.

## Plan

### Step 1: Design and Schema
- Task: Define structured canonical entity shape: span, type (from taxonomy), canonical_name, optional confidence.
- Taxonomy now includes self_* types: self_identity, self_location, self_work, self_health, self_relationships, self_preferences, self_projects for first-person cues; helper must emit these to drive self-beacon injection.
- Task: Decide storage: add structured column (JSON) and keep a derived `canonical_names` list for backward compatibility.
- Task: Update invariants: max 7 spans per turn, confidence cutoff policy, required fields.

### Step 2: Prompt and Helper
- Task: Implement the canonicalization helper prompt (spans + user_query + agent_response + summary, taxonomy).
- Task: Enforce JSON schema in responses; reject extras and cap items to 7.
- Task: Add fallback: on helper failure, return legacy lowercase spans; log the failure.

### Step 3: Consolidation Wiring
- Task: Call the canonicalization helper inside `store_and_consolidate_turn` before merge decisions.
- Task: Store both structured entities and the derived `canonical_names` list.
- Task: Ensure merge/duplicate thresholds use `canonical_names` as today, but keep legacy path if helper fails.
- Task: Emit engram_events capturing helper success/failure and applied canonical set.

### Step 4: Beacon Assignment Integration
- Task: Feed `canonical_names` into the existing beacon assignment LLM.
- Task: If canonicalization fails, fall back to legacy spans plus primordial/self beacons.
- Task: Log beacon helper outcomes and disagreements.

### Step 5: Retrieval and Beacon Discovery
- Task: Use `canonical_names` for beacon selection and keyword fallback; keep ANN fallback unchanged.
- Task: Update beacon discovery job to operate on `canonical_names` from structured entities.
- Task: Add confidence-aware filtering when selecting entities for retrieval if enabled.

### Step 6: Observability and Quality Gates
- Task: Instrument helper latency, failure rate, and drop rate from confidence cutoffs.
- Task: Log which canonical entities were used for each merge/new-card path.
- Task: Add lightweight validation to skip empty or schema-violating outputs.

### Step 7: Hardening and Cleanup
- Task: Add migration that introduces the new structured column and backfills `canonical_names` from legacy data.
- Task: Remove reliance on raw spans where safe; keep compatibility layer until data is migrated.
- Task: Add guardrails to avoid crashes on missing embeddings or helper timeouts.

## Out of Scope for This Iteration
- Relationship graphs and temporal weighting.
- Multi-modal canonicalization.
- Complex confidence-weighted retrieval policies.

## Success Criteria
- Structured canonical entities stored per turn and per card.
- Merge and beacon selection use canonical names with fewer false merges/splits.
- System remains functional under helper failure via fallbacks and logs helper health.

## Implementation Results
- Canonicalization helper added with schema enforcement, max 7 items, and fallbacks.
- Consolidation now calls the helper via the orchestrator, derives canonical names with confidence/type filtering, and falls back to legacy when needed.
- Structured canonical entities are stored on raw turns and cards; canonicalization outcomes are emitted to `engram_events`; event log renamed to `engram_events`.
