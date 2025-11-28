### Primordial Beacon `__agent_self__` (Engramic vocabulary)

- Role: anchors all agent-centric Engrams (title/summary + canonical entities + beacon list + neuroprint + source turns). Always attached to new Engrams as part of the primordial fallback.
- Where it lives: `consolidated_cards.beacon_list_json` and `beacon_registry` (immutable primordial entry).
- Recall behavior: retrieval always includes primordial anchors (`__agent_self__`, `__user_self__`) in the beacon shortlist; cards with this beacon surface first for agent-related queries.
- Consolidation behavior: `store_and_consolidate_turn` assigns the beacon on every new/merged card; access_count and beacon strength bump on hits; primordial beacons are never pruned.
- Pre-seeding (optional): you can seed Engrams about the agent (capabilities, guardrails, operating mode) tagged with `__agent_self__` to bootstrap self-knowledge; treat them as normal Engrams with summary/title/entities/beacons/embedding.
