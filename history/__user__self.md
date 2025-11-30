### Primordial Beacon `__user_self__` (Engramic vocabulary)

- Role: anchors all user-centric Engrams (title/summary + canonical entities + beacon list + neuroprint + source turns). Always attached to new Engrams as part of the primordial fallback.
- Where it lives: `consolidated_cards.beacon_list_json` and `beacon_registry` (immutable primordial entry).
- Recall behavior: retrieval always includes primordial anchors (`__user_self__`, `__agent_self__`) in the beacon shortlist; cards with this beacon surface first for user-related queries.
- Consolidation behavior: `store_and_consolidate_turn` assigns the beacon on every new/merged card; access_count and beacon strength bump on hits; primordial beacons are never pruned.
- Pre-seeding (optional): seed Engrams for known user facts (identity, pronouns, timezone, preferences, workflows) tagged with `__user_self__`; store as normal Engrams with summary/title/entities/beacons/embedding.
