# Engramic Memory Glossary & Architecture

This document describes the **Engramic** memory model used in this repo, and maps each concept to its concrete implementation in the codebase and spec doc (history/memory_consolidation.md).

The goal is to have a clear **biological-style vocabulary** (STM / LTM / engrams, etc.) that sits on top of the actual Python implementation (`engramic_orchestrator.py`, `memory_store.py`, `tools.py`, `config.json` ).

---

## 1. High-level Architecture

At a high level, the system looks like this:

* **Engramic** (this whole app)
* **AI Agent** (LLM with tools)
* **Agent Code** (Python orchestrator)
* **Embedding Server** (vectorization / “neuroprinting”)
* **Engram Store** (SQLite + FAISS + logic in `MemoryStore`)
* **Tools** (shell, file ops, etc., callable by the Agent)
* **Memory processes**:

  * **Neuroprinting** (text → vector)
  * **Recall** (fetch STM + LTM for a query)
  * **Consolidation** (update LTM after a turn)

On each turn:

1. User sends a **user query**.
2. System **neuroprints** it.
3. System runs **Recall** (fetch Engrams + STM).
4. System builds the **Activation Field** (STM + WM) and calls the **AI Agent**.
5. Agent may call **tools** internally, then produces the **Agent Final Response**.
6. System **neuroprints** the full turn and runs **Consolidation** into the **Engram Store**.

---

## 2. Glossary: Concepts ↔ Code

### 2.1 Core System Pieces

#### Engramic

* **Definition**
  The entire system: AI Agent + memory + tools + orchestration.
* **Code equivalent**
  Not a single symbol; conceptually:

  * `engramic_orchestrator.py` (orchestrator + agent loop),
  * `memory_store.py` (memory implementation),
  * `memory_consolidation.md` (design).

---

#### AI Agent (Agent)

* **Definition**
  The LLM that:

  * reads the Activation Field (system instructions + memory + user query),
  * may call tools in a loop,
  * produces the Agent Final Response.
* **Code equivalent**

  * The chat model used via `Groq` client in `engramic_orchestrator.py` (e.g. `model = "openai/gpt-oss-120b"`).
  * Called in the main loop in `code_agent_orchestrator()`.

---

#### Agent Code

* **Definition**
  The Python orchestrator that:

  * reads user input,
  * calls the Embedding Server,
  * calls the Engram Store (Recall + Consolidation),
  * builds the Activation Field,
  * runs the AI Agent + tools,
  * prints the final answer.
* **Code equivalent**

  * `engramic_orchestrator.py`, especially `code_agent_orchestrator()` and the helper functions it uses.

---

#### Embedding Server

* **Definition**
  The service that converts text into numeric vectors.
  In our vocabulary: it performs **Neuroprinting**.
* **Code equivalent**

  * The embedding calls in `_embed_text(...)` in `engramic_orchestrator.py`.
  * Configured by `embedding_provider` / `embedding_model` (default via `embedding_choice: "bge"` → BAAI/bge-m3 with FlagEmbedding/HF; optional override to Gemini: `models/gemini-embedding-001`).

---

#### Engram Store

* **Definition**
  The memory subsystem that stores:

  * Raw Turns (full interactions),
  * Engrams (consolidated memories),
  * Beacons (topic anchors),
  * ANN indexes and card events.
* **Code equivalent**

  * `memory_store.py`, class `MemoryStore`.
  * SQLite tables:

    * `raw_turns`,
    * `consolidated_cards`,
    * `beacon_registry`,
    * `engram_events`.
  * Public methods:

    * `store_and_consolidate_turn(...)`,
    * `retrieve_context(...)`,
    * `format_context(...)`,
    * maintenance jobs (pruning, beacon discovery, etc.).

---

### 2.2 Memory Units & Structures

#### Neuroprinting

* **Definition**
  The act of turning text into an embedding vector via the Embedding Server.
  The result of neuroprinting is a **neuroprint**.
* **Code equivalent**

  * `_embed_text(client, embedding_model, text, provider=...)` in `engramic_orchestrator.py`.
  * The returned embedding (a list/array of floats).

---

#### Neuroprint

* **Definition**
  A numeric vector representation of a piece of text (user query, turn, summary, etc.).
  It’s the “signal” that lets us do similarity search and store LTM.
* **Code equivalent**

  * The arrays produced by `_embed_text(...)`.
  * Stored:

    * in `raw_turns` as `embedding`,
    * in `consolidated_cards` as `embedding`.

---

#### Engram

* **Definition**
  A compact long-term memory unit (our “memory trace”).
  An Engram represents a *cluster* of related turns as a single, stable memory:

  * short **title**,
  * short **summary** (what the Agent should remember),
  * **Canonical Entities** it’s about,
  * a **beacon list** (topic anchors),
  * a single **neuroprint** representing the Engram,
  * references to **Raw Turns** that contributed to it.
* **Code equivalent**

  * One row in `consolidated_cards` table.
  * Fields include:

    * `id`,
    * `title`,
    * `summary`,
    * `entities_json` (raw spans), `structured_entities_json` (typed canonical entities), `entities_canonical_json` (derived canonical_names),
    * `beacon_list_json`,
    * `embedding`,
    * `source_turn_ids_json`,
    * `access_count`,
    * timestamps.

---

#### Raw Turn

* **Definition**
  The exact record of a single user ↔ assistant exchange: what the user said, what the assistant replied, and the embedding (neuroprint) of that combined turn.
* **Role in the system**
  Raw Turns are the ground truth the system learns from. Consolidation always starts by writing a Raw Turn, and Recall surfaces the most recent Raw Turns as short-term memory so the agent sees the freshest context.
* **How it’s wired**
  Stored durably in the raw_turns table with spans, structured canonical entities, and timestamps. Every Engram points back to the Raw Turns that shaped it, and pruning/metrics use these rows as the source of truth.

---

#### Canonical Entity

* **Definition**
  A normalized label for a concept/person/place/org.
  It removes superficial differences (case, spelling, phrasing) so the system can say:

  * “These all refer to the same thing.”
* **Role in the system**
  Canonical entities give the memory system stable handles for “what this turn is about.” They drive merge/duplicate decisions, beacon activation, and keyword fallback retrieval. Typed variants (person/org/location/tech/etc., including self_* cues) allow the system to tell personal details from topics and map self_* directly to self beacons.
* **How it’s wired**
  Each turn’s spans are lifted into structured canonical entities (span, type, canonical_name, confidence). The derived canonical_names list is kept alongside the structured records on both Raw Turns and Engrams. If the structured helper fails, a deterministic lowercase fallback preserves compatibility.

---

#### Beacon

* **Definition**
  A **topic anchor** used to route retrieval:

  * examples: `__user_self__`, `__agent_self__`, `topic/geography`, `project/keralty_portal`.
  * They represent “neural assemblies” for themes.
* **Role in the system**
  Beacons are the routing layer for memory: they decide which Engrams get pulled for a query and where new Engrams live. Primordial anchors (`__user_self__`, `__agent_self__`, etc.) keep the agent and user always represented; self_* beacons come directly from canonicalization when the user talks about themselves; a single topical beacon per turn keeps the system focused instead of drifting across many weak tags.
* **How it’s wired**
  Every Engram stores a beacon list. During Recall, the active beacon set (self/primordial + any topical matches) filters Engrams before embeddings kick in. During Consolidation, self_* beacons are injected from canonical entities, primordial anchors are always added, an LLM helper may propose at most one existing beacon or one new label, and if it returns nothing the system falls back to attaching any canonical entities that already exist in the registry. Demotion/decay later trims weak topical beacons while leaving primordial ones untouched.

---

#### Beacon list

* **Definition**
  The **set/list of beacons attached to a single Engram**.
  Example:

  * `["__user_self__", "topic/geography", "country/colombia"]`.
* **Code equivalent**

  * JSON array stored in `consolidated_cards.beacon_list_json`.

During Recall, Engrams are chosen partly by **beacon list overlap** with the beacons activated by the current query.

---

### 2.3 Cognitive Layers

#### Short-Term Memory (STM)

* **Definition**
  The recent conversation slice that’s still “in mind”: the last **N** Raw Turns (N configurable).
* **Code equivalent**

  * `MemoryStore._recent_raw_turns(limit)` in `memory_store.py`.
  * Used inside `retrieve_context(...)`, returned as `recent_raw_turns`.
  * Rendered as the “recent conversation” section in `format_context(...)`.

---

#### Working Memory (WM)

* **Definition**
  The *active* context used to process the **current** user query:

  * **System content** (instructions, tools, behavior rules),
  * **Memory Context** (retrieved Engrams + STM, formatted),
  * **Current user query** text.
* **Code equivalent**

  * Built in `code_agent_orchestrator()`:

    * `system_content` (big system prompt string),
    * `memory_block` from `format_context(...)`,
    * `user_query`.
  * Combined into the first `messages` list sent to the LLM.

---

#### Long-Term Memory (LTM)

* **Definition**
  All stored memory that persists across turns:

  * Raw Turns,
  * Engrams,
  * Beacons, ANN index, engram_events.
    LTM is updated by Consolidation and queried by Recall.
* **Code equivalent**

  * Everything managed by `MemoryStore` and its SQLite DB:

    * `raw_turns`, `consolidated_cards`, `beacon_registry`, `engram_events`,
    * FAISS index for embeddings.

---

#### Experience

* **Definition**
  The complete paired turn: the **user query** and the **Agent Final Response**. This is the “real” interaction the system remembers and learns from; consolidation always operates on this full pair, not on partials or interim tool chatter.
* **Code equivalent**

  * In the orchestrator, each turn builds `combined_text = "User: ...\nAssistant: ..."` from the user query and final agent reply.
  * `store_and_consolidate_turn(...)` receives that combined text + neuroprint and stores a Raw Turn (ground truth), then may merge or create an Engram using that same experience.
  * No tool-call intermediates or draft assistant messages are part of an Experience; only the user’s message and the final assistant message form the durable record.

---

#### Activation Field

* **Definition**
  For a given model call, the Activation Field is:

  > The sum of **STM + WM** that comes from external sources (user, system, memory).

  Concretely, for the **first call in a turn**, it consists of:

  * **System message**: `system_content` (rules + tools + memory policy).
  * **User message**:

    * Memory Context (Engrams + STM) and
    * the current user query.

  Internal tool outputs and intermediate assistant messages are **not** part of the Activation Field in our vocabulary; they live in the **inner agentic workspace**, not in memory.
* **Code equivalent**

  * The initial `messages = [{"role": "system", ...}, {"role": "user", ...}]` built in `code_agent_orchestrator()`.

---

### 2.4 Processes

#### Recall

* **Definition**
  Given a user_query and its neuroprint, retrieve:

  * relevant Engrams from LTM, and
  * the most recent Raw Turns for STM.

  Return a **Memory Context** that the Agent can use.
* **Code equivalent**

* **Narrative**

  1) Neuroprint the query, then canonicalize its entities with the LLM helper (same structured canonicalization used in consolidation).  
  2) Activate beacons for this query: score registry beacons that match those canonical entities, take the top-2 by (strength, card_count), then append primordial anchors `__user_self__`, `__agent_self__`.  
  3) Pull Engrams whose beacon lists overlap that activated set; rank by access_count and recency; keep `card_k`. If empty, try keyword on canonical entities; if still empty, fall back to ANN similarity on the query neuroprint.  
  4) Fetch the most recent Raw Turns for STM.  
  5) Format the Memory Context (Engrams + STM) and place it into Working Memory with system instructions and the current query.

---

#### Consolidation

* **Definition**
  After each non-trivial turn, update LTM by:

  * inserting a Raw Turn,
  * possibly creating a new Engram, or
  * merging into / adjusting an existing Engram (summary, beacons, embedding, access_count).
* **Code equivalent**

  * **Narrative**

  1) After the Agent responds, build the combined turn text (User + Assistant) and neuroprint it; store that Raw Turn with its embedding and extracted entities.  
  2) Canonicalize entities for this turn via the LLM helper (typed spans with `canonical_name`; deterministic fallback if needed).  
  3) Inject self beacons from any `self_*` canonical entities.  
  4) Ask the LLM beacon helper to attach at most one existing registry beacon or at most one new beacon; if it returns nothing, attach any canonical entities that already exist in the registry; always add primordial anchors.  
  5) Find candidate Engrams by canonical-entity overlap and similarity: fingerprint/duplicate/merge paths update the Engram’s summary, embedding (neuroprint), canonical entities, beacon_list, and source turn ids, bumping access_count.  
  6) If nothing matches, create a new Engram with title/summary/neuroprint, canonical entities, beacon_list, and source turn ids; seed access_count and update indexes/beacon counts.

---

### 2.6 Beacon Lifecycle

* **Demotion**
  Beacons that fall below strength 0.5 and have not been touched for 180 days are demoted: their strength is lowered, card_count reset, and they are removed from cards’ beacon lists (logged as `beacon_demote`). Primordial beacons are never demoted.
* **Immortalization**
  Beacons with strength >= 0.98 and card_count >= 100 are treated as immortal and are not decayed or demoted.
* **Code equivalent**

  * `MemoryStore.prune_stale_beacons(...)` in `memory_store.py` handles decay, demotion, and immortalization.
  * Demotion removes the beacon from card beacon lists and logs an engram_event with before/after beacons; immortals are skipped entirely.

---

#### Agent Final Response

* **Definition**
  The final assistant message the AI Agent produces for a given user query (after any internal tool calls), which is shown to the user.
* **Code equivalent**

  * In `code_agent_orchestrator()`, the final `response.choices[0].message.content` after the tool loop, printed as `Assistant: ...`.

---

### 2.5 Tools & Inner Workspace

#### Tools

* **Definition**
  External capabilities the Agent can call (e.g., shell, file operations, system info).
* **Code equivalent**

  * Tool definitions and `available_functions` in `engramic_orchestrator.py`.
  * Passed as `tools=...` into the chat completion call.

---

#### Inner Agentic Workspace

* **Definition**
  All intermediate, internal messages that exist only inside the current turn:

  * tool calls from the Agent,
  * tool outputs,
  * intermediate assistant messages in the tool loop.

  These are **not** stored in memory and **not** part of the Activation Field, in our vocabulary.
* **Code equivalent**

  * The iterative tool loop in `code_agent_orchestrator()`, where `messages` grows with `assistant` + `tool` messages during a single turn.

---

## 3. Example: “ls” then “ls” again

Here is a complete conceptual example using this vocabulary.

### 3.1 First `user_query: "ls"`

1. **User query**
   User types: `ls`.

2. **Neuroprinting**
   The query is sent to the Embedding Server → we get the **neuroprint of the query**.

3. **Recall**
   The Engram Store:

   * has no previous Engram about `ls` (first time),
   * STM may be empty or contain unrelated Raw Turns.
     Result: Memory Context has no special Engram about `ls`.

4. **Activation Field (first call)**
   Contains:

   * system rules (WM),
   * an empty/irrelevant memory section (no `ls` Engram yet),
   * the query `ls` (WM),
   * STM (if any).

5. **AI Agent behavior**
   The Agent sees `ls` in this Activation Field and decides:

   * “The user probably wants a shell directory listing.”
   * It calls the shell tool, gets the directory contents.
   * It returns an **Agent Final Response** with the listing (or an explanation).

6. **Consolidation (first `ls`)**
   After responding, the system:

   * Builds combined text: `User: ls\nAgent: <listing or explanation>`.
   * Neuroprints this combined text (neuroprint of the turn).
   * Calls `store_and_consolidate_turn(...)`:

     * Inserts a **Raw Turn** with user+agent text, its neuroprint, entities, etc.
     * Looks for matching Engrams:

       * Finds none (first `ls`).
      * Creates a **new Engram**:

        * summary (e.g., “User ran ls and saw the current directory contents”),
        * title (e.g., “Shell: ls in workspace”),
        * canonical entities (e.g. `command/ls`),
        * beacon list (e.g. `["__user_self__", "__agent_self__"]` seeded; self beacons are added if canonicalization emits self_* types; topical beacons come from the helper),
        * neuroprint for the summary,
        * `source_turn_ids` list containing this Raw Turn’s ID,
        * `access_count = 1`.

---

### 3.2 Second `user_query: "ls"` (later)

7. **User query**
   User again types: `ls`.

8. **Neuroprinting**
   This second `ls` is neuroprinted again → new **neuroprint of the query**.

9. **Recall**
   The Engram Store:

   * extracts canonical entities (`command/ls`, etc.),
   * maps those canonical entities to any matching beacons in the registry, takes the top-2 by (strength, card_count), and ensures primordial anchors `__user_self__`, `__agent_self__` are present as fallback,
   * pulls cards whose beacon lists overlap that set; if none match, it falls back to keyword overlap on canonical entities, then ANN similarity on the query neuroprint,
   * includes the existing **ls Engram** in the Memory Context (via beacon overlap in this case),
   * also includes the previous `ls` Raw Turn in STM (if it’s among the last N Raw Turns).

10. **Activation Field (second call)**
    Now the Activation Field includes:

    * system rules (WM),
    * the Engram about `ls` (LTM),
    * the previous `ls` Raw Turn in STM (if still recent),
    * the new user query `ls`.

11. **AI Agent behavior**
    The Agent again interprets “ls” as a request to list the directory:

    * calls the shell tool,
    * returns a fresh directory listing as the **Agent Final Response**.

12. **Consolidation (second `ls`)**
    After responding:

    * System builds combined text for this second turn and neuroprints it.
    * `store_and_consolidate_turn(...)`:

      * inserts a new **Raw Turn** (second `ls`).
      * compares this new turn to existing Engrams:

        * same canonical entities (`command/ls`),
        * very similar content,
        * fingerprint may match (only if the normalized combined text is effectively identical).
      * The system decides **this is the same Engram**:

        * adds the new Raw Turn’s ID to the existing Engram’s `source_turn_ids`,
        * increments the Engram’s `access_count` (its “strength”),
        * typically keeps the same summary and neuroprint (no need to change).

**Result:**

* After the first `ls`, there is **one Engram** representing “user runs `ls` here”.
* After the second `ls`, we still have **one Engram**, but:

  * it has **more Raw Turns attached** (two observations of the same behavior),
  * it has **higher access_count**, which means:

    * it is more likely to be recalled in future,
    * it is less likely to be pruned,
    * its beacons become slightly more influential in the memory system.

---

You can now use this README as a shared conceptual layer for:

* discussing changes to `MemoryStore`,
* naming configs and logging,
* writing docs and diagrams that match how the system *actually* behaves.
