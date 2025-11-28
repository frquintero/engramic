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
    * `card_events`.
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
    * `entities` / `entities_canonical`,
    * `beacon_list_json`,
    * `embedding`,
    * `source_turn_ids_json`,
    * `access_count`,
    * timestamps.

---

#### Raw Turn

* **Definition**
  One stored interaction:

  * user text,
  * agent text,
  * combined text,
  * its neuroprint,
  * entities, timestamps, conversation id, etc.
    Raw Turns feed STM and are the “ground truth” behind Engrams.
* **Code equivalent**

  * One row in `raw_turns` table.
  * Inserted in `store_and_consolidate_turn(...)`.

---

#### Canonical Entity

* **Definition**
  A normalized label for a concept/person/place/org.
  It removes superficial differences (case, spelling, phrasing) so the system can say:

  * “These all refer to the same thing.”
* **Code equivalent**

  * Extracted in `MemoryStore` (e.g. `_extract_entities`, `_canonicalize_entity`).
  * Stored as `entities` and `entities_canonical` in:

    * `raw_turns`,
    * `consolidated_cards`.

---

#### Beacon

* **Definition**
  A **topic anchor** used to route retrieval:

  * examples: `__user_self__`, `__agent_self__`, `topic/geography`, `project/keralty_portal`.
  * They represent “neural assemblies” for themes.
  Primordial beacons are seeded; each card is always anchored to `__user_self__` and `__agent_self__`. Additional topical beacons come from the LLM helper (one existing registry id or one new label), with a simple heuristic fallback only if the helper is unavailable.
  Mutability: primordial beacons are never pruned; discovered beacons are mutable (strength/card_count update, can be pruned if weak/stale, and added/removed from cards).
* **Code equivalent**

  * Strings stored in:

    * `consolidated_cards.beacon_list_json` (per-Engram),
    * `beacon_registry` (beacon stats: strength, counts).
  * Assigned via an LLM helper during consolidation: the helper sees the summary, canonical entities, current beacons (on merge), and the entire beacon registry, and returns exactly one beacon—either a single existing one or a single new label (mutually exclusive, never more than one). The new beacon is inserted into the registry immediately and attached to the engram; self-beacons (two special primordials) are added separately.

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
  * Beacons, ANN index, card_events.
    LTM is updated by Consolidation and queried by Recall.
* **Code equivalent**

  * Everything managed by `MemoryStore` and its SQLite DB:

    * `raw_turns`, `consolidated_cards`, `beacon_registry`, `card_events`,
    * FAISS index for embeddings.

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

  * `MemoryStore.retrieve_context(...)` in `memory_store.py`:

    * entity extraction + canonicalization,
    * beacon selection,
    * Engram selection (beacon-first: global top-2 + primordial, then keyword on canonical entities; ANN similarity only as a fallback),
    * `_recent_raw_turns(...)` for STM.
  * `MemoryStore.format_context(cards, recent_raw_turns, ...)` to produce the text block.

---

#### Consolidation

* **Definition**
  After each non-trivial turn, update LTM by:

  * inserting a Raw Turn,
  * possibly creating a new Engram, or
  * merging into / adjusting an existing Engram (summary, beacons, embedding, access_count).
* **Code equivalent**

  * `MemoryStore.store_and_consolidate_turn(...)`:

    * inserts into `raw_turns`,
    * chooses candidate Engrams (entities + similarity),
    * handles fingerprint/duplicate/merge/new-card logic,
    * logs events in `card_events`,
    * updates FAISS index and beacons.

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
        * beacon list (e.g. `["__user_self__", "__agent_self__"]` seeded; additional beacons only if previously registered and matching entities),
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
   * selects beacons for the query (global top-2 by strength/card_count plus primordial anchors `__user_self__`, `__agent_self__`),
   * finds the existing **ls Engram** whose beacon list matches,
   * includes it in the Memory Context,
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
