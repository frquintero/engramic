import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np


DEFAULT_DB_PATH = Path("persistent_mem/memory.db")
DEFAULT_DUPLICATE_THRESHOLD = 0.92
DEFAULT_MERGE_THRESHOLD = 0.68
DEFAULT_VECTOR_THRESHOLD = 0.38
DEFAULT_CARD_K = 4
DEFAULT_RECENT_TURN_LIMIT = 5
DEFAULT_CANON_CONFIDENCE_FLOOR = 0.35

PRIMORDIAL_BEACONS = [
    "__user_self__",
    "__agent_self__",
    "__environment__",
    "__people__",
    "__projects__",
    "__knowledge__",
    "__interests__",
]

SELF_BEACONS = [
    "__self_identity__",
    "__self_location__",
    "__self_work__",
    "__self_health__",
    "__self_relationships__",
    "__self_preferences__",
    "__self_projects__",
]

SELF_BEACON_TYPES = {
    "self_identity": "__self_identity__",
    "self_location": "__self_location__",
    "self_work": "__self_work__",
    "self_health": "__self_health__",
    "self_relationships": "__self_relationships__",
    "self_preferences": "__self_preferences__",
    "self_projects": "__self_projects__",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


_spacy_nlp = None


def _get_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy  # type: ignore

        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = spacy.blank("en")
    except Exception:
        _spacy_nlp = None
    return _spacy_nlp


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "for",
    "at",
    "with",
    "as",
    "by",
    "is",
    "it",
    "this",
    "that",
    "these",
    "those",
    "be",
    "are",
    "user",
    "assistant",
    "here",
}


def _extract_entities(text: str) -> List[str]:
    entities: List[str] = []
    nlp = _get_spacy_model()
    if nlp:
        try:
            doc = nlp(text)
            for ent in getattr(doc, "ents", []):
                cleaned = ent.text.strip()
                if cleaned:
                    entities.append(cleaned)
            if not entities:
                for chunk in getattr(doc, "noun_chunks", []):
                    cleaned = chunk.text.strip()
                    if cleaned:
                        entities.append(cleaned)
        except Exception:
            entities = []

    if not entities:
        raw_tokens = re.findall(r"[A-Za-z0-9_.\\/-]+", text)
        for tok in raw_tokens:
            cleaned = tok.strip().strip(".,;:!?()[]{}'\"")
            lowered = cleaned.lower()
            if not cleaned:
                continue
            if lowered in _STOPWORDS:
                continue
            if len(cleaned) < 3 and "." not in cleaned:
                continue
            entities.append(cleaned)

    seen = set()
    normalized: List[str] = []
    for e in entities:
        norm = e.strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(norm)
    return normalized


def _safe_load_json_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _canonicalize_entity(value: str) -> str:
    v = value.strip()
    v = v.replace("\\", "/")
    v = v.strip(".")
    v = re.sub(r"\\s+", " ", v)
    v = re.sub(r"^\\./", "", v)
    return v.lower()


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm <= 0:
        return list(arr.astype(float))
    return list((arr / norm).astype(float))


def _blend_embeddings(base: Sequence[float], new: Sequence[float], *, new_weight: float = 0.4) -> List[float]:
    if not base or not new or len(base) != len(new):
        return list(new)
    a = np.asarray(base, dtype=np.float32)
    b = np.asarray(new, dtype=np.float32)
    blended = (1.0 - new_weight) * a + new_weight * b
    return _l2_normalize(blended)


def _normalize_text_for_fingerprint(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _compute_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize_text_for_fingerprint(text).encode("utf-8")).hexdigest()


def _hash_beacon(entities: List[str]) -> str:
    payload = "::".join(sorted(entities))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _self_beacons_from_structured(structured_entities: List[Dict]) -> List[str]:
    beacons: List[str] = []
    for item in structured_entities or []:
        if not isinstance(item, dict):
            continue
        tval = item.get("type")
        if not isinstance(tval, str):
            continue
        beacon_id = SELF_BEACON_TYPES.get(tval.strip())
        if beacon_id and beacon_id not in beacons:
            beacons.append(beacon_id)
    return beacons


class AnnIndex:
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("ANN index dimension must be positive")
        self.dim = dim
        self._vectors: Dict[int, List[float]] = {}
        self._index = faiss.IndexFlatIP(dim)
        self._id_order: List[int] = []

    def upsert(self, item_id: int, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError("Vector dimension mismatch for ANN index")
        self._vectors[item_id] = _l2_normalize(vector)
        self._rebuild()

    def remove(self, item_id: int) -> None:
        if item_id in self._vectors:
            del self._vectors[item_id]
            self._rebuild()

    def _rebuild(self) -> None:
        self._index.reset()
        if not self._vectors:
            self._id_order = []
            return
        ids = list(self._vectors.keys())
        mat = np.asarray([self._vectors[i] for i in ids], dtype=np.float32)
        self._index.add(mat)
        self._id_order = ids

    def search(self, vector: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        if not self._vectors:
            return []
        if len(vector) != self.dim:
            return []
        q = np.asarray([_l2_normalize(vector)], dtype=np.float32)
        sims, idxs = self._index.search(q, top_k)
        results: List[Tuple[int, float]] = []
        for pos, sim in zip(idxs[0], sims[0]):
            if pos < 0 or pos >= len(self._id_order):
                continue
            results.append((self._id_order[pos], float(sim)))
        return results


class MemoryStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.embedding_dim: Optional[int] = None
        self.ann_index: Optional[AnnIndex] = None
        self._init_schema()
        self._ensure_primordial_beacons()
        self._load_ann_index()

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if not embedding:
            raise ValueError("Embedding vector is empty.")
        expected = self.embedding_dim or len(embedding)
        if len(embedding) != expected:
            raise ValueError(f"Embedding dimension mismatch: expected {expected}, got {len(embedding)}")
        if self.ann_index and self.ann_index.dim != expected:
            raise ValueError(f"ANN index dimension mismatch: index dim {self.ann_index.dim}, expected {expected}")

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_content TEXT NOT NULL,
                assistant_content TEXT,
                combined_text TEXT NOT NULL,
                entities_json TEXT,
                structured_entities_json TEXT,
                entities_canonical_json TEXT,
                combined_embedding_json TEXT NOT NULL,
                observation_fingerprint TEXT,
                timestamp_ms INTEGER NOT NULL,
                conversation_id TEXT
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_turns_ts ON raw_turns(timestamp_ms)")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consolidated_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                summary TEXT NOT NULL,
                entities_json TEXT,
                structured_entities_json TEXT,
                entities_canonical_json TEXT,
                embedding_json TEXT NOT NULL,
                observation_fingerprint TEXT,
                created_ms INTEGER NOT NULL,
                updated_ms INTEGER NOT NULL,
                source_turn_ids_json TEXT,
                access_count INTEGER DEFAULT 0,
                merge_version INTEGER DEFAULT 1,
                beacon_list_json TEXT
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_updated ON consolidated_cards(updated_ms)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_access ON consolidated_cards(access_count, updated_ms)")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS beacon_registry (
                beacon_id TEXT PRIMARY KEY,
                strength REAL NOT NULL,
                card_count INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                parent_beacon TEXT,
                last_touched_ms INTEGER NOT NULL
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engram_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id INTEGER,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                before_json TEXT,
                after_json TEXT,
                source_turn_id INTEGER,
                timestamp_ms INTEGER NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_engram_events_card ON engram_events(card_id)")
        self.conn.commit()
        self._maybe_migrate_from_legacy()
        self._ensure_new_columns()


    def _maybe_migrate_from_legacy(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidated_memories'"
        )
        if not cur.fetchall():
            cur_events = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='card_events'"
            )
            if cur_events.fetchall():
                rows = self.conn.execute("SELECT * FROM card_events").fetchall()
                for row in rows:
                    (
                        _id,
                        card_id,
                        event_type,
                        payload_json,
                        before_json,
                        after_json,
                        source_turn_id,
                        timestamp_ms,
                    ) = row
                    self.conn.execute(
                        """
                        INSERT INTO engram_events (card_id, event_type, payload_json, before_json, after_json, source_turn_id, timestamp_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_id,
                            event_type,
                            payload_json,
                            before_json,
                            after_json,
                            source_turn_id,
                            timestamp_ms,
                        ),
                    )
                self.conn.commit()
            return
        rows = self.conn.execute(
            """
            SELECT id, title, summary, entities_json, entities_canonical_json, embedding_json,
                   observation_fingerprint, created_ms, updated_ms, source_turn_ids_json
            FROM consolidated_memories
            """
        ).fetchall()
        for row in rows:
            (
                _cid,
                title,
                summary,
                entities_json,
                entities_canonical_json,
                embedding_json,
                observation_fingerprint,
                created_ms,
                updated_ms,
                source_turn_ids_json,
            ) = row
            self.conn.execute(
                """
                INSERT INTO consolidated_cards (
                    title, summary, entities_json, entities_canonical_json, embedding_json,
                    observation_fingerprint, created_ms, updated_ms, source_turn_ids_json,
                    access_count, merge_version, beacon_list_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    summary,
                    entities_json,
                    entities_canonical_json,
                    embedding_json,
                    observation_fingerprint,
                    created_ms,
                    updated_ms,
                    source_turn_ids_json,
                    0,
                    1,
                    json.dumps([]),
                ),
            )
        self.conn.commit()
        cur_events = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='card_events'"
        )
        if cur_events.fetchall():
            rows_events = self.conn.execute("SELECT * FROM card_events").fetchall()
            for row_ev in rows_events:
                (
                    _id,
                    card_id,
                    event_type,
                    payload_json,
                    before_json,
                    after_json,
                    source_turn_id,
                    timestamp_ms,
                ) = row_ev
                self.conn.execute(
                    """
                    INSERT INTO engram_events (card_id, event_type, payload_json, before_json, after_json, source_turn_id, timestamp_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        event_type,
                        payload_json,
                        before_json,
                        after_json,
                        source_turn_id,
                        timestamp_ms,
                    ),
                )
            self.conn.commit()

    def _ensure_new_columns(self) -> None:
        def _has_column(table: str, column: str) -> bool:
            cur = self.conn.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]
            return column in cols

        if not _has_column("raw_turns", "structured_entities_json"):
            self.conn.execute("ALTER TABLE raw_turns ADD COLUMN structured_entities_json TEXT")
        if not _has_column("consolidated_cards", "structured_entities_json"):
            self.conn.execute("ALTER TABLE consolidated_cards ADD COLUMN structured_entities_json TEXT")
        self.conn.commit()

    def _ensure_primordial_beacons(self) -> None:
        now = _now_ms()
        for beacon in PRIMORDIAL_BEACONS:
            cur = self.conn.execute("SELECT beacon_id FROM beacon_registry WHERE beacon_id = ?", (beacon,))
            if cur.fetchone():
                continue
            self.conn.execute(
                """
                INSERT INTO beacon_registry (beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (beacon, 0.9, 0, 0, None, now),
            )
        self.conn.commit()

    def _load_ann_index(self) -> None:
        cards = self._load_cards()
        if not cards:
            return
        dim = len(cards[0]["embedding"])
        if dim == 0:
            return
        self.embedding_dim = dim
        self.ann_index = AnnIndex(dim)
        for card in cards:
            emb = card.get("embedding") or []
            if len(emb) == dim:
                self.ann_index.upsert(card["id"], emb)

    def _persist_event(
        self,
        card_id: Optional[int],
        event_type: str,
        *,
        payload: Optional[Dict] = None,
        before: Optional[Dict] = None,
        after: Optional[Dict] = None,
        source_turn_id: Optional[int] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO engram_events (card_id, event_type, payload_json, before_json, after_json, source_turn_id, timestamp_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card_id,
                event_type,
                json.dumps(payload or {}),
                json.dumps(before or {}),
                json.dumps(after or {}),
                source_turn_id,
                _now_ms(),
            ),
        )
        self.conn.commit()

    def _insert_raw_turn(
        self,
        *,
        user_content: str,
        assistant_content: Optional[str],
        combined_text: str,
        combined_embedding: Sequence[float],
        entities: List[str],
        structured_entities: List[Dict],
        entities_canonical: List[str],
        observation_fingerprint: Optional[str],
        timestamp_ms: int,
        conversation_id: Optional[str],
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO raw_turns (
                user_content,
                assistant_content,
                combined_text,
                entities_json,
                structured_entities_json,
                entities_canonical_json,
                combined_embedding_json,
                observation_fingerprint,
                timestamp_ms,
                conversation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_content,
                assistant_content or "",
                combined_text,
                json.dumps(entities),
                json.dumps(structured_entities),
                json.dumps(entities_canonical),
                json.dumps(list(combined_embedding)),
                observation_fingerprint,
                timestamp_ms,
                conversation_id,
            ),
        )
        raw_turn_id = cur.lastrowid
        self.conn.commit()
        return raw_turn_id

    def _load_cards(self) -> List[Dict]:
        rows = self.conn.execute(
            """
            SELECT
                id,
                title,
                summary,
                entities_json,
                structured_entities_json,
                entities_canonical_json,
                embedding_json,
                observation_fingerprint,
                created_ms,
                updated_ms,
                source_turn_ids_json,
                access_count,
                merge_version,
                beacon_list_json
            FROM consolidated_cards
            """
        ).fetchall()
        cards: List[Dict] = []
        for row in rows:
            (
                card_id,
                title,
                summary,
                entities_json,
                structured_entities_json,
                entities_canonical_json,
                embedding_json,
                observation_fingerprint,
                created_ms,
                updated_ms,
                source_turn_ids_json,
                access_count,
                merge_version,
                beacon_list_json,
            ) = row
            cards.append(
                {
                    "id": card_id,
                    "title": title or "Memory",
                    "summary": summary or "",
                    "entities": _safe_load_json_list(entities_json),
                    "structured_entities": json.loads(structured_entities_json) if structured_entities_json else [],
                    "entities_canonical": _safe_load_json_list(entities_canonical_json),
                    "embedding": json.loads(embedding_json) if embedding_json else [],
                    "observation_fingerprint": observation_fingerprint,
                    "created_ms": created_ms,
                    "updated_ms": updated_ms,
                    "source_turn_ids": _safe_load_json_list(source_turn_ids_json),
                    "access_count": access_count or 0,
                    "merge_version": merge_version or 1,
                    "beacon_list": _safe_load_json_list(beacon_list_json),
                }
            )
        return cards

    def _update_card(
        self,
        card_id: int,
        *,
        title: Optional[str],
        summary: str,
        entities: List[str],
        structured_entities: Optional[List[Dict]],
        entities_canonical: List[str],
        embedding: Sequence[float],
        updated_ms: int,
        source_turn_ids: List[int],
        observation_fingerprint: Optional[str],
        access_count: Optional[int] = None,
        merge_version: Optional[int] = None,
        beacon_list: Optional[List[str]] = None,
    ) -> None:
        self._validate_embedding(embedding)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        self.conn.execute(
            """
            UPDATE consolidated_cards
            SET title = ?, summary = ?, entities_json = ?, structured_entities_json = ?, entities_canonical_json = ?, embedding_json = ?,
                observation_fingerprint = ?, updated_ms = ?, source_turn_ids_json = ?, access_count = ?, merge_version = ?, beacon_list_json = ?
            WHERE id = ?
            """,
            (
                title,
                summary,
                json.dumps(entities),
                json.dumps(structured_entities or []),
                json.dumps(entities_canonical),
                json.dumps(list(embedding)),
                observation_fingerprint,
                updated_ms,
                json.dumps(source_turn_ids),
                access_count,
                merge_version,
                json.dumps(beacon_list or []),
                card_id,
            ),
        )
        self.conn.commit()
        if self.ann_index:
            self.ann_index.upsert(card_id, embedding)

    def _insert_card(
        self,
        *,
        title: str,
        summary: str,
        entities: List[str],
        structured_entities: List[Dict],
        entities_canonical: List[str],
        embedding: Sequence[float],
        observation_fingerprint: Optional[str],
        created_ms: int,
        source_turn_ids: List[int],
        beacon_list: List[str],
    ) -> int:
        self._validate_embedding(embedding)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO consolidated_cards (
                title,
                summary,
                entities_json,
                structured_entities_json,
                entities_canonical_json,
                embedding_json,
                observation_fingerprint,
                created_ms,
                updated_ms,
                source_turn_ids_json,
                access_count,
                merge_version,
                beacon_list_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                summary,
                json.dumps(entities),
                json.dumps(structured_entities),
                json.dumps(entities_canonical),
                json.dumps(list(embedding)),
                observation_fingerprint,
                created_ms,
                created_ms,
                json.dumps(source_turn_ids),
                1,
                1,
                json.dumps(beacon_list),
            ),
        )
        card_id = cur.lastrowid
        self.conn.commit()
        if self.embedding_dim and len(embedding) == self.embedding_dim:
            if self.ann_index is None:
                self.ann_index = AnnIndex(self.embedding_dim)
            self.ann_index.upsert(card_id, embedding)
        return card_id

    def _beacon_registry(self) -> Dict[str, Dict]:
        rows = self.conn.execute(
            "SELECT beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms FROM beacon_registry"
        ).fetchall()
        registry: Dict[str, Dict] = {}
        for row in rows:
            beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms = row
            registry[beacon_id] = {
                "beacon_id": beacon_id,
                "strength": strength,
                "card_count": card_count,
                "generation": generation,
                "parent_beacon": parent_beacon,
                "last_touched_ms": last_touched_ms,
            }
        return registry

    def _update_beacon_counts(self, beacon_ids: List[str]) -> None:
        now = _now_ms()
        for beacon_id in beacon_ids:
            cur = self.conn.execute(
                "SELECT strength, card_count FROM beacon_registry WHERE beacon_id = ?",
                (beacon_id,),
            )
            row = cur.fetchone()
            if row:
                strength, card_count = row
                new_strength = min(1.0, strength + 0.01)
                self.conn.execute(
                    "UPDATE beacon_registry SET strength = ?, card_count = ?, last_touched_ms = ? WHERE beacon_id = ?",
                    (new_strength, card_count + 1, now, beacon_id),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO beacon_registry (beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (beacon_id, 0.6, 1, 1, None, now),
                )
        self.conn.commit()

    def _assign_beacons(
        self,
        entities_canonical: List[str],
        combined_text: str,
        *,
        summary_text: Optional[str] = None,
        beacon_assignment_fn: Optional[
            Callable[[str, List[str], Dict[str, Dict], List[str]], Dict[str, List[str]]]
        ] = None,
        current_beacons: Optional[List[str]] = None,
        forced_beacons: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        registry = self._beacon_registry()
        beacons: List[str] = []
        proposed_new: List[str] = []

        for b in forced_beacons or []:
            if not isinstance(b, str):
                continue
            bid = b.strip()
            if not bid or bid in beacons:
                continue
            beacons.append(bid)

        # Primary path: LLM-based assignment; no heuristic fallback if provided.
        if beacon_assignment_fn and summary_text:
            result = beacon_assignment_fn(
                summary_text,
                list(entities_canonical),
                dict(registry),
                list(current_beacons or []),
            )
            if not isinstance(result, dict):
                result = {}
            raw_existing = result.get("existing") if isinstance(result, dict) else []
            raw_new = result.get("proposed_new") if isinstance(result, dict) else []

            existing: List[str] = []
            if isinstance(raw_existing, list):
                for b in raw_existing:
                    if not isinstance(b, str):
                        continue
                    bid = b.strip()
                    if not bid or bid not in registry:
                        continue
                    if bid in existing:
                        continue
                    existing.append(bid)
                    if len(existing) >= 1:
                        break

            if isinstance(raw_new, str):
                raw_new = [raw_new]
            proposed: List[str] = []
            if isinstance(raw_new, list):
                for cand in raw_new:
                    if not isinstance(cand, str):
                        continue
                    cid = cand.strip()
                    if not cid:
                        continue
                    if cid in proposed:
                        continue
                    proposed.append(cid)
                    if len(proposed) >= 1:
                        break

            if existing and proposed:
                proposed = []

            if existing:
                beacons.extend(existing[:1])
            elif proposed:
                new_id = proposed[0]
                now = _now_ms()
                self.conn.execute(
                    """
                    INSERT INTO beacon_registry (beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(beacon_id) DO NOTHING
                    """,
                    (new_id, 0.6, 0, 1, None, now),
                )
                self.conn.commit()
                beacons.append(new_id)
                proposed_new.append(new_id)

        if not beacons and not proposed_new:
            for e in entities_canonical:
                if e in registry and e not in beacons:
                    beacons.append(e)

        # Always anchor with primordial context
        for p in PRIMORDIAL_BEACONS[:2]:
            if p not in beacons:
                beacons.append(p)

        return beacons, proposed_new

    def rebuild_ann_from_store(self) -> None:
        cards = self._load_cards()
        if not cards:
            self.embedding_dim = None
            self.ann_index = None
            return
        dim = len(cards[0]["embedding"])
        if dim <= 0:
            self.embedding_dim = None
            self.ann_index = None
            return
        self.embedding_dim = dim
        self.ann_index = AnnIndex(dim)
        for card in cards:
            if len(card.get("embedding") or []) == dim:
                self.ann_index.upsert(card["id"], card["embedding"])

    def beacon_discovery_job(
        self,
        *,
        min_cooccur_cards: int = 8,
        min_jaccard: float = 0.75,
    ) -> List[str]:
        cards = self._load_cards()
        if not cards:
            return []
        entity_to_cards: Dict[str, set] = {}
        for card in cards:
            canon = set(card.get("entities_canonical") or [])
            for ent in canon:
                entity_to_cards.setdefault(ent, set()).add(card["id"])
        promoted: List[str] = []
        for ent, card_set in entity_to_cards.items():
            if len(card_set) < min_cooccur_cards:
                continue
            neighbors: List[Tuple[str, float]] = []
            for other, other_set in entity_to_cards.items():
                if other == ent:
                    continue
                jac = _jaccard(card_set, other_set)
                if jac >= min_jaccard:
                    neighbors.append((other, jac))
            cluster_entities = [ent] + [n for n, _ in neighbors]
            if len(cluster_entities) < 4:
                continue
            cluster_cards = card_set
            for neigh, _jac in neighbors:
                cluster_cards = cluster_cards & entity_to_cards.get(neigh, set())
            if len(cluster_cards) < min_cooccur_cards:
                continue
            avg_jaccard = sum(j for _, j in neighbors) / float(len(neighbors)) if neighbors else min_jaccard
            strength = min(1.0, max(min_jaccard, avg_jaccard))
            beacon_id = f"__beacon_{_hash_beacon(cluster_entities)}__"
            now = _now_ms()
            self.conn.execute(
                """
                INSERT INTO beacon_registry (beacon_id, strength, card_count, generation, parent_beacon, last_touched_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(beacon_id) DO UPDATE SET strength=excluded.strength, card_count=excluded.card_count, last_touched_ms=excluded.last_touched_ms
                """,
                (beacon_id, strength, len(cluster_cards), 1, None, now),
            )
            promoted.append(beacon_id)
            for card in cards:
                if card["id"] in cluster_cards:
                    beacon_list = list({b: b for b in (card.get("beacon_list") or []) + [beacon_id]}.values())
                    self._update_card(
                        card["id"],
                        title=card.get("title"),
                        summary=card.get("summary") or "",
                        entities=card.get("entities") or [],
                        structured_entities=card.get("structured_entities") or [],
                        entities_canonical=card.get("entities_canonical") or [],
                        embedding=card.get("embedding") or [],
                        updated_ms=card.get("updated_ms") or now,
                        source_turn_ids=card.get("source_turn_ids") or [],
                        observation_fingerprint=card.get("observation_fingerprint"),
                        access_count=card.get("access_count"),
                        merge_version=card.get("merge_version"),
                        beacon_list=beacon_list,
                    )
                    self._persist_event(
                        card["id"],
                        "beacon_assignment",
                        payload={"beacon_id": beacon_id, "strength": strength},
                        before={"beacons": card.get("beacon_list")},
                        after={"beacons": beacon_list},
                    )
        self.conn.commit()
        return promoted

    def prune_stale_beacons(
        self,
        *,
        max_age_days: int = 180,
        strength_floor: float = 0.5,
        half_life_days: int = 90,
        immortal_strength: float = 0.98,
        immortal_card_count: int = 100,
    ) -> List[str]:
        cutoff_ms = _now_ms() - max_age_days * 24 * 3600 * 1000
        demoted: List[str] = []
        rows = self.conn.execute(
            "SELECT beacon_id, strength, last_touched_ms, card_count FROM beacon_registry"
        ).fetchall()
        for beacon_id, strength, last_touched_ms, card_count in rows:
            if beacon_id in PRIMORDIAL_BEACONS:
                continue
            if strength >= immortal_strength and card_count >= immortal_card_count:
                # Immortalized: preserve strength/card_count; no decay or demotion.
                continue
            age_days = max(0.0, (_now_ms() - last_touched_ms) / (24 * 3600 * 1000))
            if half_life_days > 0:
                decay_factor = 0.5 ** (age_days / half_life_days)
                strength = max(0.0, strength * decay_factor)
                self.conn.execute(
                    "UPDATE beacon_registry SET strength = ?, last_touched_ms = ? WHERE beacon_id = ?",
                    (strength, last_touched_ms, beacon_id),
                )
            if strength < strength_floor and last_touched_ms < cutoff_ms:
                # Demote: keep registry entry but strip from cards and lower strength.
                now = _now_ms()
                demoted_strength = min(strength, strength_floor * 0.5)
                self.conn.execute(
                    "UPDATE beacon_registry SET strength = ?, card_count = ?, last_touched_ms = ? WHERE beacon_id = ?",
                    (demoted_strength, 0, now, beacon_id),
                )
                cards = self._load_cards()
                for card in cards:
                    blist = card.get("beacon_list") or []
                    if beacon_id not in blist:
                        continue
                    updated = [b for b in blist if b != beacon_id]
                    self._update_card(
                        card["id"],
                        title=card.get("title"),
                        summary=card.get("summary") or "",
                        entities=card.get("entities") or [],
                        structured_entities=card.get("structured_entities") or [],
                        entities_canonical=card.get("entities_canonical") or [],
                        embedding=card.get("embedding") or [],
                        updated_ms=now,
                        source_turn_ids=card.get("source_turn_ids") or [],
                        observation_fingerprint=card.get("observation_fingerprint"),
                        access_count=card.get("access_count"),
                        merge_version=card.get("merge_version"),
                        beacon_list=updated,
                    )
                    self._persist_event(
                        card["id"],
                        "beacon_demote",
                        payload={"beacon_id": beacon_id},
                        before={"beacons": blist},
                        after={"beacons": updated},
                    )
                demoted.append(beacon_id)
        self.conn.commit()
        return demoted

    def prune_stale_cards(
        self,
        *,
        min_access_count: int = 1,
        max_age_days: int = 180,
        keep_recent_n: int = 10,
    ) -> List[int]:
        cutoff_ms = _now_ms() - max_age_days * 24 * 3600 * 1000
        rows = self.conn.execute(
            """
            SELECT id, updated_ms, access_count FROM consolidated_cards
            ORDER BY updated_ms DESC
            """
        ).fetchall()
        protected = {row[0] for row in rows[:keep_recent_n]}
        removed: List[int] = []
        for card_id, updated_ms, access_count in rows:
            if card_id in protected:
                continue
            if access_count is None:
                access_count = 0
            if access_count > min_access_count:
                continue
            if updated_ms >= cutoff_ms:
                continue
            self.conn.execute("DELETE FROM consolidated_cards WHERE id = ?", (card_id,))
            if self.ann_index:
                self.ann_index.remove(card_id)
            removed.append(card_id)
            self._persist_event(
                card_id,
                "card_prune",
                payload={"reason": "stale", "access_count": access_count},
                before={"card_id": card_id},
                after={},
            )
        self.conn.commit()
        return removed

    def store_and_consolidate_turn(
        self,
        *,
        user_content: str,
        assistant_content: Optional[str],
        combined_embedding: Sequence[float],
        conversation_id: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        embedding_strategy: str = "blend",
        embedding_fn: Optional[Callable[[str], Optional[Sequence[float]]]] = None,
        summary_fn: Optional[Callable[[str], str]] = None,
        merge_summary_fn: Optional[Callable[[str, str], str]] = None,
        title_fn: Optional[Callable[[str], str]] = None,
        contradiction_fn: Optional[Callable[[str, str], str]] = None,
        beacon_assignment_fn: Optional[
            Callable[[str, List[str], Dict[str, Dict], List[str]], Dict[str, List[str]]]
        ] = None,
        canonicalization_fn: Optional[
            Callable[[List[str], str, str, str], object]
        ] = None,
    ) -> int:
        if not user_content.strip():
            raise ValueError("Cannot store empty user content")
        if not combined_embedding:
            raise ValueError("Cannot store empty combined embedding")
        ts = timestamp_ms if timestamp_ms is not None else _now_ms()
        combined_text = f"User: {user_content.strip()}\nAssistant: {(assistant_content or '').strip()}"
        entities = _extract_entities(combined_text)
        entities_canonical: List[str] = []
        seen_canon = set()
        structured_entities: List[Dict] = []
        canon_meta: Dict = {}
        if canonicalization_fn:
            try:
                result = canonicalization_fn(
                    list(entities[:7]),
                    user_content,
                    assistant_content or "",
                    combined_text,
                )
                if isinstance(result, dict) and "entities" in result:
                    structured_entities = result.get("entities") or []
                    canon_meta = result.get("meta") or {}
                elif isinstance(result, list):
                    structured_entities = result
                    canon_meta = {"fallback_reason": "no_meta"}
                else:
                    structured_entities = []
                    canon_meta = {"fallback_reason": "invalid_result"}
            except Exception:
                structured_entities = []
                canon_meta = {"fallback_reason": "helper_exception"}
        else:
            canon_meta = {"fallback_reason": "no_helper"}

        self_beacons = _self_beacons_from_structured(structured_entities)

        allowed_types = {
            "org",
            "person",
            "location",
            "product",
            "tech",
            "domain",
            "concept",
            "event",
            "document",
            "relation",
            "action",
            "self_identity",
            "self_location",
            "self_work",
            "self_health",
            "self_relationships",
            "self_preferences",
            "self_projects",
        }
        confidence_values: List[float] = []
        if structured_entities:
            for item in structured_entities:
                if not isinstance(item, dict):
                    continue
                type_val = item.get("type")
                if type_val and type_val not in allowed_types:
                    continue
                canon_val = item.get("canonical_name")
                if not isinstance(canon_val, str):
                    continue
                conf_val = item.get("confidence")
                if isinstance(conf_val, (int, float)):
                    try:
                        conf_float = float(conf_val)
                        confidence_values.append(conf_float)
                        if conf_float < DEFAULT_CANON_CONFIDENCE_FLOOR:
                            continue
                    except Exception:
                        pass
                norm = _canonicalize_entity(canon_val)
                if not norm or norm in seen_canon:
                    continue
                seen_canon.add(norm)
                entities_canonical.append(norm)
        else:
            for e in entities:
                c = _canonicalize_entity(e)
                if not c or c in seen_canon:
                    continue
                seen_canon.add(c)
                entities_canonical.append(c)
        meta_success = canon_meta.get("success") if isinstance(canon_meta, dict) else None
        canon_event_payload = {
            "success": bool(meta_success) if meta_success is not None else bool(structured_entities),
            "fallback_reason": canon_meta.get("fallback_reason") if isinstance(canon_meta, dict) else None,
            "latency_ms": canon_meta.get("latency_ms") if isinstance(canon_meta, dict) else None,
            "entities_in": len(entities[:7]),
            "entities_out": len(entities_canonical),
        }
        if confidence_values:
            canon_event_payload["confidence_min"] = min(confidence_values)
            canon_event_payload["confidence_max"] = max(confidence_values)
            canon_event_payload["confidence_avg"] = sum(confidence_values) / float(len(confidence_values))
        self._persist_event(
            card_id=None,
            event_type="canonicalization",
            payload=canon_event_payload,
            before=None,
            after=None,
            source_turn_id=None,
        )
        observation_fingerprint = _compute_fingerprint(combined_text)
        raw_turn_id = self._insert_raw_turn(
            user_content=user_content,
            assistant_content=assistant_content,
            combined_text=combined_text,
            combined_embedding=combined_embedding,
            entities=entities,
            structured_entities=structured_entities,
            entities_canonical=entities_canonical,
            observation_fingerprint=observation_fingerprint,
            timestamp_ms=ts,
            conversation_id=conversation_id,
        )

        cards = self._load_cards()
        entity_canon_set = set(entities_canonical)
        overlap_cards: List[Dict] = []
        for card in cards:
            card_entity_set = set(card.get("entities_canonical") or [])
            if entity_canon_set and card_entity_set and entity_canon_set & card_entity_set:
                overlap_cards.append(card)

        best_entity_sim = 0.0
        best_entity_card: Optional[Dict] = None
        for card in overlap_cards:
            sim = _cosine_similarity(combined_embedding, card.get("embedding") or [])
            if sim > best_entity_sim:
                best_entity_sim = sim
                best_entity_card = card

        adaptive_threshold = max(0.68, best_entity_sim - 0.05, merge_threshold)

        ann_candidates: List[Dict] = []
        if self.ann_index and len(combined_embedding) == self.ann_index.dim:
            ann_hits = self.ann_index.search(combined_embedding, 8)
            card_map = {c["id"]: c for c in cards}
            for cid, sim in ann_hits:
                card = card_map.get(cid)
                if not card:
                    continue
                card = dict(card)
                card["_ann_sim"] = sim
                ann_candidates.append(card)

        candidate_pool: Dict[int, Dict] = {}
        for card in overlap_cards + ann_candidates:
            candidate_pool[card["id"]] = card
        candidates = list(candidate_pool.values())

        fingerprint_match_card = None
        for card in candidates:
            c_entities = set(card.get("entities_canonical") or [])
            if observation_fingerprint and card.get("observation_fingerprint") == observation_fingerprint:
                if entity_canon_set & c_entities:
                    fingerprint_match_card = card
                    break

        if fingerprint_match_card:
            source_ids = fingerprint_match_card.get("source_turn_ids") or []
            source_ids.append(raw_turn_id)
            merged_structured_map = {}
            for item in (fingerprint_match_card.get("structured_entities") or []) + (structured_entities or []):
                if not isinstance(item, dict):
                    continue
                key = ((item.get("canonical_name") or "").lower(), item.get("type"))
                if key in merged_structured_map:
                    continue
                merged_structured_map[key] = item
            merged_structured = list(merged_structured_map.values())
            beacon_list = list(
                {b: b for b in (fingerprint_match_card.get("beacon_list") or []) + self_beacons}.values()
            )
            self._persist_event(
                fingerprint_match_card["id"],
                "fingerprint_match",
                payload={"raw_turn_id": raw_turn_id},
                before={"summary": fingerprint_match_card.get("summary")},
                after={"summary": fingerprint_match_card.get("summary")},
                source_turn_id=raw_turn_id,
            )
            self._update_card(
                fingerprint_match_card["id"],
                title=fingerprint_match_card.get("title") or "Memory",
                summary=fingerprint_match_card.get("summary") or "",
                entities=fingerprint_match_card.get("entities") or [],
                structured_entities=merged_structured,
                entities_canonical=list(set(fingerprint_match_card.get("entities_canonical") or [])),
                embedding=fingerprint_match_card.get("embedding") or [],
                updated_ms=ts,
                source_turn_ids=source_ids,
                observation_fingerprint=fingerprint_match_card.get("observation_fingerprint"),
                access_count=(fingerprint_match_card.get("access_count") or 0) + 1,
                merge_version=(fingerprint_match_card.get("merge_version") or 1),
                beacon_list=beacon_list,
            )
            self._update_beacon_counts(beacon_list)
            return raw_turn_id

        best_candidate = None
        best_sim = -1.0
        for card in candidates:
            sim = _cosine_similarity(combined_embedding, card.get("embedding") or [])
            if sim > best_sim:
                best_sim = sim
                best_candidate = card

        if best_candidate:
            best_canon_set = set(best_candidate.get("entities_canonical") or [])
            if best_sim >= duplicate_threshold and best_canon_set == entity_canon_set:
                source_ids = best_candidate.get("source_turn_ids") or []
                source_ids.append(raw_turn_id)
                merged_structured_map = {}
                for item in (best_candidate.get("structured_entities") or []) + (structured_entities or []):
                    if not isinstance(item, dict):
                        continue
                    key = ((item.get("canonical_name") or "").lower(), item.get("type"))
                    if key in merged_structured_map:
                        continue
                    merged_structured_map[key] = item
                merged_structured = list(merged_structured_map.values())
                before = {"summary": best_candidate.get("summary"), "embedding": best_candidate.get("embedding")}
                beacon_list = list(
                    {b: b for b in (best_candidate.get("beacon_list") or []) + self_beacons}.values()
                )
                self._persist_event(
                    best_candidate["id"],
                    "duplicate",
                    payload={"raw_turn_id": raw_turn_id, "similarity": best_sim},
                    before=before,
                    after=before,
                    source_turn_id=raw_turn_id,
                )
                self._update_card(
                    best_candidate["id"],
                    title=best_candidate.get("title") or "Memory",
                    summary=best_candidate.get("summary") or "",
                    entities=best_candidate.get("entities") or [],
                    structured_entities=merged_structured,
                    entities_canonical=list(best_canon_set),
                    embedding=best_candidate.get("embedding") or [],
                    updated_ms=ts,
                    source_turn_ids=source_ids,
                    observation_fingerprint=best_candidate.get("observation_fingerprint"),
                    access_count=(best_candidate.get("access_count") or 0) + 1,
                    merge_version=(best_candidate.get("merge_version") or 1),
                    beacon_list=beacon_list,
                )
                self._update_beacon_counts(beacon_list)
                return raw_turn_id

        merge_candidate = None
        merge_sim = -1.0
        if best_candidate and best_sim >= adaptive_threshold:
            merge_candidate = best_candidate
            merge_sim = best_sim

        if merge_candidate:
            current_summary = merge_candidate.get("summary", "")
            merged_summary = (
                merge_summary_fn(current_summary, combined_text)
                if merge_summary_fn
                else f"{current_summary}\n- {combined_text[:400].strip()}"
            )
            contradiction = self._detect_contradiction(current_summary, combined_text)
            if contradiction and contradiction_fn:
                merged_summary = contradiction_fn(current_summary, combined_text)
            union_entities = list({e.lower(): e for e in (merge_candidate.get("entities") or []) + entities}.values())
            union_entities_canonical = list(
                {c: c for c in (merge_candidate.get("entities_canonical") or []) + entities_canonical}.values()
            )
            beacon_list = merge_candidate.get("beacon_list") or []
            new_beacons, proposed_new_beacons = self._assign_beacons(
                union_entities_canonical,
                combined_text,
                summary_text=merged_summary,
                beacon_assignment_fn=beacon_assignment_fn,
                current_beacons=beacon_list,
                forced_beacons=self_beacons,
            )
            beacon_list = list({b: b for b in beacon_list + new_beacons}.values())

            merged_structured_map = {}
            for item in (merge_candidate.get("structured_entities") or []) + (structured_entities or []):
                if not isinstance(item, dict):
                    continue
                key = ((item.get("canonical_name") or "").lower(), item.get("type"))
                if key in merged_structured_map:
                    continue
                merged_structured_map[key] = item
            merged_structured = list(merged_structured_map.values())

            card_embedding: Optional[Sequence[float]] = None
            if contradiction and contradiction_fn and embedding_fn:
                card_embedding = embedding_fn(merged_summary)
            elif embedding_strategy == "reembed_summary" and embedding_fn:
                card_embedding = embedding_fn(merged_summary)
            elif embedding_strategy == "blend":
                card_embedding = _blend_embeddings(
                    merge_candidate.get("embedding") or [], combined_embedding, new_weight=0.4
                )
            if card_embedding is None:
                card_embedding = merge_candidate.get("embedding") or []

            source_ids = merge_candidate.get("source_turn_ids") or []
            source_ids.append(raw_turn_id)
            before = {
                "summary": merge_candidate.get("summary"),
                "embedding": merge_candidate.get("embedding"),
                "beacons": merge_candidate.get("beacon_list"),
            }
            after = {
                "summary": merged_summary,
                "embedding": list(card_embedding),
                "beacons": beacon_list,
            }
            self._persist_event(
                merge_candidate["id"],
                "contradiction" if contradiction else "merge",
                payload={
                    "raw_turn_id": raw_turn_id,
                    "similarity": merge_sim,
                    "proposed_new_beacons": proposed_new_beacons,
                },
                before=before,
                after=after,
                source_turn_id=raw_turn_id,
            )

            self._update_card(
                merge_candidate["id"],
                title=merge_candidate.get("title") or (title_fn(merged_summary) if title_fn else "Memory"),
                summary=merged_summary,
                entities=union_entities,
                structured_entities=merged_structured,
                entities_canonical=union_entities_canonical,
                embedding=card_embedding,
                updated_ms=ts,
                source_turn_ids=source_ids,
                observation_fingerprint=observation_fingerprint,
                access_count=(merge_candidate.get("access_count") or 0) + 1,
                merge_version=(merge_candidate.get("merge_version") or 1) + 1,
                beacon_list=beacon_list,
            )
            self._update_beacon_counts(beacon_list)
            return raw_turn_id

        summary_text = summary_fn(combined_text) if summary_fn else combined_text[:320]
        title = title_fn(summary_text) if title_fn else "Memory"
        card_embedding: Optional[Sequence[float]] = None
        if embedding_strategy == "reembed_summary" and embedding_fn:
            card_embedding = embedding_fn(summary_text)
        if card_embedding is None:
            card_embedding = list(combined_embedding)

        beacons, proposed_new_beacons = self._assign_beacons(
            entities_canonical,
            combined_text,
            summary_text=summary_text,
            beacon_assignment_fn=beacon_assignment_fn,
            current_beacons=[],
            forced_beacons=self_beacons,
        )

        new_card_id = self._insert_card(
            title=title,
            summary=summary_text,
            entities=entities,
            structured_entities=structured_entities,
            entities_canonical=entities_canonical,
            embedding=card_embedding,
            observation_fingerprint=observation_fingerprint,
            created_ms=ts,
            source_turn_ids=[raw_turn_id],
            beacon_list=beacons,
        )
        self._persist_event(
            new_card_id,
            "new_card",
            payload={"raw_turn_id": raw_turn_id, "proposed_new_beacons": proposed_new_beacons},
            after={"summary": summary_text, "embedding": list(card_embedding), "beacons": beacons},
            source_turn_id=raw_turn_id,
        )
        self._update_beacon_counts(beacons)
        return raw_turn_id

    def _recent_raw_turns(self, limit: int) -> List[Dict]:
        rows = self.conn.execute(
            """
            SELECT id, user_content, assistant_content, timestamp_ms, entities_json
            FROM raw_turns
            ORDER BY timestamp_ms DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        turns: List[Dict] = []
        for row in rows:
            turn_id, user_content, assistant_content, ts_ms, entities_json = row
            turns.append(
                {
                    "id": turn_id,
                    "user_content": user_content,
                    "assistant_content": assistant_content or "",
                    "timestamp_ms": ts_ms,
                    "entities": _safe_load_json_list(entities_json),
                }
            )
        return turns

    def _cards_by_beacons(self, beacon_ids: List[str]) -> List[Dict]:
        if not beacon_ids:
            return []
        cards = self._load_cards()
        target = set(beacon_ids)
        return [c for c in cards if target & set(c.get("beacon_list") or [])]

    def _beacons_for_query(self, query_entities_canonical: List[str]) -> List[str]:
        registry = self._beacon_registry()
        candidates = [registry[e] for e in query_entities_canonical if e in registry]
        ranked = sorted(candidates, key=lambda x: (x["strength"], x["card_count"]), reverse=True)
        chosen: List[str] = [b["beacon_id"] for b in ranked[:2]]
        # Always ensure primordial fallback is present
        for p in PRIMORDIAL_BEACONS[:2]:
            if p not in chosen:
                chosen.append(p)
        return chosen

    def retrieve_context(
        self,
        *,
        query_text: str,
        query_embedding: Sequence[float],
        card_k: int = DEFAULT_CARD_K,
        vector_threshold: float = DEFAULT_VECTOR_THRESHOLD,
        recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
        canonicalization_fn: Optional[
            Callable[[List[str], str, str, str], object]
        ] = None,
    ) -> Dict[str, List[Dict]]:
        if not query_embedding:
            return {"cards": [], "recent_raw_turns": []}
        if canonicalization_fn is None:
            raise ValueError("canonicalization_fn is required for recall canonicalization")

        query_entities_raw = _extract_entities(query_text)
        structured_entities: List[Dict] = []
        canon_meta: Dict = {}
        try:
            result = canonicalization_fn(
                list(query_entities_raw[:7]),
                query_text,
                "",
                query_text,
            )
            if isinstance(result, dict) and "entities" in result:
                structured_entities = result.get("entities") or []
                canon_meta = result.get("meta") or {}
            elif isinstance(result, list):
                structured_entities = result
                canon_meta = {"fallback_reason": "no_meta"}
            else:
                structured_entities = []
                canon_meta = {"fallback_reason": "invalid_result"}
        except Exception as e:
            raise RuntimeError(f"Recall canonicalization failed: {e}")

        query_entities_canonical: List[str] = []
        seen_q = set()
        allowed_types = {
            "org",
            "person",
            "location",
            "product",
            "tech",
            "domain",
            "concept",
            "event",
            "document",
            "relation",
            "action",
            "self_identity",
            "self_location",
            "self_work",
            "self_health",
            "self_relationships",
            "self_preferences",
            "self_projects",
        }
        if structured_entities:
            for item in structured_entities:
                if not isinstance(item, dict):
                    continue
                type_val = item.get("type")
                if type_val and type_val not in allowed_types:
                    continue
                canon_val = item.get("canonical_name")
                if not isinstance(canon_val, str):
                    continue
                conf_val = item.get("confidence")
                if isinstance(conf_val, (int, float)):
                    try:
                        conf_float = float(conf_val)
                        if conf_float < DEFAULT_CANON_CONFIDENCE_FLOOR:
                            continue
                    except Exception:
                        pass
                norm = _canonicalize_entity(canon_val)
                if not norm or norm in seen_q:
                    continue
                seen_q.add(norm)
                query_entities_canonical.append(norm)
        if not query_entities_canonical:
            for e in query_entities_raw:
                c = _canonicalize_entity(e)
                if not c or c in seen_q:
                    continue
                seen_q.add(c)
                query_entities_canonical.append(c)

        beacon_ids = self._beacons_for_query(query_entities_canonical)
        cards = self._load_cards()
        beacon_cards = [c for c in cards if set(c.get("beacon_list") or []) & set(beacon_ids)]
        beacon_cards.sort(key=lambda c: (c.get("access_count", 0), c.get("updated_ms", 0)), reverse=True)
        final_cards = beacon_cards[:card_k]

        if not final_cards and query_entities_canonical:
            query_entity_set = set(query_entities_canonical)
            keyword_hits = [
                c
                for c in cards
                if query_entity_set & set(c.get("entities_canonical") or [])
            ]
            keyword_hits.sort(key=lambda c: c.get("updated_ms", 0), reverse=True)
            final_cards = keyword_hits[:card_k]

        if not final_cards and self.ann_index and len(query_embedding) == self.ann_index.dim:
            hits = self.ann_index.search(query_embedding, card_k * 2)
            card_map = {c["id"]: c for c in cards}
            scored: List[Tuple[float, Dict]] = []
            for cid, sim in hits:
                if sim < vector_threshold:
                    continue
                card = card_map.get(cid)
                if card:
                    scored.append((sim, card))
            scored.sort(key=lambda x: x[0], reverse=True)
            final_cards = [c for _, c in scored[:card_k]]

        for card in final_cards:
            self.conn.execute(
                "UPDATE consolidated_cards SET access_count = access_count + 1 WHERE id = ?",
                (card["id"],),
            )
        self.conn.commit()

        recent_turns = self._recent_raw_turns(recent_turn_limit)
        return {"cards": final_cards, "recent_raw_turns": recent_turns}

    def format_context(
        self,
        cards: List[Dict],
        recent_raw_turns: List[Dict],
        max_chars: int = 4000,
    ) -> str:
        parts: List[str] = []
        parts.append("### RETRIEVED LONG-TERM MEMORY (consolidated facts)")
        parts.append("### Consolidated Memory Cards")
        if cards:
            for card in cards:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(card.get("updated_ms", 0) / 1000))
                entities = ", ".join(card.get("entities") or [])
                beacons = ", ".join(card.get("beacon_list") or [])
                summary = (card.get("summary") or "").strip()
                if len(summary) > 800:
                    summary = summary[:780] + "... (truncated)"
                parts.append(
                    f"- [{card.get('id')}] {card.get('title') or 'Memory'} "
                    f"(updated {ts}) | entities: {entities or '—'} | beacons: {beacons or '—'}\n  {summary}"
                )
        else:
            parts.append("None.")

        parts.append("\n### SHORT-TERM MEMORY (recent conversation)")
        parts.append("### Recent Raw Turns")
        if recent_raw_turns:
            for turn in sorted(recent_raw_turns, key=lambda x: x["timestamp_ms"]):
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(turn["timestamp_ms"] / 1000))
                entities = ", ".join(turn.get("entities") or [])
                parts.append(
                    f"- ({ts}) User: {turn['user_content'].strip()}\n"
                    f"  Assistant: {(turn['assistant_content'] or '').strip()}\n"
                    f"  Entities: {entities or '—'}"
                )
        else:
            parts.append("None.")

        text = "\n".join(parts).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n... (truncated)"
        return text

    def _detect_contradiction(self, existing_summary: str, new_text: str) -> bool:
        existing_lower = (existing_summary or "").lower()
        new_lower = (new_text or "").lower()
        neg_markers = ["no longer", "not", "never", "stop", "stopped", "isn't", "wasn't", "aren't"]
        if not any(marker in new_lower for marker in neg_markers):
            return False
        existing_tokens = set(re.findall(r"[a-z0-9_/-]+", existing_lower))
        new_tokens = set(re.findall(r"[a-z0-9_/-]+", new_lower))
        if not existing_tokens or not new_tokens:
            return False
        overlap = existing_tokens & new_tokens
        overlap_ratio = len(overlap) / float(max(len(existing_tokens), 1))
        return overlap_ratio >= 0.3
