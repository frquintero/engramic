import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


DEFAULT_DB_PATH = Path("memory.db")
DEFAULT_DUPLICATE_THRESHOLD = 0.92
DEFAULT_MERGE_THRESHOLD = 0.75
DEFAULT_VECTOR_THRESHOLD = 0.38
DEFAULT_CARD_K = 4
DEFAULT_RECENT_TURN_LIMIT = 5
DEFAULT_CARD_EMBEDDING_STRATEGY = "reembed_summary"  # or "blend_turn"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


_spacy_nlp = None


def _get_spacy_model():
    """
    Lazy-load spaCy if available; return None if not installed.
    """
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
}


def _extract_entities(text: str) -> List[str]:
    """
    Best-effort entity/key-noun extractor.
    Prefers spaCy NER/POS; falls back to a regex-based nounish token extractor.
    """
    entities: List[str] = []
    nlp = _get_spacy_model()
    if nlp:
        try:
            doc = nlp(text)
            for ent in getattr(doc, "ents", []):
                cleaned = ent.text.strip()
                if cleaned:
                    entities.append(cleaned)
            # Fallback to noun chunks if ents are empty
            if not entities:
                for chunk in getattr(doc, "noun_chunks", []):
                    cleaned = chunk.text.strip()
                    if cleaned:
                        entities.append(cleaned)
        except Exception:
            entities = []

    if not entities:
        # Regex fallback: capture words, filenames, and slash-separated tokens.
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

    # Deduplicate while preserving order; normalize to lowercase for matching.
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


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return list(vec)
    return [v / norm for v in vec]


def _blend_embeddings(
    base: Sequence[float],
    new: Sequence[float],
    *,
    new_weight: float = 0.3,
) -> List[float]:
    if not base or not new or len(base) != len(new):
        return list(base) if base else list(new)
    w_old = 1.0 - new_weight
    blended = [w_old * a + new_weight * b for a, b in zip(base, new)]
    return _l2_normalize(blended)


def _default_summarize(text: str) -> str:
    """
    Fallback summarizer: trim and keep first ~2 sentences/lines.
    """
    text = text.strip()
    if not text:
        return ""
    # Prefer first two sentences; fall back to first 320 chars.
    parts = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(parts[:2]).strip()
    if not summary:
        summary = text[:320]
    return summary.strip()


def _default_merge_summary(existing: str, new_text: str) -> str:
    existing = (existing or "").strip()
    new_text = (new_text or "").strip()
    if not existing:
        return _default_summarize(new_text)
    if not new_text:
        return existing
    return f"{existing}\\n- {new_text[:400].strip()}"


def _default_title(text: str) -> str:
    text = text.strip()
    if not text:
        return "Memory"
    # Keep it short; prefer first few tokens.
    tokens = re.findall(r"[A-Za-z0-9_.-]+", text)
    title = " ".join(tokens[:6]) if tokens else "Memory"
    return title[:60]


class MemoryStore:
    """
    SQLite-backed memory store for raw turns and consolidated entity-centric memories.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        # Legacy tables (kept for backward compatibility)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_content TEXT NOT NULL,
                assistant_content TEXT,
                user_hash TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                conversation_id TEXT
            )
            """
        )
        self.conn.execute("DROP INDEX IF EXISTS idx_turns_user_hash")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_user_hash ON turns(user_hash)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_embeddings (
                turn_id INTEGER PRIMARY KEY,
                combined_embedding_json TEXT NOT NULL,
                FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE CASCADE
            )
            """
        )

        # New raw turns table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_content TEXT NOT NULL,
                assistant_content TEXT,
                combined_text TEXT NOT NULL,
                entities_json TEXT,
                combined_embedding_json TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                conversation_id TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_turns_ts ON raw_turns(timestamp_ms)"
        )

        # Consolidated memories table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consolidated_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                summary TEXT NOT NULL,
                entities_json TEXT,
                embedding_json TEXT NOT NULL,
                created_ms INTEGER NOT NULL,
                updated_ms INTEGER NOT NULL,
                source_turn_ids_json TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consolidated_updated ON consolidated_memories(updated_ms)"
        )
        self.conn.commit()

    def add_turn(
        self,
        *,
        user_content: str,
        assistant_content: Optional[str],
        combined_embedding: Sequence[float],
        conversation_id: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
    ) -> int:
        if not user_content.strip():
            raise ValueError("Cannot store empty user content")
        if not combined_embedding:
            raise ValueError("Cannot store empty combined embedding")
        ts = timestamp_ms if timestamp_ms is not None else _now_ms()
        user_hash = hashlib.sha256(user_content.strip().encode("utf-8")).hexdigest()
        cur = self.conn.cursor()
        # We allow duplicates now to preserve context of same commands in different situations
        cur.execute(
            """
            INSERT INTO turns (user_content, assistant_content, user_hash, timestamp_ms, conversation_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_content, assistant_content or "", user_hash, ts, conversation_id),
        )
        turn_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO turn_embeddings (turn_id, combined_embedding_json)
            VALUES (?, ?)
            """,
            (
                turn_id,
                json.dumps(list(combined_embedding)),
            ),
        )
        self.conn.commit()
        return turn_id

    def _insert_raw_turn(
        self,
        *,
        user_content: str,
        assistant_content: Optional[str],
        combined_text: str,
        combined_embedding: Sequence[float],
        entities: List[str],
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
                combined_embedding_json,
                timestamp_ms,
                conversation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_content,
                assistant_content or "",
                combined_text,
                json.dumps(entities),
                json.dumps(list(combined_embedding)),
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
            SELECT id, title, summary, entities_json, embedding_json, created_ms, updated_ms, source_turn_ids_json
            FROM consolidated_memories
            """
        ).fetchall()
        cards: List[Dict] = []
        for row in rows:
            (
                card_id,
                title,
                summary,
                entities_json,
                embedding_json,
                created_ms,
                updated_ms,
                source_ids_json,
            ) = row
            cards.append(
                {
                    "id": card_id,
                    "title": title or "Memory",
                    "summary": summary or "",
                    "entities": _safe_load_json_list(entities_json),
                    "embedding": json.loads(embedding_json) if embedding_json else [],
                    "created_ms": created_ms,
                    "updated_ms": updated_ms,
                    "source_turn_ids": _safe_load_json_list(source_ids_json),
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
        embedding: Sequence[float],
        updated_ms: int,
        source_turn_ids: List[int],
    ) -> None:
        self.conn.execute(
            """
            UPDATE consolidated_memories
            SET title = ?, summary = ?, entities_json = ?, embedding_json = ?, updated_ms = ?, source_turn_ids_json = ?
            WHERE id = ?
            """,
            (
                title,
                summary,
                json.dumps(entities),
                json.dumps(list(embedding)),
                updated_ms,
                json.dumps(source_turn_ids),
                card_id,
            ),
        )
        self.conn.commit()

    def _insert_card(
        self,
        *,
        title: str,
        summary: str,
        entities: List[str],
        embedding: Sequence[float],
        created_ms: int,
        source_turn_ids: List[int],
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO consolidated_memories (
                title,
                summary,
                entities_json,
                embedding_json,
                created_ms,
                updated_ms,
                source_turn_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                summary,
                json.dumps(entities),
                json.dumps(list(embedding)),
                created_ms,
                created_ms,
                json.dumps(source_turn_ids),
            ),
        )
        card_id = cur.lastrowid
        self.conn.commit()
        return card_id

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
        embedding_strategy: str = DEFAULT_CARD_EMBEDDING_STRATEGY,
        embedding_fn: Optional[Callable[[str], Optional[Sequence[float]]]] = None,
        summary_fn: Optional[Callable[[str], str]] = None,
        merge_summary_fn: Optional[Callable[[str, str], str]] = None,
        title_fn: Optional[Callable[[str], str]] = None,
    ) -> int:
        """
        Store a raw turn and update consolidated memories with entity-centric logic.
        """
        if not user_content.strip():
            raise ValueError("Cannot store empty user content")
        if not combined_embedding:
            raise ValueError("Cannot store empty combined embedding")

        ts = timestamp_ms if timestamp_ms is not None else _now_ms()
        combined_text = f"<user_query>{user_content}</user_query>\\n<agent_response>{assistant_content or ''}</agent_response>"
        entities = _extract_entities(combined_text)
        raw_turn_id = self._insert_raw_turn(
            user_content=user_content,
            assistant_content=assistant_content,
            combined_text=combined_text,
            combined_embedding=combined_embedding,
            entities=entities,
            timestamp_ms=ts,
            conversation_id=conversation_id,
        )

        cards = self._load_cards()
        overlap_cards: List[Dict] = []
        entity_set = {e.lower() for e in entities}

        for card in cards:
            card_entity_set = {e.lower() for e in card["entities"]}
            if not card_entity_set:
                continue
            if entity_set & card_entity_set:
                overlap_cards.append(card)

        # Identify near-duplicate or merge target
        best_candidate = None
        best_sim = -1.0
        best_entities_match = None
        for card in overlap_cards:
            card_emb = card.get("embedding") or []
            sim = _cosine_similarity(combined_embedding, card_emb)
            if sim > best_sim:
                best_sim = sim
                best_candidate = card
                best_entities_match = card.get("entities") or []

        # Near-duplicate: discard silently
        if (
            best_candidate
            and best_sim >= duplicate_threshold
            and set(e.lower() for e in best_entities_match or []) == entity_set
        ):
            return raw_turn_id

        # Merge path
        if best_candidate and best_sim >= merge_threshold:
            current_summary = best_candidate.get("summary", "")
            merge_summary = (
                merge_summary_fn(current_summary, combined_text)
                if merge_summary_fn
                else _default_merge_summary(current_summary, combined_text)
            )
            union_entities = list(
                {e.lower(): e for e in (best_candidate.get("entities") or []) + entities}.values()
            )

            card_embedding: Optional[Sequence[float]] = None
            if embedding_strategy == "reembed_summary" and embedding_fn:
                try:
                    card_embedding = embedding_fn(merge_summary)
                except Exception:
                    card_embedding = None
            if card_embedding is None:
                card_embedding = _blend_embeddings(
                    best_candidate.get("embedding") or [],
                    combined_embedding,
                    new_weight=0.3,
                )

            source_ids = best_candidate.get("source_turn_ids") or []
            source_ids.append(raw_turn_id)

            self._update_card(
                best_candidate["id"],
                title=best_candidate.get("title") or _default_title(merge_summary),
                summary=merge_summary,
                entities=union_entities,
                embedding=card_embedding,
                updated_ms=ts,
                source_turn_ids=source_ids,
            )
            return raw_turn_id

        # New card path
        summary_text = (
            summary_fn(combined_text) if summary_fn else _default_summarize(combined_text)
        )
        title = title_fn(combined_text) if title_fn else _default_title(summary_text)

        card_embedding: Optional[Sequence[float]] = None
        if embedding_strategy == "reembed_summary" and embedding_fn:
            try:
                card_embedding = embedding_fn(summary_text)
            except Exception:
                card_embedding = None
        if card_embedding is None:
            card_embedding = list(combined_embedding)

        self._insert_card(
            title=title,
            summary=summary_text,
            entities=entities,
            embedding=card_embedding,
            created_ms=ts,
            source_turn_ids=[raw_turn_id],
        )
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

    def retrieve_context(
        self,
        *,
        query_text: str,
        query_embedding: Sequence[float],
        card_k: int = DEFAULT_CARD_K,
        vector_threshold: float = DEFAULT_VECTOR_THRESHOLD,
        recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve consolidated cards (keyword-first, then vector fallback) plus recent raw turns.
        """
        if not query_embedding:
            return {"cards": [], "recent_raw_turns": []}

        query_entities = _extract_entities(query_text)
        cards = self._load_cards()
        final_cards: List[Dict] = []

        # Keyword pass
        if query_entities:
            query_entity_set = {e.lower() for e in query_entities}
            keyword_hits = [
                c
                for c in cards
                if query_entity_set & {e.lower() for e in (c.get("entities") or [])}
            ]
            keyword_hits.sort(key=lambda c: c.get("updated_ms", 0), reverse=True)
            final_cards = keyword_hits[:card_k]

        # Vector fallback
        if not final_cards:
            scored: List[Tuple[float, Dict]] = []
            for card in cards:
                card_emb = card.get("embedding") or []
                sim = _cosine_similarity(query_embedding, card_emb)
                if sim >= vector_threshold:
                    scored.append((sim, card))
            scored.sort(key=lambda x: x[0], reverse=True)
            final_cards = [c for _, c in scored[:card_k]]

        recent_turns = self._recent_raw_turns(recent_turn_limit)
        return {"cards": final_cards, "recent_raw_turns": recent_turns}

    def format_context(
        self,
        cards: List[Dict],
        recent_raw_turns: List[Dict],
        max_chars: int = 4000,
    ) -> str:
        """
        Format consolidated cards and recent raw turns for prompt injection.
        """
        parts: List[str] = []

        parts.append("### RETRIEVED LONG-TERM MEMORY (consolidated facts)")
        parts.append("### Consolidated Memory Cards")
        if cards:
            for card in cards:
                ts = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(card.get("updated_ms", 0) / 1000)
                )
                entities = ", ".join(card.get("entities") or [])
                summary = (card.get("summary") or "").strip()
                if len(summary) > 800:
                    summary = summary[:780] + "... (truncated)"
                parts.append(
                    f"- [{card.get('id')}] {card.get('title') or 'Memory'} "
                    f"(updated {ts}) | entities: {entities or '—'}\n  {summary}"
                )
        else:
            parts.append("None.")

        parts.append("\n### SHORT-TERM MEMORY (recent conversation)")
        parts.append("### Recent Raw Turns")
        if recent_raw_turns:
            for turn in sorted(recent_raw_turns, key=lambda x: x["timestamp_ms"]):
                ts = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(turn["timestamp_ms"] / 1000)
                )
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

    def retrieve_turns(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        half_life_hours: Optional[float] = None,
    ) -> List[Dict]:
        if not query_embedding or top_k <= 0:
            return []
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT t.id, t.user_content, t.assistant_content, t.timestamp_ms, t.conversation_id, e.combined_embedding_json
            FROM turns t
            JOIN turn_embeddings e ON t.id = e.turn_id
            """
        ).fetchall()
        now_ms = _now_ms()
        results: List[Tuple[float, Dict]] = []
        for row in rows:
            turn_id, user_content, assistant_content, ts_ms, conv_id, combined_embedding_json = row
            try:
                # Use combined embedding for similarity comparison
                combined_emb = json.loads(combined_embedding_json)
                combined_sim = _cosine_similarity(query_embedding, combined_emb)
            except Exception:
                combined_sim = 0.0
            
            # Revised scoring:
            # Use combined similarity as the primary signal.
            
            score = 0.0
            final_sim = combined_sim 
            
            # RECENCY BOOST
            # Give a small boost (e.g. 10%) to memories from the last 2 hours.
            # This acts as a "working memory" bridge, helping borderline relevant recent turns pass the threshold.
            # It ensures that context like "change that" works better for immediate follow-ups.
            
            base_sim = final_sim
            
            # Removed per-item threshold to rely solely on adaptive filtering
            # if base_sim < similarity_threshold:
            #     continue
            
            age_hours = max((now_ms - ts_ms) / 3_600_000.0, 0.0)
            score = base_sim
            
            if half_life_hours is not None and half_life_hours > 0:
                decay = math.exp(-age_hours / half_life_hours)
                score *= decay
            
            # Apply boost only for RANKING, not filtering.
            if age_hours < 2.0:
                score *= 1.1

            results.append(
                (
                    score,
                    {
                        "id": turn_id,
                        "user_content": user_content,
                        "assistant_content": assistant_content,
                        "timestamp_ms": ts_ms,
                        "conversation_id": conv_id,
                        "similarity": base_sim,
                        "score": score,
                        "age_hours": age_hours,
                        "combined_embedding_json": combined_embedding_json,
                    },
                )
            )
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        # --- ADAPTIVE FILTERING ---
        # Issue: Gemini has a high similarity floor (random stuff is ~0.6).
        # Fix: 
        # 1. Drop items that are too far from the best match (Relative Threshold).
        # 2. Require the best match to be "good enough" (Absolute Floor).
        
        if not results:
            return []

        # Use raw similarity for the absolute floor check, not the decayed score.
        # Older memories that are semantically perfect should still be retrievable.
        # The decayed score will only affect their ranking position.
        best_sim = max(item["similarity"] for _, item in results)
        
        # Lowered from 0.65 to 0.50 to capture more permissive matches
        MIN_BEST_SIM = 0.50 
        
        if best_sim < MIN_BEST_SIM:
             # Fallback: if the user explicitly asked for something (user_sim is high), we might want to keep it.
             return []

        # --- TOPIC-BASED EXPANSION ---
        # To retrieve all turns related to a broader topic:
        # 1. Identify the best matching turn.
        # 2. Retrieve all turns that are either highly similar to the query (>=0.6) or to the best turn's topic (>=0.7).
        # This combines direct relevance with topic clustering to avoid missing related turns.
        
        best_score, best_item = results[0]
        best_emb = json.loads(best_item["combined_embedding_json"])
        
        topic_results = []
        for row in rows:
            turn_id, user_content, assistant_content, ts_ms, conv_id, combined_embedding_json = row
            try:
                emb = json.loads(combined_embedding_json)
                query_sim = _cosine_similarity(query_embedding, emb)
                topic_sim = _cosine_similarity(best_emb, emb)
            except Exception:
                query_sim = 0.0
                topic_sim = 0.0
            
            if query_sim >= 0.6 or topic_sim >= 0.7:
                # Use query_sim as the primary score for ranking
                score = query_sim
                topic_results.append((
                    score,
                    {
                        "id": turn_id,
                        "user_content": user_content,
                        "assistant_content": assistant_content,
                        "timestamp_ms": ts_ms,
                        "conversation_id": conv_id,
                        "similarity": query_sim,
                        "score": score,
                        "age_hours": (now_ms - ts_ms) / 3_600_000.0,
                    },
                ))
        
        # Sort topic_results by score descending
        topic_results.sort(key=lambda x: x[0], reverse=True)
        
        results = topic_results
        
        # ---------------------------

        # Deduplicate results by user_content hash to avoid showing identical user queries multiple times
        # (unless they have very different assistant responses? The requirement was "non duplicative")
        # Here we deduplicate by the user_content hash to ensure variety in retrieved commands.
        seen_hashes = set()
        unique_items = []
        for _, item in results:
            h = hashlib.sha256(item["user_content"].strip().encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            unique_items.append(item)
            if len(unique_items) >= top_k:
                break

        return unique_items

    def format_turns(self, turns: List[Dict], max_chars: int = 4000) -> str:
        if not turns:
            return "No relevant past interactions retrieved."
        
        pieces: List[str] = []
        for t in sorted(turns, key=lambda x: x["timestamp_ms"]):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["timestamp_ms"] / 1000))
            sim = f"{t['similarity']:.3f}"
            
            piece = (
                f"--- Memory Turn (Timestamp: {ts}, Similarity: {sim}) ---\n"
                f"User: {t['user_content'].strip()}\n"
                f"Assistant: {(t['assistant_content'] or '').strip()}"
            )
            pieces.append(piece)
            
        text = "\n\n".join(pieces)
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n... (truncated)"
        return text
