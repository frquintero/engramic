import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from groq import Groq

from memory_store import MemoryStore
from tools import get_system_info, available_functions, tools, tools_descriptions, build_response
from workspace_executor import ensure_workspace

# Lazy-loaded BGE model instance
_bge_m3_model = None

# Load configuration
load_dotenv()

# Load configuration
try:
    with open('config.json') as f:
        config = json.load(f)
    debug = config.get('debug', False)
except FileNotFoundError:
    debug = False
    config = {}


def _embed_text(
    client: Groq,
    model: str,
    text: str,
    *,
    provider: str = "groq",
) -> Optional[Sequence[float]]:
    text = text.strip()
    if not text:
        return None
    if provider in {"huggingface", "hf", "bge"}:
        # Preferred local path: FlagEmbedding BGEM3FlagModel (handles pooling + L2)
        try:
            from FlagEmbedding import BGEM3FlagModel
            global _bge_m3_model
            if _bge_m3_model is None:
                bge_model_name = os.environ.get("BGE_MODEL_NAME", model)
                use_fp16 = os.environ.get("BGE_USE_FP16", "true").lower() in {"1", "true", "yes", "on"}
                _bge_m3_model = BGEM3FlagModel(
                    bge_model_name,
                    use_fp16=use_fp16,
                    normalize_embeddings=True,
                )
            try:
                max_len_env = os.environ.get("BGE_MAX_LENGTH")
                max_length = int(max_len_env) if max_len_env else 512  # reduce for speed; override via env if needed
            except Exception:
                max_length = 512
            res = _bge_m3_model.encode(
                [text],
                batch_size=12,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = res.get("dense_vecs")
            if hasattr(dense, "tolist"):
                dense = dense.tolist()
            if isinstance(dense, list) and len(dense) > 0:
                first = dense[0]
                if hasattr(first, "tolist"):
                    first = first.tolist()
                return first if isinstance(first, list) else None
        except Exception as e:
            print(f"BGE local embedding failed: {e}")

        # Fallback: Hugging Face Inference API
        try:
            from huggingface_hub import InferenceClient
        except Exception as e:
            print(f"huggingface_hub not available: {e}")
            return None
        hf_token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Hugging Face token not set (HUGGINGFACE_API_TOKEN or HUGGINGFACEHUB_API_TOKEN); skipping embedding.")
            return None
        try:
            try:
                timeout = float(os.environ.get("HUGGINGFACE_TIMEOUT_SECS", 3))
            except Exception:
                timeout = 3.0
            hf_client = InferenceClient(model=model, token=hf_token, timeout=timeout)
            embedding = hf_client.feature_extraction(text)
            if embedding is None:
                print("Hugging Face embedding returned None.")
                return None
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            if not isinstance(embedding, list) or len(embedding) == 0:
                print("Hugging Face embedding returned empty or invalid response.")
                return None
            return embedding
        except Exception as e:
            print(f"Hugging Face embedding request failed for model '{model}': {e}")
            return None
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set; skipping embedding.")
            return None
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        payload = {
            "model": model,
            "content": {"parts": [{"text": text}]},
        }
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=15)
            if not resp.ok:
                print(f"Gemini embedding failed ({resp.status_code}): {resp.text}")
                return None
            data = resp.json()
            embedding = data.get("embedding", {}).get("values") or data.get("embedding", {}).get("value")
            if not embedding:
                print("Gemini embedding missing values.")
                return None
            return embedding
        except Exception as e:
            print(f"Gemini embedding request failed: {e}")
            return None
    # Default: Groq embeddings
    try:
        resp = client.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding
    except Exception as e:
        print(f"Embedding request failed for model '{model}': {e}")
        return None


def _call_brief_chat_completion(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 120,
    debug: bool = False,
) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        if debug:
            print(f"Brief chat completion failed: {e}")
        return None


def _summarize_card_text(client: Groq, model: str, combined_text: str, debug: bool = False) -> str:
    system_prompt = (
        "You are a concise memory consolidator. Summarize the combined user+assistant turn into 1-2 sentences "
        "with only the new facts or decisions. Keep it short and neutral."
    )
    summary = _call_brief_chat_completion(
        client,
        model,
        system_prompt,
        combined_text,
        temperature=0.2,
        max_tokens=160,
        debug=debug,
    )
    return summary or combined_text[:320]


def _merge_card_summary(
    client: Groq,
    model: str,
    existing_summary: str,
    new_text: str,
    debug: bool = False,
) -> str:
    system_prompt = (
        "You maintain a concise memory card. Update the existing summary with ONLY new facts from the new turn. "
        "Keep it one short paragraph or bullet list, no fluff, no loss of prior facts."
    )
    user_prompt = f"Existing summary:\n{existing_summary}\n\nNew turn:\n{new_text}\n\nUpdated summary:"
    merged = _call_brief_chat_completion(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=200,
        debug=debug,
    )
    return merged or f"{existing_summary}\n- {new_text[:400]}"


def _reconcile_contradiction_summary(
    client: Groq,
    model: str,
    existing_summary: str,
    new_text: str,
    debug: bool = False,
) -> str:
    system_prompt = (
        "You are reconciling conflicting facts. Rewrite a single concise summary that preserves truth, "
        "integrates corrections, and drops superseded statements. No fluff."
    )
    user_prompt = f"Existing summary:\n{existing_summary}\n\nNew turn (may contradict):\n{new_text}\n\nReconciled summary:"
    reconciled = _call_brief_chat_completion(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=240,
        debug=debug,
    )
    return reconciled or f"{existing_summary}\nUpdated: {new_text[:400]}"


def _title_for_card(client: Groq, model: str, combined_text: str, debug: bool = False) -> str:
    system_prompt = "Generate a short 3-6 word title for this memory card. No quotes."
    title = _call_brief_chat_completion(
        client,
        model,
        system_prompt,
        combined_text,
        temperature=0.3,
        max_tokens=12,
        debug=debug,
    )
    return title or "Memory"


def _canonicalize_entities_llm(
    client: Groq,
    model: str,
    spans: Sequence[str],
    *,
    user_query: str,
    agent_response: str,
    engram_summary: str,
    debug: bool = False,
) -> Dict:
    taxonomy = [
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
    ]

    def _fallback(items: Sequence[str], reason: str) -> Dict:
        if debug:
            print(f"Canonicalization helper fallback ({reason}); using legacy spans.")
        results: List[Dict] = []
        for raw in items:
            if not isinstance(raw, str):
                continue
            cleaned = raw.strip()
            if not cleaned:
                continue
            canonical = cleaned.replace("\\", "/").strip(".").strip()
            canonical = re.sub(r"\s+", " ", canonical)
            canonical = re.sub(r"^\\./", "", canonical)
            canonical = canonical.lower()
            results.append(
                {
                    "span": cleaned,
                    "type": "concept",
                    "canonical_name": canonical,
                    "confidence": None,
                }
            )
        return {"entities": results, "meta": {"fallback_reason": reason, "success": False, "latency_ms": 0}}

    span_list = [s.strip() for s in spans if isinstance(s, str) and s.strip()]
    span_list = span_list[:7]
    if not span_list:
        return {"entities": [], "meta": {"fallback_reason": "no_spans", "success": False, "latency_ms": 0}}

    system_prompt = (
        "You are an entity canonicalization system. Given spans plus conversation context, "
        "assign a type from the taxonomy and a disambiguated canonical_name for each span. "
        "Use self_* types for first-person signals (identity, location, work, health, relationships, preferences, projects). "
        "Return only JSON; no explanations."
    )
    user_prompt = (
        "Taxonomy:\n"
        "- org: organizations, companies, institutions\n"
        "- person: individuals, roles, personas\n"
        "- location: physical and digital locations\n"
        "- product: goods, services, offerings\n"
        "- tech: technologies, tools, systems\n"
        "- domain: business domains, industries\n"
        "- concept: abstract ideas, theories\n"
        "- event: occurrences, meetings\n"
        "- document: files, reports, sources\n"
        "- relation: partnerships, hierarchies\n"
        "- action: commands, operations\n"
        "- self_identity: first-person identity/name cues\n"
        "- self_location: first-person location/residence cues\n"
        "- self_work: first-person work/job/project cues\n"
        "- self_health: first-person health/medical cues\n"
        "- self_relationships: first-person family/friend/relationship cues\n"
        "- self_preferences: first-person likes/preferences\n"
        "- self_projects: first-person project ownership cues\n\n"
        f"Context:\nUser: {user_query}\nAgent: {agent_response}\nSummary: {engram_summary}\n\n"
        f"Spans to canonicalize: {span_list}\n"
        "Return an array of objects following the schema."
    )
    schema = {
        "type": "array",
        "maxItems": 7,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "span": {"type": "string"},
                "type": {"type": "string", "enum": taxonomy},
                "canonical_name": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["span", "type", "canonical_name"],
        },
    }

    start = time.time()
    meta = {"fallback_reason": None, "success": True, "latency_ms": None}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "canonical_entities", "schema": schema},
            },
        )
        content = resp.choices[0].message.content if resp and resp.choices else None
        raw = (content or "").strip()
        if not raw:
            meta.update({"fallback_reason": "empty response", "success": False})
            return _fallback(span_list, "empty response")
        data = json.loads(raw)
    except Exception as e:
        if debug:
            print(f"Canonicalization helper failed: {e}")
        meta.update({"fallback_reason": "helper error", "success": False})
        return _fallback(span_list, "helper error")

    if not isinstance(data, list):
        meta.update({"fallback_reason": "non-list response", "success": False})
        return _fallback(span_list, "non-list response")

    results: List[Dict] = []
    for item in data[:7]:
        if not isinstance(item, dict):
            continue
        span_val = item.get("span")
        type_val = item.get("type")
        canon_val = item.get("canonical_name")
        conf_val = item.get("confidence")
        if not isinstance(span_val, str) or not span_val.strip():
            continue
        if not isinstance(type_val, str) or type_val not in taxonomy:
            continue
        if not isinstance(canon_val, str) or not canon_val.strip():
            continue
        cleaned_span = span_val.strip()
        cleaned_canon = canon_val.strip()
        confidence = None
        if isinstance(conf_val, (int, float)):
            try:
                confidence = float(conf_val)
            except Exception:
                confidence = None
        results.append(
            {
                "span": cleaned_span,
                "type": type_val,
                "canonical_name": cleaned_canon,
                "confidence": confidence,
            }
        )

    meta["latency_ms"] = int((time.time() - start) * 1000)
    if not results:
        meta.update({"fallback_reason": "validation produced empty set", "success": False})
        return _fallback(span_list, "validation produced empty set")
    return {"entities": results, "meta": meta}


def _assign_beacons_llm(
    client: Groq,
    model: str,
    summary: str,
    canonical_entities: Sequence[str],
    beacon_registry: Dict[str, Dict],
    current_beacons: Sequence[str],
    debug: bool = False,
) -> Dict[str, List[str]]:
    if not summary or not summary.strip():
        return {"existing": [], "proposed_new": []}

    registry_items = list(beacon_registry.values())
    ranked = sorted(registry_items, key=lambda x: (x.get("strength", 0), x.get("card_count", 0)), reverse=True)
    registry_lines = "\n".join(
        f"- {item.get('beacon_id')} (strength {item.get('strength', 0):.2f}, cards {item.get('card_count', 0)})"
        for item in ranked
    ) or "None"

    entities_text = ", ".join(canonical_entities) if canonical_entities else "None"
    current_beacons_text = ", ".join(current_beacons) if current_beacons else "None"
    system_prompt = (
        "You classify memory summaries into topic beacons.\n"
        "- Choose existing beacon_ids ONLY from the provided registry list.\n"
        "- Return exactly one best existing beacon OR one proposed_new beacon (mutually exclusive). Never return more than one total.\n"
        "- proposed_new is only for novel themes not covered by the registry.\n"
        "- Respond strictly in JSON using the provided schema. No explanations."
    )
    user_prompt = (
        f"Engram summary:\n{summary.strip()}\n\n"
        f"Canonical entities: {entities_text}\n\n"
        f"Existing beacon list on this card (if any): {current_beacons_text}\n\n"
        f"Beacon registry (id, strength, card_count):\n{registry_lines}\n\n"
        "Return JSON now."
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "existing": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 1,
            },
            "proposed_new": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 1,
            },
        },
        "required": ["existing", "proposed_new"],
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=220,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "beacon_assignment",
                    "schema": schema,
                },
            },
        )
        content = resp.choices[0].message.content if resp and resp.choices else None
        cleaned = (content or "").strip()
        if not cleaned:
            return {"existing": [], "proposed_new": []}
        data = json.loads(cleaned)
    except Exception as e:
        if debug:
            print(f"Beacon assigner structured call failed or parse error: {e}")
        return {"existing": [], "proposed_new": []}

    existing_raw = data.get("existing") if isinstance(data, dict) else []
    proposed_raw = data.get("proposed_new") if isinstance(data, dict) else []

    existing: List[str] = []
    if isinstance(existing_raw, list):
        for b in existing_raw:
            if not isinstance(b, str):
                continue
            bid = b.strip()
            if not bid or bid not in beacon_registry:
                continue
            existing.append(bid)
            break

    if isinstance(proposed_raw, str):
        proposed_raw = [proposed_raw]
    proposed: List[str] = []
    if isinstance(proposed_raw, list) and not existing:
        for cand in proposed_raw:
            if not isinstance(cand, str):
                continue
            cid = cand.strip()
            if not cid:
                continue
            proposed.append(cid)
            break

    return {"existing": existing, "proposed_new": proposed}
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        if debug:
            print(f"Beacon assigner parse failed: {cleaned}")
        return {"existing": [], "proposed_new": []}

    existing_raw = data.get("existing") if isinstance(data, dict) else []
    proposed_raw = data.get("proposed_new") if isinstance(data, dict) else []

    existing: List[str] = []
    if isinstance(existing_raw, list):
        for b in existing_raw:
            if not isinstance(b, str):
                continue
            bid = b.strip()
            if not bid or bid not in beacon_registry:
                continue
            if bid in existing:
                continue
            existing.append(bid)
            if len(existing) >= 3:
                break

    if isinstance(proposed_raw, str):
        proposed_raw = [proposed_raw]
    proposed: List[str] = []
    if isinstance(proposed_raw, list):
        for cand in proposed_raw:
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

    return {"existing": existing, "proposed_new": proposed}



# Code-Agent Orchestrator
def code_agent_orchestrator():
    memory_cfg = config.get("memory", {})
    memory_enabled = memory_cfg.get("enabled", True)
    memory_top_k = int(memory_cfg.get("top_k", 5))
    memory_threshold = float(memory_cfg.get("similarity_threshold", 0.2))
    memory_half_life = memory_cfg.get("half_life_hours", 48)
    memory_max_chars = int(memory_cfg.get("max_memory_chars", 4000))
    embedding_choice = memory_cfg.get("embedding_choice")
    embedding_model = memory_cfg.get("embedding_model")
    embedding_provider = memory_cfg.get("embedding_provider")
    memory_card_k = int(memory_cfg.get("card_k", memory_top_k or 4))
    memory_vector_threshold = float(memory_cfg.get("vector_threshold", 0.38))
    memory_recent_turn_limit = int(memory_cfg.get("recent_turn_limit", 5))
    memory_merge_threshold = float(memory_cfg.get("merge_threshold", 0.75))
    memory_duplicate_threshold = float(memory_cfg.get("duplicate_threshold", 0.92))
    memory_card_embedding_strategy = memory_cfg.get(
        "card_embedding_strategy", "blend"
    )
    beacon_discovery_interval = int(memory_cfg.get("beacon_discovery_interval", 100))
    beacon_prune_interval = int(memory_cfg.get("beacon_prune_interval", 500))
    beacon_prune_max_age_days = int(memory_cfg.get("beacon_prune_max_age_days", 180))
    beacon_prune_strength_floor = float(memory_cfg.get("beacon_prune_strength_floor", 0.5))
    card_prune_interval = int(memory_cfg.get("card_prune_interval", 500))
    card_prune_min_access = int(memory_cfg.get("card_prune_min_access", 1))
    card_prune_max_age_days = int(memory_cfg.get("card_prune_max_age_days", 180))
    card_prune_keep_recent = int(memory_cfg.get("card_prune_keep_recent", 10))

    # Allow a simple selector to pick a known embedding setup
    if embedding_choice:
        choice = str(embedding_choice).lower()
        if choice == "bge":
            embedding_model = "BAAI/bge-m3"
            embedding_provider = "huggingface"
        elif choice == "gemini":
            embedding_model = "models/gemini-embedding-001"
            embedding_provider = "gemini"

    def _to_serializable(obj):
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                return str(obj)
        if isinstance(obj, bytes):
            return obj.decode(errors="replace")
        return str(obj)
    # Ensure workspace exists at startup
    ensure_workspace()

    # Get current system information
    system_info = get_system_info()
    print(f"System Info: {system_info}")

    api_key = os.environ.get('GROQ_API_KEY')
    base_url_env = os.environ.get('GROQ_BASE_URL')
    base_url = None
    if base_url_env:
        # Prevent double /openai/v1/ in requests; Groq client appends the path internally
        cleaned = base_url_env.rstrip("/")
        if cleaned.endswith("/openai/v1"):
            cleaned = cleaned[: -len("/openai/v1")]
        base_url = cleaned or None
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return

    client = Groq(api_key=api_key, base_url=base_url)
    model = os.environ.get("GROQ_MODEL") or 'openai/gpt-oss-120b'
    helper_model = os.environ.get("GROQ_HELPER_MODEL") or 'openai/gpt-oss-20b'
    memory_db_path = Path(memory_cfg.get("db_path", "persistent_mem/memory.db"))
    memory_store = MemoryStore(memory_db_path) if memory_enabled else None
    conversation_id = str(uuid.uuid4())
    turn_counter = 0

    print("=== Multi-Tool Code-Agent Demo ===")
    print(f"Available tools: {', '.join(tools_descriptions.keys())}")
    print("Type 'quit' to exit.\n")

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == 'quit':
            break

        retrieved_cards = []
        recent_raw_turns = []
        memory_block = "Memory disabled."
        query_embedding = None
        if memory_enabled and memory_store:
            query_embedding = _embed_text(client, embedding_model, user_query, provider=embedding_provider)
            if query_embedding:
                ctx = memory_store.retrieve_context(
                    query_text=user_query,
                    query_embedding=query_embedding,
                    card_k=memory_card_k,
                    vector_threshold=memory_vector_threshold,
                    recent_turn_limit=memory_recent_turn_limit,
                    canonicalization_fn=lambda spans, user_text, assistant_text, summary_text: _canonicalize_entities_llm(
                        client,
                        helper_model,
                        spans,
                        user_query=user_text,
                        agent_response=assistant_text,
                        engram_summary=summary_text,
                        debug=debug,
                    ),
                )
                retrieved_cards = ctx.get("cards") or []
                recent_raw_turns = ctx.get("recent_raw_turns") or []
                memory_block = memory_store.format_context(retrieved_cards, recent_raw_turns, max_chars=memory_max_chars)
            else:
                memory_block = "Memory retrieval skipped (embedding failed)."

        # Build system content (lean: principles, guidance, tool hints, and system info)
        tools_list = "\n".join(f"{i+1}) {name}: {desc}" for i, (name, desc) in enumerate(tools_descriptions.items()))
        system_content = (
            "You are an agentic, helpful AI assistant with perfect long-term memory.\n"
            "You have access to RETRIEVED MEMORIES in every user message:\n"
            "\n"
            "### RESPONSE STYLE\n"
            "- Answer general questions (definitions, math, facts) in 1 to 4 short sentences.\n"
            "- Only be verbose if the user explicitly asks for detail ('explain everything', 'in depth', etc.).\n"
            "- Stay friendly and concise — we have been talking for weeks.\n"
            "\n"
            "### LONG-TERM MEMORY (consolidated, timeless facts — trust these completely)\n"
            "→ These are permanent, entity-grouped memory cards (people, files, projects, facts).\n"
            "→ They are always correct and up-to-date.\n"
            "→ Example: a card titled \"123.csv contents\" means you KNOW exactly what is inside that file — no need to read it again unless you suspect it changed.\n"
            "\n"
            "### SHORT-TERM MEMORY (recent conversation)\n"
            "→ Last few exact exchanges — use only for immediate references (\"that file\", \"the previous command\").\n"
            "\n"
            "### Human-like behavior when there is zero relevant memory\n"
            "# Exact-match zero memory\n"
            "\"I don’t recall us ever talking about that.\"\n"
            "\"That name/file doesn’t ring a bell.\"\n"
            "\"I’ve got no memory of that at all.\"\n"
            "\n"
            "# Vague follow-up with no anchor\n"
            "\"Sorry, I lost you — which one do you mean?\"\n"
            "\"Which Charles/file/project are we talking about?\"\n"
            "\n"
            "# After admission → proactive\n"
            "\"Want me to read the file / search / check something?\"\n"
            "\"Tell me more and I’ll remember it from now on.\"\n"
            "### Golden Rules\n"
            "- ALWAYS assume consolidated memory cards are 100% accurate for static data (file contents, names, ages, capitals, past decisions, etc.).\n"
            "- NEVER re-ask or re-tool for something already in a memory card unless the data is explicitly dynamic (weather, time, stock prices).\n"
            "- If the user refers to something by name (\"Charles\", \"123.csv\", \"Arlington\", \"the startup\"), instantly recall the relevant card — you already know everything about it.\n"
            "- When answering, behave as if you have known these facts for weeks — no \"from memory\" hedging.\n"
            "- Never hallucinate a memory that does not exist.\n"
            "- Never say “according to my memory” when the memory is empty.\n"
            "- Always admit the gap in one short, friendly sentence — then move forward.\n"
            "- Answer math, definitions, and general knowledge confidently even if not in memory.\n"
            "- Only admit memory gaps when the user is clearly referring to past conversation."
            "\n"
            "Tools (use only when truly needed):\n"
            f"{tools_list}\n"
            "\n"
            "System Context:\n"
            f" {json.dumps(system_info)}\n"
        )

        # Build user content with retrieved memories (no session buffer carryover)
        user_content = (
            f"### RETRIEVED MEMORIES (Past interactions)\n"
            f"{memory_block}\n\n"
            f"### CURRENT USER QUERY\n"
            f"{user_query}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        required_args = {
            "list_files": [],
            "read_file": ["file_path"],
            "write_file": ["file_path", "content"],
            "run_shell_pipeline": [],
            "get_system_info_inxi": [],
        }

        max_iterations = 10
        iteration = 0
        final_ai_response = None

        print(f"\n--- Processing cycle for user_query: '{user_query}' ---")

        # Initial request
        if debug: print("RAW PROMPT (initial):", json.dumps(messages, indent=2, default=_to_serializable))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.4,
        )
        try:
            raw_response = response.model_dump()
        except Exception:
            raw_response = str(response)
        if debug:
            if isinstance(raw_response, str):
                print(f"RAW RESPONSE (initial): {raw_response}")
            else:
                print("RAW RESPONSE (initial):", json.dumps(raw_response, indent=2, default=_to_serializable))

        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            messages.append(response.choices[0].message)

            if debug: print(f"\nIteration {iteration}:")

            response_message = response.choices[0].message
            if debug: print(f"AI Response: {response_message.content or 'No direct response'}")

            if response_message.tool_calls:
                if debug: print(f"AI decided to call {len(response_message.tool_calls)} tool(s):")
                for i, tool_call in enumerate(response_message.tool_calls, 1):
                    if debug: print(f"  {i}. {tool_call.function.name} with args: {tool_call.function.arguments}")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    error_payload = build_response(
                        tool=function_name,
                        success=False,
                        error_type="parse_error",
                        message=f"Invalid JSON arguments: {e}",
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_payload,
                    })
                    if debug: print(f"Argument parse error for {function_name}: {e}")
                    continue

                if function_name not in available_functions:
                    error_payload = build_response(
                        tool=function_name,
                        success=False,
                        error_type="unknown_tool",
                        message=f"Tool '{function_name}' is not available.",
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_payload,
                    })
                    if debug: print(f"Unknown tool requested: {function_name}")
                    continue

                if not isinstance(function_args, dict):
                    error_payload = build_response(
                        tool=function_name,
                        success=False,
                        error_type="validation_error",
                        message="Tool arguments must be a JSON object.",
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_payload,
                    })
                    if debug: print(f"Validation error for {function_name}: args not an object")
                    continue

                missing = [k for k in required_args.get(function_name, []) if k not in function_args]
                if function_name == "run_shell_pipeline":
                    has_pipeline = bool(function_args.get("pipeline"))
                    has_lines = function_args.get("pipeline_lines") is not None and isinstance(function_args.get("pipeline_lines"), list) and len(function_args.get("pipeline_lines")) > 0
                    if not has_pipeline and not has_lines:
                        error_payload = build_response(
                            tool=function_name,
                            success=False,
                            error_type="validation_error",
                            message="pipeline or pipeline_lines is required",
                            details={"missing": ["pipeline or pipeline_lines"]},
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": error_payload,
                        })
                        if debug: print(f"Validation error for {function_name}: missing pipeline or pipeline_lines")
                        continue
                elif missing:
                    error_payload = build_response(
                        tool=function_name,
                        success=False,
                        error_type="validation_error",
                        message=f"Missing required arguments: {', '.join(missing)}",
                        details={"missing_args": missing},
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_payload,
                    })
                    if debug: print(f"Validation error for {function_name}: missing {missing}")
                    continue

                if debug: print(f"\nExecuting {function_name}({function_args})...")
                try:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    if debug: print(f"Tool Result: {function_response}")
                except Exception as e:
                    function_response = build_response(
                        tool=function_name,
                        success=False,
                        error_type="execution_error",
                        message=str(e),
                    )
                    if debug: print(f"Execution error for {function_name}: {e}")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                }
                messages.append(tool_msg)

            # Next turn with tool results
            if debug: print("RAW PROMPT (iteration):", json.dumps([msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages], indent=2, default=_to_serializable))
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.4,
            )
            try:
                raw_response = response.model_dump()
            except Exception:
                raw_response = str(response)
            if debug:
                if isinstance(raw_response, str):
                    print(f"RAW RESPONSE (iteration): {raw_response}")
                else:
                    print("RAW RESPONSE (iteration):", json.dumps(raw_response, indent=2, default=_to_serializable))

        if iteration >= max_iterations:
            if debug: print("Max iterations reached.")
            final_ai_response = "Max iterations reached, no final response."
        else:
            final_ai_response = response.choices[0].message.content or "No final response"

        print(f"Assistant: {final_ai_response}")

        # Persist memories (user query + final response as a paired turn) if enabled
        if memory_enabled and memory_store:
            # Filter out trivial queries from being stored as "useful" memory
            trivial_queries = {"quit", "exit", "hey", "hi", "hello", "thanks", "thank you", "ok", "okay", "cool"}
            is_trivial = user_query.strip().lower() in trivial_queries
            
            if not is_trivial:
                turn_counter += 1
                combined_embedding = None
                combined_text = f"User: {user_query}\nAssistant: {final_ai_response or ''}"
                combined_embedding = _embed_text(client, embedding_model, combined_text, provider=embedding_provider)

                if combined_embedding:
                    memory_store.store_and_consolidate_turn(
                        user_content=user_query,
                        assistant_content=final_ai_response or "",
                        combined_embedding=combined_embedding,
                        conversation_id=conversation_id,
                        duplicate_threshold=memory_duplicate_threshold,
                        merge_threshold=memory_merge_threshold,
                        embedding_strategy=memory_card_embedding_strategy,
                        embedding_fn=lambda text: _embed_text(
                            client, embedding_model, text, provider=embedding_provider
                        ),
                        summary_fn=lambda text: _summarize_card_text(
                            client, helper_model, text, debug=debug
                        ),
                        merge_summary_fn=lambda existing, new: _merge_card_summary(
                            client, helper_model, existing, new, debug=debug
                        ),
                        title_fn=lambda text: _title_for_card(
                            client, helper_model, text, debug=debug
                        ),
                        contradiction_fn=lambda existing, new: _reconcile_contradiction_summary(
                            client, helper_model, existing, new, debug=debug
                        ),
                        canonicalization_fn=lambda spans, user_text, assistant_text, summary_text: _canonicalize_entities_llm(
                            client,
                            helper_model,
                            spans,
                            user_query=user_text,
                            agent_response=assistant_text,
                            engram_summary=summary_text,
                            debug=debug,
                        ),
                        beacon_assignment_fn=lambda summary, entities_canonical, registry, current_beacons: _assign_beacons_llm(
                            client,
                            helper_model,
                            summary,
                            entities_canonical,
                            registry,
                            current_beacons,
                            debug=debug,
                        ),
                    )
                else:
                    if debug: print("Embeddings missing; turn not stored.")

                # Periodic beacon maintenance
                try:
                    if beacon_discovery_interval > 0 and turn_counter % beacon_discovery_interval == 0:
                        promoted = memory_store.beacon_discovery_job()
                        if debug: print(f"Beacon discovery promoted: {promoted}")
                    if beacon_prune_interval > 0 and turn_counter % beacon_prune_interval == 0:
                        pruned = memory_store.prune_stale_beacons(
                            max_age_days=beacon_prune_max_age_days,
                            strength_floor=beacon_prune_strength_floor,
                        )
                        if debug: print(f"Beacon prune removed: {pruned}")
                    if card_prune_interval > 0 and turn_counter % card_prune_interval == 0:
                        pruned_cards = memory_store.prune_stale_cards(
                            min_access_count=card_prune_min_access,
                            max_age_days=card_prune_max_age_days,
                            keep_recent_n=card_prune_keep_recent,
                        )
                        if debug: print(f"Card prune removed: {pruned_cards}")
                except Exception as e:
                    if debug: print(f"Beacon maintenance failed: {e}")
            elif debug:
                print("Trivial query; not storing in memory.")

        if debug: print("\n" + "="*50)

if __name__ == "__main__":
    code_agent_orchestrator()
