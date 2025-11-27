import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

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
        "card_embedding_strategy", "reembed_summary"
    )

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
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return

    client = Groq(api_key=api_key)
    model = 'openai/gpt-oss-120b'
    memory_db_path = Path(memory_cfg.get("db_path", "memory.db"))
    memory_store = MemoryStore(memory_db_path) if memory_enabled else None
    conversation_id = str(uuid.uuid4())

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
                )
                retrieved_cards = ctx.get("cards") or []
                recent_raw_turns = ctx.get("recent_raw_turns") or []
                memory_block = memory_store.format_context(retrieved_cards, recent_raw_turns, max_chars=memory_max_chars)
            else:
                memory_block = "Memory retrieval skipped (embedding failed)."

        # Build system content (lean: principles, guidance, tool hints, and system info)
        tools_list = "\n".join(f"{i+1}) {name}: {desc}" for i, (name, desc) in enumerate(tools_descriptions.items()))
        system_content = (
            "You are an agentic, helpful and friendly AI assistant.\n"
            "\n"
            "- You have access to 'RETRIEVED MEMORIES' in the user message:\n"
            "  - Consolidated memory cards: long-term facts grouped by entity/topic.\n"
            "  - Recent raw turns: last few user/assistant exchanges for short-term context.\n"
            "- Trust consolidated cards for stable facts; use recent turns for immediate follow-ups like “do that”.\n"
            "- Timestamps show freshness; cards are timeless unless stated otherwise.\n"
            "Trust vs. Verify:\n"
            "   - ALWAYS TRUST high-score, recent file content from memory, recent memories for static data.\n"
            "   - VERIFY with tools if data is dynamic (content varies continually), stale, or you are unsure.\n"
            "If you need to use a tool, make sure to call it with the correct parameters.\n"
            f"Tools:\n{tools_list}\n"
            f"System Context: {json.dumps(system_info)}\n"
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
                combined_embedding = None
                combined_text = f"<user_query>{user_query}</user_query>\n<agent_response>{final_ai_response or ''}</agent_response>"
                combined_embedding = _embed_text(client, embedding_model, combined_text, provider=embedding_provider)

                if combined_embedding:
                    try:
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
                                client, model, text, debug=debug
                            ),
                            merge_summary_fn=lambda existing, new: _merge_card_summary(
                                client, model, existing, new, debug=debug
                            ),
                            title_fn=lambda text: _title_for_card(
                                client, model, text, debug=debug
                            ),
                        )
                    except Exception as e:
                        if debug: print(f"Failed to store turn: {e}")
                else:
                    if debug: print("Embeddings missing; turn not stored.")
            elif debug:
                print("Trivial query; not storing in memory.")

        if debug: print("\n" + "="*50)

if __name__ == "__main__":
    code_agent_orchestrator()
