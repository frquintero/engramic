import json
import os
from groq import Groq
from tools import get_system_info, available_functions, tools, tools_descriptions, build_response
from workspace_executor import ensure_workspace

# Load configuration
try:
    with open('config.json') as f:
        config = json.load(f)
    debug = config.get('debug', False)
except FileNotFoundError:
    debug = False



# Code-Agent Orchestrator
def code_agent_orchestrator():
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

    print("=== Multi-Tool Code-Agent Demo ===")
    print(f"Available tools: {', '.join(tools_descriptions.keys())}")
    print("Type 'quit' to exit.\n")

    # Inter-user context for continuity across cycles
    inter_user_context = []  # List of (user_query, final_ai_response) tuples

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == 'quit':
            break

        # Build system content (lean: principles, guidance, tool hints, and system info)
        tools_list = "\n".join(f"{i+1}) {name}: {desc}" for i, (name, desc) in enumerate(tools_descriptions.items()))
        system_content = (
            "You are a helpful assistant with access to these tools. Use exact tool names and parameters.\n"
            "Workspace rules: stay within the workspace (code_agent_workspace). Avoid interactive commands/shells; avoid destructive operations (rm -rf, chmod/chown)\n"
            "Return concise, structured results; if a tool fails, rely on the tool's JSON error payload rather than retrying the same invalid call.\n"
            f"Tools:\n{tools_list}\n"
            f"System Context: {json.dumps(system_info)}"
        )

        # Build user content with inter-user context
        history_summaries = ""
        if inter_user_context:
            history_summaries = "Previous Interactions:\n" + "\n".join(
                f"Query: '{q}'. Response: '{r}'." for q, r in inter_user_context[-5:]
            ) + "\n\n"

        user_content = f"{history_summaries}Current Query: {user_query}"

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

        # Append to inter-user context if we have a final response
        if final_ai_response:
            inter_user_context.append((user_query, final_ai_response))
            # Keep only last 5
            if len(inter_user_context) > 5:
                inter_user_context = inter_user_context[-5:]

        if debug: print("\n" + "="*50)

if __name__ == "__main__":
    code_agent_orchestrator()
