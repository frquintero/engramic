import json
import os
from groq import Groq
from tools import get_system_info, available_functions, tools, tools_descriptions
from workspace_executor import ensure_workspace



# Code-Agent Orchestrator
def code_agent_orchestrator():
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
    print("Available tools: list_files, git_status, git_add, git_add_all, git_commit, git_log, awk_process, read_file, write_file, get_cwd, run_shell_pipeline")
    print("Type 'quit' to exit.\n")

    # Inter-user context for continuity across cycles
    inter_user_context = []  # List of (user_query, final_ai_response) tuples

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == 'quit':
            break

        # Build system content (lean: principles, guidance, tool hints, and system info)
        system_content = (
            "You are a helpful assistant with access to various tools. Use tools when appropriate to assist with file operations, "
            "git management, text processing, system queries, and running shell pipelines inside the workspace. "
            "Tool Capabilities: list_files (directory listing; set detailed/show_hidden when needed), "
            "git_status/git_add/git_add_all/git_commit/git_log (repo management), "
            "awk_process (text processing with customizable separators), read_file/write_file (file I/O), "
            "get_cwd (current directory), run_shell_pipeline (execute full shell pipelines; params: pipeline [required], cwd, timeout_secs, "
            "max_output_chars, env, mode full|constrained). Use exact tool names and parameters. "
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

        max_iterations = 10
        iteration = 0
        final_ai_response = None

        print(f"\n--- Processing cycle for user_query: '{user_query}' ---")

        while iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}:")

            # Limit context to system + last 4 messages to avoid token limits
            if len(messages) > 5:
                messages = [messages[0]] + messages[-4:]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                print(f"API Error: {e}")
                break

            response_message = response.choices[0].message
            print(f"AI Response: {response_message.content or 'No direct response'}")

            if response_message.tool_calls:
                print(f"AI decided to call {len(response_message.tool_calls)} tool(s):")
                for i, tool_call in enumerate(response_message.tool_calls, 1):
                    print(f"  {i}. {tool_call.function.name} with args: {tool_call.function.arguments}")

            messages.append(response_message)

            if not response_message.tool_calls:
                final_ai_response = response_message.content or "No final response"
                print("No more tools needed. Final answer received.")
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"\nExecuting {function_name}({function_args})...")

                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                print(f"Tool Result: {function_response}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })

        if iteration >= max_iterations:
            print("Max iterations reached.")
            final_ai_response = "Max iterations reached, no final response."

        # Append to inter-user context if we have a final response
        if final_ai_response:
            inter_user_context.append((user_query, final_ai_response))
            # Keep only last 5
            if len(inter_user_context) > 5:
                inter_user_context = inter_user_context[-5:]

        print("\n" + "="*50)

if __name__ == "__main__":
    code_agent_orchestrator()
