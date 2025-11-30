import json
import os

from groq import Groq

api_key = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=api_key)

# ============================================================================
# Tool Implementations
# ============================================================================


def calculate(expression: str) -> str:
    """Evaluate a basic mathematical expression"""
    try:
        result = eval(expression)  # Use safe evaluation in production!
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate_compound_interest(
    principal: float, rate: float, time: float, compounds_per_year: int = 12
) -> str:
    """Calculate compound interest on an investment"""
    amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
    interest = amount - principal
    return json.dumps(
        {
            "principal": principal,
            "total_amount": round(amount, 2),
            "interest_earned": round(interest, 2),
        }
    )


def calculate_percentage(number: float, percentage: float) -> str:
    """Calculate what percentage of a number equals"""
    result = (percentage / 100) * number
    return json.dumps({"result": round(result, 2)})


# Function registry
available_functions = {
    "calculate": calculate,
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_percentage": calculate_percentage,
}

# ============================================================================
# Tool Schemas
# ============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression like '25 * 4 + 10' or '(100 - 50) / 2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_compound_interest",
            "description": "Calculate compound interest on an investment",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "The initial investment amount",
                    },
                    "rate": {
                        "type": "number",
                        "description": "The annual interest rate as a decimal (e.g., 0.05 for 5%)",
                    },
                    "time": {
                        "type": "number",
                        "description": "The time period in years",
                    },
                    "compounds_per_year": {
                        "type": "integer",
                        "description": "Number of times interest compounds per year (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["principal", "rate", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": "Calculate what a percentage of a number equals",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number", "description": "The base number"},
                    "percentage": {
                        "type": "number",
                        "description": "The percentage to calculate",
                    },
                },
                "required": ["number", "percentage"],
            },
        },
    },
]

# ============================================================================
# Agentic Loop with Multi-Tool Support
# ============================================================================

user_query = """I'm investing $10,000 at 5% annual interest for 10 years, 
compounded monthly. After 10 years, I want to withdraw 25% for a down payment. 
How much will my down payment be, and how much will remain invested?"""

messages = [
    {
        "role": "system",
        "content": "You are a financial calculator assistant. Use the provided tools to help with calculations.",
    },
    {"role": "user", "content": user_query},
]

print(f"User: {user_query}\n")

# Initial request
print("Raw prompt (initial):", json.dumps([msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages], indent=2))
response = client.chat.completions.create(
    model="openai/gpt-oss-120b", messages=messages, tools=tools, tool_choice="auto"
)
print("Raw response (initial):", json.dumps(response.model_dump(), indent=2))

# Multi-turn loop: Continue while model requests tool calls
max_iterations = 10
iteration = 0

while response.choices[0].message.tool_calls and iteration < max_iterations:
    iteration += 1
    messages.append(response.choices[0].message)

    print(
        f"Iteration {iteration}: Model called {len(response.choices[0].message.tool_calls)} tool(s)"
    )

    # Handle all tool calls from this turn
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"  → {function_name}({function_args})")

        # Execute the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(f"    ← {function_response}")

        # Add tool result to conversation
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response,
            }
        )

    # Next turn with tool results
    print(f"Raw prompt (iteration {iteration}):", json.dumps([msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages], indent=2))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(f"Raw response (iteration {iteration}):", json.dumps(response.model_dump(), indent=2))
    print()

# Final answer
print(f"Assistant: {response.choices[0].message.content}")

# Expected output:
# Iteration 1: Model called 1 tool(s)
#   → calculate_compound_interest({'principal': 10000, 'rate': 0.05, 'time': 10, 'compounds_per_year': 12})
#     ← {"principal": 10000, "total_amount": 16470.09, "interest_earned": 6470.09}
#
# Iteration 2: Model called 1 tool(s)
#   → calculate_percentage({'number': 16470.09, 'percentage': 25})
#     ← {"result": 4117.52}
#
# Iteration 3: Model called 1 tool(s)
#   → calculate({'expression': '16470.09 - 4117.52'})
#     ← {"result": 12352.57}
#
# Assistant: After 10 years, your $10,000 investment at 5% annual interest compounded monthly
# will grow to $16,470.09. Your 25% down payment will be $4,117.52, and you'll have $12,352.57
# remaining invested.
