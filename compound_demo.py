import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client with latest model version
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    default_headers={
        "Groq-Model-Version": "latest"
    }
)

# Loop to get user requests
while True:
    user_input = input("Enter your query (or 'quit'/'exit' to stop): ")
    if user_input.lower() in ['quit', 'exit']:
        print("Exiting...")
        break

    # Create a chat completion using the compound-mini model
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="groq/compound-mini",
    )

    # Print the response content
    print("Response:", completion.choices[0].message.content)
    print()  # Add a blank line for readability