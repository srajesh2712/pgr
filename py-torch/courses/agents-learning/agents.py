import json
import os
from openai import OpenAI

# Initialize client pointing to LM Studio's local server
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # LM Studio doesn't enforce key validation locally
)

# ---------------------------------------------------------
# 1. Define Local Tools (The Agent's "Hands")
# ---------------------------------------------------------
def list_files(directory="."):
    """Lists files in the target directory."""
    try:
        files = os.listdir(directory)
        return json.dumps([f for f in files if os.path.isfile(os.path.join(directory, f))])
    except Exception as e:
        return str(e)

def read_file(filename):
    """Reads content from a local text or markdown file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()[:3000]  # Cap length to manage context size
    except Exception as e:
        return str(e)

def save_report(filename, content):
    """Writes content to a file on disk."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote report to {filename}"
    except Exception as e:
        return str(e)

# Mapping functions for tool execution
TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "save_report": save_report
}

# ---------------------------------------------------------
# 2. Tool Definitions for the LLM
# ---------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files available in the workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Folder path, defaults to '.'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content from a specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save generated markdown or text content to a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Output filename, e.g., digest.md"},
                    "content": {"type": "string", "description": "Markdown or text content to write"}
                },
                "required": ["filename", "content"]
            }
        }
    }
]

# ---------------------------------------------------------
# 3. Agent Execution Loop
# ---------------------------------------------------------
def run_agent(user_prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous desktop assistant. Use the provided tools "
                "to inspect files, read information, and generate requested reports. "
                "Only answer with final statements once all tool calls are finished."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    print(f"User Request: {user_prompt}\n" + "-" * 50)

    while True:
        # Request completion from local LM Studio server
        response = client.chat.completions.create(
            model="gemma",  # LM Studio uses whichever model is active
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Check if the model called a tool
        if not msg.tool_calls:
            print(f"\n[Agent Final Answer]:\n{msg.content}")
            break

        # Process each tool request
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            print(f"[Tool Call]: Running {fn_name}({args})")
            
            # Run local Python function
            tool_result = TOOL_MAP[fn_name](**args)
            print(f"[Tool Output]: {str(tool_result)[:120]}...\n")

            # Feed tool execution results back to Gemma
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

if __name__ == "__main__":
    # Test prompt
    prompt = "Look at the files in the current folder, read any .txt files, and summarize their contents into a new file called 'summary_report.md'."
    run_agent(prompt)