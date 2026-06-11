import json
import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import config
from app.database import models

# Initialize OpenAI client pointing to local inference engine
client = OpenAI(
    base_url=config.AI_BASE_URL,
    api_key="not-needed-for-local"  # Local models typically don't require an API key
)

SYSTEM_PROMPT_TEMPLATE = """You are Lens-Brain, an offline executive logistics engine. Your sole responsibility is to convert a user's unstructured conversation stream into clear data actions and dynamic structural updates.

CONTEXT CONSTRAINTS:
- Current Timestamp: {current_timestamp}
- Calendar Windows Found: {calendar_data}
- Consolidated Session Memory (Past 24H): {memory_digest}

EXECUTION RULES:
EXECUTION RULES:
1. Always communicate back to the user in a natural, friendly, conversational tone (max two sentences). Never output paragraphs of text. Bolding task targets is mandatory.
2. Do not explain technical SQL fields, schema architecture, or JSON formatting to the user.
3. If an input is ambiguous, infer a logical layout structure first, then explicitly use your response sentences to ask for verification.
4. If a user states they finished or completed tasks, aggressively match their descriptions to the CURRENT ACTIVE TASKS list. You MUST call `update_node_status` to mark them as completed. If they finished multiple tasks, call the tool multiple times in parallel for each matching ID. Do not ask to create new nodes if they match existing active tasks.
5. If the user asks to "focus on" a project, or asks what their tasks are, do NOT list them in chat. Instead, call `update_node_urgency` with `urgency_score: 10.0` and an array of `node_ids` containing ALL active task IDs from the context that belong to or relate to that project. The frontend UI will automatically update to show them.
"""

LENS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_thought_node",
            "description": "Ingests a new piece of work or goal and connects it to the existing graph topology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The concise descriptive summary of the action item or project goal."
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["is_part_of", "blocks", "depends_on", "related_to"],
                        "description": "How this item attaches to an existing context node, if applicable."
                    },
                    "target_node_id": {
                        "type": "integer",
                        "description": "The database ID of the existing node this new item connects with."
                    },
                    "inferred_deadline": {
                        "type": "string",
                        "description": "ISO-8601 formatted date string if a deadline is explicitly mentioned or strongly implied."
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_status",
            "description": "Updates a node's operational lifecycle status, such as checking a task off or archiving it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "integer",
                        "description": "The target database unique node ID."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "on_hold", "cold_storage"],
                        "description": "The target state execution phase."
                    }
                },
                "required": ["node_id", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_urgency",
            "description": "Updates a node's urgency score to push it up or down in the frontend UI. Use a score of 10.0 to focus it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "An array of target database unique node IDs to boost."
                    },
                    "urgency_score": {
                        "type": "number",
                        "description": "The new urgency score (e.g. 10.0 for high focus, 0.0 for normal)."
                    }
                },
                "required": ["node_ids", "urgency_score"]
            }
        }
    }
]

def format_system_prompt() -> str:
    now = datetime.datetime.now().isoformat()
    
    # Fetch active tasks so the LLM actually knows their IDs
    active_nodes = models.get_active_nodes(limit=50)
    if active_nodes:
        nodes_context = "\n".join([f"[ID: {n['id']}] {n['content']}" for n in active_nodes])
    else:
        nodes_context = "No active tasks."
        
    return SYSTEM_PROMPT_TEMPLATE.format(
        current_timestamp=now,
        calendar_data="No active calendar events.",
        memory_digest=f"CURRENT ACTIVE TASKS (Use these IDs for tools!):\n{nodes_context}"
    )

def execute_tool_call(call) -> str:
    """Executes the function specified by the LLM tool call and returns a confirmation string."""
    func_name = call.function.name
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON arguments"})

    if func_name == "add_thought_node":
        content = args.get("content")
        target_date = args.get("inferred_deadline")
        
        node_id = models.add_node(content=content, status="active", target_date=target_date)
        
        rel_type = args.get("relationship_type")
        target_id = args.get("target_node_id")
        if rel_type and target_id:
            models.add_edge(parent_id=target_id, child_id=node_id, relationship=rel_type)
            
        return json.dumps({"success": True, "node_id": node_id, "action": "created"})
        
    elif func_name == "update_node_status":
        node_id = args.get("node_id")
        status = args.get("status")
        models.update_node_status(node_id=node_id, status=status)
        return json.dumps({"success": True, "node_id": node_id, "action": "updated", "status": status})
        
    elif func_name == "update_node_urgency":
        node_ids = args.get("node_ids")
        urgency_score = args.get("urgency_score")
        if not node_ids or urgency_score is None:
            return json.dumps({"error": "Missing node_ids or urgency_score"})
        
        # Ensure safe typing
        try:
            score = float(urgency_score)
            for n_id in node_ids:
                models.update_node_urgency(node_id=n_id, urgency_score=score)
            return json.dumps({"success": True, "node_ids": node_ids, "action": "urgency_updated", "urgency_score": score})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown function: {func_name}"})

def process_user_input(user_text: str) -> str:
    """Sends user input to the LLM, executes requested tools, and returns the LLM's text response."""
    messages = [
        {"role": "system", "content": format_system_prompt()},
        {"role": "user", "content": user_text}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.AI_MODEL_NAME,
            messages=messages,
            tools=LENS_TOOLS,
            tool_choice="auto"
        )
    except Exception as e:
        return f"Warning: Inference Engine unreachable. Make sure your local model is running. Details: {e}"
        
    response_message = response.choices[0].message
    
    # Check if the LLM wants to call a tool
    if response_message.tool_calls:
        # We append the assistant's message with tool calls to the history
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            tool_result = execute_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_result
            })
            
        # Add a strict reminder before the final summary to prevent reasoning models from hallucinating math formats
        messages.append({
            "role": "system", 
            "content": "Tool call completed. Now, provide the user with a natural, friendly confirmation of what you just did (max two sentences). Do NOT mention technical details like 'nodes', 'database', or 'IDs'. DO NOT use mathematical formatting like \\boxed{}."
        })
        
        # Get the final response from the LLM summarizing what it did
        try:
            final_response = client.chat.completions.create(
                model=config.AI_MODEL_NAME,
                messages=messages
            )
            return final_response.choices[0].message.content
        except Exception as e:
            return f"Action executed, but failed to generate summary. Details: {e}"
            
    return response_message.content

if __name__ == "__main__":
    # Test script
    print("Testing Engine Orchestration Layer...")
    test_input = "I need to design the new landing page, and it is blocked by finishing the Figma mockups."
    print(f"User: {test_input}")
    reply = process_user_input(test_input)
    print(f"Lens-Brain: {reply}")
    print("Current Active Nodes:", models.get_active_nodes())
