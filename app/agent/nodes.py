"""
CodeMentor AI - LangGraph Agent Nodes
========================================
Implements the three core node functions for the agent graph:

1. think_node       — Calls Gemini with full history + tool schemas.
                      Either triggers a tool call or produces a final answer.
2. tool_executor    — Reads the pending tool call, dispatches to the
                      right tool, appends result to conversation.
3. should_continue  — Edge condition: "tools" → loop, "end" → done.

Design Rationale:
- Uses the new `google.genai` SDK (google-genai package) which automatically
  handles thought_signatures required by Gemini 3.x models for multi-turn
  function calling. The deprecated `google.generativeai` package lacked this.
- Nodes are pure functions: (AgentState) → AgentState update dict.
  LangGraph merges the return dict into the state automatically.
- Tool calls are dispatched via the tool registry's get_tool_by_name(),
  keeping nodes decoupled from specific tool implementations.
- We use a stateless client but build complete conversation history per call
  so LangGraph owns the state (not the Gemini chat session).
- The iteration guard in should_continue prevents infinite loops if
  the LLM keeps requesting more tool calls.
"""

import json
from typing import Any, Literal

from google import genai
from google.genai import types as genai_types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import MAX_AGENT_ITERATIONS, AgentState, ToolCallLog
from app.core.config import get_settings
from app.core.logging import get_logger
from app.tools import get_all_tools, get_tool_by_name

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_type_string(t: str) -> str:
    """Map JSON Schema type strings to Gemini type constants."""
    mapping = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    return mapping.get(t.lower(), "STRING")


def _to_new_sdk_schema(prop_dict: dict) -> genai_types.Schema:
    """Convert a JSON-Schema property dict to a google.genai Schema object."""
    return genai_types.Schema(
        type=_map_type_string(prop_dict.get("type", "string")),
        description=prop_dict.get("description", ""),
    )


def _build_genai_tool() -> genai_types.Tool:
    """
    Build a google.genai Tool declaration from all registered tools.
    Constructed fresh on each call to pick up any registry changes.
    """
    tools = get_all_tools()
    decls = []
    for tool in tools:
        props = {
            name: _to_new_sdk_schema(info)
            for name, info in tool.args_schema.items()
        }
        decls.append(
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties=props,
                    required=list(tool.args_schema.keys()),
                ),
            )
        )
    return genai_types.Tool(function_declarations=decls)


def _build_genai_contents(messages: list) -> list:
    """
    Convert LangChain message objects to google.genai Content objects.

    Gemini expects:
      - user turns: role="user", parts=[text or function_response]
      - model turns: role="model", parts=[text or function_call]
    System prompt is injected into the first user message.
    """
    contents = []
    system_injected = False

    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, SystemMessage):
            # Injected below into the first user message
            i += 1
            continue

        if isinstance(msg, HumanMessage):
            text = msg.content
            if not system_injected:
                text = f"{SYSTEM_PROMPT}\n\n---\n\nUser: {text}"
                system_injected = True
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=text)],
                )
            )
            i += 1

        elif isinstance(msg, AIMessage):
            if msg.additional_kwargs.get("tool_calls"):
                # Restore model turn with function_call parts (including stored
                # thought_signatures that were saved in additional_kwargs)
                parts = []
                if msg.content:
                    parts.append(genai_types.Part.from_text(text=msg.content))
                for tc in msg.additional_kwargs["tool_calls"]:
                    part = genai_types.Part.from_function_call(
                        name=tc["name"],
                        args=tc["args"],
                    )
                    # Re-attach thought_signature if it was saved
                    if tc.get("thought_signature"):
                        part = genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tc["name"],
                                args=tc["args"],
                                id=tc.get("id", ""),
                            ),
                            thought_signature=tc["thought_signature"],
                        )
                    parts.append(part)
                contents.append(
                    genai_types.Content(role="model", parts=parts)
                )
            else:
                contents.append(
                    genai_types.Content(
                        role="model",
                        parts=[genai_types.Part.from_text(text=msg.content or "")],
                    )
                )
            i += 1

        elif isinstance(msg, ToolMessage):
            # Group consecutive ToolMessages into a single Content object (user turn)
            # This is required by Gemini for parallel function calling.
            tool_parts = []
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tool_msg = messages[i]
                tool_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_msg.name,
                            response={"result": tool_msg.content},
                            id=tool_msg.tool_call_id,
                        )
                    )
                )
                i += 1
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=tool_parts,
                )
            )
        else:
            i += 1

    return contents


# ---------------------------------------------------------------------------
# Node: think_node
# ---------------------------------------------------------------------------

def think_node(state: AgentState) -> dict[str, Any]:
    """
    The core reasoning node — calls Gemini and decides next action.

    Reads the full conversation history, sends it to Gemini with
    all tool schemas bound. Gemini either:
    a) Returns a tool call → we store it in messages and return
    b) Returns a final text answer → we store it in final_answer and return

    Returns a partial AgentState update dict (LangGraph merges it).
    """
    logger.info(
        "think_node: iteration=%d messages=%d",
        state["iterations"], len(state["messages"])
    )

    try:
        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        # Build tool declaration from registered tools
        tool = _build_genai_tool()

        # Convert conversation history to Gemini Content objects
        contents = _build_genai_contents(state["messages"])

        # Generate response — no chat session; we manage history in LangGraph state
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                tools=[tool],
                tool_config=genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(
                        mode="AUTO"
                    )
                ),
            ),
        )

        candidate = response.candidates[0]
        function_calls = []
        text_parts = []

        for part in candidate.content.parts:
            if part.function_call and part.function_call.name:
                fc = part.function_call
                # Store thought_signature so it can be re-attached in history
                thought_sig = None
                if hasattr(part, "thought_signature") and part.thought_signature:
                    thought_sig = part.thought_signature

                function_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args),
                    "id": fc.id or fc.name,
                    "thought_signature": thought_sig,
                })
            elif part.text:
                text_parts.append(part.text)

        if function_calls:
            # Gemini wants to call a tool
            fc = function_calls[0]
            logger.info("think_node: Gemini wants to call tool='%s'", fc["name"])

            ai_message = AIMessage(
                content="\n".join(text_parts).strip(),
                additional_kwargs={"tool_calls": function_calls},
            )
            return {
                "messages": state["messages"] + [ai_message],
                "iterations": state["iterations"] + 1,
                "final_answer": None,
            }
        else:
            # Gemini produced a final text answer
            final_text = "\n".join(text_parts).strip()
            logger.info(
                "think_node: Final answer produced (%d chars)", len(final_text)
            )

            ai_message = AIMessage(content=final_text)
            return {
                "messages": state["messages"] + [ai_message],
                "final_answer": final_text,
                "iterations": state["iterations"] + 1,
            }

    except Exception as exc:
        logger.exception("think_node error: %s", exc)
        error_msg = f"Agent error: {exc}. Please check your Gemini API key configuration."
        return {
            "final_answer": error_msg,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Node: tool_executor_node
# ---------------------------------------------------------------------------

async def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Reads the pending tool call from the last AI message,
    executes the tool, and appends the result as a ToolMessage.

    This node runs AFTER think_node signals a tool call.
    After this node, the graph loops back to think_node.

    Returns a partial AgentState update dict.
    """
    messages = state["messages"]
    last_ai_msg = messages[-1]

    if not isinstance(last_ai_msg, AIMessage):
        logger.error("tool_executor_node: last message is not AIMessage")
        return {"error": "Expected AIMessage with tool call, got something else."}

    tool_calls = last_ai_msg.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        logger.error("tool_executor_node: no tool_calls in last AIMessage")
        return {"error": "No tool calls found in AI message."}

    new_messages = list(messages)
    new_tool_logs = list(state.get("tool_calls_log", []))
    accumulated_context = state.get("retrieved_context", "")

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        logger.info(
            "tool_executor_node: executing tool='%s' args=%s",
            tool_name, tool_args
        )

        tool = get_tool_by_name(tool_name)
        if tool is None:
            result = f"Error: Tool '{tool_name}' not found. Available: search_knowledge_base, calculate, analyze_code, web_search"
            logger.error("tool_executor_node: unknown tool '%s'", tool_name)
        else:
            try:
                result = await tool.run(**tool_args)
            except Exception as exc:
                result = f"Tool '{tool_name}' failed with error: {exc}"
                logger.error("tool_executor_node: tool '%s' raised: %s", tool_name, exc)

        # Append tool result as ToolMessage
        tool_msg = ToolMessage(
            content=result,
            tool_call_id=tc.get("id", tool_name),
            name=tool_name,
        )
        new_messages.append(tool_msg)

        # Log the invocation for API response
        log_entry: ToolCallLog = {
            "tool_name": tool_name,
            "tool_input": tool_args,
            "tool_output": result[:500] + "..." if len(result) > 500 else result,
        }
        new_tool_logs.append(log_entry)

        # If this was a RAG search, accumulate retrieved context
        if tool_name == "search_knowledge_base":
            accumulated_context += f"\n\n{result}"

    return {
        "messages": new_messages,
        "tool_calls_log": new_tool_logs,
        "retrieved_context": accumulated_context.strip(),
    }


# ---------------------------------------------------------------------------
# Edge: should_continue
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Edge condition: determines whether to loop back to tools or end.

    Returns:
        "tools" → route to tool_executor_node
        "end"   → route to END (final_answer is ready)
    """
    if state.get("final_answer") is not None:
        logger.info("should_continue: final_answer set → end")
        return "end"

    if state.get("iterations", 0) >= MAX_AGENT_ITERATIONS:
        logger.warning(
            "should_continue: max iterations (%d) reached → forcing end",
            MAX_AGENT_ITERATIONS
        )
        return "end"

    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        tool_calls = messages[-1].additional_kwargs.get("tool_calls", [])
        if tool_calls:
            logger.info("should_continue: tool call pending → tools")
            return "tools"

    logger.warning("should_continue: no pending tool call and no final answer → end")
    return "end"
