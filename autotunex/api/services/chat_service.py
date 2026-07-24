# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""
Chat service — LangChain agent that discovers MCP tools and uses
Claude (via LiteLLM) to answer user questions about AutoTuneX.
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

logger = logging.getLogger("chat_service")

# Truncate individual tool results to ~12K tokens to stay within Claude's 200K context
MAX_TOOL_RESULT_CHARS = 50_000

# Process-local checkpointer so multi-turn threads retain ToolMessages (and
# thus cached config/dataset IDs) between requests. A single InMemorySaver is
# shared across all chat sessions and keyed per thread_id.
# NOTE: state is lost on process restart and NOT shared across workers. For
# multi-worker deployments, swap in SqliteSaver / PostgresSaver.
_CHECKPOINTER = InMemorySaver()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Tools whose successful completion should trigger a UI refresh, and which
# view they affect. The frontend remounts that view when it sees a "refresh" event.
TOOL_REFRESH_TARGETS: Dict[str, str] = {
    "start_tuning_job": "tunings",
    "create_config": "configs",
}

# Human-friendly status labels shown in the chat UI while each MCP tool runs.
# Keep this in sync with mcp_server.py; missing entries fall back to a title-cased tool name.
TOOL_LABELS: Dict[str, str] = {
    # Jobs
    "list_jobs": "Looking up your jobs…",
    "get_job": "Fetching job details…",
    "get_job_trials": "Loading job trials…",
    "get_trial_logs": "Reading trial logs…",
    "get_job_results": "Loading job results…",
    "get_job_assets": "Listing job assets…",
    "start_tuning_job": "Starting your tuning job…",
    # Configs
    "list_configs": "Looking up your configurations…",
    "get_config": "Fetching configuration details…",
    "get_config_template": "Loading configuration template…",
    "create_config": "Creating your configuration…",
    # Datasets
    "list_datasets": "Looking up your datasets…",
    "get_dataset": "Fetching dataset details…",
    "get_supported_dataset_types": "Checking supported dataset types…",
    # DMF
    "list_published_models": "Looking up your published models…",
    # Users
    "get_user_info": "Loading your profile…",
    "get_user_metadata": "Loading your account stats…",
}


def _friendly_label(tool_name: str) -> str:
    if tool_name in TOOL_LABELS:
        return TOOL_LABELS[tool_name]
    return f"Running {tool_name.replace('_', ' ').strip().capitalize()}…"


SYSTEM_PROMPT = (
    "You are AutoTuneX Assistant, a concise AI helper for IBM's automated "
    "LLM fine-tuning platform.\n\n"
    "User email: {user_email}\n\n"
    "RULES:\n"
    "1. Be CONVERSATIONAL. Keep responses short for explanations, but "
    "when the user asks for a list (datasets, configs, jobs, etc.) "
    "ALWAYS show the full list of names from the tool output — never "
    "summarize with just a count. Never dump long tutorials or code blocks.\n"
    "2. USE YOUR TOOLS to perform actions — do not explain how to call APIs "
    "or write sample code. You ARE the interface.\n"
    "3. GATHER INFO STEP BY STEP. If the user asks to do something but "
    "information is missing, ask for ONE missing piece at a time. "
    "For example, if they want to fine-tune a model, first check their "
    "datasets (using list_datasets), then their configs (using list_configs), "
    "then confirm before starting. IMPORTANT: if you have already called a "
    "lookup tool earlier in THIS conversation, reuse those results instead "
    "of re-calling the tool. Only re-fetch when the user asks for fresh "
    "data or references something you haven't looked up yet.\n"
    "4. Before any destructive or expensive action (starting a job, deleting "
    "something), state what you're about to do in one sentence and ask "
    "for confirmation.\n"
    "5. If a tool call fails, explain the error briefly and ask the user "
    "how to proceed.\n"
    "6. Always pass the user's email when tools require user_email.\n"
    "7. ACCURACY IS CRITICAL: When presenting data from tool results "
    "(names, IDs, counts, values), reproduce them EXACTLY as returned by "
    "the tool. NEVER invent, guess, extrapolate, or paraphrase names or "
    "IDs. If a list tool returns items, present only those exact items — "
    "do not continue patterns or fill in gaps.\n"
    "8. FORMATTING: Tool results are pre-formatted with markdown (bold "
    "names, numbered lists, inline code). Present the tool output "
    "directly — preserve the line breaks and formatting. Do NOT collapse "
    "a numbered list into a single comma-separated paragraph."
)


async def _build_agent_and_input(
    messages: List[Dict[str, Any]],
    user_email: str,
    thread_id: Optional[str] = None,
) -> Tuple[Any, List[Any], str, Dict[str, Any]]:
    """Shared setup for both blocking and streaming chat paths.

    Returns (agent, input_messages, user_input, run_config).

    When a thread_id is provided and the checkpointer already has state for it,
    only the newest user message is returned in input_messages — prior turns
    (including ToolMessages) are replayed from the checkpoint, so the model
    can reuse earlier tool results instead of re-calling list_* tools.
    """
    from services.plugins import Seam, resolve

    llm = resolve(Seam.CHAT).build_llm()

    mcp_client = MultiServerMCPClient(
        {"autotunex": {"transport": "sse", "url": MCP_SERVER_URL}}
    )
    tools = await mcp_client.get_tools()
    logger.info("Discovered %d MCP tools", len(tools))

    tool_node = ToolNode(tools, handle_tool_errors=True)
    system_prompt = SYSTEM_PROMPT.format(user_email=user_email)

    def _prepare_messages(state):
        """Prepend system prompt and truncate oversized tool results."""
        out = [SystemMessage(content=system_prompt)]
        for msg in state["messages"]:
            if (
                isinstance(msg, ToolMessage)
                and isinstance(msg.content, str)
                and len(msg.content) > MAX_TOOL_RESULT_CHARS
            ):
                out.append(
                    ToolMessage(
                        content=msg.content[:MAX_TOOL_RESULT_CHARS]
                        + "\n\n... [truncated — result too large]",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                )
            else:
                out.append(msg)
        return out

    agent = create_react_agent(
        llm, tool_node, prompt=_prepare_messages, checkpointer=_CHECKPOINTER
    )

    user_input = messages[-1].get("content", "") if messages else ""
    run_config: Dict[str, Any] = {}
    input_messages: List[Any]

    if thread_id:
        run_config = {"configurable": {"thread_id": thread_id}}
        # If the checkpointer already has prior turns for this thread, LangGraph
        # will replay them from state — we only append the new user message.
        # Otherwise (new thread, or process restarted), seed with the full
        # history from the client so the conversation isn't truncated.
        has_state = False
        try:
            snapshot = await agent.aget_state(run_config)
            has_state = bool(snapshot and snapshot.values.get("messages"))
        except Exception as e:
            logger.debug("No existing state for thread %s: %s", thread_id, e)

        if has_state:
            input_messages = [HumanMessage(content=user_input)]
        else:
            input_messages = _rehydrate_history(messages)
    else:
        # No thread_id — legacy path, send the full history every turn.
        input_messages = _rehydrate_history(messages)

    return agent, input_messages, user_input, run_config


def _rehydrate_history(messages: List[Dict[str, Any]]) -> List[Any]:
    """Convert the client's flat message list into LangChain messages."""
    out: List[Any] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    if not out or not isinstance(out[-1], HumanMessage):
        # Ensure the turn ends with a user message so the agent has something to answer.
        last = messages[-1].get("content", "") if messages else ""
        out.append(HumanMessage(content=last))
    return out


async def chat(
    messages: List[Dict[str, Any]],
    user_email: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an agentic chat turn: discover MCP tools, call OpenAI, execute tools, return response."""

    context = context or {}
    agent, input_messages, user_input, run_config = await _build_agent_and_input(
        messages, user_email, thread_id=thread_id
    )

    try:
        result = await agent.ainvoke({"messages": input_messages}, config=run_config)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        output = ai_messages[-1].content if ai_messages else ""
    except Exception as e:
        logger.exception("Chat agent failed: %s", e)
        output = (
            "I'm sorry, I encountered an error while processing your request. "
            "Please try again or rephrase your question."
        )

    return {
        "output": output,
        "context": {**context, "last_input": user_input},
    }


def _extract_token_text(chunk: Any) -> str:
    """Pull plain text out of a streamed chat-model chunk.

    Returns "" for tool-call argument chunks or empty deltas so the caller can skip them.
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


async def chat_stream(
    messages: List[Dict[str, Any]],
    user_email: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream an agent turn as a sequence of event dicts.

    Yields events of the form:
      {"type": "tool_start", "name": str, "label": str}
      {"type": "tool_end", "name": str}
      {"type": "token", "text": str}
      {"type": "done"}
      {"type": "error", "message": str}
    """
    _ = context or {}

    try:
        agent, input_messages, _user_input, run_config = await _build_agent_and_input(
            messages, user_email, thread_id=thread_id
        )
    except Exception as e:
        logger.exception("Chat agent setup failed: %s", e)
        yield {"type": "error", "message": str(e) or "Failed to initialize chat."}
        return

    # When the model resumes speaking after a tool call, its next AIMessage has no
    # leading whitespace — so without a separator the pre- and post-tool text fuse
    # ("…right IDs!Got everything!"). Track the boundary and prefix the first post-
    # tool token with a paragraph break.
    just_finished_tool = False
    streaming_answer = False

    try:
        async for event in agent.astream_events(
            {"messages": input_messages}, version="v2", config=run_config
        ):
            etype = event.get("event")
            if etype == "on_tool_start":
                name = event.get("name", "tool")
                just_finished_tool = False
                yield {
                    "type": "tool_start",
                    "name": name,
                    "label": _friendly_label(name),
                }
            elif etype == "on_tool_end":
                name = event.get("name", "tool")
                yield {"type": "tool_end", "name": name}
                target = TOOL_REFRESH_TARGETS.get(name)
                if target:
                    # Best-effort success check — ToolNode returns an error string on failure.
                    output = event.get("data", {}).get("output")
                    output_text = (
                        getattr(output, "content", output) if output is not None else ""
                    )
                    if isinstance(output_text, str) and output_text.lower().startswith(
                        "error"
                    ):
                        pass
                    else:
                        yield {"type": "refresh", "target": target}
                just_finished_tool = True
            elif etype == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk is None:
                    continue
                text = _extract_token_text(chunk)
                if not text:
                    continue
                if just_finished_tool and streaming_answer:
                    # Separator between the previous AIMessage and this one.
                    text = "\n\n" + text.lstrip()
                    just_finished_tool = False
                elif just_finished_tool:
                    just_finished_tool = False
                streaming_answer = True
                yield {"type": "token", "text": text}
        yield {"type": "done"}
    except Exception as e:
        logger.exception("Chat stream failed: %s", e)
        yield {
            "type": "error",
            "message": "I'm sorry, I encountered an error while processing your request. Please try again or rephrase your question.",
        }
