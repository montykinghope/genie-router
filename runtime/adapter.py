"""Translate between Agent Bricks invocations and the framework-native agent entrypoint."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from agent import agent as agent_module
from agent.agent import recovery_input, run_agent
from langchain.messages import AIMessageChunk
from langgraph.types import Command

from databricks_agentkit import InvocationContext
from databricks_agentkit.langgraph.responses import checkpointed_messages
from databricks_agentkit.langgraph.session_store import thread_config
from databricks_agentkit.runtime.auth import AuthError


# Content-block types that hold the model's private thinking rather than its answer.
_REASONING_TYPES = frozenset(
    {"reasoning", "reasoning_content", "thinking", "redacted_thinking"}
)

_JSON = json.JSONDecoder()


def _is_reasoning_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in _REASONING_TYPES


def _display_content(content: Any) -> Any:
    """Strip the model's reasoning blocks, returning just the text meant for the caller.

    Why this is needed: streaming chunks carry reasoning as a *list* of content blocks, but
    ``AIMessageChunk.content`` is typed as a string. When LangChain adds the chunks together it
    string-concatenates them, so the aggregated message arrives with serialized JSON glued to
    the front of the real answer:

        [{"type": "reasoning", "summary": [...]}][{"type": "reasoning", ...}]The answer is 42.

    That lands in ``/api/invocations`` responses, so every client sees it, not just the bundled
    chat UI. The reasoning summaries are empty in practice (text is ``""``, only a signature is
    populated), so nothing useful is lost by dropping them.

    Anything that isn't recognisably reasoning is passed through untouched, so a genuine answer
    that happens to begin with ``[`` is safe.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if _is_reasoning_block(block):
                continue
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)

    if isinstance(content, str):
        remaining = content
        # Peel off each leading JSON array that contains only reasoning blocks.
        while remaining.startswith("["):
            try:
                value, end = _JSON.raw_decode(remaining)
            except ValueError:
                break
            if isinstance(value, list) and value and all(map(_is_reasoning_block, value)):
                remaining = remaining[end:]
            else:
                break
        return remaining

    return content


def _clean_message(message: Any) -> dict[str, Any]:
    """Serialize a LangChain message for the wire, with reasoning blocks removed."""
    dumped = message.model_dump()
    if "content" in dumped:
        dumped["content"] = _display_content(dumped["content"])
    return dumped


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"messages": value}
    if not isinstance(value, dict):
        raise ValueError("input must be a message list or an object")
    return value


def _session_id(context: InvocationContext) -> str:
    value = context.session_id
    if not isinstance(value, str) or not value:
        raise ValueError("session_id must be provided as a top-level invocation field")
    return value


def _actor(payload: dict[str, Any], session_id: str) -> str:
    value = payload.get("actor") or session_id
    if not isinstance(value, str) or not value:
        raise ValueError("actor must be a non-empty string")
    return value


def _agent_input(payload: dict[str, Any]) -> Any:
    if (resume := payload.get("resume")) is not None:
        return Command(resume=resume)
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return {"messages": messages}


async def invoke(value: Any, context: InvocationContext) -> dict:
    payload = _payload(value)
    _check_user_input(payload, context)
    return await _invoke_agent(_agent_input(payload), payload, context)


async def recover(value: Any, context: InvocationContext) -> dict:
    payload = _payload(value)
    _check_user_input(payload, context)
    if getattr(context, "request_auth", None) is not None:
        raise AuthError(
            "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            "Request-user invocations cannot be recovered in the background.",
            400,
        )
    session_id = _session_id(context)
    actor = _actor(payload, session_id)
    agent_input = await recovery_input(
        _agent_input(payload),
        session_id=session_id,
        actor=actor,
        invocation_id=context.invocation_id,
    )
    return await _invoke_agent(agent_input, payload, context)


def _check_user_input(payload: dict[str, Any], context: InvocationContext) -> None:
    auth = getattr(context, "request_auth", None)
    if auth is not None and any(
        payload.get(key) is not None for key in ("resume", "approval", "approvals")
    ):
        raise AuthError(
            "MCP_USER_AUTH_HITL_UNSUPPORTED",
            "Request-user invocations do not support approval or resume input.",
            400,
        )


async def _invoke_agent(
    agent_input: Any,
    payload: dict[str, Any],
    context: InvocationContext,
) -> dict:
    session_id = _session_id(context)
    actor = _actor(payload, session_id)
    auth = getattr(context, "request_auth", None)
    if auth and payload.get("actor"):
        actor = auth.namespace("actor", actor)
    user_auth = auth is not None
    run_kwargs = {"workspace_client_for": auth.client_for} if user_auth else {}
    model = payload.get("model")
    model = model if isinstance(model, str) else None
    outputs = []
    if agent_input is None:
        graph = await agent_module.create_agent_graph(actor, model, **run_kwargs)
        outputs = [
            {"type": "message", "message": _clean_message(message)}
            for message in await checkpointed_messages(
                graph, thread_config(session_id, actor), context.invocation_id
            )
        ]
        run_kwargs["graph"] = graph
    async for event in _serialize_events(
        run_agent(
            agent_input,
            session_id=session_id,
            actor=actor,
            model=model,
            invocation_id=context.invocation_id,
            **run_kwargs,
        )
    ):
        if user_auth and event.get("type") == "interrupt":
            raise AuthError(
                "MCP_USER_AUTH_HITL_UNSUPPORTED",
                "Request-user invocations do not support paused approvals.",
                400,
            )
        await context.emit(event)
        if event.get("type") in ("message", "interrupt"):
            outputs.append(event)

    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [event["message"] if event["type"] == "message" else event for event in outputs],
        "status": "interrupted" if interrupted else "completed",
    }


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    async for mode, payload in async_stream:
        if mode == "updates":
            if interrupts := payload.get("__interrupt__"):
                for item in interrupts:
                    yield {"type": "interrupt", "id": item.id, "value": item.value}
                continue
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for message in messages:
                    yield {"type": "message", "message": _clean_message(message)}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    # A chunk can be pure reasoning, which cleans down to nothing. Emitting an
                    # empty delta would render as a blank token in the client, so skip it.
                    if display := _display_content(content):
                        yield {"type": "delta", "content": display, "id": chunk.id}
            except (KeyError, IndexError, TypeError):
                continue
