"""The Genie agents this router routes between, and the trace lineage for every call.

TO ADD OR CHANGE A GENIE AGENT, edit GENIE_AGENTS below. Nothing else needs to change:
the tools are generated from that list, and agent/tools/__init__.py discovers them
automatically. Remember to add a matching line to SYSTEM_PROMPT in agent/agent.py.

Why the Genie agents are configured in code here rather than bound with
`agentbricks tools add genie-agent <space_id>`:

1. **Routing descriptions.** The CLI-bound tools are all named `<id>_ask` and share one
   generic description, so the only routing signal the model gets is a space ID. Tool
   descriptions are what a router routes on, so each agent gets a written description here.
2. **Trace lineage.** This module tags each call's space / conversation / message IDs onto
   the MLflow trace, so you can match the router's traces to Genie's own conversation
   records. See "Reading the traces" in README.md.

The Genie calls themselves reuse Databricks' own `GenieAgent` (polling with backoff, timeouts
and submission-failure handling), rather than reimplementing any of that here.
"""

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import mlflow
from databricks_agentkit.runtime.genie import GenieAgent
from langchain_core.tools import BaseTool, StructuredTool


@dataclass(frozen=True)
class GenieAgentConfig:
    """One Genie agent the router can send a question to.

    name:        the identifier the MODEL uses to call this tool. Not the Genie agent's name
                 and never sent to Genie -- it is a function name in the tool-calling protocol,
                 so it must match ^[a-zA-Z0-9_-]{1,128}$: letters, digits, underscores and
                 hyphens, no spaces. Beyond that it's free. Mirroring the Genie agent's real
                 name with underscores (`NYC_Taxi_Trips`) keeps traces readable, since this
                 value is what shows up as `routed_to` and inside the trace tag keys.
    space_id:    the Genie space ID. Find it in the Genie agent's URL in the workspace:
                 /genie/rooms/<space_id>
    description: THE ROUTING SIGNAL. The model reads this to decide whether a question
                 belongs to this agent, so be specific about the data it covers. Name the
                 entities and the kinds of question, not just the department. Vague
                 descriptions are the single most common cause of bad routing.
    """

    name: str
    space_id: str
    description: str


# ---------------------------------------------------------------------------
# CONFIGURE YOUR GENIE AGENTS HERE
# ---------------------------------------------------------------------------
# The space IDs below are placeholders and will fail with a clear error until you replace
# them. Two agents are shown because routing needs at least two choices to be meaningful,
# but you can list as many as you like.
GENIE_AGENTS = [
    GenieAgentConfig(
        name="ask_sales_genie",
        space_id="<your-sales-genie-space-id>",
        description=(
            "Answer questions about sales orders, revenue, pipeline and customer accounts.\n\n"
            "Use for: revenue by region, product or period; order volumes and order status; "
            "pipeline and deal stages; customer and account questions; sales targets and "
            "attainment. Pass the user's question through unchanged."
        ),
    ),
    GenieAgentConfig(
        name="ask_ops_genie",
        space_id="<your-ops-genie-space-id>",
        description=(
            "Answer questions about operations, inventory, fulfilment and logistics.\n\n"
            "Use for: stock levels and inventory by location; shipments, deliveries and "
            "fulfilment times; warehouse throughput; supplier and lead-time questions; "
            "backorders and returns. Pass the user's question through unchanged."
        ),
    ),
]

_PLACEHOLDER_PREFIX = "<your-"

# How many result rows to put in front of the model. Genie result sets can be large and the
# router only needs enough to answer; raise it if your questions return wide summaries.
_MAX_ROWS = 50

_BY_NAME = {config.name: config for config in GENIE_AGENTS}

# auth="app" calls Genie as the deployed agent's service principal, which must be granted
# CAN RUN on every Genie agent listed above. To call as the signed-in end user instead, set
# [auth.user] in agent.toml and grant each user access to the Genie agents directly.
_AGENTS = {
    config.name: GenieAgent(config.space_id, auth="app")
    for config in GENIE_AGENTS
    if not config.space_id.startswith(_PLACEHOLDER_PREFIX)
}

# Accumulates the Genie calls made during each in-flight trace, keyed by trace_id. Tags on an
# active trace can't be read back reliably, so we keep our own tally and rewrite the full list
# each time. Bounded so a long-lived server doesn't leak.
_calls_by_trace: "OrderedDict[str, list[dict]]" = OrderedDict()
_calls_lock = threading.Lock()
_MAX_TRACKED_TRACES = 512


def _tag_lineage(tool_name: str, result: dict[str, Any]) -> None:
    """Record this Genie call on the current trace.

    `genie.calls` holds every call made during the request, so a question that needed two
    Genie agents keeps full lineage instead of only the last tool call. The flat
    `genie.<tool>.*` tags exist so the common single-hop case stays easy to filter in SQL.
    """
    call = {
        "tool": tool_name,
        "space_id": result.get("space_id"),
        "conversation_id": result.get("conversation_id"),
        "message_id": result.get("message_id"),
    }

    span = mlflow.get_current_active_span()
    trace_id = getattr(span, "trace_id", None)

    if trace_id is None:
        calls = [call]
    else:
        with _calls_lock:
            tracked = _calls_by_trace.setdefault(trace_id, [])
            tracked.append(call)
            _calls_by_trace.move_to_end(trace_id)
            while len(_calls_by_trace) > _MAX_TRACKED_TRACES:
                _calls_by_trace.popitem(last=False)
            calls = list(tracked)

    tags = {
        "invoked_by": "genie-router",
        "genie.calls": json.dumps(calls),
    }
    for field in ("space_id", "conversation_id", "message_id"):
        if call[field]:
            tags[f"genie.{tool_name}.{field}"] = call[field]
    mlflow.update_current_trace(tags=tags)


def _render_rows(payload: dict[str, Any]) -> str:
    """Format the rows from GenieAgent.query_result as a text table."""
    columns = [c.get("name", "?") for c in payload.get("columns") or []]
    rows = payload.get("rows") or []
    if not rows:
        return "The query returned no rows."

    lines = [" | ".join(columns)] if columns else []
    for row in rows[:_MAX_ROWS]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))

    total = payload.get("total_row_count")
    if payload.get("truncated") or (total is not None and total > len(rows[:_MAX_ROWS])):
        shown = min(len(rows), _MAX_ROWS)
        lines.append(f"... (showing {shown} of {total if total is not None else 'many'} rows)")
    return "\n".join(lines)


async def _render_answer(tool_name: str, result: dict[str, Any]) -> str:
    """Turn Genie's response into a single text answer for the router."""
    status = result.get("status")

    if result.get("timed_out"):
        return (
            f"Genie is still working on this question (status {status}). "
            "The question was submitted; do not resubmit it."
        )
    if status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
        error = (result.get("error") or {}).get("message") or "no detail given"
        return f"Genie could not answer this question (status {status}): {error}"

    agent = _AGENTS[tool_name]
    conversation_id = result.get("conversation_id")
    message_id = result.get("message_id")
    parts: list[str] = []

    for attachment in result.get("attachments") or []:
        if "text" in attachment:
            content = (attachment.get("text") or {}).get("content")
            if content:
                parts.append(content)
        elif "query" in attachment:
            query = attachment.get("query") or {}
            if query.get("description"):
                parts.append(query["description"])
            if query.get("query"):
                parts.append(f"SQL Genie ran:\n{query['query']}")
            # Only query attachments carry an attachment_id; text ones don't.
            attachment_id = attachment.get("attachment_id")
            if attachment_id:
                try:
                    rows = await agent.query_result(conversation_id, message_id, attachment_id)
                    parts.append("Result:\n" + _render_rows(rows))
                except Exception as exc:  # surface it rather than silently drop the rows
                    parts.append(f"(Could not fetch query result: {exc})")

    if not parts:
        return "Genie returned no answer for this question."
    if result.get("deep_link"):
        parts.append(f"Open in Genie: {result['deep_link']}")
    return "\n\n".join(parts)


async def _ask(tool_name: str, question: str) -> str:
    """Send a question to one Genie agent, tag its lineage, and return the answer."""
    config = _BY_NAME[tool_name]
    if tool_name not in _AGENTS:
        raise ValueError(
            f"{tool_name} still has the placeholder space ID {config.space_id!r}. "
            "Set a real Genie space ID in GENIE_AGENTS in agent/tools/genie_router.py "
            "(find it in the Genie agent's workspace URL: /genie/rooms/<space_id>)."
        )

    with mlflow.start_span(name=tool_name, span_type="TOOL") as span:
        span.set_inputs({"tool": tool_name, "question": question})

        result = await _AGENTS[tool_name].ask(question)

        # Tag before rendering, so a rendering failure still leaves the IDs on the trace.
        _tag_lineage(tool_name, result)
        span.set_attributes(
            {
                "genie.space_id": result.get("space_id"),
                "genie.conversation_id": result.get("conversation_id"),
                "genie.message_id": result.get("message_id"),
            }
        )

        answer = await _render_answer(tool_name, result)
        span.set_outputs({"status": result.get("status"), "answer": answer})
        return answer


_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _check_name(config: GenieAgentConfig) -> None:
    """Reject a tool name the model API will refuse, at startup rather than mid-question.

    The name goes straight into the model's tool definitions, which only allow letters, digits,
    underscores and hyphens. A name with a space or a dot deploys and starts up perfectly
    normally, then fails the first actual question with a bare HTTP 400 that doesn't obviously
    point back here:

        tools.0.custom.name: String should match pattern '^[a-zA-Z0-9_-]{1,128}$'

    So check it while we still have somewhere useful to point.
    """
    if not _VALID_TOOL_NAME.match(config.name):
        suggestion = re.sub(r"[^a-zA-Z0-9_-]+", "_", config.name).strip("_")
        raise ValueError(
            f"Genie agent name {config.name!r} is not usable as a tool name. Names may only "
            "contain letters, digits, underscores and hyphens -- no spaces or dots -- because "
            "this is the identifier the model uses to call the tool, so it has to be a valid "
            "function name.\n"
            f"Fix it in GENIE_AGENTS in agent/tools/genie_router.py. Closest valid version of "
            f"your name: {suggestion}\n"
            "Remember to update the matching line in SYSTEM_PROMPT in agent/agent.py too."
        )


def _build_tool(config: GenieAgentConfig) -> BaseTool:
    """Turn one GenieAgentConfig into a LangChain tool the model can call."""
    _check_name(config)

    async def ask(question: str) -> str:
        return await _ask(config.name, question)

    return StructuredTool.from_function(
        coroutine=ask,
        name=config.name,
        description=config.description,
    )


# Discovered by agent/tools/__init__.py:all_tools() and passed to the model.
GENIE_TOOLS = [_build_tool(config) for config in GENIE_AGENTS]
