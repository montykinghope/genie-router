"""The Genie router agent.

This file holds the two things you change to make the router yours:

1. ``SYSTEM_PROMPT`` — the routing rules.
2. ``MODEL`` — the model that makes the routing decision.

The Genie agents themselves are configured in ``agent/tools/genie_router.py``.

Everything below those two settings is plumbing: it builds a LangGraph agent whose only
available actions are the Genie tools, then streams its events. You should not need to
touch it to add a Genie agent or change the routing.
"""

import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent

# Importing the tools package auto-registers every tool module in agent/tools/.
from agent.tools import all_tools
from databricks_agentkit import workspace_client, workspace_headers
from databricks_agentkit.langgraph import (
    checkpointer,
    configure_tracing,
    start_trace,
    thread_config,
)
from databricks_agentkit.langgraph.session_store import (
    invocation_id_from_metadata,
    invocation_metadata,
)
from databricks_agentkit.runtime.tool_manifest import resolve_model_service

# ---------------------------------------------------------------------------
# 1. The model
# ---------------------------------------------------------------------------
# A Unity Catalog model service in the `system.ai` schema, called through the Unity Gateway
# (see use_ai_gateway=True below). Any `system.ai.*` chat model your workspace exposes works
# here; list them with:
#
#     databricks api get /api/2.1/unity-catalog/model-services?parent=schemas/system.ai
#
# `agentbricks experimental models bind` can point this at a model service you own, which lets
# `agentbricks experimental models set` change the model without a code change or redeploy.
MODEL = resolve_model_service() or "system.ai.claude-sonnet-5-5"

# ---------------------------------------------------------------------------
# 2. The routing rules
# ---------------------------------------------------------------------------
# Most of the routing signal comes from the tool descriptions in agent/tools/genie_router.py —
# the model reads those to decide which Genie agent owns a question. This prompt covers the
# edges a description can't express: what to do with a question that needs two agents, and
# what to do with one that belongs to neither.
#
# When you add a Genie agent, add a matching line to the list below.
SYSTEM_PROMPT = """You are a router. You do not answer data questions yourself.

You have these Genie agents, each covering a different dataset:
- ask_sales_genie: sales orders, revenue, pipeline and customer accounts.
- ask_ops_genie: operations, inventory, fulfilment and logistics.

Rules:
1. Pick exactly one tool per question, based on which dataset holds the answer.
2. Pass the user's question through unchanged. Do not rewrite or decompose it.
3. Report the tool's answer, including the SQL it ran and the result rows.
4. If a question genuinely needs more than one dataset, call the tools one at a time and say
   which figure came from which agent.
5. If no dataset covers the question, say so and name the topics you do cover.
6. Never invent figures. Every number you report must come from a tool result. If a tool
   reports a failure or a timeout, say that plainly instead of guessing.
"""


# ---------------------------------------------------------------------------
# Plumbing below here.
# ---------------------------------------------------------------------------


class _RoutedChatDatabricks(ChatDatabricks):
    """Forward account-host workspace routing to the underlying OpenAI clients."""

    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    configure_tracing()


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable.
    """
    try:
        workspace_client()
    except Exception as e:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = (
            f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
        )
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
            "Fix one of:\n"
            "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
            "  • run `databricks auth login --profile <name>` to create one, or\n"
            "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
            f"(underlying error: {e})"
        ) from e


async def create_agent_graph(
    actor: str,
    model: str | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
):
    """Build the LangGraph agent: the routing prompt plus one tool per Genie agent.

    ``actor`` is the identity this run belongs to; it partitions conversation state.

    ``model`` selects the gateway model for this run, falling back to ``MODEL``. The agent is
    rebuilt per turn, so the model can vary request to request.
    """
    return create_agent(
        # use_ai_gateway routes to the Unity Catalog AI Gateway (`<host>/ai-gateway/mlflow/v1`), so
        # `endpoint` is a `system.ai.*` model service name rather than a serving-endpoint name.
        model=_RoutedChatDatabricks(
            endpoint=model or MODEL,
            workspace_client=workspace_client(),
            use_ai_gateway=True,
        ),
        tools=all_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer(),
    )


async def recovery_input(
    agent_input: Any,
    *,
    session_id: str,
    actor: str,
    invocation_id: str,
) -> Any:
    """Return the input that should be passed to ``run_agent`` after worker loss.

    A checkpoint tagged with this invocation means LangGraph can continue by receiving ``None``.
    Otherwise recovery replays the original application input.
    """
    checkpoint = await checkpointer().aget_tuple(thread_config(session_id, actor))
    current_invocation_checkpointed = bool(
        checkpoint and invocation_id_from_metadata(checkpoint.metadata) == invocation_id
    )
    return None if current_invocation_checkpointed else agent_input


async def run_agent(
    agent_input: Any,
    *,
    session_id: str,
    actor: str | None = None,
    model: str | None = None,
    invocation_id: str | None = None,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
    graph: Any = None,
) -> AsyncGenerator[Any, None]:
    """Run the agent and yield native LangGraph stream events.

    This is the framework-native entrypoint. It has no dependency on Agent Bricks request or context
    types, so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
    if graph is None:
        graph = await create_agent_graph(actor, model, workspace_client_for=workspace_client_for)
    config = thread_config(session_id, actor)
    if invocation_id:
        config["metadata"] = invocation_metadata(invocation_id)

    last_update: Any = None
    # This span is the root of the trace. The Genie tools hang their lineage tags off it; see
    # "Reading the traces" in README.md.
    with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
        async for event in graph.astream(
            input=agent_input,
            config=config,
            stream_mode=["updates", "messages"],
            durability="sync",
        ):
            if event[0] == "updates":
                last_update = event[1]
            yield event
        if span is not None and last_update is not None:
            span.set_outputs(last_update)
