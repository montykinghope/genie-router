"""Serve the Genie router agent over the Agent Bricks durable runtime.

The deployed app is API-only: it exposes /api/invocations and nothing else.

The browser chat UI is opt-in, behind ENABLE_CHAT_UI, so the app can be run as an API-only
service. Remove that variable from app.yaml to serve only /api/invocations.
"""

import os
from pathlib import Path

import uvicorn
from agent.agent import configure
from dotenv import load_dotenv
from runtime.adapter import invoke, recover

from databricks_agentkit import DurableAgentServer

# override=False so injected DATABRICKS_* (from `agentbricks dev -p` or the deploy platform) win
# over a checked-in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
configure()


def _chat_ui_enabled() -> bool:
    return os.getenv("ENABLE_CHAT_UI", "").strip().lower() in ("1", "true", "yes")


app = DurableAgentServer()
app.invoke(invoke)
app.recover(recover)

if _chat_ui_enabled():
    from runtime.ui import install_ui

    install_ui(app)


def main() -> None:
    uvicorn.run("runtime.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
