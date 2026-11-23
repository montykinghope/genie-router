"""Agent tools package.

Every module in this folder is auto-imported, and every LangChain ``BaseTool`` it exposes is
collected by ``all_tools()``, which ``agent/agent.py`` passes to the model. A module can expose
tools either as module-level objects (e.g. a ``@tool``-decorated function) or as a list of them
(the way ``genie_router.py`` generates one tool per configured Genie agent).

Drop a new ``*.py`` in here and its tools are picked up automatically — no wiring to edit.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterable

from langchain_core.tools import BaseTool


def _tools_in(obj: object) -> Iterable[BaseTool]:
    """Yield the tools held by a module member: either the tool itself, or a list of them."""
    if isinstance(obj, BaseTool):
        yield obj
    elif isinstance(obj, (list, tuple)):
        yield from (item for item in obj if isinstance(item, BaseTool))


def all_tools() -> list[BaseTool]:
    """Every BaseTool exposed across the modules in this package."""
    tools: list[BaseTool] = []
    for module in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{module.name}")
        for _, member in inspect.getmembers(mod):
            for found in _tools_in(member):
                if found not in tools:  # a tool exposed twice is collected once
                    tools.append(found)
    return tools
