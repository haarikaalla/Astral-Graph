"""MCP permission scoping.

Two layers of defence:

1. :func:`assert_allowed` — hard check against :data:`core.mcp_registry.AGENT_PERMISSIONS`.
2. :class:`PermissionBroker` — per-agent facade handed to each agent so it cannot
   even *name* a server outside its scope, plus plan sanitisation that silently
   drops out-of-scope calls a model tries to invent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.mcp_client import MCPHub, MCPPermissionError, ToolResult
from core.mcp_registry import allowed_servers
from core.telemetry import get_logger, log_event
from guardrails.schemas import ToolCall

log = get_logger("guardrails.permissions")


def assert_allowed(agent: str, server: str) -> None:
    if server not in allowed_servers(agent):
        raise MCPPermissionError(
            f"DENIED: agent '{agent}' -> server '{server}'. "
            f"Allowed: {sorted(allowed_servers(agent)) or 'none'}"
        )


@dataclass
class PermissionBroker:
    """Scoped MCP facade for a single agent."""

    agent: str
    hub: MCPHub
    denied: list[str] = field(default_factory=list)
    call_log: list[str] = field(default_factory=list)

    @property
    def servers(self) -> list[str]:
        return sorted(allowed_servers(self.agent))

    def can(self, server: str) -> bool:
        return server in allowed_servers(self.agent)

    async def catalogue(self) -> str:
        return await self.hub.tool_catalogue(self.servers)

    async def call(self, server: str, tool: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        assert_allowed(self.agent, server)
        result = await self.hub.call(self.agent, server, tool, arguments or {})
        self.call_log.append(f"{server}.{tool}")
        return result

    def sanitise(self, calls: list[ToolCall]) -> tuple[list[ToolCall], list[str]]:
        """Drop planned calls that violate scope; return (kept, rejected_descriptions)."""
        kept: list[ToolCall] = []
        rejected: list[str] = []
        for call in calls:
            if self.can(call.server):
                kept.append(call)
            else:
                message = f"{self.agent} -> {call.qualified} (out of scope)"
                rejected.append(message)
                self.denied.append(message)
                log_event(log, "permission_denied", agent=self.agent, server=call.server,
                          tool=call.tool)
        return kept, rejected

    def report(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "allowed_servers": self.servers,
            "calls_made": self.call_log,
            "denied": self.denied,
        }
