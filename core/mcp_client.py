"""MCP client layer — the only way any agent is allowed to reach data.

``MCPHub`` owns one long-lived stdio session per server. Each session runs inside
its own dedicated asyncio task (owning its anyio task group), so multiple agents
can issue concurrent ``call_tool`` requests without cross-task cancel-scope errors.

Responsibilities
----------------
* lazy connect + connection reuse + graceful shutdown
* tool discovery / catalogue rendering for prompts
* permission scoping (agent -> allowed servers) and per-session call budget
* prompt-injection scanning of tool output before it reaches the LLM
* per-call tracing (latency, ok/fail) for the benchmark report
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config import REPO_ROOT, get_settings
from core.mcp_registry import ServerSpec, allowed_servers, build_registry
from core.telemetry import Trace, get_logger, log_event, span

log = get_logger("mcp")

CONNECT_TIMEOUT = 60.0
CALL_TIMEOUT = 90.0


class MCPPermissionError(PermissionError):
    """An agent tried to reach a server it is not scoped for."""


class MCPBudgetExceeded(RuntimeError):
    """The session-wide tool-call budget was exhausted."""


@dataclass
class ToolInfo:
    server: str
    name: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified(self) -> str:
        return f"{self.server}.{self.name}"


@dataclass
class ToolResult:
    server: str
    tool: str
    ok: bool
    data: Any
    raw_text: str = ""
    error: str = ""
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def brief(self, limit: int = 4000) -> str:
        text = json.dumps(self.data, indent=2, default=str) if self.ok else f"ERROR: {self.error}"
        return text[:limit] + ("\n...[truncated]" if len(text) > limit else "")


class _ServerConnection:
    """One stdio MCP session pinned to a dedicated task."""

    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.tools: list[ToolInfo] = []
        self.error: str = ""
        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            await self._ready.wait()
            return
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.spec.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            self.error = f"timed out connecting to MCP server '{self.spec.name}'"
        if self.error:
            raise ConnectionError(self.error)

    async def _run(self) -> None:
        params = StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env={**os.environ, **self.spec.env},
            cwd=str(REPO_ROOT),
        )
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.tools = [
                        ToolInfo(
                            server=self.spec.name,
                            name=t.name,
                            description=(t.description or "").strip(),
                            schema=_schema_of(t),
                        )
                        for t in listing.tools
                    ]
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log_event(log, "server_start_failed", server=self.spec.name, error=self.error)
        finally:
            self._session = None
            self._ready.set()

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise ConnectionError(self.error or f"server '{self.spec.name}' is not connected")
        return await asyncio.wait_for(
            self._session.call_tool(tool, arguments), timeout=CALL_TIMEOUT
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None


class MCPHub:
    """Facade used by every agent. One hub per question/session."""

    def __init__(self, trace: Trace | None = None, *, servers: set[str] | None = None) -> None:
        self.settings = get_settings()
        self.registry = build_registry()
        self.trace = trace
        self.allowed_subset = servers
        self._connections: dict[str, _ServerConnection] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.calls_made = 0
        self.failed_servers: dict[str, str] = {}

    # ---- lifecycle -------------------------------------------------------
    async def __aenter__(self) -> "MCPHub":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.gather(
            *(conn.stop() for conn in self._connections.values()), return_exceptions=True
        )
        self._connections.clear()

    # ---- connection ------------------------------------------------------
    async def connect(self, server: str) -> _ServerConnection | None:
        spec = self.registry.get(server)
        if spec is None:
            self.failed_servers[server] = "unknown server"
            return None
        if not spec.enabled:
            self.failed_servers[server] = "disabled (missing npx or API key)"
            return None
        if self.allowed_subset is not None and server not in self.allowed_subset:
            raise MCPPermissionError(f"server '{server}' not in this hub's scope")
        if server in self.failed_servers:
            return None

        lock = self._locks.setdefault(server, asyncio.Lock())
        async with lock:
            conn = self._connections.get(server)
            if conn is not None:
                return conn
            conn = _ServerConnection(spec)
            try:
                with span(self.trace, f"connect:{server}", "mcp_connect", server=server):
                    await conn.start()
            except Exception as exc:  # noqa: BLE001
                self.failed_servers[server] = str(exc)
                log_event(log, "connect_failed", server=server, error=str(exc))
                return None
            self._connections[server] = conn
            log_event(log, "server_connected", server=server, tools=len(conn.tools))
            return conn

    async def warmup(self, servers: list[str]) -> dict[str, bool]:
        results = await asyncio.gather(
            *(self.connect(s) for s in servers), return_exceptions=True
        )
        return {
            server: isinstance(result, _ServerConnection)
            for server, result in zip(servers, results)
        }

    # ---- discovery -------------------------------------------------------
    async def list_tools(self, servers: list[str]) -> list[ToolInfo]:
        tools: list[ToolInfo] = []
        for server in servers:
            conn = await self.connect(server)
            if conn is not None:
                tools.extend(conn.tools)
        return tools

    async def tool_catalogue(self, servers: list[str]) -> str:
        """Human/LLM readable catalogue used inside agent system prompts."""
        lines: list[str] = []
        for tool in await self.list_tools(servers):
            required = (tool.schema.get("required") or []) if isinstance(tool.schema, dict) else []
            props = (tool.schema.get("properties") or {}) if isinstance(tool.schema, dict) else {}
            args = ", ".join(
                f"{k}: {v.get('type', 'any')}" + ("*" if k in required else "")
                for k, v in props.items()
            )
            head = (tool.description or "").splitlines()[0] if tool.description else ""
            lines.append(f"- {tool.qualified}({args}) — {head}")
        return "\n".join(lines) or "- (no tools available)"

    # ---- invocation ------------------------------------------------------
    async def call(
        self,
        agent: str,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Invoke ``server.tool`` on behalf of ``agent`` with full guardrail checks."""
        arguments = arguments or {}
        if server not in allowed_servers(agent):
            raise MCPPermissionError(
                f"agent '{agent}' is not permitted to call server '{server}' "
                f"(allowed: {sorted(allowed_servers(agent)) or 'none'})"
            )
        if self.calls_made >= self.settings.max_tool_calls_per_session:
            raise MCPBudgetExceeded(
                f"tool-call budget of {self.settings.max_tool_calls_per_session} exhausted"
            )

        self.calls_made += 1
        with span(
            self.trace,
            f"{server}.{tool}",
            "mcp_tool",
            server=server,
            tool=tool,
            agent=agent,
            args=_redact(arguments),
        ) as sp:
            conn = await self.connect(server)
            if conn is None:
                sp["_ok"] = False
                reason = self.failed_servers.get(server, "unavailable")
                sp["error"] = reason
                return ToolResult(server, tool, False, None, error=reason)

            try:
                raw = await conn.call(tool, arguments)
            except Exception as exc:  # noqa: BLE001
                sp["_ok"] = False
                sp["error"] = str(exc)
                return ToolResult(server, tool, False, None, error=f"{type(exc).__name__}: {exc}")

            result = _parse(server, tool, raw)
            sp["_ok"] = result.ok
            if not result.ok:
                sp["error"] = result.error[:300]
            if result.warnings:
                sp["warnings"] = result.warnings
            return result


# --------------------------------------------------------------------------- #
# Result parsing + safety scanning
# --------------------------------------------------------------------------- #

INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "you are now",
    "system prompt",
    "reveal your instructions",
    "<|im_start|>",
    "developer mode",
    "exfiltrate",
)


def scan_for_injection(text: str) -> list[str]:
    """Flag prompt-injection attempts inside third-party tool output."""
    lowered = text.lower()
    return [m for m in INJECTION_MARKERS if m in lowered]


def _schema_of(tool: Any) -> dict[str, Any]:
    """``input_schema`` in mcp 2.x, ``inputSchema`` in 1.x."""
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    return dict(schema) if isinstance(schema, dict) else {}


def _parse(server: str, tool: str, raw: Any) -> ToolResult:
    texts: list[str] = []
    for block in getattr(raw, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    joined = "\n".join(texts)

    payload: Any = getattr(raw, "structured_content", None) or getattr(
        raw, "structuredContent", None
    )
    if payload is None and joined:
        try:
            payload = json.loads(joined)
        except (json.JSONDecodeError, TypeError):
            payload = joined
    if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
        payload = payload["result"]

    warnings = scan_for_injection(joined)
    is_error = bool(getattr(raw, "is_error", False) or getattr(raw, "isError", False))
    if is_error:
        return ToolResult(server, tool, False, None, joined, error=joined or "tool error",
                          warnings=warnings)

    # Our custom servers use the {"ok": bool, "data": ..., "source": ...} envelope.
    if isinstance(payload, dict) and "ok" in payload:
        if not payload.get("ok", False):
            return ToolResult(
                server, tool, False, payload, joined,
                error=str(payload.get("error", "tool reported failure")), warnings=warnings,
            )
        return ToolResult(server, tool, True, payload, joined, warnings=warnings)

    return ToolResult(server, tool, True, payload, joined, warnings=warnings)


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for key, value in arguments.items():
        if any(marker in key.lower() for marker in ("key", "token", "secret", "password")):
            redacted[key] = "***"
        else:
            text = str(value)
            redacted[key] = text if len(text) <= 200 else text[:200] + "..."
    return redacted
