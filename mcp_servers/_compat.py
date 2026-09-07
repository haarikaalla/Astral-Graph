"""Compatibility shim for the ``mcp`` Python SDK.

The SDK renamed ``FastMCP`` to ``MCPServer`` in v2. Importing through this module
keeps the servers running on both v1 and v2 installations.
"""

from __future__ import annotations

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as MCPServer
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[assignment]

__all__ = ["MCPServer"]
