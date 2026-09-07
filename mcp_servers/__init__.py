"""AstralGraph custom MCP servers (official `mcp` Python SDK, stdio transport).

Each server is a standalone process. Agents talk to them **only** through the MCP
protocol — no module in ``/agents`` is allowed to make an HTTP call.
"""
