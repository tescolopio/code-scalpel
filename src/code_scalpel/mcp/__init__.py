"""
Code Scalpel MCP Server - Model Context Protocol integration.

This module provides a fully MCP-compliant server that exposes Code Scalpel's
analysis capabilities through the official MCP protocol.

Supports:
- stdio transport (preferred for local integration)
- Streamable HTTP transport (for network deployment)
"""

from .server import mcp, run_server

__all__ = ["mcp", "run_server"]
