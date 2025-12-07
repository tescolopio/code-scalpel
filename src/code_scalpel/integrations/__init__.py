"""
Code Scalpel Integrations - Integration wrappers for AI agent frameworks.

This module provides integration wrappers for various AI agent frameworks
including Autogen, CrewAI, LangChain, and Claude.

For MCP (Model Context Protocol) integration, use:
    from code_scalpel.mcp import mcp, run_server

For the legacy REST API server (not MCP-compliant), use:
    from code_scalpel.integrations.rest_api_server import create_app
"""

from .autogen import AnalysisResult, AutogenCodeAnalysisAgent, AutogenScalpel
from .crewai import CrewAIScalpel, RefactorResult

# Import REST API server with backward-compatible names
from .rest_api_server import MCPServerConfig, create_app
from .rest_api_server import run_server as run_rest_server

__all__ = [
    # Autogen integration
    "AutogenScalpel",
    "AutogenCodeAnalysisAgent",  # Backward compatibility alias
    "AnalysisResult",
    # CrewAI integration
    "CrewAIScalpel",
    "RefactorResult",
    # REST API Server (legacy, not MCP-compliant)
    "create_app",
    "run_rest_server",
    "MCPServerConfig",
]
