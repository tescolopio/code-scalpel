"""
Code Scalpel Integrations - Integration wrappers for AI agent frameworks.

This module provides integration wrappers for various AI agent frameworks
including Autogen, CrewAI, LangChain, and Claude, as well as an MCP server
for HTTP-based agent queries.
"""

from .autogen import AutogenScalpel, AnalysisResult, AutogenCodeAnalysisAgent
from .crewai import CrewAIScalpel, RefactorResult
from .mcp_server import create_app, run_server, MCPServerConfig

__all__ = [
    # Autogen integration
    "AutogenScalpel",
    "AutogenCodeAnalysisAgent",  # Backward compatibility alias
    "AnalysisResult",
    
    # CrewAI integration
    "CrewAIScalpel",
    "RefactorResult",
    
    # MCP Server
    "create_app",
    "run_server",
    "MCPServerConfig",
]
