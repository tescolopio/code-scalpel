"""
Code Scalpel Agents - AI Agent Framework for Code Analysis.

This module provides specialized AI agents that demonstrate how to use Code Scalpel's
MCP tools for various code analysis and improvement tasks.

Agents follow the OODA loop (Observe, Orient, Decide, Act) and use MCP tools to:
- Observe: Gather information about codebases
- Orient: Analyze and understand context
- Decide: Determine optimal actions
- Act: Execute changes safely with verification

Available Agents:
- CodeReviewAgent: Comprehensive code quality and security review
- SecurityAgent: Specialized security vulnerability detection and remediation
- OptimizationAgent: Performance analysis and optimization suggestions

Example Usage:
    from code_scalpel.agents import CodeReviewAgent

    agent = CodeReviewAgent(workspace_root="/path/to/project")
    result = await agent.execute_ooda_loop("src/main.py")
"""

from .base_agent import BaseCodeAnalysisAgent, AgentContext
from .code_review_agent import CodeReviewAgent
from .security_agent import SecurityAgent
from .optimazation_agent import OptimizationAgent

__all__ = [
    "BaseCodeAnalysisAgent",
    "AgentContext",
    "CodeReviewAgent",
    "SecurityAgent",
    "OptimizationAgent"
]