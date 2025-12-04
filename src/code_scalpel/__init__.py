"""
Code Scalpel - AI Agent toolkit for code analysis using ASTs, PDGs, and Symbolic Execution.

Code Scalpel provides precision tools for AI-driven code analysis and transformation,
enabling AI agents to perform deep analysis and surgical modifications of code.

Quick Start:
    >>> from code_scalpel import CodeAnalyzer
    >>> analyzer = CodeAnalyzer()
    >>> result = analyzer.analyze("def hello(): return 42")
    >>> print(result.metrics.num_functions)
    1

For MCP server:
    >>> from code_scalpel import run_server
    >>> run_server(port=8080)

For AI agent integrations:
    >>> from code_scalpel.integrations import AutogenScalpel, CrewAIScalpel
"""

__version__ = "0.1.0"
__author__ = "Tim Escolopio"
__email__ = "3dtsus@gmail.com"

# Core analysis
# AST tools
from .ast_tools import (
    ASTAnalyzer,
    ASTBuilder,
    ClassMetrics,
    FunctionMetrics,
    build_ast,
    build_ast_from_file,
)
from .code_analyzer import (
    AnalysisLevel,
    AnalysisMetrics,
    AnalysisResult,
    CodeAnalyzer,
    DeadCodeItem,
    RefactorSuggestion,
    analyze_code,
)

# MCP Server
from .integrations.mcp_server import (
    MCPServerConfig,
    create_app,
    run_server,
)

# PDG tools
from .pdg_tools import (
    PDGAnalyzer,
    PDGBuilder,
    build_pdg,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core analysis
    "CodeAnalyzer",
    "AnalysisResult",
    "AnalysisLevel",
    "AnalysisMetrics",
    "DeadCodeItem",
    "RefactorSuggestion",
    "analyze_code",
    # AST tools
    "ASTAnalyzer",
    "ASTBuilder",
    "FunctionMetrics",
    "ClassMetrics",
    "build_ast",
    "build_ast_from_file",
    # PDG tools
    "PDGBuilder",
    "PDGAnalyzer",
    "build_pdg",
    # MCP Server
    "create_app",
    "run_server",
    "MCPServerConfig",
]
