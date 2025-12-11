# src/symbolic_execution_tools/__init__.py
"""
Symbolic Execution Tools for Code Scalpel.

v0.3.0 "The Mathematician" - Security Analysis Edition

This module provides symbolic execution capabilities for Python code analysis.
Building on v0.2.0 "Redemption", this release adds STRING SUPPORT and
SECURITY ANALYSIS for detecting vulnerabilities.

Working Features (v0.3.0):
    - SymbolicAnalyzer: Main entry point for symbolic analysis
    - SecurityAnalyzer: Taint-based vulnerability detection
    - ConstraintSolver: Z3-powered satisfiability checking
    - SymbolicInterpreter: Path exploration with smart forking
    - TypeInferenceEngine: Int/Bool/String type tracking
    - TaintTracker: Data flow tracking for security analysis

Security Detectors:
    - SQL Injection (CWE-89)
    - Cross-Site Scripting (CWE-79)
    - Path Traversal (CWE-22)
    - Command Injection (CWE-78)

Sanitizer Support (v0.3.1):
    - Built-in sanitizer registry (html.escape, shlex.quote, etc.)
    - Custom sanitizer registration via register_sanitizer()
    - Configuration via pyproject.toml [tool.code-scalpel.sanitizers]
    - Type coercion (int, float, bool) fully clears taint

Current Limitations:
    - Float type not yet supported for symbolic execution
    - Loops bounded to 10 iterations
    - Function calls are stubbed (not symbolically executed)

For production use cases with full type support:
    - code_scalpel.ast_tools (AST analysis)
    - code_scalpel.pdg_tools (Program Dependence Graphs)

Example (Symbolic Analysis):
    >>> from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer
    >>> analyzer = SymbolicAnalyzer()
    >>> result = analyzer.analyze("x = 5; y = x * 2 if x > 0 else -x")
    >>> print(f"Paths: {result.total_paths}, Feasible: {result.feasible_count}")

Example (Security Analysis):
    >>> from code_scalpel.symbolic_execution_tools import analyze_security
    >>> result = analyze_security('''
    ...     user_id = request.args.get("id")
    ...     cursor.execute("SELECT * FROM users WHERE id=" + user_id)
    ... ''')
    >>> if result.has_vulnerabilities:
    ...     print(result.summary())

Example (Custom Sanitizer):
    >>> from code_scalpel.symbolic_execution_tools import (
    ...     register_sanitizer, SecuritySink
    ... )
    >>> register_sanitizer(
    ...     "my_sanitize_sql",
    ...     clears_sinks={SecuritySink.SQL_QUERY},
    ...     full_clear=False
    ... )
"""

import warnings

from .constraint_solver import ConstraintSolver
from .engine import SymbolicExecutionEngine, SymbolicAnalyzer

# v0.3.0: Security Analysis
from .taint_tracker import (
    TaintTracker,
    TaintSource,
    TaintLevel,
    SecuritySink,
    TaintInfo,
    TaintedValue,
    Vulnerability,
    # v0.3.1: Sanitizer Support
    SanitizerInfo,
    SANITIZER_REGISTRY,
    register_sanitizer,
    load_sanitizers_from_config,
)
from .security_analyzer import (
    SecurityAnalyzer,
    SecurityAnalysisResult,
    analyze_security,
    find_sql_injections,
    find_xss,
    find_command_injections,
    find_path_traversals,
)

# Emit info on import so users know about limitations
warnings.warn(
    "symbolic_execution_tools v1.2.0 (Stable). "
    "Supports Int/Bool/String. See docs for type limitations.",
    category=UserWarning,
    stacklevel=2,
)

__all__ = [
    # Core symbolic execution
    "ConstraintSolver",
    "SymbolicExecutionEngine",
    "SymbolicAnalyzer",
    # v0.3.0: Security Analysis
    "TaintTracker",
    "TaintSource",
    "TaintLevel",
    "SecuritySink",
    "TaintInfo",
    "TaintedValue",
    "Vulnerability",
    "SecurityAnalyzer",
    "SecurityAnalysisResult",
    "analyze_security",
    "find_sql_injections",
    "find_xss",
    "find_command_injections",
    "find_path_traversals",
    # v0.3.1: Sanitizer Support
    "SanitizerInfo",
    "SANITIZER_REGISTRY",
    "register_sanitizer",
    "load_sanitizers_from_config",
]
