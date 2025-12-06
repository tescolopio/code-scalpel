# src/symbolic_execution_tools/__init__.py
"""
Symbolic Execution Tools for Code Scalpel.

🆕 BETA STATUS (v0.2.0 "Redemption")

This module provides symbolic execution capabilities for Python code analysis.
The "Redemption" release (v0.2.0) brings working symbolic execution with:

✅ Working Features:
- SymbolicAnalyzer: Main entry point for symbolic analysis
- ConstraintSolver: Z3-powered satisfiability checking
- SymbolicInterpreter: Path exploration with smart forking
- TypeInferenceEngine: Int/Bool type tracking

⚠️ Current Limitations (Phase 1):
- Int and Bool types only (no floats, strings, lists yet)
- Loops bounded to 10 iterations
- Function calls are stubbed (not symbolically executed)

For production use cases with full type support:
- code_scalpel.ast_tools (AST analysis)
- code_scalpel.pdg_tools (Program Dependence Graphs)
- code_scalpel.code_analyzer (High-level analysis)

Example:
    >>> from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer
    >>> analyzer = SymbolicAnalyzer()
    >>> result = analyzer.analyze("x = 5; y = x * 2 if x > 0 else -x")
    >>> print(f"Paths: {result.total_paths}, Feasible: {result.feasible_count}")
"""

import warnings

# Emit warning on import so users know about limitations
warnings.warn(
    "symbolic_execution_tools is BETA (v0.2.0). "
    "Supports Int/Bool only. See docs for limitations.",
    category=UserWarning,
    stacklevel=2,
)

from .constraint_solver import ConstraintSolver
from .engine import SymbolicExecutionEngine
from .model_checker import ModelChecker
from .path_explorer import PathExplorer
from .result_analyzer import ResultAnalyzer
from .symbolic_executor import SymbolicExecutor
from .test_generator import TestGenerator

__all__ = [
    "ConstraintSolver",
    "SymbolicExecutionEngine",
    "PathExplorer",
    "ResultAnalyzer",
    "ModelChecker",
    "SymbolicExecutor",
    "TestGenerator",
]
