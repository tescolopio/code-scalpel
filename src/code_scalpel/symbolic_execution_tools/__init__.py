# src/symbolic_execution_tools/__init__.py
"""
Symbolic Execution Tools for Code Scalpel.

⚠️  EXPERIMENTAL / ALPHA STATUS ⚠️

This module is under active development and is NOT production-ready.
The symbolic execution engine is incomplete and will fail on most inputs.

Known Limitations:
- SymbolicExecutionEngine.execute() crashes due to missing _infer_type method
- ConstraintSolver lacks a solve()/check() method
- Path exploration is incomplete

For production use cases, please use the stable modules:
- code_scalpel.ast_tools (AST analysis)
- code_scalpel.pdg_tools (Program Dependence Graphs)
- code_scalpel.code_analyzer (High-level analysis)

This module is included for experimental use and to show our roadmap direction.
Contributions welcome! See ROADMAP.md for planned improvements.
"""

import warnings

# Emit warning on import so users know this is experimental
warnings.warn(
    "symbolic_execution_tools is EXPERIMENTAL and incomplete. "
    "The engine will fail on most inputs. Use ast_tools or pdg_tools for production.",
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
