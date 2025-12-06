"""
Normalizers - Convert language-specific AST/CST to Unified IR.

Each normalizer converts its language's native parse tree to our Unified IR.
The IR preserves source_language for semantic dispatch.

Available Normalizers:
    - PythonNormalizer: Python ast.* -> IR
    - JavaScriptNormalizer: tree-sitter CST -> IR (v0.4.0)
"""

from .base import BaseNormalizer
from .python_normalizer import PythonNormalizer

__all__ = [
    "BaseNormalizer",
    "PythonNormalizer",
]
