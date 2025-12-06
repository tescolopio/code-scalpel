"""
Unified Intermediate Representation (IR) for Multi-Language Analysis.

This module provides a language-agnostic IR that normalizes AST/CST structures
from different languages (Python, JavaScript, TypeScript) into a common format.

Architecture:
    Source Code -> Language Parser -> Normalizer -> Unified IR -> Analysis Engine

Key Design Decisions (RFC-003):
    1. STRUCTURE is normalized (IRBinaryOp, IRAssign, etc.)
    2. SEMANTICS are NOT normalized (delegated to LanguageSemantics)
    3. Source language is preserved for semantic dispatch

Example:
    >>> from code_scalpel.ir import PythonNormalizer, IRBinaryOp
    >>> normalizer = PythonNormalizer()
    >>> ir = normalizer.normalize("x = 1 + 2")
    >>> isinstance(ir.body[0].value, IRBinaryOp)
    True

Modules:
    nodes: IR node dataclasses (IRModule, IRFunction, IRBinaryOp, etc.)
    operators: Operator enums (BinaryOperator, CompareOperator, etc.)
    normalizers: Language-specific normalizers (PythonNormalizer, etc.)
    semantics: Language-specific behavior (PythonSemantics, JavaScriptSemantics)
"""

from .nodes import (
    # Base
    IRNode,
    SourceLocation,
    # Statements
    IRModule,
    IRFunctionDef,
    IRClassDef,
    IRIf,
    IRFor,
    IRWhile,
    IRReturn,
    IRAssign,
    IRAugAssign,
    IRExprStmt,
    IRPass,
    IRBreak,
    IRContinue,
    # Expressions
    IRExpr,
    IRBinaryOp,
    IRUnaryOp,
    IRCompare,
    IRBoolOp,
    IRCall,
    IRAttribute,
    IRSubscript,
    IRName,
    IRConstant,
    IRList,
    IRDict,
    IRParameter,
)

from .operators import (
    BinaryOperator,
    UnaryOperator,
    CompareOperator,
    BoolOperator,
)

from .semantics import (
    LanguageSemantics,
    PythonSemantics,
    JavaScriptSemantics,
)

from .normalizers import (
    BaseNormalizer,
    PythonNormalizer,
)

__all__ = [
    # Nodes
    "IRNode",
    "SourceLocation",
    "IRModule",
    "IRFunctionDef",
    "IRClassDef",
    "IRIf",
    "IRFor",
    "IRWhile",
    "IRReturn",
    "IRAssign",
    "IRAugAssign",
    "IRExprStmt",
    "IRPass",
    "IRBreak",
    "IRContinue",
    "IRExpr",
    "IRBinaryOp",
    "IRUnaryOp",
    "IRCompare",
    "IRBoolOp",
    "IRCall",
    "IRAttribute",
    "IRSubscript",
    "IRName",
    "IRConstant",
    "IRList",
    "IRDict",
    "IRParameter",
    # Operators
    "BinaryOperator",
    "UnaryOperator",
    "CompareOperator",
    "BoolOperator",
    # Semantics
    "LanguageSemantics",
    "PythonSemantics",
    "JavaScriptSemantics",
    # Normalizers
    "BaseNormalizer",
    "PythonNormalizer",
]
