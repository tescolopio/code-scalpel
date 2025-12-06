"""
Unified IR Node Definitions.

These dataclasses represent the STRUCTURE of programs in a language-agnostic way.
All nodes track their source_language for semantic dispatch.

Design Principles:
    1. Minimal - Only include nodes needed for analysis
    2. Semantic - Represent meaning, not syntax (no commas, parentheses, etc.)
    3. Typed - All fields have type annotations
    4. Immutable-ish - Use dataclasses with default_factory for lists

Node Categories:
    - Statements: IRModule, IRFunctionDef, IRIf, IRFor, IRWhile, IRAssign, etc.
    - Expressions: IRBinaryOp, IRUnaryOp, IRCall, IRName, IRConstant, etc.
    - Special: IRParameter, SourceLocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .operators import (
    BinaryOperator,
    UnaryOperator,
    CompareOperator,
    BoolOperator,
    AugAssignOperator,
)


# =============================================================================
# Source Location
# =============================================================================

@dataclass
class SourceLocation:
    """
    Source code location for error reporting and debugging.
    
    Attributes:
        line: 1-indexed line number
        column: 0-indexed column offset
        end_line: End line (optional, for multi-line nodes)
        end_column: End column (optional)
        filename: Source file path (optional)
    """
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    filename: Optional[str] = None
    
    def __str__(self) -> str:
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"line {self.line}, col {self.column}"


# =============================================================================
# Base Node
# =============================================================================

@dataclass
class IRNode:
    """
    Base class for all Unified IR nodes.
    
    Every node tracks:
        - loc: Source location for error messages
        - source_language: Original language for semantic dispatch
        - _metadata: Extensible dict for analysis passes
    
    The source_language field is CRITICAL for correct semantics:
        - Python "5" + 3 -> TypeError
        - JavaScript "5" + 3 -> "53"
    
    Analysis engines use source_language to select the right LanguageSemantics.
    """
    loc: Optional[SourceLocation] = None
    source_language: str = "unknown"
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_metadata(self, key: str, value: Any) -> "IRNode":
        """Add metadata and return self for chaining."""
        self._metadata[key] = value
        return self


# =============================================================================
# Statement Nodes
# =============================================================================

@dataclass
class IRModule(IRNode):
    """
    Root node representing a source file/module.
    
    Attributes:
        body: List of top-level statements
        docstring: Module docstring if present
    """
    body: List[IRNode] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class IRFunctionDef(IRNode):
    """
    Function definition.
    
    Covers:
        - Python: def foo(), async def foo()
        - JavaScript: function foo(), async function foo(), arrow functions
    
    Attributes:
        name: Function name (empty string for anonymous functions)
        params: List of parameters
        body: List of statements in function body
        return_type: Type annotation string if present (Python type hints, TS types)
        is_async: Whether function is async
        is_generator: Whether function is a generator (yield)
        decorators: List of decorator expressions (Python-specific)
    """
    name: str = ""
    params: List["IRParameter"] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    decorators: List["IRExpr"] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class IRClassDef(IRNode):
    """
    Class definition.
    
    Covers:
        - Python: class Foo(Base)
        - JavaScript: class Foo extends Base
    
    Attributes:
        name: Class name
        bases: Base classes/superclasses
        body: Class body (methods, properties)
        decorators: Class decorators (Python-specific)
    """
    name: str = ""
    bases: List["IRExpr"] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    decorators: List["IRExpr"] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class IRIf(IRNode):
    """
    If statement with optional elif/else chain.
    
    The orelse field contains:
        - Empty list: no else
        - Single IRIf: elif chain
        - Other statements: else block
    
    Attributes:
        test: Condition expression
        body: Statements in if-true branch
        orelse: Statements in else branch (may be another IRIf for elif)
    """
    test: "IRExpr" = None
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)


@dataclass
class IRFor(IRNode):
    """
    For loop (iteration over collection).
    
    Covers:
        - Python: for x in iterable
        - JavaScript: for (let x of iterable), for (let x in obj)
    
    Attributes:
        target: Loop variable (IRName or destructuring pattern)
        iter: Iterable expression
        body: Loop body statements
        orelse: Python's for-else clause (empty for JS)
        is_for_in: True for JS for-in (iterate keys), False for for-of (iterate values)
    """
    target: "IRExpr" = None
    iter: "IRExpr" = None
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)
    is_for_in: bool = False  # JS for-in vs for-of


@dataclass
class IRWhile(IRNode):
    """
    While loop.
    
    Attributes:
        test: Condition expression
        body: Loop body statements
        orelse: Python's while-else clause (empty for JS)
    """
    test: "IRExpr" = None
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)


@dataclass
class IRReturn(IRNode):
    """
    Return statement.
    
    Attributes:
        value: Return value expression (None for bare return)
    """
    value: Optional["IRExpr"] = None


@dataclass
class IRAssign(IRNode):
    """
    Assignment statement.
    
    Covers:
        - Python: x = 1, x = y = 1
        - JavaScript: let x = 1, const x = 1, x = 1
    
    Attributes:
        targets: Assignment targets (multiple for chained assignment)
        value: Value being assigned
        declaration_kind: "let", "const", "var" for JS; None for Python
    """
    targets: List["IRExpr"] = field(default_factory=list)
    value: "IRExpr" = None
    declaration_kind: Optional[str] = None  # "let", "const", "var" for JS


@dataclass
class IRAugAssign(IRNode):
    """
    Augmented assignment (+=, -=, etc.).
    
    Attributes:
        target: Assignment target
        op: Augmented assignment operator
        value: Value to combine with target
    """
    target: "IRExpr" = None
    op: AugAssignOperator = None
    value: "IRExpr" = None


@dataclass
class IRExprStmt(IRNode):
    """
    Expression statement (expression used as statement).
    
    Example: function call as statement: `print("hello")`
    
    Attributes:
        value: The expression
    """
    value: "IRExpr" = None


@dataclass
class IRPass(IRNode):
    """
    Pass/no-op statement.
    
    Python: pass
    JavaScript: ; (empty statement)
    """
    pass


@dataclass
class IRBreak(IRNode):
    """Break statement - exit loop."""
    pass


@dataclass
class IRContinue(IRNode):
    """Continue statement - skip to next iteration."""
    pass


# =============================================================================
# Expression Nodes
# =============================================================================

@dataclass
class IRExpr(IRNode):
    """
    Base class for expression nodes.
    
    Expressions produce values and can be nested.
    """
    pass


@dataclass
class IRBinaryOp(IRExpr):
    """
    Binary operation (a + b, a * b, etc.).
    
    IMPORTANT: The operator is structural, not semantic.
    BinaryOperator.ADD means "the add operation" but its BEHAVIOR
    depends on source_language (Python vs JS string coercion).
    
    Attributes:
        left: Left operand
        op: Binary operator
        right: Right operand
    """
    left: IRExpr = None
    op: BinaryOperator = None
    right: IRExpr = None


@dataclass
class IRUnaryOp(IRExpr):
    """
    Unary operation (-x, not x, ~x, etc.).
    
    Attributes:
        op: Unary operator
        operand: The operand expression
    """
    op: UnaryOperator = None
    operand: IRExpr = None


@dataclass
class IRCompare(IRExpr):
    """
    Comparison operation.
    
    Supports chained comparisons (Python: a < b < c).
    JavaScript comparisons are single: ops=[EQ], comparators=[right].
    
    Attributes:
        left: Left-most value
        ops: List of comparison operators
        comparators: List of values to compare (parallel to ops)
    
    Example (Python a < b < c):
        left=a, ops=[LT, LT], comparators=[b, c]
        Means: a < b AND b < c
    """
    left: IRExpr = None
    ops: List[CompareOperator] = field(default_factory=list)
    comparators: List[IRExpr] = field(default_factory=list)


@dataclass
class IRBoolOp(IRExpr):
    """
    Boolean/logical operation (and, or).
    
    Short-circuit evaluation semantics depend on source_language.
    
    Attributes:
        op: Boolean operator (AND or OR)
        values: List of operands (at least 2)
    """
    op: BoolOperator = None
    values: List[IRExpr] = field(default_factory=list)


@dataclass
class IRCall(IRExpr):
    """
    Function/method call.
    
    Attributes:
        func: The callable expression (IRName, IRAttribute, etc.)
        args: Positional arguments
        kwargs: Keyword arguments (name -> value)
    """
    func: IRExpr = None
    args: List[IRExpr] = field(default_factory=list)
    kwargs: Dict[str, IRExpr] = field(default_factory=dict)


@dataclass
class IRAttribute(IRExpr):
    """
    Attribute access (obj.attr).
    
    Attributes:
        value: The object expression
        attr: The attribute name
    """
    value: IRExpr = None
    attr: str = ""


@dataclass
class IRSubscript(IRExpr):
    """
    Subscript/index access (obj[key]).
    
    Attributes:
        value: The object expression
        slice: The index/key expression
    """
    value: IRExpr = None
    slice: IRExpr = None


@dataclass
class IRName(IRExpr):
    """
    Variable/identifier reference.
    
    Attributes:
        id: The variable name
    """
    id: str = ""


@dataclass
class IRConstant(IRExpr):
    """
    Literal constant value.
    
    Covers:
        - Numbers: 42, 3.14
        - Strings: "hello", 'world'
        - Booleans: True/False, true/false
        - None/null/undefined
    
    Attributes:
        value: The Python-native value
        raw: Original source representation (for preserving "undefined" vs "null")
    """
    value: Any = None
    raw: Optional[str] = None  # Preserves "undefined" vs "null" distinction


@dataclass
class IRList(IRExpr):
    """
    List/Array literal.
    
    Attributes:
        elements: List of element expressions
    """
    elements: List[IRExpr] = field(default_factory=list)


@dataclass
class IRDict(IRExpr):
    """
    Dictionary/Object literal.
    
    Attributes:
        keys: List of key expressions (None for spread: {...x})
        values: List of value expressions
    """
    keys: List[Optional[IRExpr]] = field(default_factory=list)
    values: List[IRExpr] = field(default_factory=list)


@dataclass 
class IRParameter(IRNode):
    """
    Function parameter.
    
    Attributes:
        name: Parameter name
        type_annotation: Type annotation string if present
        default: Default value expression if present
        is_rest: True for rest parameters (*args, ...args)
        is_keyword_only: True for Python keyword-only params (after *)
    """
    name: str = ""
    type_annotation: Optional[str] = None
    default: Optional[IRExpr] = None
    is_rest: bool = False
    is_keyword_only: bool = False


# =============================================================================
# Type Aliases for Convenience
# =============================================================================

# Any statement node
IRStmt = Union[
    IRModule, IRFunctionDef, IRClassDef, IRIf, IRFor, IRWhile,
    IRReturn, IRAssign, IRAugAssign, IRExprStmt, IRPass, IRBreak, IRContinue
]

# Any node
AnyIRNode = Union[IRStmt, IRExpr, IRParameter]
