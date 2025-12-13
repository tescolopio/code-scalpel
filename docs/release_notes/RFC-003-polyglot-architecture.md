# RFC-003: The Polyglot Architecture

**Status:** IMPLEMENTED (Phase 1)  
**Version:** 0.4.0 Target  
**Author:** Code Scalpel Team  
**Date:** 2025-12-06

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| IR Nodes | COMPLETE | `src/code_scalpel/ir/nodes.py` |
| Operators | COMPLETE | `src/code_scalpel/ir/operators.py` |
| PythonNormalizer | COMPLETE | `src/code_scalpel/ir/normalizers/python_normalizer.py` |
| PythonSemantics | COMPLETE | `src/code_scalpel/ir/semantics.py` |
| JavaScriptSemantics | COMPLETE | `src/code_scalpel/ir/semantics.py` |
| JavaScriptNormalizer | PENDING | v0.4.0 Phase 2 |
| IR-based Interpreter | PENDING | v0.4.0 Phase 3 |

**Test Coverage:** 30 new tests (521 total)

## Abstract

This RFC defines the architecture for multi-language support in Code Scalpel. The core insight is that different parsers (Python `ast`, tree-sitter) produce structurally different outputs (AST vs CST), but our analysis engines expect a consistent format. We solve this with a **Unified IR (Intermediate Representation)** layer that normalizes all inputs into a common node format.

## Problem Statement

### The CST vs AST Chasm

| Parser | Output Type | Node Example | Children |
|--------|-------------|--------------|----------|
| Python `ast` | **Abstract** Syntax Tree | `ast.BinOp` | `left`, `op`, `right` (semantic) |
| tree-sitter | **Concrete** Syntax Tree | `binary_expression` | `left`, operator *token*, `right`, parentheses, whitespace |

**The Problem:** Our `SymbolicInterpreter` expects `ast.BinOp`. tree-sitter gives `binary_expression` with "noise" nodes (tokens, punctuation).

**The Risk:** Feeding tree-sitter nodes directly to the interpreter will cause:
- `AttributeError: 'Node' object has no attribute 'op'`
- Logic errors from misinterpreting CST structure
- Maintenance nightmare with language-specific handlers throughout the codebase

### Current Architecture (Python-only)

```
                    Python Source
                         │
                         ▼
                   ┌───────────┐
                   │  ast.parse │
                   └─────┬─────┘
                         │
                         ▼
                    ast.Module
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   TypeInference   SymbolicInterp   SecurityAnalyzer
```

The interpreter directly walks `ast.*` nodes:

```python
# Current: Tightly coupled to Python ast
def _visit_BinOp(self, node: ast.BinOp, state: SymbolicState):
    left = self._evaluate_expr(node.left, state)
    right = self._evaluate_expr(node.right, state)
    op = node.op  # ast.Add, ast.Sub, etc.
```

## Proposed Solution: Unified IR

### Design Principle: "Parse Once, Analyze Everywhere"

```
    Python Source          JavaScript Source        TypeScript Source
         │                       │                        │
         ▼                       ▼                        ▼
   ┌───────────┐          ┌────────────┐          ┌────────────────┐
   │  ast.parse │          │ tree-sitter │          │  tree-sitter   │
   └─────┬─────┘          │ -javascript │          │  -typescript   │
         │                 └──────┬─────┘          └───────┬────────┘
         │                        │                        │
         ▼                        ▼                        ▼
   ┌───────────┐          ┌────────────┐          ┌────────────────┐
   │  Python    │          │ JavaScript  │          │   TypeScript   │
   │ Normalizer │          │  Normalizer │          │   Normalizer   │
   └─────┬─────┘          └──────┬─────┘          └───────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  Unified IR   │
                          │   (scalpel)   │
                          └──────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        TypeInference    SymbolicInterp    SecurityAnalyzer
```

### Unified IR Node Types

We define a minimal set of IR nodes that capture the **semantics** (not syntax) of programs:

```python
# src/code_scalpel/ir/nodes.py

from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from enum import Enum, auto


class IRNodeKind(Enum):
    """All supported IR node types."""
    # Statements
    MODULE = auto()
    FUNCTION_DEF = auto()
    CLASS_DEF = auto()
    IF = auto()
    FOR = auto()
    WHILE = auto()
    RETURN = auto()
    ASSIGN = auto()
    AUG_ASSIGN = auto()
    EXPR_STMT = auto()
    PASS = auto()
    BREAK = auto()
    CONTINUE = auto()
    
    # Expressions
    BINARY_OP = auto()
    UNARY_OP = auto()
    COMPARE = auto()
    BOOL_OP = auto()
    CALL = auto()
    ATTRIBUTE = auto()
    SUBSCRIPT = auto()
    NAME = auto()
    CONSTANT = auto()
    LIST = auto()
    DICT = auto()
    
    # Special
    ARGUMENT = auto()
    PARAMETER = auto()


class BinaryOperator(Enum):
    """Binary operators normalized across languages."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOOR_DIV = "//"  # Python: //, JS: Math.floor(a/b)
    MOD = "%"
    POW = "**"        # Python: **, JS: **
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"


class CompareOperator(Enum):
    """Comparison operators normalized across languages."""
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IS = "is"          # Python only
    IS_NOT = "is not"  # Python only
    IN = "in"          # Python only
    NOT_IN = "not in"  # Python only
    STRICT_EQ = "==="  # JS only
    STRICT_NE = "!=="  # JS only


class UnaryOperator(Enum):
    """Unary operators normalized across languages."""
    NEG = "-"
    POS = "+"
    NOT = "not"    # Python: not, JS: !
    INVERT = "~"   # Bitwise invert


class BoolOperator(Enum):
    """Boolean operators normalized across languages."""
    AND = "and"    # Python: and, JS: &&
    OR = "or"      # Python: or, JS: ||


@dataclass
class SourceLocation:
    """Source location for error reporting."""
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    filename: Optional[str] = None


@dataclass
class IRNode:
    """
    Base class for all Unified IR nodes.
    
    Every node tracks:
    - kind: The semantic node type
    - loc: Source location for error messages
    - source_language: Original language (for language-specific semantics)
    """
    kind: IRNodeKind
    loc: Optional[SourceLocation] = None
    source_language: str = "unknown"
    
    # Metadata for analysis passes
    _metadata: dict = field(default_factory=dict)


@dataclass
class IRModule(IRNode):
    """Root node - a source file/module."""
    body: List[IRNode] = field(default_factory=list)
    kind: IRNodeKind = IRNodeKind.MODULE


@dataclass 
class IRFunctionDef(IRNode):
    """Function definition."""
    name: str = ""
    params: List['IRParameter'] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    return_type: Optional[str] = None  # Type annotation if present
    is_async: bool = False
    kind: IRNodeKind = IRNodeKind.FUNCTION_DEF


@dataclass
class IRParameter(IRNode):
    """Function parameter."""
    name: str = ""
    type_annotation: Optional[str] = None
    default: Optional['IRExpr'] = None
    kind: IRNodeKind = IRNodeKind.PARAMETER


@dataclass
class IRIf(IRNode):
    """If statement with optional elif/else."""
    test: 'IRExpr' = None
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)  # elif/else chain
    kind: IRNodeKind = IRNodeKind.IF


@dataclass
class IRFor(IRNode):
    """For loop (iteration)."""
    target: 'IRExpr' = None  # Loop variable
    iter: 'IRExpr' = None    # Iterable
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)  # Python's for-else
    kind: IRNodeKind = IRNodeKind.FOR


@dataclass
class IRWhile(IRNode):
    """While loop."""
    test: 'IRExpr' = None
    body: List[IRNode] = field(default_factory=list)
    orelse: List[IRNode] = field(default_factory=list)  # Python's while-else
    kind: IRNodeKind = IRNodeKind.WHILE


@dataclass
class IRReturn(IRNode):
    """Return statement."""
    value: Optional['IRExpr'] = None
    kind: IRNodeKind = IRNodeKind.RETURN


@dataclass
class IRAssign(IRNode):
    """Assignment statement."""
    targets: List['IRExpr'] = field(default_factory=list)
    value: 'IRExpr' = None
    kind: IRNodeKind = IRNodeKind.ASSIGN


@dataclass
class IRAugAssign(IRNode):
    """Augmented assignment (+=, -=, etc.)."""
    target: 'IRExpr' = None
    op: BinaryOperator = None
    value: 'IRExpr' = None
    kind: IRNodeKind = IRNodeKind.AUG_ASSIGN


# Expression nodes

@dataclass
class IRExpr(IRNode):
    """Base for expression nodes."""
    pass


@dataclass
class IRBinaryOp(IRExpr):
    """Binary operation (a + b, a * b, etc.)."""
    left: IRExpr = None
    op: BinaryOperator = None
    right: IRExpr = None
    kind: IRNodeKind = IRNodeKind.BINARY_OP


@dataclass
class IRUnaryOp(IRExpr):
    """Unary operation (-x, not x, etc.)."""
    op: UnaryOperator = None
    operand: IRExpr = None
    kind: IRNodeKind = IRNodeKind.UNARY_OP


@dataclass
class IRCompare(IRExpr):
    """
    Comparison operation.
    
    Supports chained comparisons (Python: a < b < c).
    JS comparisons are always single: ops=[EQ], comparators=[right].
    """
    left: IRExpr = None
    ops: List[CompareOperator] = field(default_factory=list)
    comparators: List[IRExpr] = field(default_factory=list)
    kind: IRNodeKind = IRNodeKind.COMPARE


@dataclass
class IRBoolOp(IRExpr):
    """Boolean operation (and, or)."""
    op: BoolOperator = None
    values: List[IRExpr] = field(default_factory=list)
    kind: IRNodeKind = IRNodeKind.BOOL_OP


@dataclass
class IRCall(IRExpr):
    """Function/method call."""
    func: IRExpr = None  # The callable (name or attribute)
    args: List[IRExpr] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)  # keyword arguments
    kind: IRNodeKind = IRNodeKind.CALL


@dataclass
class IRAttribute(IRExpr):
    """Attribute access (obj.attr)."""
    value: IRExpr = None  # The object
    attr: str = ""        # The attribute name
    kind: IRNodeKind = IRNodeKind.ATTRIBUTE


@dataclass
class IRSubscript(IRExpr):
    """Subscript access (obj[key])."""
    value: IRExpr = None  # The object
    slice: IRExpr = None  # The index/key
    kind: IRNodeKind = IRNodeKind.SUBSCRIPT


@dataclass
class IRName(IRExpr):
    """Variable reference."""
    id: str = ""
    kind: IRNodeKind = IRNodeKind.NAME


@dataclass
class IRConstant(IRExpr):
    """
    Literal constant value.
    
    Python and JS constants are normalized:
    - Python True/False -> bool
    - JS true/false -> bool
    - Python None -> None
    - JS null/undefined -> None (with metadata distinguishing)
    """
    value: Any = None
    kind: IRNodeKind = IRNodeKind.CONSTANT


@dataclass
class IRList(IRExpr):
    """List/Array literal."""
    elements: List[IRExpr] = field(default_factory=list)
    kind: IRNodeKind = IRNodeKind.LIST


@dataclass
class IRDict(IRExpr):
    """Dictionary/Object literal."""
    keys: List[IRExpr] = field(default_factory=list)
    values: List[IRExpr] = field(default_factory=list)
    kind: IRNodeKind = IRNodeKind.DICT
```

### Normalizer Interface

Each language implements a normalizer that converts its native AST/CST to Unified IR:

```python
# src/code_scalpel/ir/normalizers/base.py

from abc import ABC, abstractmethod
from typing import Union
from ..nodes import IRModule


class BaseNormalizer(ABC):
    """
    Base class for language-specific normalizers.
    
    A normalizer converts a language's native AST/CST into Unified IR.
    """
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name (e.g., 'python', 'javascript')."""
        pass
    
    @abstractmethod
    def normalize(self, source: str) -> IRModule:
        """
        Parse source code and normalize to Unified IR.
        
        Args:
            source: Source code string
            
        Returns:
            IRModule representing the normalized AST
            
        Raises:
            SyntaxError: If source cannot be parsed
        """
        pass
    
    @abstractmethod
    def normalize_node(self, node) -> Union['IRNode', list]:
        """
        Normalize a single native AST node to IR.
        
        Args:
            node: Native AST/CST node
            
        Returns:
            IRNode or list of IRNodes
        """
        pass
```

### Python Normalizer (Reference Implementation)

```python
# src/code_scalpel/ir/normalizers/python_normalizer.py

import ast
from typing import Union, List

from .base import BaseNormalizer
from ..nodes import (
    IRModule, IRFunctionDef, IRParameter, IRIf, IRFor, IRWhile,
    IRReturn, IRAssign, IRAugAssign, IRBinaryOp, IRUnaryOp,
    IRCompare, IRBoolOp, IRCall, IRAttribute, IRSubscript,
    IRName, IRConstant, IRList, IRDict, IRNode, IRExpr,
    BinaryOperator, UnaryOperator, CompareOperator, BoolOperator,
    SourceLocation,
)


class PythonNormalizer(BaseNormalizer):
    """Normalizes Python ast.* nodes to Unified IR."""
    
    @property
    def language(self) -> str:
        return "python"
    
    def normalize(self, source: str) -> IRModule:
        tree = ast.parse(source)
        return self._normalize_module(tree)
    
    def _normalize_module(self, node: ast.Module) -> IRModule:
        return IRModule(
            body=[self.normalize_node(stmt) for stmt in node.body],
            source_language="python",
        )
    
    def normalize_node(self, node: ast.AST) -> Union[IRNode, List[IRNode]]:
        """Dispatch to appropriate normalizer based on node type."""
        method_name = f"_normalize_{node.__class__.__name__}"
        method = getattr(self, method_name, self._normalize_unknown)
        return method(node)
    
    def _make_loc(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            line=getattr(node, 'lineno', 0),
            column=getattr(node, 'col_offset', 0),
            end_line=getattr(node, 'end_lineno', None),
            end_column=getattr(node, 'end_col_offset', None),
        )
    
    # === Statement Normalizers ===
    
    def _normalize_FunctionDef(self, node: ast.FunctionDef) -> IRFunctionDef:
        return IRFunctionDef(
            name=node.name,
            params=[self._normalize_arg(arg) for arg in node.args.args],
            body=[self.normalize_node(stmt) for stmt in node.body],
            return_type=ast.unparse(node.returns) if node.returns else None,
            is_async=False,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_arg(self, node: ast.arg) -> IRParameter:
        return IRParameter(
            name=node.arg,
            type_annotation=ast.unparse(node.annotation) if node.annotation else None,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_If(self, node: ast.If) -> IRIf:
        return IRIf(
            test=self.normalize_node(node.test),
            body=[self.normalize_node(stmt) for stmt in node.body],
            orelse=[self.normalize_node(stmt) for stmt in node.orelse],
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_Return(self, node: ast.Return) -> IRReturn:
        return IRReturn(
            value=self.normalize_node(node.value) if node.value else None,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_Assign(self, node: ast.Assign) -> IRAssign:
        return IRAssign(
            targets=[self.normalize_node(t) for t in node.targets],
            value=self.normalize_node(node.value),
            loc=self._make_loc(node),
            source_language="python",
        )
    
    # === Expression Normalizers ===
    
    def _normalize_BinOp(self, node: ast.BinOp) -> IRBinaryOp:
        return IRBinaryOp(
            left=self.normalize_node(node.left),
            op=self._normalize_binary_op(node.op),
            right=self.normalize_node(node.right),
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_binary_op(self, op: ast.operator) -> BinaryOperator:
        mapping = {
            ast.Add: BinaryOperator.ADD,
            ast.Sub: BinaryOperator.SUB,
            ast.Mult: BinaryOperator.MUL,
            ast.Div: BinaryOperator.DIV,
            ast.FloorDiv: BinaryOperator.FLOOR_DIV,
            ast.Mod: BinaryOperator.MOD,
            ast.Pow: BinaryOperator.POW,
            ast.BitAnd: BinaryOperator.BIT_AND,
            ast.BitOr: BinaryOperator.BIT_OR,
            ast.BitXor: BinaryOperator.BIT_XOR,
            ast.LShift: BinaryOperator.LSHIFT,
            ast.RShift: BinaryOperator.RSHIFT,
        }
        return mapping.get(type(op), BinaryOperator.ADD)
    
    def _normalize_Compare(self, node: ast.Compare) -> IRCompare:
        return IRCompare(
            left=self.normalize_node(node.left),
            ops=[self._normalize_cmp_op(op) for op in node.ops],
            comparators=[self.normalize_node(c) for c in node.comparators],
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_cmp_op(self, op: ast.cmpop) -> CompareOperator:
        mapping = {
            ast.Eq: CompareOperator.EQ,
            ast.NotEq: CompareOperator.NE,
            ast.Lt: CompareOperator.LT,
            ast.LtE: CompareOperator.LE,
            ast.Gt: CompareOperator.GT,
            ast.GtE: CompareOperator.GE,
            ast.Is: CompareOperator.IS,
            ast.IsNot: CompareOperator.IS_NOT,
            ast.In: CompareOperator.IN,
            ast.NotIn: CompareOperator.NOT_IN,
        }
        return mapping.get(type(op), CompareOperator.EQ)
    
    def _normalize_Name(self, node: ast.Name) -> IRName:
        return IRName(
            id=node.id,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_Constant(self, node: ast.Constant) -> IRConstant:
        return IRConstant(
            value=node.value,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_Call(self, node: ast.Call) -> IRCall:
        return IRCall(
            func=self.normalize_node(node.func),
            args=[self.normalize_node(arg) for arg in node.args],
            kwargs={kw.arg: self.normalize_node(kw.value) for kw in node.keywords if kw.arg},
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_Attribute(self, node: ast.Attribute) -> IRAttribute:
        return IRAttribute(
            value=self.normalize_node(node.value),
            attr=node.attr,
            loc=self._make_loc(node),
            source_language="python",
        )
    
    def _normalize_unknown(self, node: ast.AST) -> IRNode:
        """Fallback for unsupported nodes."""
        raise NotImplementedError(
            f"Python node type {node.__class__.__name__} not yet supported"
        )
```

### JavaScript Normalizer (v0.4.0 Target)

```python
# src/code_scalpel/ir/normalizers/javascript_normalizer.py

import tree_sitter
import tree_sitter_javascript as ts_js

from .base import BaseNormalizer
from ..nodes import (
    IRModule, IRFunctionDef, IRParameter, IRIf, IRFor, IRWhile,
    IRReturn, IRAssign, IRBinaryOp, IRUnaryOp, IRCompare, IRBoolOp,
    IRCall, IRAttribute, IRSubscript, IRName, IRConstant, IRList, IRDict,
    BinaryOperator, UnaryOperator, CompareOperator, BoolOperator,
    SourceLocation, IRNode,
)


class JavaScriptNormalizer(BaseNormalizer):
    """
    Normalizes tree-sitter JavaScript CST to Unified IR.
    
    Key insight: tree-sitter gives us a CST with "noise" nodes (tokens, punctuation).
    We extract only the semantic nodes and convert them to IR.
    """
    
    def __init__(self):
        self._language = tree_sitter.Language(ts_js.language())
        self._parser = tree_sitter.Parser(self._language)
    
    @property
    def language(self) -> str:
        return "javascript"
    
    def normalize(self, source: str) -> IRModule:
        tree = self._parser.parse(source.encode('utf-8'))
        return self._normalize_program(tree.root_node, source)
    
    def _normalize_program(self, node, source: str) -> IRModule:
        """Normalize the root 'program' node."""
        body = []
        for child in node.children:
            if child.type in ('function_declaration', 'lexical_declaration', 
                             'variable_declaration', 'expression_statement',
                             'if_statement', 'for_statement', 'while_statement',
                             'return_statement'):
                body.append(self.normalize_node(child, source))
        return IRModule(body=body, source_language="javascript")
    
    def normalize_node(self, node, source: str) -> IRNode:
        """Dispatch based on tree-sitter node type."""
        method_name = f"_normalize_{node.type}"
        method = getattr(self, method_name, self._normalize_unknown)
        return method(node, source)
    
    def _get_text(self, node, source: str) -> str:
        """Extract text content from a node."""
        return source[node.start_byte:node.end_byte]
    
    def _make_loc(self, node) -> SourceLocation:
        return SourceLocation(
            line=node.start_point[0] + 1,  # tree-sitter is 0-indexed
            column=node.start_point[1],
            end_line=node.end_point[0] + 1,
            end_column=node.end_point[1],
        )
    
    # === Statement Normalizers ===
    
    def _normalize_function_declaration(self, node, source: str) -> IRFunctionDef:
        """
        tree-sitter structure:
        function_declaration [
            'function',           <- keyword token (skip)
            identifier,           <- name
            formal_parameters,    <- params
            statement_block       <- body
        ]
        """
        name = ""
        params = []
        body = []
        
        for child in node.children:
            if child.type == 'identifier':
                name = self._get_text(child, source)
            elif child.type == 'formal_parameters':
                params = self._normalize_formal_parameters(child, source)
            elif child.type == 'statement_block':
                body = self._normalize_statement_block(child, source)
        
        return IRFunctionDef(
            name=name,
            params=params,
            body=body,
            loc=self._make_loc(node),
            source_language="javascript",
        )
    
    def _normalize_formal_parameters(self, node, source: str) -> list:
        """Extract parameters, skipping punctuation."""
        params = []
        for child in node.children:
            if child.type == 'identifier':
                params.append(IRParameter(
                    name=self._get_text(child, source),
                    loc=self._make_loc(child),
                    source_language="javascript",
                ))
        return params
    
    def _normalize_statement_block(self, node, source: str) -> list:
        """Normalize a { ... } block, skipping braces."""
        body = []
        for child in node.children:
            if child.type not in ('{', '}'):
                normalized = self.normalize_node(child, source)
                if normalized:
                    body.append(normalized)
        return body
    
    def _normalize_if_statement(self, node, source: str) -> IRIf:
        """
        tree-sitter structure:
        if_statement [
            'if',
            parenthesized_expression,  <- test (includes parens)
            statement_block,           <- body
            else_clause?               <- optional else/else if
        ]
        """
        test = None
        body = []
        orelse = []
        
        for child in node.children:
            if child.type == 'parenthesized_expression':
                # Unwrap the parentheses to get the actual expression
                test = self._unwrap_parenthesized(child, source)
            elif child.type == 'statement_block':
                body = self._normalize_statement_block(child, source)
            elif child.type == 'else_clause':
                orelse = self._normalize_else_clause(child, source)
        
        return IRIf(
            test=test,
            body=body,
            orelse=orelse,
            loc=self._make_loc(node),
            source_language="javascript",
        )
    
    def _unwrap_parenthesized(self, node, source: str):
        """Remove parentheses wrapper to get inner expression."""
        for child in node.children:
            if child.type not in ('(', ')'):
                return self.normalize_node(child, source)
        return None
    
    def _normalize_return_statement(self, node, source: str) -> IRReturn:
        """Extract return value, skipping 'return' keyword and semicolon."""
        value = None
        for child in node.children:
            if child.type not in ('return', ';'):
                value = self.normalize_node(child, source)
                break
        return IRReturn(value=value, loc=self._make_loc(node), source_language="javascript")
    
    # === Expression Normalizers ===
    
    def _normalize_binary_expression(self, node, source: str) -> IRBinaryOp:
        """
        tree-sitter structure:
        binary_expression [
            left_expr,
            operator_token,  <- '+', '-', '*', etc.
            right_expr
        ]
        """
        left = None
        op = None
        right = None
        
        for child in node.children:
            if child.is_named and left is None:
                left = self.normalize_node(child, source)
            elif not child.is_named:  # operator token
                op = self._map_binary_op(self._get_text(child, source))
            elif child.is_named:
                right = self.normalize_node(child, source)
        
        # Handle comparison operators separately
        if op in (CompareOperator.EQ, CompareOperator.NE, CompareOperator.LT,
                  CompareOperator.LE, CompareOperator.GT, CompareOperator.GE,
                  CompareOperator.STRICT_EQ, CompareOperator.STRICT_NE):
            return IRCompare(
                left=left,
                ops=[op],
                comparators=[right],
                loc=self._make_loc(node),
                source_language="javascript",
            )
        
        return IRBinaryOp(
            left=left,
            op=op,
            right=right,
            loc=self._make_loc(node),
            source_language="javascript",
        )
    
    def _map_binary_op(self, op_str: str):
        """Map JS operator string to IR operator enum."""
        mapping = {
            '+': BinaryOperator.ADD,
            '-': BinaryOperator.SUB,
            '*': BinaryOperator.MUL,
            '/': BinaryOperator.DIV,
            '%': BinaryOperator.MOD,
            '**': BinaryOperator.POW,
            '&': BinaryOperator.BIT_AND,
            '|': BinaryOperator.BIT_OR,
            '^': BinaryOperator.BIT_XOR,
            '<<': BinaryOperator.LSHIFT,
            '>>': BinaryOperator.RSHIFT,
            # Comparisons (returned as CompareOperator)
            '==': CompareOperator.EQ,
            '!=': CompareOperator.NE,
            '===': CompareOperator.STRICT_EQ,
            '!==': CompareOperator.STRICT_NE,
            '<': CompareOperator.LT,
            '<=': CompareOperator.LE,
            '>': CompareOperator.GT,
            '>=': CompareOperator.GE,
        }
        return mapping.get(op_str, BinaryOperator.ADD)
    
    def _normalize_identifier(self, node, source: str) -> IRName:
        return IRName(
            id=self._get_text(node, source),
            loc=self._make_loc(node),
            source_language="javascript",
        )
    
    def _normalize_number(self, node, source: str) -> IRConstant:
        text = self._get_text(node, source)
        # Parse as int or float
        value = float(text) if '.' in text else int(text)
        return IRConstant(value=value, loc=self._make_loc(node), source_language="javascript")
    
    def _normalize_string(self, node, source: str) -> IRConstant:
        text = self._get_text(node, source)
        # Remove quotes
        if text.startswith(("'", '"', '`')):
            text = text[1:-1]
        return IRConstant(value=text, loc=self._make_loc(node), source_language="javascript")
    
    def _normalize_true(self, node, source: str) -> IRConstant:
        return IRConstant(value=True, loc=self._make_loc(node), source_language="javascript")
    
    def _normalize_false(self, node, source: str) -> IRConstant:
        return IRConstant(value=False, loc=self._make_loc(node), source_language="javascript")
    
    def _normalize_null(self, node, source: str) -> IRConstant:
        return IRConstant(value=None, loc=self._make_loc(node), source_language="javascript")
    
    def _normalize_call_expression(self, node, source: str) -> IRCall:
        """
        tree-sitter structure:
        call_expression [
            function_expr,    <- the callable
            arguments         <- (arg1, arg2, ...)
        ]
        """
        func = None
        args = []
        
        for child in node.children:
            if child.type == 'arguments':
                args = self._normalize_arguments(child, source)
            elif child.type not in ('(', ')'):
                func = self.normalize_node(child, source)
        
        return IRCall(
            func=func,
            args=args,
            loc=self._make_loc(node),
            source_language="javascript",
        )
    
    def _normalize_arguments(self, node, source: str) -> list:
        """Extract arguments, skipping punctuation."""
        args = []
        for child in node.children:
            if child.type not in ('(', ')', ','):
                args.append(self.normalize_node(child, source))
        return args
    
    def _normalize_unknown(self, node, source: str) -> None:
        """Skip unknown nodes (comments, whitespace, etc.)."""
        return None
```

## Migration Path

### Phase 1: Introduce IR Without Breaking Changes

1. Create `src/code_scalpel/ir/` module with node definitions
2. Implement `PythonNormalizer` as 1:1 mapping from `ast`
3. Add `--use-ir` flag to CLI (opt-in)
4. Run both paths in parallel, assert identical results

### Phase 2: Add JavaScript Support

1. Implement `JavaScriptNormalizer`
2. Update `SymbolicInterpreter` to walk `IRNode` instead of `ast.AST`
3. Add `--language` flag to CLI

### Phase 3: Deprecate Direct AST Access

1. Route all analysis through IR
2. Deprecate `ast`-specific APIs
3. Remove `ast` coupling from interpreter

## Open Questions

1. **Type Coercion:** JS `"5" + 3 = "53"` vs Python `"5" + 3 = TypeError`. How do we handle this in symbolic execution?

2. **Scoping:** JS has `var` (function scope), `let`/`const` (block scope). Python has only function/module scope. Do we normalize scope rules?

3. **This/Self:** JS `this` is dynamic, Python `self` is explicit. Does IR need a special node for receiver binding?

## References

- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Python ast: https://docs.python.org/3/library/ast.html
- Babel AST (inspiration for IR design): https://github.com/babel/babel/tree/main/packages/babel-types
