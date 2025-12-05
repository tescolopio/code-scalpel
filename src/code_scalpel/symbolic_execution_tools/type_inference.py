"""
Type Inference Engine - Milestone M1
=====================================

Infers Z3-compatible types from Python AST.

Scope (per RFC-001, Gall's Law):
- Integer literals and operations → INT
- Boolean literals and comparisons → BOOL
- Everything else → UNKNOWN

This is a flow-insensitive analysis: we track the LAST assignment to each variable.
"""

import ast
from enum import Enum, auto
from typing import Dict, Optional, Union
from dataclasses import dataclass

from z3 import IntSort, BoolSort, Sort


class InferredType(Enum):
    """
    Types that can be inferred from Python code.
    
    Phase 1 only supports INT and BOOL.
    UNKNOWN means we cannot determine the type (unsupported construct).
    """
    INT = auto()
    BOOL = auto()
    UNKNOWN = auto()
    
    def to_z3_sort(self) -> Sort:
        """Convert to Z3 sort. Raises if UNKNOWN."""
        if self == InferredType.INT:
            return IntSort()
        elif self == InferredType.BOOL:
            return BoolSort()
        else:
            raise ValueError(f"Cannot convert {self.name} to Z3 sort")
    
    def __repr__(self) -> str:
        return f"InferredType.{self.name}"


class TypeInferenceEngine:
    """
    Infers types for variables in Python code.
    
    Usage:
        engine = TypeInferenceEngine()
        types = engine.infer("x = 1\\ny = x > 0")
        # types = {"x": InferredType.INT, "y": InferredType.BOOL}
    
    Design:
        - Flow-insensitive: tracks last assignment per variable
        - Conservative: unknown operations produce UNKNOWN
        - Taint propagation: UNKNOWN + anything = UNKNOWN
    """
    
    def __init__(self):
        self._types: Dict[str, InferredType] = {}
    
    def infer(self, code: str) -> Dict[str, InferredType]:
        """
        Infer types for all variables in the code.
        
        Args:
            code: Python source code as string
            
        Returns:
            Dict mapping variable names to their inferred types
        """
        self._types = {}
        
        if not code.strip():
            return {}
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        
        self._visit(tree)
        return self._types.copy()
    
    def _visit(self, node: ast.AST) -> None:
        """Visit AST node and its children."""
        if isinstance(node, ast.Assign):
            self._handle_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._handle_aug_assign(node)
        
        # Visit children
        for child in ast.iter_child_nodes(node):
            self._visit(child)
    
    def _handle_assign(self, node: ast.Assign) -> None:
        """Handle assignment: x = expr or x = y = expr"""
        value_type = self._infer_expr_type(node.value)
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._types[target.id] = value_type
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                # Tuple unpacking: a, b = ... → both UNKNOWN
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self._types[elt.id] = InferredType.UNKNOWN
    
    def _handle_aug_assign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment: x += 1"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            current_type = self._types.get(var_name, InferredType.UNKNOWN)
            value_type = self._infer_expr_type(node.value)
            
            # Result type depends on operation and operands
            result_type = self._combine_types_for_binop(
                current_type, node.op, value_type
            )
            self._types[var_name] = result_type
    
    def _infer_expr_type(self, node: ast.expr) -> InferredType:
        """Infer the type of an expression."""
        
        # Integer literals
        if isinstance(node, ast.Constant):
            return self._infer_constant_type(node.value)
        
        # Legacy Num node (Python 3.7 compat, but we're 3.9+)
        if isinstance(node, ast.Num):
            return self._infer_constant_type(node.n)
        
        # Name reference: look up in our type map
        if isinstance(node, ast.Name):
            return self._types.get(node.id, InferredType.UNKNOWN)
        
        # Unary operations: -x, not x, +x, ~x
        if isinstance(node, ast.UnaryOp):
            return self._infer_unary_type(node)
        
        # Binary operations: x + y, x * y, etc.
        if isinstance(node, ast.BinOp):
            return self._infer_binop_type(node)
        
        # Boolean operations: x and y, x or y
        if isinstance(node, ast.BoolOp):
            return self._infer_boolop_type(node)
        
        # Comparisons: x < y, x == y, 1 < x < 10
        if isinstance(node, ast.Compare):
            return InferredType.BOOL
        
        # IfExp: x if cond else y
        if isinstance(node, ast.IfExp):
            # Both branches should have same type ideally
            # For now, return UNKNOWN (conservative)
            return InferredType.UNKNOWN
        
        # Everything else: function calls, attribute access, subscript, etc.
        # → UNKNOWN (Phase 1 limitation)
        return InferredType.UNKNOWN
    
    def _infer_constant_type(self, value) -> InferredType:
        """Infer type from a constant value."""
        if isinstance(value, bool):
            # IMPORTANT: bool check MUST come before int
            # because isinstance(True, int) is True in Python!
            return InferredType.BOOL
        elif isinstance(value, int):
            return InferredType.INT
        else:
            # float, str, bytes, None, etc. → UNKNOWN
            return InferredType.UNKNOWN
    
    def _infer_unary_type(self, node: ast.UnaryOp) -> InferredType:
        """Infer type of unary operation."""
        operand_type = self._infer_expr_type(node.operand)
        
        if isinstance(node.op, ast.Not):
            # not x → Bool (regardless of operand in Python, but we're strict)
            return InferredType.BOOL
        
        elif isinstance(node.op, (ast.UAdd, ast.USub)):
            # +x, -x → same type as operand (if Int, stays Int)
            if operand_type == InferredType.INT:
                return InferredType.INT
            return InferredType.UNKNOWN
        
        elif isinstance(node.op, ast.Invert):
            # ~x → Int (bitwise invert)
            if operand_type == InferredType.INT:
                return InferredType.INT
            return InferredType.UNKNOWN
        
        return InferredType.UNKNOWN
    
    def _infer_binop_type(self, node: ast.BinOp) -> InferredType:
        """Infer type of binary operation."""
        left_type = self._infer_expr_type(node.left)
        right_type = self._infer_expr_type(node.right)
        
        return self._combine_types_for_binop(left_type, node.op, right_type)
    
    def _combine_types_for_binop(
        self, 
        left: InferredType, 
        op: ast.operator, 
        right: InferredType
    ) -> InferredType:
        """Determine result type of binary operation."""
        
        # UNKNOWN taints everything
        if left == InferredType.UNKNOWN or right == InferredType.UNKNOWN:
            return InferredType.UNKNOWN
        
        # Arithmetic operators on Int → Int
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)):
            if left == InferredType.INT and right == InferredType.INT:
                return InferredType.INT
            return InferredType.UNKNOWN
        
        # True division always returns float → UNKNOWN in Phase 1
        if isinstance(op, ast.Div):
            return InferredType.UNKNOWN
        
        # Bitwise operators on Int → Int
        if isinstance(op, (ast.BitOr, ast.BitXor, ast.BitAnd, ast.LShift, ast.RShift)):
            if left == InferredType.INT and right == InferredType.INT:
                return InferredType.INT
            return InferredType.UNKNOWN
        
        # Matrix mult (@) → UNKNOWN
        if isinstance(op, ast.MatMult):
            return InferredType.UNKNOWN
        
        return InferredType.UNKNOWN
    
    def _infer_boolop_type(self, node: ast.BoolOp) -> InferredType:
        """Infer type of boolean operation (and/or)."""
        # In Python, `and` and `or` return one of their operands
        # But for symbolic execution, we treat the result as Bool
        # (conservative but correct for our use case)
        
        # Check all operands are Bool
        for value in node.values:
            operand_type = self._infer_expr_type(value)
            if operand_type == InferredType.UNKNOWN:
                return InferredType.UNKNOWN
        
        return InferredType.BOOL
