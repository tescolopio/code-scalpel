from typing import Dict, List, Set, Optional, Union, Any, Tuple, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import ast
import z3
import inspect
from collections import defaultdict
import logging
import sympy
from functools import lru_cache
import hashlib

T = TypeVar('T')

class SymbolicType(Enum):
    """Types supported in symbolic execution."""
    INT = 'int'
    REAL = 'real'
    BOOL = 'bool'
    STR = 'str'
    ARRAY = 'array'
    OBJECT = 'object'
    FUNCTION = 'function'

@dataclass
class TypeInfo:
    """Information about a type."""
    base_type: SymbolicType
    constraints: List[Any] = None
    dimensions: Optional[List[int]] = None  # For arrays
    fields: Optional[Dict[str, 'TypeInfo']] = None  # For objects

class SymbolicUtils:
    """Advanced utilities for symbolic execution."""
    
    @staticmethod
    def ast_to_z3(node: ast.AST, 
                  symbolic_state: Dict[str, Any],
                  type_info: Dict[str, TypeInfo] = None) -> z3.ExprRef:
        """
        Convert AST to Z3 expression with type inference.
        
        Args:
            node: AST node
            symbolic_state: Current symbolic state
            type_info: Optional type information
        
        Returns:
            Z3 expression
        """
        if isinstance(node, ast.Name):
            return SymbolicUtils._handle_name(node, symbolic_state, type_info)
        elif isinstance(node, ast.Constant):
            return SymbolicUtils._handle_constant(node)
        elif isinstance(node, ast.BinOp):
            return SymbolicUtils._handle_binop(node, symbolic_state, type_info)
        elif isinstance(node, ast.Compare):
            return SymbolicUtils._handle_compare(node, symbolic_state, type_info)
        elif isinstance(node, ast.BoolOp):
            return SymbolicUtils._handle_boolop(node, symbolic_state, type_info)
        elif isinstance(node, ast.UnaryOp):
            return SymbolicUtils._handle_unaryop(node, symbolic_state, type_info)
        elif isinstance(node, ast.Call):
            return SymbolicUtils._handle_call(node, symbolic_state, type_info)
        elif isinstance(node, ast.Subscript):
            return SymbolicUtils._handle_subscript(node, symbolic_state, type_info)
        elif isinstance(node, ast.Attribute):
            return SymbolicUtils._handle_attribute(node, symbolic_state, type_info)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    @staticmethod
    def infer_type(node: ast.AST, 
                   type_context: Dict[str, TypeInfo]) -> TypeInfo:
        """
        Infer type information from AST node.
        
        Args:
            node: AST node
            type_context: Known type information
            
        Returns:
            Inferred type information
        """
        if isinstance(node, ast.Name):
            return type_context.get(node.id, TypeInfo(SymbolicType.INT))
        elif isinstance(node, ast.Constant):
            return SymbolicUtils._infer_constant_type(node.value)
        elif isinstance(node, ast.BinOp):
            left = SymbolicUtils.infer_type(node.left, type_context)
            right = SymbolicUtils.infer_type(node.right, type_context)
            return SymbolicUtils._combine_types(left, right, node.op)
        elif isinstance(node, ast.Call):
            return SymbolicUtils._infer_call_type(node, type_context)
        return TypeInfo(SymbolicType.INT)  # Default

    @staticmethod
    def simplify_expression(expr: z3.ExprRef) -> z3.ExprRef:
        """
        Simplify a Z3 expression.
        
        Args:
            expr: Z3 expression to simplify
            
        Returns:
            Simplified expression
        """
        tactics = [
            'ctx-solver-simplify',
            'propagate-values',
            'ctx-simplify',
            'elim-uncnstr'
        ]
        
        for tactic in tactics:
            try:
                expr = z3.Tactic(tactic)(expr).as_expr()
            except z3.Z3Exception:
                continue
                
        return expr

    @staticmethod
    def get_path_condition(node: ast.AST, 
                          symbolic_state: Dict[str, Any]) -> z3.BoolRef:
        """
        Generate path condition for an AST node.
        
        Args:
            node: AST node
            symbolic_state: Current symbolic state
            
        Returns:
            Path condition as Z3 boolean expression
        """
        conditions = []
        
        if isinstance(node, ast.If):
            test_expr = SymbolicUtils.ast_to_z3(node.test, symbolic_state)
            conditions.append(test_expr)
            
        for child in ast.iter_child_nodes(node):
            child_cond = SymbolicUtils.get_path_condition(child, symbolic_state)
            if child_cond is not None:
                conditions.append(child_cond)
                
        if conditions:
            return z3.And(*conditions)
        return None

    @staticmethod
    @lru_cache(maxsize=1024)
    def compute_expression_hash(expr: z3.ExprRef) -> str:
        """Compute hash of Z3 expression for caching."""
        return hashlib.sha256(str(expr).encode()).hexdigest()[:16]

    @staticmethod
    def _handle_name(node: ast.Name,
                    symbolic_state: Dict[str, Any],
                    type_info: Dict[str, TypeInfo]) -> z3.ExprRef:
        """Handle variable names."""
        if node.id in symbolic_state:
            return symbolic_state[node.id]
            
        # Create new symbolic variable with inferred type
        var_type = type_info.get(node.id, TypeInfo(SymbolicType.INT))
        if var_type.base_type == SymbolicType.INT:
            var = z3.Int(node.id)
        elif var_type.base_type == SymbolicType.REAL:
            var = z3.Real(node.id)
        elif var_type.base_type == SymbolicType.BOOL:
            var = z3.Bool(node.id)
        else:
            var = z3.Int(node.id)  # Default to int
            
        symbolic_state[node.id] = var
        return var

    @staticmethod
    def _handle_constant(node: ast.Constant) -> z3.ExprRef:
        """Handle constant values."""
        if isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        elif isinstance(node.value, int):
            return z3.IntVal(node.value)
        elif isinstance(node.value, float):
            return z3.RealVal(node.value)
        elif isinstance(node.value, str):
            return z3.StringVal(node.value)
        elif node.value is None:
            return None
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

    @staticmethod
    def _handle_binop(node: ast.BinOp,
                     symbolic_state: Dict[str, Any],
                     type_info: Dict[str, TypeInfo]) -> z3.ExprRef:
        """Handle binary operations."""
        left = SymbolicUtils.ast_to_z3(node.left, symbolic_state, type_info)
        right = SymbolicUtils.ast_to_z3(node.right, symbolic_state, type_info)
        
        operation_map = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.Div: lambda x, y: x / y,
            ast.FloorDiv: lambda x, y: x / y,
            ast.Mod: lambda x, y: x % y,
            ast.Pow: lambda x, y: z3.Power(x, y),
            ast.LShift: lambda x, y: x << y,
            ast.RShift: lambda x, y: x >> y,
            ast.BitOr: lambda x, y: x | y,
            ast.BitXor: lambda x, y: x ^ y,
            ast.BitAnd: lambda x, y: x & y
        }
        
        if type(node.op) not in operation_map:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
            
        return operation_map[type(node.op)](left, right)

    @staticmethod
    def _handle_compare(node: ast.Compare,
                       symbolic_state: Dict[str, Any],
                       type_info: Dict[str, TypeInfo]) -> z3.BoolRef:
        """Handle comparison operations."""
        comparisons = []
        left = SymbolicUtils.ast_to_z3(node.left, symbolic_state, type_info)
        
        for op, right_node in zip(node.ops, node.comparators):
            right = SymbolicUtils.ast_to_z3(right_node, symbolic_state, type_info)
            
            op_map = {
                ast.Eq: lambda x, y: x == y,
                ast.NotEq: lambda x, y: x != y,
                ast.Lt: lambda x, y: x < y,
                ast.LtE: lambda x, y: x <= y,
                ast.Gt: lambda x, y: x > y,
                ast.GtE: lambda x, y: x >= y,
                ast.Is: lambda x, y: x == y,
                ast.IsNot: lambda x, y: x != y,
                ast.In: lambda x, y: z3.Select(y, x) == True,
                ast.NotIn: lambda x, y: z3.Select(y, x) == False
            }
            
            if type(op) not in op_map:
                raise ValueError(f"Unsupported comparison operator: {type(op)}")
                
            comparisons.append(op_map[type(op)](left, right))
            left = right
            
        return z3.And(*comparisons) if comparisons else None

    @staticmethod
    def _handle_boolop(node: ast.BoolOp,
                      symbolic_state: Dict[str, Any],
                      type_info: Dict[str, TypeInfo]) -> z3.BoolRef:
        """Handle boolean operations."""
        values = [
            SymbolicUtils.ast_to_z3(value, symbolic_state, type_info)
            for value in node.values
        ]
        
        if isinstance(node.op, ast.And):
            return z3.And(*values)
        elif isinstance(node.op, ast.Or):
            return z3.Or(*values)
        else:
            raise ValueError(f"Unsupported boolean operator: {type(node.op)}")

    @staticmethod
    def _handle_unaryop(node: ast.UnaryOp,
                       symbolic_state: Dict[str, Any],
                       type_info: Dict[str, TypeInfo]) -> z3.ExprRef:
        """Handle unary operations."""
        operand = SymbolicUtils.ast_to_z3(node.operand, symbolic_state, type_info)
        
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.Not):
            return z3.Not(operand)
        elif isinstance(node.op, ast.Invert):
            return ~operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

    @staticmethod
    def _handle_call(node: ast.Call,
                    symbolic_state: Dict[str, Any],
                    type_info: Dict[str, TypeInfo]) -> z3.ExprRef:
        """Handle function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            args = [
                SymbolicUtils.ast_to_z3(arg, symbolic_state, type_info)
                for arg in node.args
            ]
            
            # Handle built-in functions
            if func_name == 'abs':
                return z3.If(args[0] >= 0, args[0], -args[0])
            elif func_name == 'min':
                return z3.If(args[0] <= args[1], args[0], args[1])
            elif func_name == 'max':
                return z3.If(args[0] >= args[1], args[0], args[1])
                
        raise ValueError(f"Unsupported function call: {ast.dump(node)}")

    @staticmethod
    def _handle_subscript(node: ast.Subscript,
                         symbolic_state: Dict[str, Any],
                         type_info: Dict[str, TypeInfo]) -> z3.ExprRef:
        """Handle array/object subscripting."""
        array = SymbolicUtils.ast_to_z3(node.value, symbolic_state, type_info)
        index = SymbolicUtils.ast_to_z3(node.slice, symbolic_state, type_info)
        
        # Handle array access
        if isinstance(array, z3.Array):
            return z3.Select(array, index)
            
        raise ValueError(f"Unsupported subscript operation: {ast.dump(node)}")

    @staticmethod
    def _infer_constant_type(value: Any) -> TypeInfo:
        """Infer type information for a constant value."""
        if isinstance(value, bool):
            return TypeInfo(SymbolicType.BOOL)
        elif isinstance(value, int):
            return TypeInfo(SymbolicType.INT)
        elif isinstance(value, float):
            return TypeInfo(SymbolicType.REAL)
        elif isinstance(value, str):
            return TypeInfo(SymbolicType.STR)
        elif value is None:
            return TypeInfo(SymbolicType.OBJECT)
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")

    @staticmethod
    def _combine_types(left: TypeInfo, right: TypeInfo, op: ast.operator) -> TypeInfo:
        """Combine types for binary operations."""
        type_precedence = {
            SymbolicType.INT: 1,
            SymbolicType.REAL: 2,
            SymbolicType.BOOL: 0,
            SymbolicType.STR: 3
        }
        
        left_prec = type_precedence[left.base_type]
        right_prec = type_precedence[right.base_type]
        
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
                return TypeInfo(
                    SymbolicType.REAL if max(left_prec, right_prec) >= 2
                    else SymbolicType.INT
                )
        elif isinstance(op, (ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)):
            return TypeInfo(SymbolicType.INT)
        elif isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
            return TypeInfo(SymbolicType.BOOL)
        else:
            raise ValueError(f"Unsupported binary operator: {op}")