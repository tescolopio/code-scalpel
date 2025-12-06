"""
Python Normalizer - Convert Python ast.* to Unified IR.

This normalizer provides a 1:1 mapping from Python's ast module
to the Unified IR. It serves as the reference implementation.

Python's ast module produces an Abstract Syntax Tree, which maps
cleanly to our IR without needing to filter "noise" nodes.
"""

from __future__ import annotations

import ast
from typing import Any, List, Optional, Union

from .base import BaseNormalizer
from ..nodes import (
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
    IRNode,
    IRExpr,
    SourceLocation,
)
from ..operators import (
    BinaryOperator,
    UnaryOperator,
    CompareOperator,
    BoolOperator,
    AugAssignOperator,
)


class PythonNormalizer(BaseNormalizer):
    """
    Normalizes Python ast.* nodes to Unified IR.
    
    This is a straightforward mapping since Python's ast is already abstract.
    The main work is converting Python-specific node types to IR node types.
    
    Example:
        >>> normalizer = PythonNormalizer()
        >>> ir = normalizer.normalize('''
        ... def add(a, b):
        ...     return a + b
        ... ''')
        >>> ir.body[0].name
        'add'
    """
    
    def __init__(self):
        self._filename: str = "<string>"
    
    @property
    def language(self) -> str:
        return "python"
    
    def normalize(self, source: str, filename: str = "<string>") -> IRModule:
        """Parse Python source and normalize to IR."""
        self._filename = filename
        
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            raise
        
        return self._normalize_Module(tree)
    
    def normalize_node(self, node: ast.AST) -> Union[IRNode, List[IRNode], None]:
        """Dispatch to appropriate normalizer based on node type."""
        method_name = f"_normalize_{node.__class__.__name__}"
        method = getattr(self, method_name, None)
        
        if method is None:
            raise NotImplementedError(
                f"Python AST node type '{node.__class__.__name__}' "
                f"is not yet supported in IR normalization"
            )
        
        return method(node)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _make_loc(self, node: ast.AST) -> Optional[SourceLocation]:
        """Extract source location from ast node."""
        if not hasattr(node, 'lineno'):
            return None
        
        return SourceLocation(
            line=node.lineno,
            column=getattr(node, 'col_offset', 0),
            end_line=getattr(node, 'end_lineno', None),
            end_column=getattr(node, 'end_col_offset', None),
            filename=self._filename,
        )
    
    def _normalize_body(self, body: List[ast.stmt]) -> List[IRNode]:
        """Normalize a list of statements."""
        result = []
        for stmt in body:
            normalized = self.normalize_node(stmt)
            if normalized is not None:
                if isinstance(normalized, list):
                    result.extend(normalized)
                else:
                    result.append(normalized)
        return result
    
    # =========================================================================
    # Module
    # =========================================================================
    
    def _normalize_Module(self, node: ast.Module) -> IRModule:
        """Normalize module (root node)."""
        # Extract docstring if present
        docstring = None
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        return IRModule(
            body=self._normalize_body(node.body),
            docstring=docstring,
            loc=None,
            source_language=self.language,
        )
    
    # =========================================================================
    # Statements
    # =========================================================================
    
    def _normalize_FunctionDef(self, node: ast.FunctionDef) -> IRFunctionDef:
        """Normalize function definition."""
        # Extract docstring
        docstring = None
        body = node.body
        if (body and 
            isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
            docstring = body[0].value.value
        
        return IRFunctionDef(
            name=node.name,
            params=self._normalize_arguments(node.args),
            body=self._normalize_body(node.body),
            return_type=ast.unparse(node.returns) if node.returns else None,
            is_async=False,
            is_generator=any(
                isinstance(n, (ast.Yield, ast.YieldFrom))
                for n in ast.walk(node)
            ),
            decorators=[self.normalize_node(d) for d in node.decorator_list],
            docstring=docstring,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> IRFunctionDef:
        """Normalize async function definition."""
        # Reuse FunctionDef logic but set is_async=True
        result = self._normalize_FunctionDef(node)  # type: ignore
        result.is_async = True
        return result
    
    def _normalize_arguments(self, args: ast.arguments) -> List[IRParameter]:
        """Normalize function arguments."""
        params = []
        
        # Regular args
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            default_idx = i - defaults_offset
            default = None
            if default_idx >= 0:
                default = self.normalize_node(args.defaults[default_idx])
            
            params.append(IRParameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default=default,
                is_rest=False,
                is_keyword_only=False,
                loc=self._make_loc(arg),
                source_language=self.language,
            ))
        
        # *args
        if args.vararg:
            params.append(IRParameter(
                name=args.vararg.arg,
                type_annotation=ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                is_rest=True,
                loc=self._make_loc(args.vararg),
                source_language=self.language,
            ))
        
        # Keyword-only args
        kw_defaults_map = {i: d for i, d in enumerate(args.kw_defaults) if d is not None}
        for i, arg in enumerate(args.kwonlyargs):
            default = None
            if i in kw_defaults_map:
                default = self.normalize_node(kw_defaults_map[i])
            
            params.append(IRParameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default=default,
                is_keyword_only=True,
                loc=self._make_loc(arg),
                source_language=self.language,
            ))
        
        return params
    
    def _normalize_ClassDef(self, node: ast.ClassDef) -> IRClassDef:
        """Normalize class definition."""
        # Extract docstring
        docstring = None
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        return IRClassDef(
            name=node.name,
            bases=[self.normalize_node(b) for b in node.bases],
            body=self._normalize_body(node.body),
            decorators=[self.normalize_node(d) for d in node.decorator_list],
            docstring=docstring,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_If(self, node: ast.If) -> IRIf:
        """Normalize if statement."""
        return IRIf(
            test=self.normalize_node(node.test),
            body=self._normalize_body(node.body),
            orelse=self._normalize_body(node.orelse),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_For(self, node: ast.For) -> IRFor:
        """Normalize for loop."""
        return IRFor(
            target=self.normalize_node(node.target),
            iter=self.normalize_node(node.iter),
            body=self._normalize_body(node.body),
            orelse=self._normalize_body(node.orelse),
            is_for_in=False,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_While(self, node: ast.While) -> IRWhile:
        """Normalize while loop."""
        return IRWhile(
            test=self.normalize_node(node.test),
            body=self._normalize_body(node.body),
            orelse=self._normalize_body(node.orelse),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Return(self, node: ast.Return) -> IRReturn:
        """Normalize return statement."""
        return IRReturn(
            value=self.normalize_node(node.value) if node.value else None,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Assign(self, node: ast.Assign) -> IRAssign:
        """Normalize assignment statement."""
        return IRAssign(
            targets=[self.normalize_node(t) for t in node.targets],
            value=self.normalize_node(node.value),
            declaration_kind=None,  # Python doesn't have let/const/var
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_AnnAssign(self, node: ast.AnnAssign) -> IRAssign:
        """Normalize annotated assignment (x: int = 5)."""
        return IRAssign(
            targets=[self.normalize_node(node.target)],
            value=self.normalize_node(node.value) if node.value else None,
            declaration_kind=None,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_AugAssign(self, node: ast.AugAssign) -> IRAugAssign:
        """Normalize augmented assignment (+=, -=, etc.)."""
        return IRAugAssign(
            target=self.normalize_node(node.target),
            op=self._map_aug_assign_op(node.op),
            value=self.normalize_node(node.value),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Expr(self, node: ast.Expr) -> IRExprStmt:
        """Normalize expression statement."""
        return IRExprStmt(
            value=self.normalize_node(node.value),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Pass(self, node: ast.Pass) -> IRPass:
        """Normalize pass statement."""
        return IRPass(
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Break(self, node: ast.Break) -> IRBreak:
        """Normalize break statement."""
        return IRBreak(
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Continue(self, node: ast.Continue) -> IRContinue:
        """Normalize continue statement."""
        return IRContinue(
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    # Import statements - normalize to expression statements for now
    def _normalize_Import(self, node: ast.Import) -> IRExprStmt:
        """Normalize import statement (placeholder)."""
        # For now, represent as a call to __import__
        return IRExprStmt(
            value=IRCall(
                func=IRName(id="__import__", source_language=self.language),
                args=[IRConstant(value=alias.name, source_language=self.language) 
                      for alias in node.names],
                source_language=self.language,
            ),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_ImportFrom(self, node: ast.ImportFrom) -> IRExprStmt:
        """Normalize from...import statement (placeholder)."""
        return IRExprStmt(
            value=IRCall(
                func=IRName(id="__import__", source_language=self.language),
                args=[IRConstant(value=node.module or "", source_language=self.language)],
                source_language=self.language,
            ),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    # =========================================================================
    # Expressions
    # =========================================================================
    
    def _normalize_BinOp(self, node: ast.BinOp) -> IRBinaryOp:
        """Normalize binary operation."""
        return IRBinaryOp(
            left=self.normalize_node(node.left),
            op=self._map_binary_op(node.op),
            right=self.normalize_node(node.right),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Await(self, node: ast.Await) -> IRCall:
        """Normalize await expression as a special call."""
        return IRCall(
            func=IRName(id="__await__", source_language=self.language),
            args=[self.normalize_node(node.value)],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_UnaryOp(self, node: ast.UnaryOp) -> IRUnaryOp:
        """Normalize unary operation."""
        return IRUnaryOp(
            op=self._map_unary_op(node.op),
            operand=self.normalize_node(node.operand),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Compare(self, node: ast.Compare) -> IRCompare:
        """Normalize comparison operation."""
        return IRCompare(
            left=self.normalize_node(node.left),
            ops=[self._map_cmp_op(op) for op in node.ops],
            comparators=[self.normalize_node(c) for c in node.comparators],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_BoolOp(self, node: ast.BoolOp) -> IRBoolOp:
        """Normalize boolean operation (and/or)."""
        return IRBoolOp(
            op=self._map_bool_op(node.op),
            values=[self.normalize_node(v) for v in node.values],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Call(self, node: ast.Call) -> IRCall:
        """Normalize function call."""
        kwargs = {}
        for kw in node.keywords:
            if kw.arg:  # Named keyword
                kwargs[kw.arg] = self.normalize_node(kw.value)
            # **kwargs is complex, skip for now
        
        return IRCall(
            func=self.normalize_node(node.func),
            args=[self.normalize_node(arg) for arg in node.args],
            kwargs=kwargs,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Attribute(self, node: ast.Attribute) -> IRAttribute:
        """Normalize attribute access."""
        return IRAttribute(
            value=self.normalize_node(node.value),
            attr=node.attr,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Subscript(self, node: ast.Subscript) -> IRSubscript:
        """Normalize subscript access."""
        return IRSubscript(
            value=self.normalize_node(node.value),
            slice=self.normalize_node(node.slice),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Name(self, node: ast.Name) -> IRName:
        """Normalize variable reference."""
        return IRName(
            id=node.id,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Constant(self, node: ast.Constant) -> IRConstant:
        """Normalize literal constant."""
        return IRConstant(
            value=node.value,
            raw=repr(node.value),
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_List(self, node: ast.List) -> IRList:
        """Normalize list literal."""
        return IRList(
            elements=[self.normalize_node(elt) for elt in node.elts],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Tuple(self, node: ast.Tuple) -> IRList:
        """Normalize tuple literal (treated as list in IR)."""
        result = IRList(
            elements=[self.normalize_node(elt) for elt in node.elts],
            loc=self._make_loc(node),
            source_language=self.language,
        )
        result._metadata["is_tuple"] = True
        return result
    
    def _normalize_Dict(self, node: ast.Dict) -> IRDict:
        """Normalize dict literal."""
        return IRDict(
            keys=[self.normalize_node(k) if k else None for k in node.keys],
            values=[self.normalize_node(v) for v in node.values],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_IfExp(self, node: ast.IfExp) -> IRCall:
        """
        Normalize ternary expression (x if cond else y).
        
        Represented as a special call for now.
        """
        return IRCall(
            func=IRName(id="__ternary__", source_language=self.language),
            args=[
                self.normalize_node(node.test),
                self.normalize_node(node.body),
                self.normalize_node(node.orelse),
            ],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    def _normalize_Lambda(self, node: ast.Lambda) -> IRFunctionDef:
        """Normalize lambda expression."""
        return IRFunctionDef(
            name="",  # Anonymous
            params=self._normalize_arguments(node.args),
            body=[IRReturn(
                value=self.normalize_node(node.body),
                source_language=self.language,
            )],
            is_async=False,
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    # Slice is used in subscripts
    def _normalize_Slice(self, node: ast.Slice) -> IRCall:
        """Normalize slice (a:b:c)."""
        return IRCall(
            func=IRName(id="slice", source_language=self.language),
            args=[
                self.normalize_node(node.lower) if node.lower else IRConstant(value=None, source_language=self.language),
                self.normalize_node(node.upper) if node.upper else IRConstant(value=None, source_language=self.language),
                self.normalize_node(node.step) if node.step else IRConstant(value=None, source_language=self.language),
            ],
            loc=self._make_loc(node),
            source_language=self.language,
        )
    
    # =========================================================================
    # Operator Mappings
    # =========================================================================
    
    def _map_binary_op(self, op: ast.operator) -> BinaryOperator:
        """Map Python ast operator to IR BinaryOperator."""
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
            ast.MatMult: BinaryOperator.MATMUL,
        }
        return mapping.get(type(op), BinaryOperator.ADD)
    
    def _map_unary_op(self, op: ast.unaryop) -> UnaryOperator:
        """Map Python ast unary operator to IR UnaryOperator."""
        mapping = {
            ast.UAdd: UnaryOperator.POS,
            ast.USub: UnaryOperator.NEG,
            ast.Not: UnaryOperator.NOT,
            ast.Invert: UnaryOperator.INVERT,
        }
        return mapping.get(type(op), UnaryOperator.NEG)
    
    def _map_cmp_op(self, op: ast.cmpop) -> CompareOperator:
        """Map Python ast comparison operator to IR CompareOperator."""
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
    
    def _map_bool_op(self, op: ast.boolop) -> BoolOperator:
        """Map Python ast boolean operator to IR BoolOperator."""
        mapping = {
            ast.And: BoolOperator.AND,
            ast.Or: BoolOperator.OR,
        }
        return mapping.get(type(op), BoolOperator.AND)
    
    def _map_aug_assign_op(self, op: ast.operator) -> AugAssignOperator:
        """Map Python ast operator to IR AugAssignOperator."""
        mapping = {
            ast.Add: AugAssignOperator.ADD,
            ast.Sub: AugAssignOperator.SUB,
            ast.Mult: AugAssignOperator.MUL,
            ast.Div: AugAssignOperator.DIV,
            ast.FloorDiv: AugAssignOperator.FLOOR_DIV,
            ast.Mod: AugAssignOperator.MOD,
            ast.Pow: AugAssignOperator.POW,
            ast.BitAnd: AugAssignOperator.BIT_AND,
            ast.BitOr: AugAssignOperator.BIT_OR,
            ast.BitXor: AugAssignOperator.BIT_XOR,
            ast.LShift: AugAssignOperator.LSHIFT,
            ast.RShift: AugAssignOperator.RSHIFT,
            ast.MatMult: AugAssignOperator.MATMUL,
        }
        return mapping.get(type(op), AugAssignOperator.ADD)
