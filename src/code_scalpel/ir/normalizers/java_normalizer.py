"""
Java Normalizer - Converts Java CST (tree-sitter) to Unified IR.

This module implements the normalization logic for Java, mapping verbose
Java constructs to the simplified Unified IR.

Key Mappings:
- method_declaration -> IRFunction
- class_declaration -> IRClass
- variable_declarator -> IRAssignment
- if_statement -> IRIf
- while_statement -> IRWhile
- for_statement -> IRFor
"""

from typing import Any, List, Optional

from ..nodes import (
    IRAssign,
    IRBinaryOp,
    IRCall,
    IRClassDef,
    IRFunctionDef,
    IRIf,
    IRConstant,
    IRModule,
    IRName,
    IRReturn,
    IRWhile,
    IRParameter,
    IRNode,
)
from ..operators import BinaryOperator
from .base import BaseNormalizer
from .tree_sitter_visitor import TreeSitterVisitor
import tree_sitter_java
from tree_sitter import Language, Parser


class JavaVisitor(TreeSitterVisitor):
    """Visitor that converts Java CST nodes to IR nodes."""

    language = "java"

    def __init__(self, source: str = ""):
        super().__init__()
        self.ctx.source = source

    def _get_node_type(self, node: Any) -> str:
        return node.type

    def _get_text(self, node: Any) -> str:
        return self.ctx.source[node.start_byte : node.end_byte]

    def _get_location(self, node: Any) -> Any:
        # Simplified location for now
        return None

    def _get_children(self, node: Any) -> List[Any]:
        return node.children

    def _get_named_children(self, node: Any) -> List[Any]:
        return [c for c in node.children if c.is_named]

    def _get_child_by_field(self, node: Any, field_name: str) -> Optional[Any]:
        return node.child_by_field_name(field_name)

    def _get_children_by_field(self, node: Any, field_name: str) -> List[Any]:
        return node.children_by_field_name(field_name)

    def visit_program(self, node: Any) -> IRModule:
        """Root node of a Java file."""
        body = []
        for child in node.children:
            # Skip comments and whitespace
            if not child.is_named:
                continue

            res = self.visit(child)
            if res:
                if isinstance(res, list):
                    body.extend(res)
                else:
                    body.append(res)

        return IRModule(body=body, source_language=self.language)

    def visit_class_declaration(self, node: Any) -> IRClassDef:
        """
        class MyClass { ... }
        """
        name_node = node.child_by_field_name("name")
        name = self.get_text(name_node) if name_node else "Anonymous"

        body_node = node.child_by_field_name("body")
        body = []

        if body_node:
            for child in body_node.children:
                if child.type == "method_declaration":
                    method = self.visit(child)
                    if method:
                        body.append(method)
                elif child.type == "field_declaration":
                    # Field declarations can be multiple: int x, y;
                    # For now, we might just skip or handle simply
                    pass

        return IRClassDef(
            name=name,
            bases=[],  # Java extends/implements logic could go here
            body=body,
            source_language=self.language,
        )

    def visit_method_declaration(self, node: Any) -> IRFunctionDef:
        """
        public void myMethod(int x) { ... }
        """
        name_node = node.child_by_field_name("name")
        name = self.get_text(name_node)

        # Parameters
        params = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            for child in params_node.children:
                if child.type == "formal_parameter":
                    p_name = child.child_by_field_name("name")
                    if p_name:
                        params.append(IRParameter(name=self.get_text(p_name)))

        # Body
        body_node = node.child_by_field_name("body")
        body_stmts = []
        if body_node:
            # visit_block returns List[IRNode]
            body_stmts = self.visit(body_node)

        return IRFunctionDef(
            name=name,
            params=params,
            body=body_stmts,
            return_type=None,  # TODO: Extract return type
            source_language=self.language,
        )

    def visit_block(self, node: Any) -> List[IRNode]:
        """{ stmt1; stmt2; }"""
        statements = []
        for child in node.children:
            if not child.is_named:
                continue
            stmt = self.visit(child)
            if stmt:
                statements.append(stmt)
        return statements

    def visit_local_variable_declaration(self, node: Any) -> Any:
        """
        int x = 5;
        int x, y = 10;
        """
        # This node contains type and declarators
        # We want to return a list of assignments/declarations
        declarators = []

        for child in node.children:
            if child.type == "variable_declarator":
                declarators.append(self.visit(child))

        # If single declarator, return it. If multiple, return list?
        # IRBlock expects statements.
        # For simplicity, let's return the first one or a list if multiple
        if len(declarators) == 1:
            return declarators[0]
        return declarators

    def visit_variable_declarator(self, node: Any) -> IRAssign:
        """x = 5"""
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")

        target = IRName(id=self.get_text(name_node))
        value = self.visit(value_node) if value_node else IRConstant(value=None)

        return IRAssign(targets=[target], value=value)

    def visit_expression_statement(self, node: Any) -> Any:
        """x = 5; or func();"""
        # Usually wraps an assignment or method invocation
        for child in node.children:
            if child.type != ";":
                return self.visit(child)
        return None

    def visit_assignment_expression(self, node: Any) -> IRAssign:
        """x = y"""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")

        return IRAssign(targets=[self.visit(left)], value=self.visit(right))

    def visit_method_invocation(self, node: Any) -> IRCall:
        """obj.method(arg)"""
        name_node = node.child_by_field_name("name")
        object_node = node.child_by_field_name("object")
        args_node = node.child_by_field_name("arguments")

        func_name = self.get_text(name_node)
        if object_node:
            # method call on object
            obj_text = self.get_text(object_node)
            # We might represent this as a specialized IR node or just a name "obj.method"
            # For now, let's use IRName with the full dotted path if simple
            func_name = f"{obj_text}.{func_name}"

        args = []
        if args_node:
            for child in args_node.children:
                if child.is_named:
                    args.append(self.visit(child))

        return IRCall(func=IRName(id=func_name), args=args, kwargs={})

    def visit_if_statement(self, node: Any) -> IRIf:
        """if (cond) { ... } else { ... }"""
        condition_node = node.child_by_field_name("condition")
        consequence_node = node.child_by_field_name("consequence")
        alternative_node = node.child_by_field_name("alternative")

        condition = self.visit(condition_node) if condition_node else None
        consequence = self.visit(consequence_node) if consequence_node else []
        alternative = self.visit(alternative_node) if alternative_node else []

        # Helper to ensure list of nodes
        def to_list(n):
            if isinstance(n, list):
                return n
            return [n] if n else []

        return IRIf(
            test=condition, body=to_list(consequence), orelse=to_list(alternative)
        )

    def visit_while_statement(self, node: Any) -> IRWhile:
        """while (cond) { ... }"""
        condition = self.visit(node.child_by_field_name("condition"))
        body = self.visit(node.child_by_field_name("body"))

        def to_list(n):
            if isinstance(n, list):
                return n
            return [n] if n else []

        return IRWhile(test=condition, body=to_list(body))

    def visit_return_statement(self, node: Any) -> IRReturn:
        """return x;"""
        # return statement has children, usually 'return' keyword and expression and ';'
        expr = None
        for child in node.children:
            if child.is_named:
                expr = self.visit(child)
                break

        return IRReturn(value=expr)

    def visit_binary_expression(self, node: Any) -> IRBinaryOp:
        """a + b"""
        left = self.visit(node.child_by_field_name("left"))
        right = self.visit(node.child_by_field_name("right"))
        operator_text = self.get_text(node.child_by_field_name("operator"))

        # Map operator text to BinaryOperator enum if possible, or just use text for now if allowed
        # The IRBinaryOp expects a BinaryOperator enum usually, but let's check definition.
        # It says op: BinaryOperator = None.
        # For now, we might need a mapping.
        # Simplified mapping:
        op_map = {
            "+": BinaryOperator.ADD,
            "-": BinaryOperator.SUB,
            "*": BinaryOperator.MUL,
            "/": BinaryOperator.DIV,
            "%": BinaryOperator.MOD,
        }
        op = op_map.get(
            operator_text, BinaryOperator.ADD
        )  # Default to ADD if unknown for now

        return IRBinaryOp(left=left, op=op, right=right)

    def visit_identifier(self, node: Any) -> IRName:
        return IRName(id=self.get_text(node))

    def visit_decimal_integer_literal(self, node: Any) -> IRConstant:
        return IRConstant(value=int(self.get_text(node)))

    def visit_string_literal(self, node: Any) -> IRConstant:
        # Strip quotes
        text = self.get_text(node)
        return IRConstant(value=text[1:-1])

    def visit_true(self, node: Any) -> IRConstant:
        return IRConstant(value=True)

    def visit_false(self, node: Any) -> IRConstant:
        return IRConstant(value=False)


class JavaNormalizer(BaseNormalizer):
    """Normalizes Java source code to Unified IR."""

    language = "java"

    def __init__(self):
        self.JAVA_LANGUAGE = Language(tree_sitter_java.language())
        self.parser = Parser()
        self.parser.language = self.JAVA_LANGUAGE
        self._visitor: Optional[JavaVisitor] = None

    def normalize(self, source: str, filename: str = "<string>") -> IRModule:
        tree = self.parser.parse(bytes(source, "utf8"))
        self._visitor = JavaVisitor(source)
        return self._visitor.visit(tree.root_node)

    def normalize_node(self, node: Any) -> Any:
        """Normalize a single tree-sitter node to IR."""
        if self._visitor is None:
            raise RuntimeError("normalize() must be called before normalize_node()")
        return self._visitor.visit(node)
