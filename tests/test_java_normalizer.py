import unittest
from unittest.mock import MagicMock, patch

from code_scalpel.ir.normalizers.java_normalizer import JavaNormalizer, JavaVisitor
from code_scalpel.ir.nodes import (
    IRFunctionDef, IRClassDef, IRIf, IRWhile, IRCall,
    IRAssign, IRName, IRConstant, IRReturn, IRBinaryOp
)
from code_scalpel.ir.operators import BinaryOperator


class TestJavaNormalizer(unittest.TestCase):
    """Tests for JavaNormalizer and JavaVisitor."""

    def test_visitor_structure(self):
        """Test visitor creates proper IRFunctionDef from method_declaration."""
        visitor = JavaVisitor("source code placeholder")
        
        mock_method = MagicMock()
        mock_method.type = "method_declaration"
        
        name_node = MagicMock()
        visitor.get_text = MagicMock(return_value="myMethod")
        mock_method.child_by_field_name.side_effect = lambda name: name_node if name == "name" else None
        
        result = visitor.visit_method_declaration(mock_method)
        self.assertIsInstance(result, IRFunctionDef)
        self.assertEqual(result.name, "myMethod")
        self.assertEqual(result.source_language, "java")

    def test_visit_binary_expression(self):
        """Test binary expression visitor maps operators correctly."""
        visitor = JavaVisitor("a + b")
        
        mock_node = MagicMock()
        mock_node.type = "binary_expression"
        
        left = MagicMock()
        right = MagicMock()
        op = MagicMock()
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "left": left, "right": right, "operator": op
        }.get(name)
        
        visitor.visit = MagicMock(return_value="mock_ir")
        visitor.get_text = MagicMock(return_value="+")
        
        result = visitor.visit_binary_expression(mock_node)
        self.assertEqual(result.op, BinaryOperator.ADD)

    def test_visit_class_declaration(self):
        """Test class declaration produces IRClassDef."""
        visitor = JavaVisitor("class MyClass {}")
        
        mock_node = MagicMock()
        mock_node.type = "class_declaration"
        
        name_node = MagicMock()
        body_node = MagicMock()
        body_node.children = []
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "name": name_node, "body": body_node
        }.get(name)
        
        visitor.get_text = MagicMock(return_value="MyClass")
        
        result = visitor.visit_class_declaration(mock_node)
        self.assertIsInstance(result, IRClassDef)
        self.assertEqual(result.name, "MyClass")
        self.assertEqual(result.source_language, "java")
        self.assertEqual(result.bases, [])

    def test_visit_if_statement(self):
        """Test if statement produces IRIf."""
        visitor = JavaVisitor("if (x) { y; }")
        
        mock_node = MagicMock()
        mock_node.type = "if_statement"
        
        condition = MagicMock()
        consequence = MagicMock()
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "condition": condition,
            "consequence": consequence,
            "alternative": None
        }.get(name)
        
        mock_condition_ir = MagicMock()
        mock_body_ir = [MagicMock()]
        
        visitor.visit = MagicMock(side_effect=lambda n: {
            id(condition): mock_condition_ir,
            id(consequence): mock_body_ir
        }.get(id(n)))
        
        result = visitor.visit_if_statement(mock_node)
        self.assertIsInstance(result, IRIf)
        self.assertEqual(result.test, mock_condition_ir)
        self.assertEqual(result.body, mock_body_ir)
        self.assertEqual(result.orelse, [])

    def test_visit_while_statement(self):
        """Test while statement produces IRWhile."""
        visitor = JavaVisitor("while (x) { y; }")
        
        mock_node = MagicMock()
        mock_node.type = "while_statement"
        
        condition = MagicMock()
        body = MagicMock()
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "condition": condition,
            "body": body
        }.get(name)
        
        mock_condition_ir = MagicMock()
        mock_body_ir = [MagicMock()]
        
        visitor.visit = MagicMock(side_effect=lambda n: {
            id(condition): mock_condition_ir,
            id(body): mock_body_ir
        }.get(id(n)))
        
        result = visitor.visit_while_statement(mock_node)
        self.assertIsInstance(result, IRWhile)
        self.assertEqual(result.test, mock_condition_ir)
        self.assertEqual(result.body, mock_body_ir)

    def test_visit_return_statement(self):
        """Test return statement produces IRReturn."""
        visitor = JavaVisitor("return 42;")
        
        mock_node = MagicMock()
        mock_node.type = "return_statement"
        
        expr_child = MagicMock()
        expr_child.is_named = True
        mock_node.children = [expr_child]
        
        mock_expr_ir = IRConstant(value=42)
        visitor.visit = MagicMock(return_value=mock_expr_ir)
        
        result = visitor.visit_return_statement(mock_node)
        self.assertIsInstance(result, IRReturn)
        self.assertEqual(result.value, mock_expr_ir)

    def test_visit_method_invocation(self):
        """Test method invocation produces IRCall."""
        visitor = JavaVisitor("obj.method(arg)")
        
        mock_node = MagicMock()
        mock_node.type = "method_invocation"
        
        name_node = MagicMock()
        object_node = MagicMock()
        args_node = MagicMock()
        
        arg_child = MagicMock()
        arg_child.is_named = True
        args_node.children = [arg_child]
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "name": name_node,
            "object": object_node,
            "arguments": args_node
        }.get(name)
        
        visitor.get_text = MagicMock(side_effect=lambda n: {
            id(name_node): "method",
            id(object_node): "obj"
        }.get(id(n)))
        
        mock_arg_ir = MagicMock()
        visitor.visit = MagicMock(return_value=mock_arg_ir)
        
        result = visitor.visit_method_invocation(mock_node)
        self.assertIsInstance(result, IRCall)
        self.assertEqual(result.func.id, "obj.method")
        self.assertEqual(result.args, [mock_arg_ir])

    def test_visit_variable_declarator(self):
        """Test variable declarator produces IRAssign."""
        visitor = JavaVisitor("int x = 5;")
        
        mock_node = MagicMock()
        mock_node.type = "variable_declarator"
        
        name_node = MagicMock()
        value_node = MagicMock()
        
        mock_node.child_by_field_name.side_effect = lambda name: {
            "name": name_node,
            "value": value_node
        }.get(name)
        
        visitor.get_text = MagicMock(return_value="x")
        mock_value_ir = IRConstant(value=5)
        visitor.visit = MagicMock(return_value=mock_value_ir)
        
        result = visitor.visit_variable_declarator(mock_node)
        self.assertIsInstance(result, IRAssign)
        self.assertEqual(result.targets[0].id, "x")
        self.assertEqual(result.value, mock_value_ir)

    def test_visit_identifier(self):
        """Test identifier produces IRName."""
        visitor = JavaVisitor("myVar")
        
        mock_node = MagicMock()
        mock_node.type = "identifier"
        
        visitor.get_text = MagicMock(return_value="myVar")
        
        result = visitor.visit_identifier(mock_node)
        self.assertIsInstance(result, IRName)
        self.assertEqual(result.id, "myVar")

    def test_visit_decimal_integer_literal(self):
        """Test integer literal produces IRConstant."""
        visitor = JavaVisitor("42")
        
        mock_node = MagicMock()
        mock_node.type = "decimal_integer_literal"
        
        visitor.get_text = MagicMock(return_value="42")
        
        result = visitor.visit_decimal_integer_literal(mock_node)
        self.assertIsInstance(result, IRConstant)
        self.assertEqual(result.value, 42)

    def test_visit_string_literal(self):
        """Test string literal produces IRConstant with stripped quotes."""
        visitor = JavaVisitor('"hello"')
        
        mock_node = MagicMock()
        mock_node.type = "string_literal"
        
        visitor.get_text = MagicMock(return_value='"hello"')
        
        result = visitor.visit_string_literal(mock_node)
        self.assertIsInstance(result, IRConstant)
        self.assertEqual(result.value, "hello")

    def test_visit_true(self):
        """Test true literal produces IRConstant(True)."""
        visitor = JavaVisitor("true")
        
        mock_node = MagicMock()
        mock_node.type = "true"
        
        result = visitor.visit_true(mock_node)
        self.assertIsInstance(result, IRConstant)
        self.assertEqual(result.value, True)

    def test_visit_false(self):
        """Test false literal produces IRConstant(False)."""
        visitor = JavaVisitor("false")
        
        mock_node = MagicMock()
        mock_node.type = "false"
        
        result = visitor.visit_false(mock_node)
        self.assertIsInstance(result, IRConstant)
        self.assertEqual(result.value, False)

    def test_binary_operators_mapping(self):
        """Test all binary operator mappings."""
        visitor = JavaVisitor("a op b")
        
        operators = [
            ("+", BinaryOperator.ADD),
            ("-", BinaryOperator.SUB),
            ("*", BinaryOperator.MUL),
            ("/", BinaryOperator.DIV),
            ("%", BinaryOperator.MOD),
        ]
        
        for op_str, expected_op in operators:
            mock_node = MagicMock()
            left = MagicMock()
            right = MagicMock()
            op = MagicMock()
            
            mock_node.child_by_field_name.side_effect = lambda name: {
                "left": left, "right": right, "operator": op
            }.get(name)
            
            visitor.visit = MagicMock(return_value="mock_ir")
            visitor.get_text = MagicMock(return_value=op_str)
            
            result = visitor.visit_binary_expression(mock_node)
            self.assertEqual(result.op, expected_op, f"Failed for operator {op_str}")


if __name__ == "__main__":
    unittest.main()
