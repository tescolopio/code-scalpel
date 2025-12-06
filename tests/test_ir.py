"""
Tests for the Unified IR module.

Tests cover:
- IR node creation and properties
- PythonNormalizer correctness
- Language semantics differences
"""

import pytest

from code_scalpel.ir import (
    # Nodes
    IRModule,
    IRFunctionDef,
    IRIf,
    IRAssign,
    IRBinaryOp,
    IRCompare,
    IRName,
    IRConstant,
    IRParameter,
    SourceLocation,
    # Operators
    BinaryOperator,
    CompareOperator,
    # Semantics
    PythonSemantics,
    JavaScriptSemantics,
    # Normalizers
    PythonNormalizer,
)


class TestIRNodes:
    """Test IR node creation and properties."""
    
    def test_source_location(self):
        """Test SourceLocation string representation."""
        loc = SourceLocation(line=10, column=5, filename="test.py")
        assert str(loc) == "test.py:10:5"
        
        loc_no_file = SourceLocation(line=10, column=5)
        assert str(loc_no_file) == "line 10, col 5"
    
    def test_ir_node_metadata(self):
        """Test metadata chaining."""
        node = IRName(id="x", source_language="python")
        result = node.with_metadata("tainted", True)
        
        assert result is node  # Returns self
        assert node._metadata["tainted"] is True
    
    def test_ir_binary_op(self):
        """Test binary operation node."""
        left = IRConstant(value=1, source_language="python")
        right = IRConstant(value=2, source_language="python")
        
        binop = IRBinaryOp(
            left=left,
            op=BinaryOperator.ADD,
            right=right,
            source_language="python",
        )
        
        assert binop.left.value == 1
        assert binop.right.value == 2
        assert binop.op == BinaryOperator.ADD
        assert binop.source_language == "python"
    
    def test_ir_function_def(self):
        """Test function definition node."""
        func = IRFunctionDef(
            name="add",
            params=[
                IRParameter(name="a", source_language="python"),
                IRParameter(name="b", source_language="python"),
            ],
            body=[],
            is_async=False,
            source_language="python",
        )
        
        assert func.name == "add"
        assert len(func.params) == 2
        assert func.params[0].name == "a"
        assert not func.is_async


class TestPythonNormalizer:
    """Test Python AST to IR normalization."""
    
    def setup_method(self):
        self.normalizer = PythonNormalizer()
    
    def test_simple_assignment(self):
        """Test normalizing simple assignment."""
        ir = self.normalizer.normalize("x = 42")
        
        assert isinstance(ir, IRModule)
        assert len(ir.body) == 1
        
        assign = ir.body[0]
        assert isinstance(assign, IRAssign)
        assert len(assign.targets) == 1
        assert assign.targets[0].id == "x"
        assert assign.value.value == 42
    
    def test_binary_operation(self):
        """Test normalizing binary operations."""
        ir = self.normalizer.normalize("result = 1 + 2 * 3")
        
        assign = ir.body[0]
        binop = assign.value
        
        assert isinstance(binop, IRBinaryOp)
        assert binop.op == BinaryOperator.ADD
        # Right side should be another BinOp (2 * 3)
        assert isinstance(binop.right, IRBinaryOp)
        assert binop.right.op == BinaryOperator.MUL
    
    def test_comparison(self):
        """Test normalizing comparisons."""
        ir = self.normalizer.normalize("result = x > 10")
        
        assign = ir.body[0]
        compare = assign.value
        
        assert isinstance(compare, IRCompare)
        assert compare.ops[0] == CompareOperator.GT
        assert compare.left.id == "x"
        assert compare.comparators[0].value == 10
    
    def test_chained_comparison(self):
        """Test normalizing Python's chained comparisons."""
        ir = self.normalizer.normalize("result = 0 < x < 10")
        
        compare = ir.body[0].value
        
        assert isinstance(compare, IRCompare)
        assert len(compare.ops) == 2
        assert compare.ops[0] == CompareOperator.LT
        assert compare.ops[1] == CompareOperator.LT
    
    def test_function_definition(self):
        """Test normalizing function definitions."""
        code = '''
def greet(name, greeting="Hello"):
    return greeting + " " + name
'''
        ir = self.normalizer.normalize(code)
        
        func = ir.body[0]
        assert isinstance(func, IRFunctionDef)
        assert func.name == "greet"
        assert len(func.params) == 2
        assert func.params[0].name == "name"
        assert func.params[1].name == "greeting"
        assert func.params[1].default is not None
        assert func.params[1].default.value == "Hello"
    
    def test_if_statement(self):
        """Test normalizing if statements."""
        code = '''
if x > 10:
    y = 1
else:
    y = 2
'''
        ir = self.normalizer.normalize(code)
        
        if_stmt = ir.body[0]
        assert isinstance(if_stmt, IRIf)
        assert isinstance(if_stmt.test, IRCompare)
        assert len(if_stmt.body) == 1
        assert len(if_stmt.orelse) == 1
    
    def test_source_language_propagation(self):
        """Test that source_language is set on all nodes."""
        ir = self.normalizer.normalize("x = 1 + 2")
        
        assert ir.source_language == "python"
        assert ir.body[0].source_language == "python"
        assert ir.body[0].value.source_language == "python"
        assert ir.body[0].value.left.source_language == "python"
    
    def test_source_location(self):
        """Test that source locations are preserved."""
        ir = self.normalizer.normalize("x = 42\ny = 10")
        
        # First assignment should be on line 1
        loc1 = ir.body[0].loc
        assert loc1.line == 1
        
        # Second assignment should be on line 2
        loc2 = ir.body[1].loc
        assert loc2.line == 2


class TestLanguageSemantics:
    """Test language-specific semantic differences."""
    
    def setup_method(self):
        self.py = PythonSemantics()
        self.js = JavaScriptSemantics()
    
    # === String + Number: The Classic Trap ===
    
    def test_string_plus_number_python(self):
        """Python: '5' + 3 raises TypeError."""
        with pytest.raises(TypeError):
            self.py.binary_add("5", 3)
    
    def test_string_plus_number_javascript(self):
        """JavaScript: '5' + 3 = '53' (string coercion)."""
        result = self.js.binary_add("5", 3)
        assert result == "53"
    
    def test_number_plus_string_javascript(self):
        """JavaScript: 3 + '5' = '35' (string wins)."""
        result = self.js.binary_add(3, "5")
        assert result == "35"
    
    # === Modulo Sign Difference ===
    
    def test_modulo_negative_python(self):
        """Python: -7 % 3 = 2 (sign of divisor)."""
        result = self.py.binary_mod(-7, 3)
        assert result == 2
    
    def test_modulo_negative_javascript(self):
        """JavaScript: -7 % 3 = -1 (sign of dividend)."""
        result = self.js.binary_mod(-7, 3)
        assert result == -1
    
    # === Truthiness Differences ===
    
    def test_empty_list_truthy_python(self):
        """Python: [] is falsy."""
        assert self.py.to_boolean([]) is False
    
    def test_empty_list_truthy_javascript(self):
        """JavaScript: [] is TRUTHY (one of the big gotchas)."""
        assert self.js.to_boolean([]) is True
    
    def test_empty_string_falsy_both(self):
        """Both languages: '' is falsy."""
        assert self.py.to_boolean("") is False
        assert self.js.to_boolean("") is False
    
    # === Equality Differences ===
    
    def test_loose_equality_python(self):
        """Python: 1 == '1' is False (no coercion)."""
        assert self.py.compare_eq(1, "1") is False
    
    def test_loose_equality_javascript(self):
        """JavaScript: 1 == '1' is True (type coercion)."""
        assert self.js.compare_eq(1, "1") is True
    
    def test_strict_equality_javascript(self):
        """JavaScript: 1 === '1' is False (no coercion)."""
        assert self.js.compare_strict_eq(1, "1") is False
    
    # === Boolean Operations ===
    
    def test_and_short_circuit_python(self):
        """Python: 0 and 'hello' returns 0."""
        result = self.py.bool_and(0, "hello")
        assert result == 0
    
    def test_or_short_circuit_python(self):
        """Python: 0 or 'hello' returns 'hello'."""
        result = self.py.bool_or(0, "hello")
        assert result == "hello"
    
    # === Division ===
    
    def test_division_by_zero_javascript(self):
        """JavaScript: 5 / 0 = Infinity (not error)."""
        result = self.js.binary_div(5, 0)
        assert result == float('inf')


class TestNormalizerEdgeCases:
    """Test edge cases in normalization."""
    
    def setup_method(self):
        self.normalizer = PythonNormalizer()
    
    def test_lambda(self):
        """Test normalizing lambda expressions."""
        ir = self.normalizer.normalize("f = lambda x: x + 1")
        
        assign = ir.body[0]
        func = assign.value
        
        assert isinstance(func, IRFunctionDef)
        assert func.name == ""  # Anonymous
        assert len(func.params) == 1
        assert func.params[0].name == "x"
    
    def test_class_definition(self):
        """Test normalizing class definitions."""
        code = '''
class Person:
    def __init__(self, name):
        self.name = name
'''
        ir = self.normalizer.normalize(code)
        
        from code_scalpel.ir import IRClassDef
        cls = ir.body[0]
        assert isinstance(cls, IRClassDef)
        assert cls.name == "Person"
        assert len(cls.body) >= 1
    
    def test_async_function(self):
        """Test normalizing async functions."""
        code = '''
async def fetch_data():
    return await get_data()
'''
        ir = self.normalizer.normalize(code)
        
        func = ir.body[0]
        assert isinstance(func, IRFunctionDef)
        assert func.is_async is True
    
    def test_syntax_error_propagation(self):
        """Test that syntax errors are propagated."""
        with pytest.raises(SyntaxError):
            self.normalizer.normalize("def broken(")
