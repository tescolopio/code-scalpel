"""
Tests for JavaScript Normalizer.

This module tests the tree-sitter based JavaScript to IR normalization.
"""
import pytest

# Check if tree-sitter-javascript is available
try:
    import tree_sitter_javascript  # noqa: F401

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE, reason="tree-sitter-javascript not installed"
)


@pytest.fixture
def normalizer():
    """Create a JavaScriptNormalizer instance."""
    from code_scalpel.ir.normalizers.javascript_normalizer import JavaScriptNormalizer

    return JavaScriptNormalizer()


class TestBasicNormalization:
    """Test basic JavaScript constructs."""

    def test_simple_function(self, normalizer):
        """Test simple function declaration."""
        code = "function add(a, b) { return a + b; }"
        ir = normalizer.normalize(code)

        assert ir.source_language == "javascript"
        assert len(ir.body) == 1

        func = ir.body[0]
        assert func.__class__.__name__ == "IRFunctionDef"
        assert func.name == "add"
        assert len(func.params) == 2
        assert func.params[0].name == "a"
        assert func.params[1].name == "b"

    def test_arrow_function(self, normalizer):
        """Test arrow function expression."""
        code = "const add = (a, b) => a + b;"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        assign = ir.body[0]
        assert assign.__class__.__name__ == "IRAssign"
        # Arrow functions normalize to IRFunctionDef (no separate IRLambda in our IR)
        assert assign.value.__class__.__name__ == "IRFunctionDef"

    def test_variable_declaration_const(self, normalizer):
        """Test const variable declaration."""
        code = "const PI = 3.14159;"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        assign = ir.body[0]
        assert assign.__class__.__name__ == "IRAssign"
        # Note: declaration_kind tracking not yet implemented
        assert assign.targets[0].id == "PI"
        assert assign.value.value == 3.14159

    def test_variable_declaration_let(self, normalizer):
        """Test let variable declaration."""
        code = "let count = 0;"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        assign = ir.body[0]
        assert assign.__class__.__name__ == "IRAssign"
        # Note: declaration_kind tracking not yet implemented
        assert assign.targets[0].id == "count"

    def test_variable_declaration_var(self, normalizer):
        """Test var variable declaration."""
        code = "var x = 42;"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        assign = ir.body[0]
        assert assign.__class__.__name__ == "IRAssign"
        # Note: declaration_kind tracking not yet implemented
        assert assign.targets[0].id == "x"


class TestExpressions:
    """Test JavaScript expressions."""

    def test_binary_expression_arithmetic(self, normalizer):
        """Test arithmetic binary expressions."""
        code = "const result = 1 + 2;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRBinaryOp"
        assert assign.value.op.name == "ADD"

    def test_binary_expression_comparison(self, normalizer):
        """Test comparison expressions."""
        code = "const isGreater = x > 5;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRCompare"

    def test_unary_expression(self, normalizer):
        """Test unary expressions."""
        code = "const neg = -x;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRUnaryOp"
        # NEG is used in our IR (maps from USUB in Python)
        assert assign.value.op.name == "NEG"

    def test_logical_expression_and(self, normalizer):
        """Test logical AND expression."""
        code = "const both = a && b;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRBoolOp"
        assert assign.value.op.name == "AND"

    def test_logical_expression_or(self, normalizer):
        """Test logical OR expression."""
        code = "const either = a || b;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRBoolOp"
        assert assign.value.op.name == "OR"

    def test_call_expression(self, normalizer):
        """Test function call expression."""
        code = "console.log('hello');"
        ir = normalizer.normalize(code)

        # Expression statement containing call
        assert len(ir.body) == 1
        expr_stmt = ir.body[0]
        assert expr_stmt.__class__.__name__ == "IRExprStmt"
        assert expr_stmt.value.__class__.__name__ == "IRCall"

    def test_member_expression(self, normalizer):
        """Test member access expression."""
        code = "const name = obj.name;"
        ir = normalizer.normalize(code)

        assign = ir.body[0]
        assert assign.value.__class__.__name__ == "IRAttribute"


class TestControlFlow:
    """Test JavaScript control flow statements."""

    def test_if_statement_simple(self, normalizer):
        """Test simple if statement."""
        code = "if (x > 0) { console.log(x); }"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        if_stmt = ir.body[0]
        assert if_stmt.__class__.__name__ == "IRIf"
        assert if_stmt.test is not None
        assert len(if_stmt.body) == 1

    def test_while_loop(self, normalizer):
        """Test while loop."""
        code = "while (count > 0) { count--; }"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        while_stmt = ir.body[0]
        assert while_stmt.__class__.__name__ == "IRWhile"
        assert while_stmt.test is not None
        assert len(while_stmt.body) == 1

    def test_return_statement(self, normalizer):
        """Test return statement."""
        code = "function f() { return 42; }"
        ir = normalizer.normalize(code)

        func = ir.body[0]
        ret_stmt = func.body[0]
        assert ret_stmt.__class__.__name__ == "IRReturn"
        assert ret_stmt.value.value == 42


class TestClasses:
    """Test JavaScript class declarations."""

    def test_simple_class(self, normalizer):
        """Test simple class declaration."""
        code = """
        class Dog {
            bark() {
                return "woof";
            }
        }
        """
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        cls = ir.body[0]
        assert cls.__class__.__name__ == "IRClassDef"
        assert cls.name == "Dog"
        assert len(cls.body) == 1

        method = cls.body[0]
        assert method.__class__.__name__ == "IRFunctionDef"
        assert method.name == "bark"

    def test_class_with_constructor(self, normalizer):
        """Test class with constructor."""
        code = """
        class Animal {
            constructor(name) {
                this.name = name;
            }
        }
        """
        ir = normalizer.normalize(code)

        cls = ir.body[0]
        constructor = cls.body[0]
        assert constructor.__class__.__name__ == "IRFunctionDef"
        assert constructor.name == "constructor"
        assert len(constructor.params) == 1


class TestAugmentedAssignment:
    """Test augmented assignment operators."""

    def test_increment(self, normalizer):
        """Test increment operator."""
        code = "i++;"
        ir = normalizer.normalize(code)

        assert len(ir.body) == 1
        # Expression statements wrap the IRAugAssign
        expr_stmt = ir.body[0]
        aug = expr_stmt.value
        assert aug.__class__.__name__ == "IRAugAssign"
        assert aug.op.name == "ADD"

    def test_decrement(self, normalizer):
        """Test decrement operator."""
        code = "count--;"
        ir = normalizer.normalize(code)

        expr_stmt = ir.body[0]
        aug = expr_stmt.value
        assert aug.__class__.__name__ == "IRAugAssign"
        assert aug.op.name == "SUB"

    def test_plus_equals(self, normalizer):
        """Test += operator."""
        code = "sum += value;"
        ir = normalizer.normalize(code)

        expr_stmt = ir.body[0]
        aug = expr_stmt.value
        assert aug.__class__.__name__ == "IRAugAssign"
        assert aug.op.name == "ADD"


class TestConstants:
    """Test JavaScript literals and constants."""

    def test_number_integer(self, normalizer):
        """Test integer constant."""
        code = "const x = 42;"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value == 42

    def test_number_float(self, normalizer):
        """Test float constant."""
        code = "const pi = 3.14;"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value == 3.14

    def test_string_single_quotes(self, normalizer):
        """Test single-quoted string."""
        code = "const s = 'hello';"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value == "hello"

    def test_string_double_quotes(self, normalizer):
        """Test double-quoted string."""
        code = 'const s = "world";'
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value == "world"

    def test_boolean_true(self, normalizer):
        """Test true constant."""
        code = "const flag = true;"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value is True

    def test_boolean_false(self, normalizer):
        """Test false constant."""
        code = "const flag = false;"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value is False

    def test_null(self, normalizer):
        """Test null constant."""
        code = "const x = null;"
        ir = normalizer.normalize(code)

        assert ir.body[0].value.value is None

    def test_undefined(self, normalizer):
        """Test undefined constant."""
        code = "const x = undefined;"
        ir = normalizer.normalize(code)

        # undefined maps to None in our IR
        assert ir.body[0].value.value is None


class TestSourceLanguageMarking:
    """Test that IR nodes are marked with source_language."""

    def test_module_has_language(self, normalizer):
        """Test module is marked as JavaScript."""
        code = "const x = 1;"
        ir = normalizer.normalize(code)

        assert ir.source_language == "javascript"

    def test_function_has_language(self, normalizer):
        """Test function is marked as JavaScript."""
        code = "function f() { return 1; }"
        ir = normalizer.normalize(code)

        func = ir.body[0]
        assert func.source_language == "javascript"

    def test_class_has_language(self, normalizer):
        """Test class is marked as JavaScript."""
        code = "class C {}"
        ir = normalizer.normalize(code)

        cls = ir.body[0]
        assert cls.source_language == "javascript"


class TestSourceLocations:
    """Test that source locations are preserved."""

    def test_function_location(self, normalizer):
        """Test function has location info."""
        code = "function f() { return 1; }"
        ir = normalizer.normalize(code)

        func = ir.body[0]
        assert func.loc is not None
        assert func.loc.line == 1
        assert func.loc.column == 0

    def test_multiline_locations(self, normalizer):
        """Test multiline code locations."""
        code = """function f() {
    return 1;
}"""
        ir = normalizer.normalize(code)

        func = ir.body[0]
        assert func.loc.line == 1
        # End line should capture the function span
        assert func.loc.end_line >= 3
