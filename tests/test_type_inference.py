"""
Test Suite for Type Inference Engine - Milestone M1
====================================================

TDD: These tests define the behavior BEFORE implementation.

Scope (per RFC-001):
- Integer literals and operations → Z3 Int
- Boolean literals and comparisons → Z3 Bool
- Everything else → None (unsupported)

Gall's Law: Start simple. Strings, lists, objects are OUT OF SCOPE.
"""

import ast
import pytest
from z3 import IntSort, BoolSort

# This import will fail until we implement the module
from code_scalpel.symbolic_execution_tools.type_inference import (
    TypeInferenceEngine,
    InferredType,
)


class TestIntegerInference:
    """Test cases for integer type inference."""

    def test_integer_literal(self):
        """x = 42 → x is Int"""
        engine = TypeInferenceEngine()
        code = "x = 42"
        result = engine.infer(code)
        
        assert "x" in result
        assert result["x"] == InferredType.INT

    def test_negative_integer_literal(self):
        """x = -10 → x is Int"""
        engine = TypeInferenceEngine()
        code = "x = -10"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_integer_addition(self):
        """x = 1; y = x + 2 → both are Int"""
        engine = TypeInferenceEngine()
        code = "x = 1\ny = x + 2"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT
        assert result["y"] == InferredType.INT

    def test_integer_arithmetic_chain(self):
        """a = 1; b = 2; c = a + b * 3 → all Int"""
        engine = TypeInferenceEngine()
        code = "a = 1\nb = 2\nc = a + b * 3"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.INT
        assert result["b"] == InferredType.INT
        assert result["c"] == InferredType.INT

    def test_integer_modulo(self):
        """x = 10 % 3 → x is Int"""
        engine = TypeInferenceEngine()
        code = "x = 10 % 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_integer_floor_division(self):
        """x = 10 // 3 → x is Int"""
        engine = TypeInferenceEngine()
        code = "x = 10 // 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT


class TestBooleanInference:
    """Test cases for boolean type inference."""

    def test_bool_literal_true(self):
        """x = True → x is Bool"""
        engine = TypeInferenceEngine()
        code = "x = True"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.BOOL

    def test_bool_literal_false(self):
        """x = False → x is Bool"""
        engine = TypeInferenceEngine()
        code = "x = False"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.BOOL

    def test_comparison_produces_bool(self):
        """x = 1; y = x > 10 → x is Int, y is Bool"""
        engine = TypeInferenceEngine()
        code = "x = 1\ny = x > 10"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT
        assert result["y"] == InferredType.BOOL

    def test_equality_produces_bool(self):
        """x = 5; y = x == 5 → y is Bool"""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = x == 5"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.BOOL

    def test_boolean_and(self):
        """a = True; b = False; c = a and b → all Bool"""
        engine = TypeInferenceEngine()
        code = "a = True\nb = False\nc = a and b"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.BOOL
        assert result["b"] == InferredType.BOOL
        assert result["c"] == InferredType.BOOL

    def test_boolean_or(self):
        """x = True or False → x is Bool"""
        engine = TypeInferenceEngine()
        code = "x = True or False"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.BOOL

    def test_boolean_not(self):
        """x = True; y = not x → both Bool"""
        engine = TypeInferenceEngine()
        code = "x = True\ny = not x"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.BOOL
        assert result["y"] == InferredType.BOOL

    def test_chained_comparison(self):
        """x = 5; y = 1 < x < 10 → y is Bool"""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = 1 < x < 10"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.BOOL


class TestUnsupportedTypes:
    """Test that unsupported types return None/UNKNOWN."""

    def test_string_literal_supported(self):
        """x = 'hello' → x is STRING (supported in v0.3.0)"""
        engine = TypeInferenceEngine()
        code = "x = 'hello'"
        result = engine.infer(code)
        
        # v0.3.0: Strings are now supported
        assert result["x"] == InferredType.STRING

    def test_float_literal_unsupported(self):
        """x = 3.14 → x is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "x = 3.14"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_list_literal_unsupported(self):
        """x = [1, 2, 3] → x is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "x = [1, 2, 3]"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_dict_literal_unsupported(self):
        """x = {'a': 1} → x is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "x = {'a': 1}"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_function_call_unsupported(self):
        """x = some_func() → x is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "x = some_func()"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_attribute_access_unsupported(self):
        """x = obj.attr → x is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "x = obj.attr"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN


class TestTypePropagation:
    """Test that types propagate through operations correctly."""

    def test_unknown_propagates_through_addition(self):
        """x = unknown(); y = x + 1 → y is UNKNOWN (tainted)"""
        engine = TypeInferenceEngine()
        code = "x = unknown()\ny = x + 1"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN
        assert result["y"] == InferredType.UNKNOWN

    def test_int_plus_unknown_is_unknown(self):
        """a = 1; b = func(); c = a + b → c is UNKNOWN"""
        engine = TypeInferenceEngine()
        code = "a = 1\nb = func()\nc = a + b"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.INT
        assert result["b"] == InferredType.UNKNOWN
        assert result["c"] == InferredType.UNKNOWN

    def test_reassignment_updates_type(self):
        """x = 1; x = True → x is Bool (last assignment wins)"""
        engine = TypeInferenceEngine()
        code = "x = 1\nx = True"
        result = engine.infer(code)
        
        # In flow-insensitive analysis, we track last assignment
        # For symbolic execution, this is the "current" type
        assert result["x"] == InferredType.BOOL


class TestComplexExpressions:
    """Test complex but valid expressions."""

    def test_nested_arithmetic(self):
        """x = (1 + 2) * (3 - 4) → x is Int"""
        engine = TypeInferenceEngine()
        code = "x = (1 + 2) * (3 - 4)"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_comparison_chain_in_assignment(self):
        """a = 5; b = 10; c = a < b → c is Bool"""
        engine = TypeInferenceEngine()
        code = "a = 5\nb = 10\nc = a < b"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.INT
        assert result["b"] == InferredType.INT
        assert result["c"] == InferredType.BOOL

    def test_boolean_expression_with_ints(self):
        """x = 5; y = 10; z = x > 0 and y < 20 → z is Bool"""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = 10\nz = x > 0 and y < 20"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT
        assert result["y"] == InferredType.INT
        assert result["z"] == InferredType.BOOL


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_code(self):
        """Empty code returns empty dict."""
        engine = TypeInferenceEngine()
        code = ""
        result = engine.infer(code)
        
        assert result == {}

    def test_only_comments(self):
        """Code with only comments returns empty dict."""
        engine = TypeInferenceEngine()
        code = "# This is a comment\n# Another comment"
        result = engine.infer(code)
        
        assert result == {}

    def test_multiple_targets_assignment(self):
        """a = b = 1 → both are Int"""
        engine = TypeInferenceEngine()
        code = "a = b = 1"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.INT
        assert result["b"] == InferredType.INT

    def test_augmented_assignment(self):
        """x = 1; x += 2 → x is still Int"""
        engine = TypeInferenceEngine()
        code = "x = 1\nx += 2"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_unary_negative(self):
        """x = 5; y = -x → y is Int"""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = -x"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.INT

    def test_unary_not(self):
        """x = True; y = not x → y is Bool"""
        engine = TypeInferenceEngine()
        code = "x = True\ny = not x"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.BOOL


class TestZ3Conversion:
    """Test that inferred types can produce valid Z3 sorts."""

    def test_int_to_z3_sort(self):
        """InferredType.INT maps to z3.IntSort()"""
        assert InferredType.INT.to_z3_sort() == IntSort()

    def test_bool_to_z3_sort(self):
        """InferredType.BOOL maps to z3.BoolSort()"""
        assert InferredType.BOOL.to_z3_sort() == BoolSort()

    def test_unknown_to_z3_sort_raises(self):
        """InferredType.UNKNOWN cannot be converted to Z3."""
        with pytest.raises(ValueError, match="Cannot convert UNKNOWN"):
            InferredType.UNKNOWN.to_z3_sort()


# =============================================================================
# SECTION: Coverage Completeness Tests
# =============================================================================

class TestCoverageCompleteness:
    """Tests to achieve 100% coverage on type_inference.py."""

    def test_string_to_z3_sort(self):
        """InferredType.STRING maps to z3.StringSort()"""
        from z3 import StringSort
        assert InferredType.STRING.to_z3_sort() == StringSort()

    def test_inferred_type_repr(self):
        """InferredType has readable repr."""
        assert repr(InferredType.INT) == "InferredType.INT"
        assert repr(InferredType.BOOL) == "InferredType.BOOL"
        assert repr(InferredType.STRING) == "InferredType.STRING"

    def test_syntax_error_returns_empty(self):
        """Code with syntax error returns empty dict."""
        engine = TypeInferenceEngine()
        code = "x = ("  # Unclosed paren
        result = engine.infer(code)
        
        assert result == {}

    def test_tuple_unpacking(self):
        """a, b = 1, 2 → both are UNKNOWN (conservative)."""
        engine = TypeInferenceEngine()
        code = "a, b = 1, 2"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.UNKNOWN
        assert result["b"] == InferredType.UNKNOWN

    def test_list_unpacking(self):
        """[a, b] = [1, 2] → both are UNKNOWN (conservative)."""
        engine = TypeInferenceEngine()
        code = "[a, b] = [1, 2]"
        result = engine.infer(code)
        
        assert result["a"] == InferredType.UNKNOWN
        assert result["b"] == InferredType.UNKNOWN

    def test_string_literal(self):
        """x = 'hello' → x is STRING."""
        engine = TypeInferenceEngine()
        code = "x = 'hello'"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.STRING

    def test_float_literal_is_unknown(self):
        """x = 3.14 → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = 3.14"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_none_literal_is_unknown(self):
        """x = None → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = None"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_ternary_expression_is_unknown(self):
        """x = 1 if True else 2 → x is UNKNOWN (conservative)."""
        engine = TypeInferenceEngine()
        code = "x = 1 if True else 2"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_function_call_is_unknown(self):
        """x = foo() → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = foo()"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_unary_plus(self):
        """x = 5; y = +x → y is Int."""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = +x"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.INT

    def test_unary_plus_unknown(self):
        """x = foo(); y = +x → y is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "y = +foo()"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.UNKNOWN

    def test_unary_invert(self):
        """x = 5; y = ~x → y is Int."""
        engine = TypeInferenceEngine()
        code = "x = 5\ny = ~x"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.INT

    def test_unary_invert_unknown(self):
        """x = foo(); y = ~x → y is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "y = ~foo()"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.UNKNOWN

    def test_unary_fallback_unknown(self):
        """Unary operator on non-int is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = 3.14\ny = -x"
        result = engine.infer(code)
        
        assert result["y"] == InferredType.UNKNOWN

    def test_string_concatenation(self):
        """x = 'a' + 'b' → x is STRING."""
        engine = TypeInferenceEngine()
        code = "x = 'a' + 'b'"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.STRING

    def test_string_int_add_unknown(self):
        """x = 'a' + 1 → x is UNKNOWN (type mismatch)."""
        engine = TypeInferenceEngine()
        code = "a = 'hello'\nb = 1\nx = a + b"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_string_repetition(self):
        """x = 'a' * 3 → x is STRING."""
        engine = TypeInferenceEngine()
        code = "x = 'a' * 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.STRING

    def test_int_string_multiplication(self):
        """x = 3 * 'a' → x is STRING."""
        engine = TypeInferenceEngine()
        code = "x = 3 * 'a'"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.STRING

    def test_string_mult_unknown(self):
        """x = 'a' * 'b' → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "a = 'hello'\nb = 'world'\nx = a * b"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_int_power(self):
        """x = 2 ** 3 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 2 ** 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_true_division_is_unknown(self):
        """x = 10 / 3 → x is UNKNOWN (returns float)."""
        engine = TypeInferenceEngine()
        code = "x = 10 / 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_bitwise_or(self):
        """x = 5 | 3 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 5 | 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_bitwise_xor(self):
        """x = 5 ^ 3 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 5 ^ 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_bitwise_and(self):
        """x = 5 & 3 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 5 & 3"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_left_shift(self):
        """x = 5 << 2 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 5 << 2"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_right_shift(self):
        """x = 20 >> 2 → x is INT."""
        engine = TypeInferenceEngine()
        code = "x = 20 >> 2"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.INT

    def test_bitwise_on_non_int_unknown(self):
        """x = 'a' | 'b' → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "a = 'hello'\nb = 'world'\nx = a | b"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_matrix_mult_unknown(self):
        """x = a @ b → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = a @ b"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_boolop_with_unknown_operand(self):
        """x = foo() and True → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "x = foo() and True"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN

    def test_subtraction_non_int_unknown(self):
        """x = 'a' - 'b' → x is UNKNOWN."""
        engine = TypeInferenceEngine()
        code = "a = 'hello'\nb = 'world'\nx = a - b"
        result = engine.infer(code)
        
        assert result["x"] == InferredType.UNKNOWN
