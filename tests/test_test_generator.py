"""Tests for the Test Generator.

Tests the conversion of symbolic execution results to pytest/unittest code.
"""

import ast
import pytest


class TestTestGenerator:
    """Tests for the TestGenerator class."""

    def test_generate_simple_function(self):
        """Test generating tests for a simple function."""
        from code_scalpel.generators import TestGenerator

        code = """
def classify(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="classify")

        assert result.function_name == "classify"
        assert result.test_cases  # At least one test case
        assert result.pytest_code  # Generated code

    def test_generate_autodetect_function(self):
        """Test that function name is auto-detected."""
        from code_scalpel.generators import TestGenerator

        code = """
def my_function(a, b):
    return a + b
"""
        generator = TestGenerator()
        result = generator.generate(code)

        assert result.function_name == "my_function"

    def test_generate_pytest_format(self):
        """Test pytest code generation format."""
        from code_scalpel.generators import TestGenerator

        code = """
def add(x, y):
    return x + y
"""
        generator = TestGenerator(framework="pytest")
        result = generator.generate(code, function_name="add")

        # Check pytest structure
        assert "import pytest" in result.pytest_code
        assert "def test_" in result.pytest_code
        assert "assert" in result.pytest_code

    def test_generate_unittest_format(self):
        """Test unittest code generation format."""
        from code_scalpel.generators import TestGenerator

        code = """
def add(x, y):
    return x + y
"""
        generator = TestGenerator(framework="unittest")
        result = generator.generate(code, function_name="add")

        # Check unittest structure
        assert "import unittest" in result.unittest_code
        assert "class Test" in result.unittest_code
        assert "unittest.TestCase" in result.unittest_code
        assert "def test_" in result.unittest_code

    def test_generate_with_branches(self):
        """Test that branches generate multiple test cases."""
        from code_scalpel.generators import TestGenerator

        code = """
def check(x: int):
    if x > 10:
        return "big"
    return "small"
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="check")

        # Should have test cases for both branches (requires type hint)
        assert len(result.test_cases) >= 2

    def test_test_case_has_inputs(self):
        """Test that test cases include input values."""
        from code_scalpel.generators import TestGenerator

        code = """
def double(n):
    if n > 0:
        return n * 2
    return 0
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="double")

        # Each test case should have inputs
        for tc in result.test_cases:
            assert isinstance(tc.inputs, dict)

    def test_invalid_framework_raises(self):
        """Test that invalid framework raises ValueError."""
        from code_scalpel.generators import TestGenerator

        with pytest.raises(ValueError, match="Unsupported framework"):
            TestGenerator(framework="invalid")

    def test_generated_code_is_valid_python(self):
        """Test that generated pytest code is valid Python."""
        from code_scalpel.generators import TestGenerator

        code = """
def is_even(n):
    return n % 2 == 0
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="is_even")

        # Should parse without errors
        ast.parse(result.pytest_code)

    def test_test_case_structure(self):
        """Test TestCase dataclass structure."""
        from code_scalpel.generators.test_generator import TestCase

        tc = TestCase(
            path_id=0,
            function_name="test_func",
            inputs={"x": 5},
            expected_behavior="Returns value",
            path_conditions=["x > 0"],
            description="Positive path",
        )

        assert tc.path_id == 0
        assert tc.function_name == "test_func"
        assert tc.inputs == {"x": 5}

    def test_test_case_to_pytest(self):
        """Test converting TestCase to pytest function."""
        from code_scalpel.generators.test_generator import TestCase

        tc = TestCase(
            path_id=1,
            function_name="classify",
            inputs={"x": 5},
            expected_behavior="Returns positive",
            path_conditions=["x > 0"],
            description="Tests positive branch",
        )

        pytest_code = tc.to_pytest(0)

        assert "def test_classify_path_1():" in pytest_code
        assert "x = 5" in pytest_code
        assert "result = classify(x=x)" in pytest_code

    def test_generated_suite_structure(self):
        """Test GeneratedTestSuite structure."""
        from code_scalpel.generators.test_generator import GeneratedTestSuite, TestCase

        suite = GeneratedTestSuite(
            function_name="my_func",
            test_cases=[
                TestCase(
                    path_id=0,
                    function_name="my_func",
                    inputs={},
                    expected_behavior="",
                    path_conditions=[],
                    description="Test",
                )
            ],
            source_code="def my_func(): pass",
            language="python",
            framework="pytest",
        )

        assert suite.function_name == "my_func"
        assert len(suite.test_cases) == 1

    def test_generate_from_symbolic_result(self):
        """Test generating tests from a pre-computed symbolic result."""
        from code_scalpel.generators import TestGenerator

        code = """
def abs_val(x):
    if x < 0:
        return -x
    return x
"""
        symbolic_result = {
            "paths": [
                {
                    "path_id": 0,
                    "conditions": ["x < 0"],
                    "state": {"x": -5},
                    "reachable": True,
                },
                {
                    "path_id": 1,
                    "conditions": ["not (x < 0)"],
                    "state": {"x": 5},
                    "reachable": True,
                },
            ],
            "symbolic_vars": ["x"],
            "constraints": ["x < 0"],
        }

        generator = TestGenerator()
        result = generator.generate_from_symbolic_result(
            symbolic_result, code, "abs_val"
        )

        assert result.function_name == "abs_val"
        assert len(result.test_cases) == 2
        # Check inputs are populated
        assert result.test_cases[0].inputs.get("x") == -5
        assert result.test_cases[1].inputs.get("x") == 5


class TestTestGeneratorEdgeCases:
    """Edge case tests for TestGenerator."""

    def test_empty_function(self):
        """Test generating tests for function with no branches."""
        from code_scalpel.generators import TestGenerator

        code = """
def identity(x):
    return x
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="identity")

        # Should still generate at least one test
        assert len(result.test_cases) >= 1

    def test_function_no_params(self):
        """Test generating tests for function with no parameters."""
        from code_scalpel.generators import TestGenerator

        code = """
def get_constant():
    return 42
"""
        generator = TestGenerator()
        result = generator.generate(code, function_name="get_constant")

        assert result.function_name == "get_constant"
        # Should generate a test even without inputs
        assert result.pytest_code

    def test_multiple_functions_selects_first(self):
        """Test that first non-private function is selected."""
        from code_scalpel.generators import TestGenerator

        code = """
def _private():
    pass

def public_func():
    return 1

def another_func():
    return 2
"""
        generator = TestGenerator()
        result = generator.generate(code)

        assert result.function_name == "public_func"

    def test_syntax_error_handling(self):
        """Test handling of syntax errors in code."""
        from code_scalpel.generators import TestGenerator

        code = "def broken("

        generator = TestGenerator()
        result = generator.generate(code, function_name="broken")

        # Should handle gracefully
        assert result.function_name == "broken"
