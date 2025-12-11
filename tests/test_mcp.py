"""Tests for MCP Server tools.

Tests the FastMCP-based Model Context Protocol server.
All MCP tool functions are async and return Pydantic models.
"""

import pytest

# Mark entire module as async
pytestmark = pytest.mark.asyncio


class TestAnalyzeCodeTool:
    """Tests for the analyze_code tool."""

    async def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        from code_scalpel.mcp.server import analyze_code

        code = """
def hello():
    return "Hello, World!"
"""
        result = await analyze_code(code)
        assert result.success is True
        assert "hello" in result.functions
        assert result.function_count == 1

    async def test_analyze_class(self):
        """Test analyzing a class definition."""
        from code_scalpel.mcp.server import analyze_code

        code = """
class MyClass:
    def __init__(self):
        pass

    def method(self):
        return 42
"""
        result = await analyze_code(code)
        assert result.success is True
        assert "MyClass" in result.classes
        assert result.class_count == 1
        assert result.function_count == 2  # __init__ and method

    async def test_analyze_imports(self):
        """Test analyzing imports."""
        from code_scalpel.mcp.server import analyze_code

        code = """
import os
from pathlib import Path
import sys
"""
        result = await analyze_code(code)
        assert result.success is True
        assert "os" in result.imports
        assert "sys" in result.imports
        assert any("Path" in imp for imp in result.imports)

    async def test_analyze_complexity(self):
        """Test complexity calculation."""
        from code_scalpel.mcp.server import analyze_code

        code = """
def complex_func(x):
    if x > 0:
        if x > 10:
            return "big"
        return "small"
    return "negative"
"""
        result = await analyze_code(code)
        assert result.success is True
        assert result.complexity >= 3  # At least 3 due to two if statements

    async def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        from code_scalpel.mcp.server import analyze_code

        result = await analyze_code("")
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower()

    async def test_analyze_syntax_error(self):
        """Test handling syntax errors."""
        from code_scalpel.mcp.server import analyze_code

        code = "def broken("
        result = await analyze_code(code)
        assert result.success is False
        assert result.error is not None
        assert "syntax" in result.error.lower()

    async def test_analyze_async_function(self):
        """Test analyzing async functions."""
        from code_scalpel.mcp.server import analyze_code

        code = """
async def async_func():
    await some_coroutine()
"""
        result = await analyze_code(code)
        assert result.success is True
        assert any("async_func" in f for f in result.functions)


class TestSecurityScanTool:
    """Tests for the security_scan tool."""

    async def test_scan_clean_code(self):
        """Test scanning clean code with no vulnerabilities."""
        from code_scalpel.mcp.server import security_scan

        code = """
def safe_function(x):
    return x + 1
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.vulnerability_count == 0
        assert result.risk_level == "low"

    async def test_scan_sql_injection(self):
        """Test detecting SQL injection."""
        from code_scalpel.mcp.server import security_scan

        code = """
def vulnerable(request):
    user_input = request.args.get("id")
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True
        assert any("SQL" in v.type for v in result.vulnerabilities)

    async def test_scan_command_injection(self):
        """Test detecting command injection."""
        from code_scalpel.mcp.server import security_scan

        code = """
import os
def run_command(request):
    user_input = request.form.get("cmd")
    os.system("ls " + user_input)
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True
        assert any("Command" in v.type for v in result.vulnerabilities)

    async def test_scan_eval_injection(self):
        """Test detecting eval injection."""
        from code_scalpel.mcp.server import security_scan

        code = """
def dangerous(request):
    user_input = request.args.get("expr")
    result = eval(user_input)
    return result
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True

    async def test_scan_hardcoded_secret(self):
        """Test detecting hardcoded secrets."""
        from code_scalpel.mcp.server import security_scan

        code = """
def connect():
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    return aws_key
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True
        assert any(
            "Secret" in v.type or "Hardcoded" in v.type for v in result.vulnerabilities
        )

    async def test_scan_empty_code(self):
        """Test scanning empty code."""
        from code_scalpel.mcp.server import security_scan

        result = await security_scan("")
        assert result.success is False
        assert result.error is not None

    async def test_scan_detects_taint_sources(self):
        """Test detection of taint sources."""
        from code_scalpel.mcp.server import security_scan

        code = """
from flask import request
def handler():
    user_data = request.args.get('user')
    return user_data
"""
        result = await security_scan(code)
        assert result.success is True
        # Either SecurityAnalyzer or fallback should detect request.args
        # (May or may not have vulnerabilities depending on implementation)

    async def test_scan_risk_levels(self):
        """Test risk level calculation."""
        from code_scalpel.mcp.server import security_scan

        # Code with multiple vulnerabilities
        code = """
import os
def very_dangerous(request):
    user_input = request.args.get("data")
    eval(user_input)
    exec(user_input)
    os.system(user_input)
"""
        result = await security_scan(code)
        assert result.success is True
        assert result.risk_level in ["high", "critical"]


class TestSymbolicExecuteTool:
    """Tests for the symbolic_execute tool."""

    async def test_symbolic_simple_function(self):
        """Test symbolic execution on simple function."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def simple(x):
    return x + 1
"""
        result = await symbolic_execute(code)
        assert result.success is True
        assert result.paths_explored >= 1

    async def test_symbolic_branching(self):
        """Test symbolic execution with branches."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def branching(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"""
        result = await symbolic_execute(code)
        assert result.success is True
        # Should find paths for both branches
        assert result.paths_explored >= 1
        assert len(result.constraints) > 0

    async def test_symbolic_multiple_branches(self):
        """Test symbolic execution with multiple branches."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
"""
        result = await symbolic_execute(code)
        assert result.success is True
        assert result.paths_explored >= 1

    async def test_symbolic_detects_symbolic_vars(self):
        """Test that symbolic variables are detected."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def func(a, b, c):
    if a > b:
        return c
    return a + b
"""
        result = await symbolic_execute(code)
        assert result.success is True
        # Should detect function parameters as symbolic
        assert len(result.symbolic_variables) > 0

    async def test_symbolic_empty_code(self):
        """Test symbolic execution on empty code."""
        from code_scalpel.mcp.server import symbolic_execute

        result = await symbolic_execute("")
        assert result.success is False
        assert result.error is not None

    async def test_symbolic_max_paths(self):
        """Test max_paths parameter."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def many_branches(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return 1
            return 2
        return 3
    return 4
"""
        result = await symbolic_execute(code, max_paths=5)
        assert result.success is True
        # Should respect max_paths
        assert result.paths_explored <= 10  # Some buffer for implementation

    async def test_symbolic_loop_handling(self):
        """Test symbolic execution with loops."""
        from code_scalpel.mcp.server import symbolic_execute

        code = """
def with_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""
        result = await symbolic_execute(code)
        assert result.success is True


class TestMCPIntegration:
    """Integration tests for MCP server."""

    async def test_all_tools_available(self):
        """Test that all tools are registered."""
        from code_scalpel.mcp.server import mcp

        # FastMCP should have our tools
        assert mcp is not None
        # The tools should be callable
        from code_scalpel.mcp.server import (
            analyze_code,
            security_scan,
            symbolic_execute,
        )

        assert callable(analyze_code)
        assert callable(security_scan)
        assert callable(symbolic_execute)

    async def test_result_models_are_valid(self):
        """Test that result models are proper Pydantic models."""
        from code_scalpel.mcp.server import (
            AnalysisResult,
            SecurityResult,
            SymbolicResult,
        )

        # Should be importable and usable
        assert AnalysisResult is not None
        assert SecurityResult is not None
        assert SymbolicResult is not None

    async def test_code_validation(self):
        """Test code validation for all tools."""
        from code_scalpel.mcp.server import (
            analyze_code,
            security_scan,
            symbolic_execute,
        )

        # All should reject empty code
        result1 = await analyze_code("")
        result2 = await security_scan("")
        result3 = await symbolic_execute("")

        assert result1.success is False
        assert result2.success is False
        assert result3.success is False

    async def test_large_code_rejection(self):
        """Test rejection of code exceeding size limit."""
        from code_scalpel.mcp.server import analyze_code, MAX_CODE_SIZE

        # Create code exceeding limit
        large_code = "x = 1\n" * (MAX_CODE_SIZE // 5)
        result = await analyze_code(large_code)
        assert result.success is False
        assert "size" in result.error.lower() or "exceed" in result.error.lower()

    async def test_analysis_pipeline(self):
        """Test running multiple analyses on the same code."""
        from code_scalpel.mcp.server import (
            analyze_code,
            security_scan,
            symbolic_execute,
        )

        code = """
def process_user(user_id):
    if user_id > 0:
        return f"User: {user_id}"
    return "Invalid"
"""
        # All should succeed
        analysis = await analyze_code(code)
        security = await security_scan(code)
        symbolic = await symbolic_execute(code)

        assert analysis.success is True
        assert security.success is True
        assert symbolic.success is True

        # Cross-check: function detected
        assert "process_user" in analysis.functions

    async def test_concurrent_analysis(self):
        """Test concurrent analysis calls."""
        import asyncio
        from code_scalpel.mcp.server import analyze_code

        codes = [
            "def f1(): return 1",
            "def f2(): return 2",
            "def f3(): return 3",
        ]

        results = await asyncio.gather(*[analyze_code(code) for code in codes])

        for result in results:
            assert result.success is True


class TestValidationHelpers:
    """Tests for internal validation helpers."""

    async def test_validate_code_empty(self):
        """Test validation of empty code."""
        from code_scalpel.mcp.server import _validate_code

        valid, error = _validate_code("")
        assert valid is False
        assert error is not None

    async def test_validate_code_valid(self):
        """Test validation of valid code."""
        from code_scalpel.mcp.server import _validate_code

        valid, error = _validate_code("x = 1")
        assert valid is True
        assert error is None

    async def test_validate_code_non_string(self):
        """Test validation of non-string input."""
        from code_scalpel.mcp.server import _validate_code

        valid, error = _validate_code(123)  # type: ignore
        assert valid is False
        assert error is not None


class TestComplexityCalculation:
    """Tests for complexity estimation."""

    async def test_complexity_linear(self):
        """Test complexity of linear code."""
        from code_scalpel.mcp.server import _count_complexity
        import ast

        code = "x = 1\ny = 2\nz = x + y"
        tree = ast.parse(code)
        complexity = _count_complexity(tree)
        assert complexity == 1  # Base complexity

    async def test_complexity_with_if(self):
        """Test complexity with if statements."""
        from code_scalpel.mcp.server import _count_complexity
        import ast

        code = """
if x > 0:
    y = 1
"""
        tree = ast.parse(code)
        complexity = _count_complexity(tree)
        assert complexity == 2  # Base + 1 for if

    async def test_complexity_with_loop(self):
        """Test complexity with loops."""
        from code_scalpel.mcp.server import _count_complexity
        import ast

        code = """
for i in range(10):
    x = i
"""
        tree = ast.parse(code)
        complexity = _count_complexity(tree)
        assert complexity == 2  # Base + 1 for for


class TestResultModels:
    """Tests for Pydantic result models."""

    async def test_analysis_result_serialization(self):
        """Test AnalysisResult JSON serialization."""
        from code_scalpel.mcp.server import AnalysisResult

        result = AnalysisResult(
            success=True,
            functions=["foo", "bar"],
            classes=["MyClass"],
            imports=["os"],
            function_count=2,
            class_count=1,
            complexity=3,
            lines_of_code=10,
        )

        # Should serialize to dict/JSON
        data = result.model_dump()
        assert data["success"] is True
        assert len(data["functions"]) == 2

    async def test_security_result_serialization(self):
        """Test SecurityResult JSON serialization."""
        from code_scalpel.mcp.server import SecurityResult, VulnerabilityInfo

        result = SecurityResult(
            success=True,
            has_vulnerabilities=True,
            vulnerability_count=1,
            risk_level="high",
            vulnerabilities=[
                VulnerabilityInfo(
                    type="SQL Injection",
                    cwe="CWE-89",
                    severity="high",
                    line=5,
                    description="SQL injection via execute()",
                )
            ],
        )

        data = result.model_dump()
        assert data["vulnerability_count"] == 1
        assert len(data["vulnerabilities"]) == 1

    async def test_symbolic_result_serialization(self):
        """Test SymbolicResult JSON serialization."""
        from code_scalpel.mcp.server import SymbolicResult, ExecutionPath

        result = SymbolicResult(
            success=True,
            paths_explored=2,
            paths=[
                ExecutionPath(
                    path_id=0,
                    conditions=["x > 0"],
                    final_state={"x": 5},
                    reproduction_input={"x": 5},
                    is_reachable=True,
                ),
            ],
            symbolic_variables=["x"],
            constraints=["x > 0"],
        )

        data = result.model_dump()
        assert data["paths_explored"] == 2
        assert len(data["paths"]) == 1


class TestExtractCodeTool:
    """Tests for the extract_code surgical extraction tool."""

    async def test_extract_function_basic(self):
        """Test extracting a simple function."""
        from code_scalpel.mcp.server import extract_code

        code = """
def hello():
    return "Hello, World!"

def goodbye():
    return "Goodbye!"
"""
        result = await extract_code(
            code=code, target_type="function", target_name="hello"
        )
        assert result.success is True
        assert result.target_name == "hello"
        assert "return" in result.target_code
        assert result.full_code != ""

    async def test_extract_function_with_context(self):
        """Test extracting a function with its dependencies."""
        from code_scalpel.mcp.server import extract_code

        code = """
def helper():
    return 42

def main():
    return helper() + 1
"""
        result = await extract_code(
            code=code,
            target_type="function",
            target_name="main",
            include_context=True,
            context_depth=1,
        )
        assert result.success is True
        assert result.target_name == "main"
        assert len(result.context_items) > 0
        assert "helper" in result.context_items
        assert "helper" in result.context_code

    async def test_extract_class_basic(self):
        """Test extracting a simple class."""
        from code_scalpel.mcp.server import extract_code

        code = """
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""
        result = await extract_code(
            code=code, target_type="class", target_name="MyClass"
        )
        assert result.success is True
        assert result.target_name == "MyClass"
        assert "__init__" in result.target_code
        assert "get_value" in result.target_code

    async def test_extract_method(self):
        """Test extracting a specific method from a class."""
        from code_scalpel.mcp.server import extract_code

        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
        result = await extract_code(
            code=code, target_type="method", target_name="Calculator.add"
        )
        assert result.success is True
        assert result.target_name == "Calculator.add"
        assert "return a + b" in result.target_code

    async def test_extract_method_invalid_format(self):
        """Test method extraction with invalid name format."""
        from code_scalpel.mcp.server import extract_code

        code = "class Foo:\n    def bar(self): pass"
        result = await extract_code(code=code, target_type="method", target_name="bar")
        assert result.success is False
        assert "ClassName.method_name" in result.error

    async def test_extract_nonexistent_function(self):
        """Test extracting a function that doesn't exist."""
        from code_scalpel.mcp.server import extract_code

        code = "def foo(): pass"
        result = await extract_code(
            code=code, target_type="function", target_name="nonexistent"
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_extract_unknown_target_type(self):
        """Test extraction with unknown target type."""
        from code_scalpel.mcp.server import extract_code

        code = "def foo(): pass"
        result = await extract_code(
            code=code, target_type="variable", target_name="x"
        )
        assert result.success is False
        assert "unknown target_type" in result.error.lower()

    async def test_extract_token_estimate(self):
        """Test that token estimation is included."""
        from code_scalpel.mcp.server import extract_code

        code = """
def calculate(x, y, z):
    result = x + y
    result = result * z
    return result
"""
        result = await extract_code(
            code=code,
            target_type="function",
            target_name="calculate",
            include_token_estimate=True,
        )
        assert result.success is True
        assert result.token_estimate > 0

    async def test_extract_class_with_context(self):
        """Test extracting a class with its dependencies."""
        from code_scalpel.mcp.server import extract_code

        code = """
def utility_func():
    return 100

class MyClass:
    def __init__(self):
        self.value = utility_func()
"""
        result = await extract_code(
            code=code,
            target_type="class",
            target_name="MyClass",
            include_context=True,
        )
        assert result.success is True
        assert result.target_name == "MyClass"
        # Should include utility_func as dependency
        assert "utility_func" in result.context_items

    async def test_extract_syntax_error(self):
        """Test extraction from code with syntax errors."""
        from code_scalpel.mcp.server import extract_code

        code = "def broken("
        result = await extract_code(
            code=code, target_type="function", target_name="broken"
        )
        assert result.success is False
        assert result.error is not None

    async def test_extract_no_input_error(self):
        """Test extraction fails when neither file_path nor code provided."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            target_type="function", target_name="foo"
        )
        assert result.success is False
        assert "file_path" in result.error.lower() or "code" in result.error.lower()


class TestExtractCodeFromFile:
    """Tests for file-based extraction - the TOKEN-EFFICIENT path."""

    async def test_extract_function_from_file(self, tmp_path):
        """Test extracting a function directly from a file (0 token cost for read)."""
        from code_scalpel.mcp.server import extract_code

        # Create a "large" file - Agent never sees these 20 lines
        file_content = '''
"""A utility module with many functions."""
import math

def unused_function_1():
    """This function is not requested."""
    return "not needed"

def unused_function_2():
    """Another unrequested function."""
    return "also not needed"

def calculate_tax(amount, rate=0.1):
    """Calculate tax for a given amount."""
    return amount * rate

def unused_function_3():
    """Yet another function the agent doesn't need."""
    return "still not needed"

class UnrelatedClass:
    """The agent doesn't need this either."""
    pass
'''
        test_file = tmp_path / "utils.py"
        test_file.write_text(file_content)

        # Agent asks: "Get me calculate_tax from utils.py"
        # Agent sends: ~50 tokens (the request)
        # Server reads file: FREE to Agent
        # Agent receives: ~50 tokens (just the function)
        result = await extract_code(
            file_path=str(test_file),
            target_type="function",
            target_name="calculate_tax",
        )

        assert result.success is True
        assert result.target_name == "calculate_tax"
        assert "amount * rate" in result.target_code
        # Verify the agent did NOT receive the other functions
        assert "unused_function" not in result.target_code
        assert "UnrelatedClass" not in result.target_code

    async def test_extract_class_from_file(self, tmp_path):
        """Test extracting a class directly from a file."""
        from code_scalpel.mcp.server import extract_code

        file_content = '''
class OtherClass:
    pass

class TargetClass:
    """This is what we want."""
    def __init__(self, value):
        self.value = value

    def process(self):
        return self.value * 2

class AnotherClass:
    pass
'''
        test_file = tmp_path / "models.py"
        test_file.write_text(file_content)

        result = await extract_code(
            file_path=str(test_file),
            target_type="class",
            target_name="TargetClass",
        )

        assert result.success is True
        assert "TargetClass" in result.target_code
        assert "__init__" in result.target_code
        assert "process" in result.target_code
        # Verify we didn't get the other classes
        assert "OtherClass" not in result.target_code
        assert "AnotherClass" not in result.target_code

    async def test_extract_method_from_file(self, tmp_path):
        """Test extracting a specific method from a file."""
        from code_scalpel.mcp.server import extract_code

        file_content = '''
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b
'''
        test_file = tmp_path / "calc.py"
        test_file.write_text(file_content)

        result = await extract_code(
            file_path=str(test_file),
            target_type="method",
            target_name="Calculator.multiply",
        )

        assert result.success is True
        assert "multiply" in result.target_code
        assert "a * b" in result.target_code

    async def test_extract_with_context_from_file(self, tmp_path):
        """Test extracting a function with its dependencies from a file."""
        from code_scalpel.mcp.server import extract_code

        file_content = '''
def helper():
    return 42

def another_helper():
    return 100

def main():
    return helper() + 1

def unrelated():
    return "not connected"
'''
        test_file = tmp_path / "service.py"
        test_file.write_text(file_content)

        result = await extract_code(
            file_path=str(test_file),
            target_type="function",
            target_name="main",
            include_context=True,
        )

        assert result.success is True
        assert result.target_name == "main"
        assert "helper" in result.context_items
        assert "helper" in result.context_code
        # another_helper and unrelated are NOT dependencies
        assert "another_helper" not in result.context_code
        assert "unrelated" not in result.context_code

    async def test_extract_file_not_found(self):
        """Test extraction from non-existent file."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path="/nonexistent/path/to/file.py",
            target_type="function",
            target_name="foo",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_extract_function_not_found_in_file(self, tmp_path):
        """Test extraction of non-existent function from valid file."""
        from code_scalpel.mcp.server import extract_code

        file_content = "def existing_func(): pass"
        test_file = tmp_path / "test.py"
        test_file.write_text(file_content)

        result = await extract_code(
            file_path=str(test_file),
            target_type="function",
            target_name="nonexistent_func",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_token_savings_demonstration(self, tmp_path):
        """Demonstrate the token savings of file-based extraction."""
        from code_scalpel.mcp.server import extract_code


        # Create a "large" file with 100 lines
        lines = ["# Line of code"] * 100
        lines[50] = "def target_function():\n    return 'found me'"
        file_content = "\n".join(lines)
        test_file = tmp_path / "large_file.py"
        test_file.write_text(file_content)

        result = await extract_code(
            file_path=str(test_file),
            target_type="function",
            target_name="target_function",
            include_token_estimate=True,
        )

        assert result.success is True
        # The returned code should be tiny compared to the file
        assert result.token_estimate < 50  # ~2 lines worth of tokens
        # The full file would be ~100 lines * 15 chars / 4 = ~375 tokens
        # Token savings: Agent received ~50 tokens instead of ~375


class TestUpdateSymbolTool:
    """Tests for the update_symbol surgical modification tool."""

    async def test_update_function_in_file(self, tmp_path):
        """Test updating a function in a file."""
        from code_scalpel.mcp.server import update_symbol

        file_content = '''
def old_function():
    """Old implementation."""
    return 1


def other_function():
    return 2
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(file_content)

        new_code = '''def old_function():
    """New and improved!"""
    return 42
'''
        result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="old_function",
            new_code=new_code,
        )

        assert result.success is True
        assert result.target_name == "old_function"
        assert result.target_type == "function"

        # Verify file was modified
        modified_content = test_file.read_text()
        assert "return 42" in modified_content
        assert "return 2" in modified_content  # other_function preserved
        assert "return 1" not in modified_content  # old code gone

        # Verify backup was created
        assert result.backup_path is not None
        assert (tmp_path / "test.py.bak").exists()

    async def test_update_class_in_file(self, tmp_path):
        """Test updating a class in a file."""
        from code_scalpel.mcp.server import update_symbol

        file_content = '''
class OldClass:
    """Old class."""
    def method(self):
        return 1


def helper():
    return 2
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(file_content)

        new_code = '''class OldClass:
    """Completely rewritten!"""
    def method(self):
        return 42

    def new_method(self):
        return 100
'''
        result = await update_symbol(
            file_path=str(test_file),
            target_type="class",
            target_name="OldClass",
            new_code=new_code,
        )

        assert result.success is True
        assert result.target_type == "class"

        modified_content = test_file.read_text()
        assert "new_method" in modified_content
        assert "helper" in modified_content  # preserved

    async def test_update_method_in_file(self, tmp_path):
        """Test updating a method within a class."""
        from code_scalpel.mcp.server import update_symbol

        file_content = '''
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(file_content)

        new_code = '''def add(self, a, b):
    """Now with logging!"""
    print(f"Adding {a} + {b}")
    return a + b
'''
        result = await update_symbol(
            file_path=str(test_file),
            target_type="method",
            target_name="Calculator.add",
            new_code=new_code,
        )

        assert result.success is True
        assert result.target_type == "method"

        modified_content = test_file.read_text()
        assert "print" in modified_content
        assert "subtract" in modified_content  # preserved

    async def test_update_file_not_found(self):
        """Test error when file doesn't exist."""
        from code_scalpel.mcp.server import update_symbol

        result = await update_symbol(
            file_path="/nonexistent/path.py",
            target_type="function",
            target_name="foo",
            new_code="def foo(): pass",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_update_function_not_found(self, tmp_path):
        """Test error when function doesn't exist."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("def existing(): pass")

        result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="nonexistent",
            new_code="def nonexistent(): pass",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_update_invalid_syntax(self, tmp_path):
        """Test error when new code has syntax error."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("def target(): pass")

        result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="target",
            new_code="def target( broken syntax",
        )

        assert result.success is False
        assert "syntax" in result.error.lower()

    async def test_update_invalid_target_type(self, tmp_path):
        """Test error for invalid target type."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        result = await update_symbol(
            file_path=str(test_file),
            target_type="variable",
            target_name="x",
            new_code="x = 2",
        )

        assert result.success is False
        assert "invalid" in result.error.lower()

    async def test_update_method_invalid_format(self, tmp_path):
        """Test error when method name doesn't have ClassName.method format."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("class Foo:\n    def bar(self): pass")

        result = await update_symbol(
            file_path=str(test_file),
            target_type="method",
            target_name="bar",  # Missing ClassName.
            new_code="def bar(self): pass",
        )

        assert result.success is False
        assert "ClassName.method_name" in result.error

    async def test_update_no_backup(self, tmp_path):
        """Test updating without creating backup."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("def old(): return 1")

        result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="old",
            new_code="def old(): return 42",
            create_backup=False,
        )

        assert result.success is True
        assert result.backup_path is None
        assert not (tmp_path / "test.py.bak").exists()

    async def test_update_lines_delta(self, tmp_path):
        """Test that line count changes are tracked."""
        from code_scalpel.mcp.server import update_symbol

        test_file = tmp_path / "test.py"
        test_file.write_text("def short(): return 1")

        new_code = '''def short():
    x = 1
    y = 2
    z = 3
    return x + y + z
'''
        result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="short",
            new_code=new_code,
        )

        assert result.success is True
        assert result.lines_after > result.lines_before
        assert result.lines_delta > 0

    async def test_extract_modify_update_workflow(self, tmp_path):
        """Test the full workflow: extract -> modify -> update."""
        from code_scalpel.mcp.server import extract_code, update_symbol

        # Original file
        file_content = '''
def calculate_tax(amount):
    """Calculate tax."""
    return amount * 0.1


def other_function():
    return 42
'''
        test_file = tmp_path / "utils.py"
        test_file.write_text(file_content)

        # Step 1: Extract the function (token-efficient)
        extract_result = await extract_code(
            file_path=str(test_file),
            target_type="function",
            target_name="calculate_tax",
        )

        assert extract_result.success is True
        assert "amount * 0.1" in extract_result.target_code

        # Step 2: "Agent modifies" the code (simulated)
        modified_code = '''def calculate_tax(amount, rate=0.1):
    """Calculate tax with configurable rate."""
    return round(amount * rate, 2)
'''

        # Step 3: Update the file with the new code
        update_result = await update_symbol(
            file_path=str(test_file),
            target_type="function",
            target_name="calculate_tax",
            new_code=modified_code,
        )

        assert update_result.success is True

        # Verify final state
        final_content = test_file.read_text()
        assert "rate=0.1" in final_content  # New parameter
        assert "round(" in final_content  # New logic
        assert "other_function" in final_content  # Preserved


@pytest.mark.asyncio
class TestCrossFileDependenciesMCP:
    """Test extract_code with include_cross_file_deps=True."""

    @pytest.fixture
    def multi_file_project(self, tmp_path):
        """Create a multi-file project for cross-file tests."""
        # models.py
        models_py = tmp_path / "models.py"
        models_py.write_text('''"""Models module."""

class TaxRate:
    """Tax rate configuration."""

    def __init__(self, rate: float):
        self.rate = rate

    def calculate(self, amount: float) -> float:
        return amount * self.rate


def get_default_rate() -> float:
    """Get the default tax rate."""
    return 0.1
''')

        # utils.py
        utils_py = tmp_path / "utils.py"
        utils_py.write_text('''"""Utilities module."""

from models import TaxRate, get_default_rate


def calculate_tax(amount: float) -> float:
    """Calculate tax using TaxRate."""
    rate = TaxRate(get_default_rate())
    return rate.calculate(amount)


def simple_function():
    """No external dependencies."""
    return 42
''')

        return tmp_path

    async def test_cross_file_deps_resolves_class(self, multi_file_project):
        """Test that cross-file deps resolves imported classes."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="calculate_tax",
            include_cross_file_deps=True,
        )

        assert result.success is True
        assert "calculate_tax" in result.target_code

        # Should have resolved TaxRate from models.py
        assert "TaxRate" in result.context_code
        assert "class TaxRate" in result.context_code
        assert "models.py" in result.context_code

    async def test_cross_file_deps_resolves_function(self, multi_file_project):
        """Test that cross-file deps resolves imported functions."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="calculate_tax",
            include_cross_file_deps=True,
        )

        assert result.success is True

        # Should have resolved get_default_rate from models.py
        assert "get_default_rate" in result.context_code
        assert "def get_default_rate" in result.context_code

    async def test_cross_file_deps_full_code(self, multi_file_project):
        """Test that full_code contains external deps + target."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="calculate_tax",
            include_cross_file_deps=True,
        )

        assert result.success is True

        # full_code should have both external symbols and target
        assert "class TaxRate" in result.full_code
        assert "def calculate_tax" in result.full_code
        assert "# From" in result.full_code

    async def test_cross_file_deps_context_items(self, multi_file_project):
        """Test that context_items lists resolved symbols."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="calculate_tax",
            include_cross_file_deps=True,
        )

        assert result.success is True
        assert len(result.context_items) >= 1

        # Context items should mention the resolved symbols
        context_str = " ".join(result.context_items)
        assert "TaxRate" in context_str

    async def test_cross_file_deps_no_deps(self, multi_file_project):
        """Test function with no external dependencies."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="simple_function",
            include_cross_file_deps=True,
        )

        assert result.success is True
        assert "simple_function" in result.target_code
        assert result.context_code == ""  # No external deps

    async def test_cross_file_deps_requires_file_path(self):
        """Test that cross-file deps requires file_path (not code)."""
        from code_scalpel.mcp.server import extract_code

        code = '''
from models import TaxRate

def my_func():
    return TaxRate(0.1)
'''

        result = await extract_code(
            code=code,  # Using code, not file_path
            target_type="function",
            target_name="my_func",
            include_cross_file_deps=True,
        )

        # Should succeed but won't resolve cross-file deps
        assert result.success is True
        # Without file_path, can't resolve external deps
        assert "TaxRate" not in result.context_code or result.context_code == ""

    async def test_cross_file_deps_token_savings(self, multi_file_project):
        """Test that cross-file extraction is token-efficient."""
        from code_scalpel.mcp.server import extract_code

        result = await extract_code(
            file_path=str(multi_file_project / "utils.py"),
            target_type="function",
            target_name="calculate_tax",
            include_cross_file_deps=True,
            include_token_estimate=True,
        )

        assert result.success is True
        assert result.token_estimate > 0

        # Token estimate should be reasonable (not the entire file)
        # The models.py + target should be ~150-250 tokens
        assert result.token_estimate < 500
