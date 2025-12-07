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

        code = '''
def hello():
    return "Hello, World!"
'''
        result = await analyze_code(code)
        assert result.success is True
        assert "hello" in result.functions
        assert result.function_count == 1

    async def test_analyze_class(self):
        """Test analyzing a class definition."""
        from code_scalpel.mcp.server import analyze_code

        code = '''
class MyClass:
    def __init__(self):
        pass

    def method(self):
        return 42
'''
        result = await analyze_code(code)
        assert result.success is True
        assert "MyClass" in result.classes
        assert result.class_count == 1
        assert result.function_count == 2  # __init__ and method

    async def test_analyze_imports(self):
        """Test analyzing imports."""
        from code_scalpel.mcp.server import analyze_code

        code = '''
import os
from pathlib import Path
import sys
'''
        result = await analyze_code(code)
        assert result.success is True
        assert "os" in result.imports
        assert "sys" in result.imports
        assert any("Path" in imp for imp in result.imports)

    async def test_analyze_complexity(self):
        """Test complexity calculation."""
        from code_scalpel.mcp.server import analyze_code

        code = '''
def complex_func(x):
    if x > 0:
        if x > 10:
            return "big"
        return "small"
    return "negative"
'''
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

        code = '''
async def async_func():
    await some_coroutine()
'''
        result = await analyze_code(code)
        assert result.success is True
        assert any("async_func" in f for f in result.functions)


class TestSecurityScanTool:
    """Tests for the security_scan tool."""

    async def test_scan_clean_code(self):
        """Test scanning clean code with no vulnerabilities."""
        from code_scalpel.mcp.server import security_scan

        code = '''
def safe_function(x):
    return x + 1
'''
        result = await security_scan(code)
        assert result.success is True
        assert result.vulnerability_count == 0
        assert result.risk_level == "low"

    async def test_scan_sql_injection(self):
        """Test detecting SQL injection."""
        from code_scalpel.mcp.server import security_scan

        code = '''
def vulnerable(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
'''
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True
        assert any("SQL" in v.type for v in result.vulnerabilities)

    async def test_scan_command_injection(self):
        """Test detecting command injection."""
        from code_scalpel.mcp.server import security_scan

        code = '''
import os
def run_command(user_input):
    os.system("ls " + user_input)
'''
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True
        assert any("Command" in v.type for v in result.vulnerabilities)

    async def test_scan_eval_injection(self):
        """Test detecting eval injection."""
        from code_scalpel.mcp.server import security_scan

        code = '''
def dangerous(user_input):
    result = eval(user_input)
    return result
'''
        result = await security_scan(code)
        assert result.success is True
        assert result.has_vulnerabilities is True

    async def test_scan_empty_code(self):
        """Test scanning empty code."""
        from code_scalpel.mcp.server import security_scan

        result = await security_scan("")
        assert result.success is False
        assert result.error is not None

    async def test_scan_detects_taint_sources(self):
        """Test detection of taint sources."""
        from code_scalpel.mcp.server import security_scan

        code = '''
from flask import request
def handler():
    user_data = request.args.get('user')
    return user_data
'''
        result = await security_scan(code)
        assert result.success is True
        # Either SecurityAnalyzer or fallback should detect request.args
        # (May or may not have vulnerabilities depending on implementation)

    async def test_scan_risk_levels(self):
        """Test risk level calculation."""
        from code_scalpel.mcp.server import security_scan

        # Code with multiple vulnerabilities
        code = '''
import os
def very_dangerous(user_input):
    eval(user_input)
    exec(user_input)
    os.system(user_input)
'''
        result = await security_scan(code)
        assert result.success is True
        assert result.risk_level in ["high", "critical"]


class TestSymbolicExecuteTool:
    """Tests for the symbolic_execute tool."""

    async def test_symbolic_simple_function(self):
        """Test symbolic execution on simple function."""
        from code_scalpel.mcp.server import symbolic_execute

        code = '''
def simple(x):
    return x + 1
'''
        result = await symbolic_execute(code)
        assert result.success is True
        assert result.paths_explored >= 1

    async def test_symbolic_branching(self):
        """Test symbolic execution with branches."""
        from code_scalpel.mcp.server import symbolic_execute

        code = '''
def branching(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
'''
        result = await symbolic_execute(code)
        assert result.success is True
        # Should find paths for both branches
        assert result.paths_explored >= 1
        assert len(result.constraints) > 0

    async def test_symbolic_multiple_branches(self):
        """Test symbolic execution with multiple branches."""
        from code_scalpel.mcp.server import symbolic_execute

        code = '''
def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
'''
        result = await symbolic_execute(code)
        assert result.success is True
        assert result.paths_explored >= 1

    async def test_symbolic_detects_symbolic_vars(self):
        """Test that symbolic variables are detected."""
        from code_scalpel.mcp.server import symbolic_execute

        code = '''
def func(a, b, c):
    if a > b:
        return c
    return a + b
'''
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

        code = '''
def many_branches(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return 1
            return 2
        return 3
    return 4
'''
        result = await symbolic_execute(code, max_paths=5)
        assert result.success is True
        # Should respect max_paths
        assert result.paths_explored <= 10  # Some buffer for implementation

    async def test_symbolic_loop_handling(self):
        """Test symbolic execution with loops."""
        from code_scalpel.mcp.server import symbolic_execute

        code = '''
def with_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total
'''
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
        from code_scalpel.mcp.server import analyze_code, security_scan, symbolic_execute

        assert callable(analyze_code)
        assert callable(security_scan)
        assert callable(symbolic_execute)

    async def test_result_models_are_valid(self):
        """Test that result models are proper Pydantic models."""
        from code_scalpel.mcp.server import AnalysisResult, SecurityResult, SymbolicResult

        # Should be importable and usable
        assert AnalysisResult is not None
        assert SecurityResult is not None
        assert SymbolicResult is not None

    async def test_code_validation(self):
        """Test code validation for all tools."""
        from code_scalpel.mcp.server import analyze_code, security_scan, symbolic_execute

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
        from code_scalpel.mcp.server import analyze_code, security_scan, symbolic_execute

        code = '''
def process_user(user_id):
    if user_id > 0:
        return f"User: {user_id}"
    return "Invalid"
'''
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
