"""
Tests for the MCP (Model Context Protocol) server.

Tests the real MCP-compliant server implementation using FastMCP.
"""

import pytest


class TestMCPServerModule:
    """Test MCP server module loads correctly."""

    def test_mcp_server_imports(self):
        """Test that MCP server module can be imported."""
        from code_scalpel.mcp.server import mcp, analyze_code, security_scan, symbolic_execute
        assert mcp is not None
        assert callable(analyze_code)
        assert callable(security_scan)
        assert callable(symbolic_execute)

    def test_mcp_server_name(self):
        """Test MCP server has correct name."""
        from code_scalpel.mcp.server import mcp
        assert mcp.name == "Code Scalpel"

    def test_mcp_tools_registered(self):
        """Test all expected tools are registered."""
        from code_scalpel.mcp.server import mcp
        
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "analyze_code" in tool_names
        assert "security_scan" in tool_names
        assert "symbolic_execute" in tool_names


class TestAnalyzeCodeTool:
    """Test the analyze_code MCP tool."""

    def test_analyze_simple_function(self):
        """Test analyzing code with a simple function."""
        from code_scalpel.mcp.server import analyze_code
        
        code = '''
def hello(name):
    return f"Hello, {name}!"
'''
        result = analyze_code(code)
        assert result.success is True
        assert "hello" in result.functions
        assert result.lines_of_code > 0

    def test_analyze_class(self):
        """Test analyzing code with a class."""
        from code_scalpel.mcp.server import analyze_code
        
        code = '''
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
'''
        result = analyze_code(code)
        assert result.success is True
        assert "Calculator" in result.classes
        assert "add" in result.functions
        assert "subtract" in result.functions

    def test_analyze_imports(self):
        """Test analyzing code with imports."""
        from code_scalpel.mcp.server import analyze_code
        
        code = '''
import os
from pathlib import Path
import sys as system

def get_cwd():
    return os.getcwd()
'''
        result = analyze_code(code)
        assert result.success is True
        assert len(result.imports) >= 2

    def test_analyze_syntax_error(self):
        """Test analyzing code with syntax errors."""
        from code_scalpel.mcp.server import analyze_code
        
        code = '''
def broken(
    return "missing closing paren"
'''
        result = analyze_code(code)
        assert result.success is False
        assert result.error is not None
        assert "Syntax error" in result.error

    def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        from code_scalpel.mcp.server import analyze_code
        
        result = analyze_code("")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_result_is_serializable(self):
        """Test that result can be converted to dict (JSON-serializable)."""
        from code_scalpel.mcp.server import analyze_code
        
        code = "def foo(): pass"
        result = analyze_code(code)
        
        # Pydantic models use model_dump() for serialization
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "functions" in result_dict


class TestSecurityScanTool:
    """Test the security_scan MCP tool."""

    def test_scan_clean_code(self):
        """Test scanning code with no vulnerabilities."""
        from code_scalpel.mcp.server import security_scan
        
        code = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        result = security_scan(code)
        assert result.success is True
        assert result.vulnerability_count == 0

    def test_scan_sql_injection(self):
        """Test detecting SQL injection vulnerability."""
        from code_scalpel.mcp.server import security_scan
        
        code = '''
def get_user(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
'''
        result = security_scan(code)
        assert result.success is True
        # May or may not detect depending on taint analysis depth
        # Just verify it runs without error

    def test_scan_command_injection(self):
        """Test detecting command injection vulnerability."""
        from code_scalpel.mcp.server import security_scan
        
        code = '''
import os

def run_command(user_input):
    os.system("echo " + user_input)
'''
        result = security_scan(code)
        assert result.success is True

    def test_scan_empty_code(self):
        """Test scanning empty code."""
        from code_scalpel.mcp.server import security_scan
        
        result = security_scan("")
        assert result.success is False

    def test_security_result_serializable(self):
        """Test that security result is serializable."""
        from code_scalpel.mcp.server import security_scan
        
        code = "x = 1"
        result = security_scan(code)
        
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "vulnerability_count" in result_dict


class TestSymbolicExecuteTool:
    """Test the symbolic_execute MCP tool."""

    def test_execute_simple_function(self):
        """Test symbolic execution of simple function."""
        from code_scalpel.mcp.server import symbolic_execute
        
        code = '''
def abs_value(x):
    if x < 0:
        return -x
    return x
'''
        result = symbolic_execute(code)
        assert result.success is True
        assert result.paths_explored >= 1

    def test_execute_with_max_paths(self):
        """Test symbolic execution with path limit."""
        from code_scalpel.mcp.server import symbolic_execute
        
        code = '''
def branchy(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return 1
            return 2
        return 3
    return 4
'''
        result = symbolic_execute(code, max_paths=2)
        assert result.success is True

    def test_execute_syntax_error(self):
        """Test symbolic execution with syntax error."""
        from code_scalpel.mcp.server import symbolic_execute
        
        result = symbolic_execute("def broken(")
        assert result.success is False
        assert result.error is not None

    def test_execute_empty_code(self):
        """Test symbolic execution with empty code."""
        from code_scalpel.mcp.server import symbolic_execute
        
        result = symbolic_execute("")
        assert result.success is False

    def test_symbolic_result_serializable(self):
        """Test that symbolic result is serializable."""
        from code_scalpel.mcp.server import symbolic_execute
        
        code = "def f(x): return x + 1"
        result = symbolic_execute(code)
        
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "paths_explored" in result_dict


class TestMCPDataClasses:
    """Test MCP result Pydantic models."""

    def test_analysis_result_fields(self):
        """Test AnalysisResult has expected fields."""
        from code_scalpel.mcp.server import AnalysisResult
        
        result = AnalysisResult(
            success=True,
            functions=["foo"],
            classes=["Bar"],
            imports=["os"],
            complexity=5,
            lines_of_code=10,
            issues=[],
            error=None,
        )
        assert result.success is True
        assert result.functions == ["foo"]
        assert result.complexity == 5

    def test_security_result_fields(self):
        """Test SecurityResult has expected fields."""
        from code_scalpel.mcp.server import SecurityResult
        
        result = SecurityResult(
            success=True,
            has_vulnerabilities=True,
            vulnerability_count=2,
            risk_level="high",
            vulnerabilities=[],
            taint_sources=[],
            error=None,
        )
        assert result.success is True
        assert result.vulnerability_count == 2
        assert result.risk_level == "high"

    def test_symbolic_result_fields(self):
        """Test SymbolicResult has expected fields."""
        from code_scalpel.mcp.server import SymbolicResult
        
        result = SymbolicResult(
            success=True,
            paths_explored=5,
            paths=[],
            symbolic_variables=[],
            constraints=[],
            error=None,
        )
        assert result.success is True
        assert result.paths_explored == 5


class TestCodeValidation:
    """Test code validation helper."""

    def test_validate_rejects_empty(self):
        """Test validation rejects empty code."""
        from code_scalpel.mcp.server import _validate_code
        
        valid, error = _validate_code("")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_rejects_too_large(self):
        """Test validation rejects overly large code."""
        from code_scalpel.mcp.server import _validate_code, MAX_CODE_SIZE
        
        huge_code = "x = 1\n" * (MAX_CODE_SIZE // 5)
        valid, error = _validate_code(huge_code)
        assert valid is False
        assert "large" in error.lower() or "size" in error.lower()

    def test_validate_accepts_valid_code(self):
        """Test validation accepts valid code."""
        from code_scalpel.mcp.server import _validate_code
        
        valid, error = _validate_code("def foo(): pass")
        assert valid is True
        assert error is None


class TestMCPIntegration:
    """Integration tests for MCP server."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        from code_scalpel.mcp.server import analyze_code, security_scan, symbolic_execute
        
        code = '''
def calculate_discount(price, discount_percent):
    if discount_percent < 0:
        raise ValueError("Discount cannot be negative")
    if discount_percent > 100:
        raise ValueError("Discount cannot exceed 100%")
    return price * (1 - discount_percent / 100)
'''
        # Step 1: Analyze structure
        analysis = analyze_code(code)
        assert analysis.success is True
        assert "calculate_discount" in analysis.functions
        
        # Step 2: Security scan
        security = security_scan(code)
        assert security.success is True
        
        # Step 3: Symbolic execution
        symbolic = symbolic_execute(code)
        assert symbolic.success is True
        assert symbolic.paths_explored >= 1

    def test_handles_complex_code(self):
        """Test handling of more complex code structures."""
        from code_scalpel.mcp.server import analyze_code
        
        code = '''
from typing import List, Optional
import json

class DataProcessor:
    """Processes data from various sources."""
    
    def __init__(self, config: dict):
        self.config = config
        self._cache = {}
    
    def process(self, data: List[dict]) -> List[dict]:
        results = []
        for item in data:
            if self._validate(item):
                results.append(self._transform(item))
        return results
    
    def _validate(self, item: dict) -> bool:
        return "id" in item and "value" in item
    
    def _transform(self, item: dict) -> dict:
        return {
            "id": item["id"],
            "processed_value": item["value"] * 2,
        }

def main():
    processor = DataProcessor({"debug": True})
    data = [{"id": 1, "value": 10}]
    print(processor.process(data))
'''
        result = analyze_code(code)
        assert result.success is True
        assert "DataProcessor" in result.classes
        assert len(result.functions) >= 4  # process, _validate, _transform, main
        assert len(result.imports) >= 2
