"""
Tests for Code Scalpel integrations (AutogenScalpel, CrewAIScalpel, MCP Server).
"""

import asyncio
import os
import sys
import unittest

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from code_scalpel.integrations import (
    AnalysisResult,
    AutogenScalpel,
    CrewAIScalpel,
    RefactorResult,
)
from code_scalpel.integrations.mcp_server import create_app


class TestAutogenScalpel(unittest.TestCase):
    """Tests for AutogenScalpel wrapper."""

    def setUp(self):
        self.scalpel = AutogenScalpel()
        self.sample_code = """
def BadFunc():
    eval("print('hello')")
    return True
"""

    def test_analyze_async_success(self):
        """Test async analysis returns successful result."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertIsInstance(result, AnalysisResult)
        self.assertTrue(result.ast_analysis.get("parsed"))
        self.assertIsNone(result.error)

    def test_analyze_async_detects_security_issues(self):
        """Test that analysis detects security issues like eval()."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertGreater(result.ast_analysis.get("security_issues_count", 0), 0)

    def test_analyze_async_detects_naming_issues(self):
        """Test that analysis detects naming convention issues."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertGreater(result.ast_analysis.get("style_issues_count", 0), 0)

    def test_analyze_async_generates_suggestions(self):
        """Test that analysis generates suggestions."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertGreater(len(result.suggestions), 0)

    def test_analyze_async_syntax_error(self):
        """Test that syntax errors are handled gracefully."""
        bad_code = "def foo( return 42"

        async def run_test():
            result = await self.scalpel.analyze_async(bad_code)
            return result

        result = asyncio.run(run_test())
        self.assertFalse(result.ast_analysis.get("parsed"))
        self.assertIsNotNone(result.error)

    def test_refactor_async_success(self):
        """Test async refactoring returns successful result."""

        async def run_test():
            result = await self.scalpel.refactor_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertIsInstance(result, AnalysisResult)
        self.assertIsNotNone(result.refactored_code)

    def test_result_to_dict(self):
        """Test that result can be converted to dictionary."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        result_dict = result.to_dict()
        self.assertIn("code", result_dict)
        self.assertIn("ast_analysis", result_dict)
        self.assertIn("suggestions", result_dict)

    def test_get_tool_description(self):
        """Test that tool description is returned correctly."""
        description = self.scalpel.get_tool_description()
        self.assertIn("name", description)
        self.assertIn("description", description)


class TestCrewAIScalpel(unittest.TestCase):
    """Tests for CrewAIScalpel wrapper."""

    def setUp(self):
        self.scalpel = CrewAIScalpel()
        self.sample_code = """
def calculateSum(numbers):
    for i in range(len(numbers)):
        for j in range(len(numbers[i])):
            for k in range(len(numbers[i][j])):
                for l in range(len(numbers[i][j][k])):
                    pass
    return 0
"""

    def test_analyze_success(self):
        """Test synchronous analysis."""
        result = self.scalpel.analyze(self.sample_code)
        self.assertIsInstance(result, RefactorResult)
        self.assertTrue(result.success)

    def test_analyze_async_success(self):
        """Test async analysis."""

        async def run_test():
            result = await self.scalpel.analyze_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertIsInstance(result, RefactorResult)
        self.assertTrue(result.success)

    def test_analyze_detects_deep_nesting(self):
        """Test that analysis detects deep nesting issues."""
        result = self.scalpel.analyze(self.sample_code)
        # Check for deep nesting in issues
        has_nesting_issue = any(
            issue.get("category") == "deep_nesting" for issue in result.issues
        )
        self.assertTrue(has_nesting_issue)

    def test_analyze_security_async(self):
        """Test async security analysis."""
        code_with_eval = "eval('1+1')"

        async def run_test():
            result = await self.scalpel.analyze_security_async(code_with_eval)
            return result

        result = asyncio.run(run_test())
        self.assertTrue(result["success"])
        self.assertEqual(result["risk_level"], "medium")
        self.assertGreater(len(result["issues"]), 0)

    def test_refactor_success(self):
        """Test synchronous refactoring."""
        result = self.scalpel.refactor(self.sample_code)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.refactored_code)

    def test_refactor_async_success(self):
        """Test async refactoring."""

        async def run_test():
            result = await self.scalpel.refactor_async(self.sample_code)
            return result

        result = asyncio.run(run_test())
        self.assertTrue(result.success)
        self.assertIsNotNone(result.refactored_code)

    def test_result_to_dict(self):
        """Test that result can be converted to dictionary."""
        result = self.scalpel.analyze(self.sample_code)
        result_dict = result.to_dict()
        self.assertIn("original_code", result_dict)
        self.assertIn("analysis", result_dict)
        self.assertIn("success", result_dict)

    def test_get_crewai_tools(self):
        """Test that CrewAI tools are returned correctly."""
        tools = self.scalpel.get_crewai_tools()
        self.assertEqual(len(tools), 3)
        tool_names = [t["name"] for t in tools]
        self.assertIn("analyze_code", tool_names)
        self.assertIn("refactor_code", tool_names)
        self.assertIn("security_scan", tool_names)

    def test_risk_level_calculation(self):
        """Test risk level calculation through security analysis."""
        # Low risk - code with no security issues
        clean_code = "def foo(): return 42"
        result = self.scalpel.analyze_security(clean_code)
        self.assertEqual(result["risk_level"], "low")

        # Medium risk - one dangerous function
        medium_code = "eval('1+1')"
        result = self.scalpel.analyze_security(medium_code)
        self.assertEqual(result["risk_level"], "medium")


class TestMCPServer(unittest.TestCase):
    """Tests for MCP Server endpoints."""

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.sample_code = """
def BadFunc():
    eval("test")
    return True
"""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "code-scalpel-mcp")

    def test_analyze_endpoint_success(self):
        """Test analyze endpoint with valid code."""
        response = self.client.post("/analyze", json={"code": self.sample_code})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("analysis", data)
        self.assertIn("issues", data)
        self.assertIn("processing_time_ms", data)

    def test_analyze_endpoint_missing_code(self):
        """Test analyze endpoint with missing code field."""
        response = self.client.post("/analyze", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_analyze_endpoint_empty_body(self):
        """Test analyze endpoint with empty body."""
        response = self.client.post("/analyze")
        # Flask returns 415 (Unsupported Media Type) when no JSON is provided
        # or 400 (Bad Request) depending on configuration
        self.assertIn(response.status_code, [400, 415])

    def test_refactor_endpoint_success(self):
        """Test refactor endpoint with valid code."""
        response = self.client.post(
            "/refactor", json={"code": self.sample_code, "task": "improve naming"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("refactored_code", data)

    def test_security_endpoint_success(self):
        """Test security endpoint with valid code."""
        response = self.client.post("/security", json={"code": self.sample_code})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("risk_level", data)
        self.assertIn("recommendations", data)

    def test_response_time_under_2s(self):
        """Test that response time is under 2 seconds."""
        import time

        start = time.time()
        response = self.client.post("/analyze", json={"code": self.sample_code})
        elapsed = time.time() - start

        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 2.0, "Response time should be under 2 seconds")


if __name__ == "__main__":
    unittest.main()
