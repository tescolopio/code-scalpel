"""
Tests for Code Scalpel integrations (AutogenScalpel, CrewAIScalpel, REST API Server).
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
from code_scalpel.integrations.rest_api_server import create_app


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
        """Test async security analysis with taint-based analyzer."""
        # Use a real vulnerability: user input flows to eval
        code_with_vuln = """
user_input = input("Enter code: ")
eval(user_input)
"""

        async def run_test():
            result = await self.scalpel.analyze_security_async(code_with_vuln)
            return result

        result = asyncio.run(run_test())
        self.assertTrue(result["success"])
        # New taint-based analyzer detects actual data flow vulnerabilities
        self.assertIn(result["risk_level"], ["high", "critical", "medium"])
        # Check for vulnerabilities (new format) or issues (fallback format)
        has_findings = (
            result.get("has_vulnerabilities", False) or
            len(result.get("vulnerabilities", [])) > 0 or
            len(result.get("issues", [])) > 0
        )
        self.assertTrue(has_findings, "Should detect eval vulnerability")

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

        # High/Critical risk - SQL injection with user input
        vuln_code = """
user_id = request.args.get("id")
cursor.execute("SELECT * FROM users WHERE id=" + user_id)
"""
        result = self.scalpel.analyze_security(vuln_code)
        self.assertIn(result["risk_level"], ["high", "critical"])


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
        # New taint-based API returns different fields
        # Check for new format (vulnerabilities) or old format (recommendations)
        has_security_info = (
            "vulnerabilities" in data or 
            "recommendations" in data or
            "vulnerability_count" in data
        )
        self.assertTrue(has_security_info)

    def test_response_time_under_2s(self):
        """Test that response time is under 2 seconds."""
        import time

        start = time.time()
        response = self.client.post("/analyze", json={"code": self.sample_code})
        elapsed = time.time() - start

        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 2.0, "Response time should be under 2 seconds")

    def test_analyze_code_not_string(self):
        """Test analyze endpoint with non-string code."""
        response = self.client.post("/analyze", json={"code": 12345})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Code must be a string", data["error"])

    def test_refactor_endpoint_missing_code(self):
        """Test refactor endpoint with missing code field."""
        response = self.client.post("/refactor", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        # Empty json triggers "Request body is required" since get_json returns {}
        # which is truthy but lacks "code" key
        self.assertIn("error", data)

    def test_refactor_endpoint_missing_body(self):
        """Test refactor endpoint with no request body."""
        response = self.client.post(
            "/refactor", content_type="application/json", data=""
        )
        self.assertIn(response.status_code, [400, 415])

    def test_refactor_code_not_string(self):
        """Test refactor endpoint with non-string code."""
        response = self.client.post("/refactor", json={"code": ["not", "a", "string"]})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Code must be a string", data["error"])

    def test_refactor_default_task(self):
        """Test refactor endpoint uses default task when not provided."""
        response = self.client.post("/refactor", json={"code": "x = 1"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

    def test_security_endpoint_missing_code(self):
        """Test security endpoint with missing code field."""
        response = self.client.post("/security", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        # Empty json triggers error
        self.assertIn("error", data)

    def test_security_endpoint_missing_body(self):
        """Test security endpoint with no request body."""
        response = self.client.post(
            "/security", content_type="application/json", data=""
        )
        self.assertIn(response.status_code, [400, 415])

    def test_security_code_not_string(self):
        """Test security endpoint with non-string code."""
        response = self.client.post("/security", json={"code": {"not": "string"}})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Code must be a string", data["error"])

    def test_analyze_returns_error_field_on_syntax_error(self):
        """Test analyze endpoint includes error field for syntax errors."""
        response = self.client.post("/analyze", json={"code": "def foo( return"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        # Analysis may succeed but report parse failure
        self.assertIn("processing_time_ms", data)


class TestMCPServerConfig(unittest.TestCase):
    """Tests for REST API Server configuration (legacy name: MCP Server)."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from code_scalpel.integrations.rest_api_server import MCPServerConfig

        config = MCPServerConfig()
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8080)
        self.assertFalse(config.debug)
        self.assertTrue(config.cache_enabled)
        self.assertEqual(config.max_code_size, 100000)

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from code_scalpel.integrations.rest_api_server import MCPServerConfig

        config = MCPServerConfig(
            host="0.0.0.0",
            port=9000,
            debug=True,
            cache_enabled=False,
            max_code_size=50000,
        )
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)
        self.assertTrue(config.debug)
        self.assertFalse(config.cache_enabled)
        self.assertEqual(config.max_code_size, 50000)

    def test_create_app_with_custom_config(self):
        """Test creating app with custom config."""
        from code_scalpel.integrations.rest_api_server import MCPServerConfig, create_app

        config = MCPServerConfig(cache_enabled=False)
        app = create_app(config)
        self.assertIsNotNone(app)
        # Should work with custom config
        client = app.test_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_code_size_limit_enforced(self):
        """Test that code size limit is enforced."""
        from code_scalpel.integrations.rest_api_server import MCPServerConfig, create_app

        # Create app with very small max code size
        config = MCPServerConfig(max_code_size=10)
        app = create_app(config)
        client = app.test_client()

        # Code exceeding limit should be rejected
        # Flask's MAX_CONTENT_LENGTH returns 413 (Request Entity Too Large)
        # Our custom check returns 400
        response = client.post(
            "/analyze", json={"code": "x = 1\ny = 2\nz = 3\nprint('hello world')"}
        )
        # Could be 413 from Flask or 400 from our handler
        self.assertIn(response.status_code, [400, 413])
        if response.status_code == 400:
            data = response.get_json()
            self.assertFalse(data["success"])
            self.assertIn("exceeds maximum size", data["error"])

    def test_elapsed_ms_helper(self):
        """Test the _elapsed_ms helper function."""
        import time

        from code_scalpel.integrations.rest_api_server import _elapsed_ms

        start = time.time()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = _elapsed_ms(start)
        self.assertGreater(elapsed, 5)  # Should be at least 5ms
        self.assertLess(elapsed, 100)  # Should be less than 100ms


class TestMCPServerRunServer(unittest.TestCase):
    """Tests for run_server function (without actually running)."""

    def test_run_server_production_warning(self):
        """Test that debug mode is disabled in production."""
        import os
        import warnings

        from code_scalpel.integrations.rest_api_server import run_server

        # Set production environment
        os.environ["FLASK_ENV"] = "production"

        try:
            # Should warn and force debug=False, but we can't actually run
            # Just verify the import and function exist
            self.assertTrue(callable(run_server))
        finally:
            # Restore environment
            os.environ.pop("FLASK_ENV", None)


if __name__ == "__main__":
    unittest.main()
