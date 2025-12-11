"""
CrewAIScalpel - Integration wrapper for CrewAI with Code Scalpel analysis capabilities.

This module provides the CrewAIScalpel class that wraps Code Scalpel's
AST analysis capabilities for use with CrewAI agents and tools.

v0.3.1: Now includes taint-based SecurityAnalyzer and SymbolicAnalyzer.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RefactorResult:
    """Result of code refactoring analysis."""

    original_code: str
    analysis: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    refactored_code: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original_code": self.original_code,
            "analysis": self.analysis,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "refactored_code": self.refactored_code,
            "success": self.success,
            "error": self.error,
        }


class CrewAIScalpel:
    """
    Wrapper class that integrates Code Scalpel's analysis capabilities
    with CrewAI agents for async code analysis and refactoring.

    This class provides async methods suitable for CrewAI tool integration,
    including code analysis, security scanning, and refactoring capabilities.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the CrewAIScalpel wrapper.

        Args:
            cache_enabled: Whether to cache AST parsing results for performance.
        """
        # Import ASTAnalyzer - handle different import contexts
        try:
            from ..ast_tools.analyzer import ASTAnalyzer
        except (ImportError, ValueError):
            try:
                from ast_tools.analyzer import ASTAnalyzer
            except ImportError:
                # Direct import as fallback
                import os
                import sys

                src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from ast_tools.analyzer import ASTAnalyzer
        self.analyzer = ASTAnalyzer(cache_enabled=cache_enabled)
        self._cache_enabled = cache_enabled

    async def analyze_async(self, code: str) -> RefactorResult:
        """
        Perform async code analysis using AST tools.

        Args:
            code: Python source code to analyze.

        Returns:
            RefactorResult containing analysis details.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, code)

    def analyze(self, code: str) -> RefactorResult:
        """
        Synchronous code analysis (for non-async contexts).

        Args:
            code: Python source code to analyze.

        Returns:
            RefactorResult containing analysis details.
        """
        return self._analyze_sync(code)

    def _analyze_sync(self, code: str) -> RefactorResult:
        """
        Synchronous code analysis implementation.

        Args:
            code: Python source code to analyze.

        Returns:
            RefactorResult containing analysis details.
        """
        result = RefactorResult(original_code=code)

        try:
            # Parse to AST
            tree = self.analyzer.parse_to_ast(code)

            # Perform style analysis
            style_issues = self.analyzer.analyze_code_style(tree)

            # Find security issues
            security_issues = self.analyzer.find_security_issues(tree)

            # Combine all issues
            result.issues = [
                {"type": "style", "category": k, "details": v}
                for k, v in style_issues.items()
                if v
            ]
            result.issues.extend(
                [{"type": "security", **issue} for issue in security_issues]
            )

            # Generate suggestions
            result.suggestions = self._generate_suggestions(
                style_issues, security_issues
            )

            # Store analysis metadata
            result.analysis = {
                "parsed": True,
                "total_issues": len(result.issues),
                "style_issues": sum(len(v) for v in style_issues.values()),
                "security_issues": len(security_issues),
            }

        except SyntaxError as e:
            result.success = False
            result.error = f"Syntax error: {str(e)}"
            result.analysis = {"parsed": False, "error": str(e)}
        except Exception as e:
            result.success = False
            result.error = f"Analysis error: {str(e)}"
            result.analysis = {"parsed": False, "error": str(e)}

        return result

    async def refactor_async(
        self, code: str, task_description: str = "improve code quality"
    ) -> RefactorResult:
        """
        Perform async code refactoring based on analysis.

        Args:
            code: Python source code to refactor.
            task_description: Description of the refactoring task.

        Returns:
            RefactorResult with refactored code.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._refactor_sync, code, task_description
        )

    def refactor(
        self, code: str, task_description: str = "improve code quality"
    ) -> RefactorResult:
        """
        Synchronous code refactoring (for non-async contexts).

        Args:
            code: Python source code to refactor.
            task_description: Description of the refactoring task.

        Returns:
            RefactorResult with refactored code.
        """
        return self._refactor_sync(code, task_description)

    def _refactor_sync(self, code: str, task_description: str) -> RefactorResult:
        """
        Synchronous refactoring implementation.

        Args:
            code: Python source code to refactor.
            task_description: Description of the refactoring task.

        Returns:
            RefactorResult with refactored code.
        """
        # First perform analysis
        result = self._analyze_sync(code)

        if not result.success:
            return result

        try:
            tree = self.analyzer.parse_to_ast(code)
            # Return regenerated code - refactoring logic can be extended
            result.refactored_code = self.analyzer.ast_to_code(tree)
        except Exception as e:
            result.success = False
            result.error = f"Refactoring error: {str(e)}"

        return result

    # =========================================================================
    # Symbolic Execution (v0.3.0+)
    # =========================================================================

    async def analyze_symbolic_async(self, code: str) -> dict[str, Any]:
        """
        Perform async symbolic execution analysis.

        Args:
            code: Python source code to analyze.

        Returns:
            Dictionary with symbolic execution results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_symbolic, code)

    def analyze_symbolic(self, code: str) -> dict[str, Any]:
        """
        Perform symbolic execution analysis to find execution paths and edge cases.

        v0.3.0: Uses Z3-powered symbolic execution engine to:
        - Enumerate all possible execution paths
        - Generate concrete test inputs for each path
        - Identify dead code and unreachable branches

        Args:
            code: Python source code to analyze.

        Returns:
            Dictionary with symbolic execution results including:
            - paths: List of execution paths with constraints
            - test_inputs: Generated test cases
            - dead_code: Unreachable code segments
        """
        try:
            from ..symbolic_execution_tools import SymbolicAnalyzer

            analyzer = SymbolicAnalyzer()
            result = analyzer.analyze(code)

            # Extract path information
            paths_info = []
            for i, path in enumerate(result.paths[:10]):  # Limit to 10 paths
                path_data = {
                    "path_id": i,
                    "feasible": path.is_feasible
                    if hasattr(path, "is_feasible")
                    else True,
                }
                # Try to get variables from path
                if hasattr(path, "variables"):
                    path_data["variables"] = {
                        k: str(v) for k, v in path.variables.items()
                    }
                elif hasattr(path, "state") and hasattr(
                    path.state, "get_all_variables"
                ):
                    path_data["variables"] = {
                        k: str(v) for k, v in path.state.get_all_variables().items()
                    }
                paths_info.append(path_data)

            return {
                "success": True,
                "total_paths": result.total_paths,
                "feasible_paths": result.feasible_count,
                "infeasible_paths": result.infeasible_count,
                "paths": paths_info,
                "all_variables": {k: str(v) for k, v in result.all_variables.items()}
                if result.all_variables
                else {},
                "analyzer": "z3-symbolic",
            }
        except ImportError as e:
            return {
                "success": False,
                "error": f"Symbolic execution tools not available: {e}",
                "paths": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "paths": [],
            }

    # =========================================================================
    # Security Analysis (v0.3.1 - Taint-based)
    # =========================================================================

    async def analyze_security_async(self, code: str) -> dict[str, Any]:
        """
        Perform async security-focused analysis.

        Args:
            code: Python source code to analyze.

        Returns:
            Dictionary with security analysis results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_security, code)

    def analyze_security(self, code: str) -> dict[str, Any]:
        """
        Perform synchronous security-focused analysis using taint tracking.

        v0.3.1: Now uses the SecurityAnalyzer with taint-based vulnerability
        detection (SQL injection, XSS, command injection, path traversal).

        Args:
            code: Python source code to analyze.

        Returns:
            Dictionary with security analysis results including:
            - vulnerabilities: List of detected vulnerabilities with CWE IDs
            - taint_flows: Data flow paths from sources to sinks
            - risk_level: Overall risk assessment
        """
        try:
            # Use the new taint-based SecurityAnalyzer (v0.3.0+)
            try:
                from ..symbolic_execution_tools import analyze_security as taint_analyze

                result = taint_analyze(code)

                vulnerabilities = [v.to_dict() for v in result.vulnerabilities]

                return {
                    "success": True,
                    "vulnerabilities": vulnerabilities,
                    "vulnerability_count": result.vulnerability_count,
                    "has_vulnerabilities": result.has_vulnerabilities,
                    "sql_injections": len(result.get_sql_injections()),
                    "xss": len(result.get_xss()),
                    "command_injections": len(result.get_command_injections()),
                    "path_traversals": len(result.get_path_traversals()),
                    "risk_level": self._calculate_risk_from_vulns(vulnerabilities),
                    "summary": result.summary()
                    if result.has_vulnerabilities
                    else "No vulnerabilities detected",
                }
            except ImportError:
                # Fallback to AST-based analysis if symbolic tools not available
                tree = self.analyzer.parse_to_ast(code)
                security_issues = self.analyzer.find_security_issues(tree)

                return {
                    "success": True,
                    "issues": security_issues,
                    "risk_level": self._calculate_risk_level(security_issues),
                    "recommendations": self._get_security_recommendations(
                        security_issues
                    ),
                    "analyzer": "ast-based (fallback)",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "vulnerabilities": [],
                "risk_level": "unknown",
            }

    def _calculate_risk_from_vulns(self, vulnerabilities: list[dict]) -> str:
        """Calculate risk level from vulnerability list."""
        if not vulnerabilities:
            return "low"

        # Check for critical vulnerabilities
        critical_types = {"SQL Injection", "Command Injection"}
        high_types = {"Cross-Site Scripting (XSS)", "Path Traversal"}

        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", "")
            if vuln_type in critical_types:
                return "critical"
            if vuln_type in high_types:
                return "high"

        return "medium" if vulnerabilities else "low"

    def _analyze_security_sync(self, code: str) -> dict[str, Any]:
        """
        Deprecated: Use analyze_security() instead.

        Synchronous security analysis implementation.

        Args:
            code: Python source code to analyze.

        Returns:
            Dictionary with security analysis results.
        """
        return self.analyze_security(code)

    def _generate_suggestions(
        self, style_issues: dict[str, list[str]], security_issues: list[dict[str, Any]]
    ) -> list[str]:
        """
        Generate suggestions based on analysis results.

        Args:
            style_issues: Dictionary of style issues by category.
            security_issues: List of security issue dictionaries.

        Returns:
            List of suggestion strings.
        """
        suggestions = []

        # Style-based suggestions
        if style_issues.get("long_functions"):
            suggestions.append(
                "Break down long functions into smaller, single-purpose functions."
            )

        if style_issues.get("deep_nesting"):
            suggestions.append(
                "Reduce nesting depth using early returns or extracting methods."
            )

        if style_issues.get("naming_conventions"):
            suggestions.append(
                "Follow PEP 8 naming conventions for better code readability."
            )

        # Security-based suggestions
        dangerous_funcs = [
            issue
            for issue in security_issues
            if issue.get("type") == "dangerous_function"
        ]
        if dangerous_funcs:
            funcs = ", ".join({i.get("function", "") for i in dangerous_funcs})
            suggestions.append(
                f"Replace dangerous functions ({funcs}) with safer alternatives."
            )

        sql_issues = [
            issue for issue in security_issues if issue.get("type") == "sql_injection"
        ]
        if sql_issues:
            suggestions.append(
                "Use parameterized queries instead of string formatting for SQL."
            )

        return suggestions

    def _calculate_risk_level(self, security_issues: list[dict[str, Any]]) -> str:
        """
        Calculate overall security risk level.

        Args:
            security_issues: List of security issue dictionaries.

        Returns:
            Risk level string ("low", "medium", "high", "critical").
        """
        if not security_issues:
            return "low"

        # Count issue types
        dangerous_count = sum(
            1 for i in security_issues if i.get("type") == "dangerous_function"
        )
        sql_count = sum(1 for i in security_issues if i.get("type") == "sql_injection")

        total_critical = dangerous_count + sql_count

        if total_critical >= 3:
            return "critical"
        elif total_critical >= 2:
            return "high"
        elif total_critical >= 1:
            return "medium"
        return "low"

    def _get_security_recommendations(
        self, security_issues: list[dict[str, Any]]
    ) -> list[str]:
        """
        Get specific security recommendations.

        Args:
            security_issues: List of security issue dictionaries.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        for issue in security_issues:
            issue_type = issue.get("type")
            if issue_type == "dangerous_function":
                func = issue.get("function", "unknown")
                if "eval" in func or "exec" in func:
                    recommendations.append(
                        f"Replace '{func}' with ast.literal_eval or a safer parser."
                    )
                elif "os.system" in func or "subprocess" in func:
                    recommendations.append(
                        f"Replace '{func}' with subprocess.run with shell=False."
                    )
                elif "pickle" in func:
                    recommendations.append(
                        "Use json or other safe serialization instead of pickle."
                    )
            elif issue_type == "sql_injection":
                recommendations.append(
                    "Use parameterized queries (?, %s) instead of string formatting."
                )

        return list(set(recommendations))  # Remove duplicates

    def get_crewai_tools(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for CrewAI integration.

        Returns:
            List of tool definition dictionaries for CrewAI.
        """
        return [
            {
                "name": "analyze_code",
                "description": (
                    "Analyzes Python code for style issues, security vulnerabilities, "
                    "and improvement opportunities using AST parsing."
                ),
                "func": self.analyze,
            },
            {
                "name": "refactor_code",
                "description": (
                    "Refactors Python code based on analysis to improve quality "
                    "and fix identified issues."
                ),
                "func": self.refactor,
            },
            {
                "name": "security_scan",
                "description": (
                    "Performs security-focused analysis to identify vulnerabilities "
                    "like dangerous function usage and SQL injection."
                ),
                "func": self.analyze_security,
            },
        ]
