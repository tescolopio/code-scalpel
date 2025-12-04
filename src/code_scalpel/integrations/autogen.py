"""
AutogenScalpel - Integration wrapper for Autogen with Code Scalpel analysis capabilities.

This module provides the AutogenScalpel class that wraps Code Scalpel's
AST analysis capabilities for use with Autogen agents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AnalysisResult:
    """Result of code analysis."""

    code: str
    ast_analysis: dict[str, Any] = field(default_factory=dict)
    security_issues: list[dict[str, Any]] = field(default_factory=list)
    style_issues: dict[str, list[str]] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    refactored_code: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "code": self.code,
            "ast_analysis": self.ast_analysis,
            "security_issues": self.security_issues,
            "style_issues": self.style_issues,
            "suggestions": self.suggestions,
            "refactored_code": self.refactored_code,
            "error": self.error,
        }


class AutogenScalpel:
    """
    Wrapper class that integrates Code Scalpel's analysis capabilities
    with Autogen agents for async code analysis.

    This class provides async methods for analyzing code using AST,
    detecting security issues, and suggesting refactoring improvements.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the AutogenScalpel wrapper.

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

    async def analyze_async(self, code: str) -> AnalysisResult:
        """
        Perform async code analysis using AST tools.

        Args:
            code: Python source code to analyze.

        Returns:
            AnalysisResult containing analysis details.
        """
        # Run the synchronous analysis in a thread pool to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, code)

    def _analyze_sync(self, code: str) -> AnalysisResult:
        """
        Synchronous code analysis implementation.

        Args:
            code: Python source code to analyze.

        Returns:
            AnalysisResult containing analysis details.
        """
        result = AnalysisResult(code=code)

        try:
            # Parse to AST
            tree = self.analyzer.parse_to_ast(code)

            # Perform style analysis
            result.style_issues = self.analyzer.analyze_code_style(tree)

            # Find security issues
            result.security_issues = self.analyzer.find_security_issues(tree)

            # Generate suggestions based on analysis
            result.suggestions = self._generate_suggestions(result)

            # Collect basic AST info
            result.ast_analysis = {
                "parsed": True,
                "style_issues_count": sum(len(v) for v in result.style_issues.values()),
                "security_issues_count": len(result.security_issues),
            }

        except SyntaxError as e:
            result.error = f"Syntax error: {str(e)}"
            result.ast_analysis = {"parsed": False, "error": str(e)}
        except Exception as e:
            result.error = f"Analysis error: {str(e)}"
            result.ast_analysis = {"parsed": False, "error": str(e)}

        return result

    async def refactor_async(
        self, code: str, refactor_type: str = "auto"
    ) -> AnalysisResult:
        """
        Perform async code refactoring based on analysis.

        Args:
            code: Python source code to refactor.
            refactor_type: Type of refactoring ("auto", "simplify", "optimize").

        Returns:
            AnalysisResult with refactored code.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._refactor_sync, code, refactor_type
        )

    def _refactor_sync(self, code: str, refactor_type: str) -> AnalysisResult:
        """
        Synchronous refactoring implementation.

        Args:
            code: Python source code to refactor.
            refactor_type: Type of refactoring.

        Returns:
            AnalysisResult with refactored code.
        """
        # First perform analysis
        result = self._analyze_sync(code)

        if result.error:
            return result

        try:
            tree = self.analyzer.parse_to_ast(code)
            # For now, return the same code - refactoring logic can be extended
            result.refactored_code = self.analyzer.ast_to_code(tree)
        except Exception as e:
            result.error = f"Refactoring error: {str(e)}"

        return result

    def _generate_suggestions(self, result: AnalysisResult) -> list[str]:
        """
        Generate suggestions based on analysis results.

        Args:
            result: AnalysisResult from analysis.

        Returns:
            List of suggestion strings.
        """
        suggestions = []

        # Add suggestions based on style issues
        for issue_type, _issues in result.style_issues.items():
            if issue_type == "long_functions":
                suggestions.append(
                    "Consider breaking down long functions into smaller, focused functions."
                )
            elif issue_type == "deep_nesting":
                suggestions.append(
                    "Consider reducing nesting depth using early returns or guard clauses."
                )
            elif issue_type == "naming_conventions":
                suggestions.append(
                    "Follow PEP 8 naming conventions: snake_case for functions, CamelCase for classes."
                )

        # Add suggestions based on security issues
        if result.security_issues:
            for issue in result.security_issues:
                if issue.get("type") == "dangerous_function":
                    suggestions.append(
                        f"Avoid using dangerous function '{issue.get('function')}' - consider safer alternatives."
                    )
                elif issue.get("type") == "sql_injection":
                    suggestions.append(
                        "Use parameterized queries to prevent SQL injection vulnerabilities."
                    )

        return suggestions

    def get_tool_description(self) -> dict[str, str]:
        """
        Get description for use as an Autogen tool.

        Returns:
            Tool description dictionary for Autogen integration.
        """
        return {
            "name": "code_scalpel_analyzer",
            "description": (
                "Analyzes Python code using AST parsing to detect style issues, "
                "security vulnerabilities, and suggest improvements."
            ),
            "parameters": {
                "code": "Python source code to analyze",
                "refactor": "Whether to include refactoring suggestions (optional)",
            },
        }


# Backward compatibility alias
AutogenCodeAnalysisAgent = AutogenScalpel
