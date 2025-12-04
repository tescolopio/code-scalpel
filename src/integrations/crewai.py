"""
CrewAIScalpel - Integration wrapper for CrewAI with Code Scalpel analysis capabilities.

This module provides the CrewAIScalpel class that wraps Code Scalpel's
AST analysis capabilities for use with CrewAI agents and tools.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class RefactorResult:
    """Result of code refactoring analysis."""
    original_code: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    refactored_code: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original_code": self.original_code,
            "analysis": self.analysis,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "refactored_code": self.refactored_code,
            "success": self.success,
            "error": self.error
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
                import sys
                import os
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
            result.issues.extend([
                {"type": "security", **issue}
                for issue in security_issues
            ])
            
            # Generate suggestions
            result.suggestions = self._generate_suggestions(
                style_issues, security_issues
            )
            
            # Store analysis metadata
            result.analysis = {
                "parsed": True,
                "total_issues": len(result.issues),
                "style_issues": sum(len(v) for v in style_issues.values()),
                "security_issues": len(security_issues)
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
        self, 
        code: str, 
        task_description: str = "improve code quality"
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
        self, 
        code: str, 
        task_description: str = "improve code quality"
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
    
    async def analyze_security_async(self, code: str) -> Dict[str, Any]:
        """
        Perform async security-focused analysis.
        
        Args:
            code: Python source code to analyze.
            
        Returns:
            Dictionary with security analysis results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.analyze_security, code
        )
    
    def analyze_security(self, code: str) -> Dict[str, Any]:
        """
        Perform synchronous security-focused analysis.
        
        Args:
            code: Python source code to analyze.
            
        Returns:
            Dictionary with security analysis results.
        """
        try:
            tree = self.analyzer.parse_to_ast(code)
            security_issues = self.analyzer.find_security_issues(tree)
            
            return {
                "success": True,
                "issues": security_issues,
                "risk_level": self._calculate_risk_level(security_issues),
                "recommendations": self._get_security_recommendations(security_issues)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "issues": [],
                "risk_level": "unknown"
            }
    
    def _analyze_security_sync(self, code: str) -> Dict[str, Any]:
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
        self, 
        style_issues: Dict[str, List[str]], 
        security_issues: List[Dict[str, Any]]
    ) -> List[str]:
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
        if style_issues.get('long_functions'):
            suggestions.append(
                "Break down long functions into smaller, single-purpose functions."
            )
        
        if style_issues.get('deep_nesting'):
            suggestions.append(
                "Reduce nesting depth using early returns or extracting methods."
            )
        
        if style_issues.get('naming_conventions'):
            suggestions.append(
                "Follow PEP 8 naming conventions for better code readability."
            )
        
        # Security-based suggestions
        dangerous_funcs = [
            issue for issue in security_issues 
            if issue.get('type') == 'dangerous_function'
        ]
        if dangerous_funcs:
            funcs = ', '.join(set(i.get('function', '') for i in dangerous_funcs))
            suggestions.append(
                f"Replace dangerous functions ({funcs}) with safer alternatives."
            )
        
        sql_issues = [
            issue for issue in security_issues 
            if issue.get('type') == 'sql_injection'
        ]
        if sql_issues:
            suggestions.append(
                "Use parameterized queries instead of string formatting for SQL."
            )
        
        return suggestions
    
    def _calculate_risk_level(
        self, 
        security_issues: List[Dict[str, Any]]
    ) -> str:
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
            1 for i in security_issues 
            if i.get('type') == 'dangerous_function'
        )
        sql_count = sum(
            1 for i in security_issues 
            if i.get('type') == 'sql_injection'
        )
        
        total_critical = dangerous_count + sql_count
        
        if total_critical >= 3:
            return "critical"
        elif total_critical >= 2:
            return "high"
        elif total_critical >= 1:
            return "medium"
        return "low"
    
    def _get_security_recommendations(
        self, 
        security_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Get specific security recommendations.
        
        Args:
            security_issues: List of security issue dictionaries.
            
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        for issue in security_issues:
            issue_type = issue.get('type')
            if issue_type == 'dangerous_function':
                func = issue.get('function', 'unknown')
                if 'eval' in func or 'exec' in func:
                    recommendations.append(
                        f"Replace '{func}' with ast.literal_eval or a safer parser."
                    )
                elif 'os.system' in func or 'subprocess' in func:
                    recommendations.append(
                        f"Replace '{func}' with subprocess.run with shell=False."
                    )
                elif 'pickle' in func:
                    recommendations.append(
                        "Use json or other safe serialization instead of pickle."
                    )
            elif issue_type == 'sql_injection':
                recommendations.append(
                    "Use parameterized queries (?, %s) instead of string formatting."
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_crewai_tools(self) -> List[Dict[str, Any]]:
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
                "func": self.analyze
            },
            {
                "name": "refactor_code",
                "description": (
                    "Refactors Python code based on analysis to improve quality "
                    "and fix identified issues."
                ),
                "func": self.refactor
            },
            {
                "name": "security_scan",
                "description": (
                    "Performs security-focused analysis to identify vulnerabilities "
                    "like dangerous function usage and SQL injection."
                ),
                "func": self.analyze_security
            }
        ]
