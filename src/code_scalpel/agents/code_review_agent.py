"""
Code Review Agent - Demonstrates comprehensive code review using Code Scalpel MCP tools.

This agent performs automated code review by:
1. Observing file structure and complexity
2. Analyzing security vulnerabilities
3. Checking code quality and best practices
4. Suggesting improvements with safe refactoring
"""

import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseCodeAnalysisAgent


class CodeReviewAgent(BaseCodeAnalysisAgent):
    """
    AI agent specialized in comprehensive code review.

    Uses Code Scalpel MCP tools to perform thorough analysis:
    - File structure analysis
    - Security vulnerability detection
    - Code quality assessment
    - Refactoring suggestions with safety verification
    """

    def __init__(self, workspace_root: Optional[str] = None):
        super().__init__(workspace_root)
        self.quality_thresholds = {
            "max_function_length": 30,
            "max_class_length": 200,
            "max_complexity_score": 10,
            "min_test_coverage": 80
        }

    async def observe(self, target: str) -> Dict[str, Any]:
        """Observe the target file and gather comprehensive information."""
        self.logger.info(f"Observing file: {target}")

        # Get file context
        file_info = await self.observe_file(target)
        if not file_info.get("success"):
            return file_info

        # Analyze security
        security_info = await self.analyze_code_security("# File analysis placeholder")
        # Note: In real usage, would read file content and analyze

        # Find symbol references for key functions
        symbol_analysis = {}
        for func_name in file_info.get("functions", [])[:3]:  # Analyze first 3 functions
            refs = await self.find_symbol_usage(func_name, self.context.workspace_root)
            if refs.get("success"):
                symbol_analysis[func_name] = refs

        return {
            "success": True,
            "file_info": file_info,
            "security_scan": security_info,
            "symbol_analysis": symbol_analysis,
            "target": target
        }

    async def orient(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze observations and identify issues and opportunities."""
        self.logger.info("Analyzing observations for code quality issues")

        file_info = observations.get("file_info", {})
        security_info = observations.get("security_info", {})

        issues = []
        suggestions = []

        # Analyze file structure
        structure_issues = self._analyze_file_structure(file_info)
        issues.extend(structure_issues)

        # Analyze security
        security_issues = self._analyze_security(security_info)
        issues.extend(security_issues)

        # Analyze symbol usage
        symbol_issues = self._analyze_symbol_usage(observations.get("symbol_analysis", {}))
        issues.extend(symbol_issues)

        # Generate improvement suggestions
        suggestions = self._generate_improvements(issues, file_info)

        return {
            "success": True,
            "issues": issues,
            "suggestions": suggestions,
            "severity_breakdown": self._categorize_issues(issues),
            "quality_score": self._calculate_quality_score(issues, file_info)
        }

    async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Decide which improvements to implement based on analysis."""
        self.logger.info("Deciding on code improvements to implement")

        issues = analysis.get("issues", [])
        suggestions = analysis.get("suggestions", [])

        # Prioritize issues by severity and feasibility
        high_priority = [i for i in issues if i.get("severity") == "high"]
        medium_priority = [i for i in issues if i.get("severity") == "medium"]
        low_priority = [i for i in issues if i.get("severity") == "low"]

        # For this demo, focus on high-priority security and quality issues
        actionable_items = []

        for issue in high_priority + medium_priority[:2]:  # Limit to top 2 medium
            if issue.get("actionable", False):
                actionable_items.append({
                    "issue": issue,
                    "action": self._create_action_plan(issue),
                    "confidence": issue.get("confidence", 0.8)
                })

        return {
            "success": True,
            "actionable_items": actionable_items,
            "total_actions": len(actionable_items),
            "prioritized_count": len(actionable_items),
            "estimated_effort": self._estimate_effort(actionable_items)
        }

    async def act(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided improvements safely."""
        self.logger.info("Executing code improvements")

        actionable_items = decisions.get("actionable_items", [])
        results = []

        for item in actionable_items:
            action = item.get("action", {})
            action_type = action.get("type")

            if action_type == "refactor_function":
                result = await self._execute_function_refactor(action)
            elif action_type == "add_security_check":
                result = await self._execute_security_improvement(action)
            elif action_type == "improve_documentation":
                result = await self._execute_documentation_improvement(action)
            else:
                result = {"success": False, "error": f"Unknown action type: {action_type}"}

            results.append({
                "action": action,
                "result": result,
                "success": result.get("success", False)
            })

        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)

        return {
            "success": success_count > 0 or total_count == 0,  # Success if actions succeeded or no actions needed
            "results": results,
            "success_rate": success_count / total_count if total_count > 0 else 1.0,
            "summary": f"Executed {success_count}/{total_count} improvements successfully"
        }

    # Analysis Helper Methods

    def _analyze_file_structure(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze file structure for quality issues."""
        issues = []

        complexity = file_info.get("complexity_score", 0)
        if complexity > self.quality_thresholds["max_complexity_score"]:
            issues.append({
                "type": "complexity",
                "severity": "medium",
                "description": f"File complexity score ({complexity}) exceeds threshold ({self.quality_thresholds['max_complexity_score']})",
                "actionable": True,
                "confidence": 0.9
            })

        function_count = len(file_info.get("functions", []))
        if function_count > 10:
            issues.append({
                "type": "structure",
                "severity": "low",
                "description": f"File has {function_count} functions - consider splitting into multiple files",
                "actionable": False,
                "confidence": 0.7
            })

        return issues

    def _analyze_security(self, security_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze security scan results."""
        issues = []

        vulnerabilities = security_info.get("vulnerabilities", [])
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low")
            issues.append({
                "type": "security",
                "severity": severity,
                "description": f"Security vulnerability: {vuln.get('description', 'Unknown')}",
                "actionable": True,
                "confidence": 0.95,
                "vulnerability": vuln
            })

        return issues

    def _analyze_symbol_usage(self, symbol_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze symbol usage patterns."""
        issues = []

        for symbol_name, refs in symbol_analysis.items():
            total_refs = refs.get("total_references", 0)
            if total_refs == 0:
                issues.append({
                    "type": "dead_code",
                    "severity": "low",
                    "description": f"Function '{symbol_name}' appears to be unused",
                    "actionable": False,
                    "confidence": 0.6
                })
            elif total_refs > 20:
                issues.append({
                    "type": "complexity",
                    "severity": "medium",
                    "description": f"Function '{symbol_name}' is used in {total_refs} places - consider simplifying",
                    "actionable": True,
                    "confidence": 0.8
                })

        return issues

    def _generate_improvements(self, issues: List[Dict[str, Any]], file_info: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []

        has_security_issues = any(i["type"] == "security" for i in issues)
        if has_security_issues:
            suggestions.append("Address security vulnerabilities before other improvements")

        complexity_issues = [i for i in issues if i["type"] == "complexity"]
        if complexity_issues:
            suggestions.append("Consider breaking down complex functions into smaller, focused functions")

        structure_issues = [i for i in issues if i["type"] == "structure"]
        if structure_issues:
            suggestions.append("Consider organizing code into multiple modules for better maintainability")

        return suggestions

    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize issues by severity."""
        breakdown = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            breakdown[severity] += 1
        return breakdown

    def _calculate_quality_score(self, issues: List[Dict[str, Any]], file_info: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        base_score = 100

        # Deduct points for issues
        severity_penalties = {"high": 20, "medium": 10, "low": 2}
        for issue in issues:
            penalty = severity_penalties.get(issue.get("severity", "low"), 2)
            base_score -= penalty

        # Bonus for good practices
        if file_info.get("has_security_issues") == False:
            base_score += 5

        return max(0, min(100, base_score))

    def _create_action_plan(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Create an action plan for an issue."""
        issue_type = issue.get("type")

        if issue_type == "security":
            return {
                "type": "add_security_check",
                "description": "Add input validation or sanitization",
                "target": issue.get("vulnerability", {}).get("location", "unknown")
            }
        elif issue_type == "complexity":
            return {
                "type": "refactor_function",
                "description": "Break down complex function into smaller functions",
                "target": "complex_function"
            }
        else:
            return {
                "type": "improve_documentation",
                "description": "Add documentation and comments",
                "target": "file"
            }

    def _estimate_effort(self, actionable_items: List[Dict[str, Any]]) -> str:
        """Estimate effort required for improvements."""
        total_items = len(actionable_items)
        if total_items == 0:
            return "minimal"
        elif total_items <= 2:
            return "low"
        elif total_items <= 5:
            return "medium"
        else:
            return "high"

    # Action Execution Methods

    async def _execute_function_refactor(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function refactoring action."""
        # This would use extract_code and update_symbol tools
        # For demo purposes, return success
        self.logger.info(f"Executing function refactor: {action}")
        return {"success": True, "message": "Function refactor completed"}

    async def _execute_security_improvement(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a security improvement action."""
        # This would use simulate_refactor and update_symbol tools
        self.logger.info(f"Executing security improvement: {action}")
        return {"success": True, "message": "Security improvement applied"}

    async def _execute_documentation_improvement(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation improvement action."""
        # This would use update_symbol to add docstrings
        self.logger.info(f"Executing documentation improvement: {action}")
        return {"success": True, "message": "Documentation improved"}
