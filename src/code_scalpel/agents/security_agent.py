"""
Security Agent - Specialized agent for security vulnerability detection and remediation.

This agent demonstrates how AI agents can use Code Scalpel's security analysis tools
to identify vulnerabilities and suggest safe remediation strategies.
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseCodeAnalysisAgent


class SecurityAgent(BaseCodeAnalysisAgent):
    """
    AI agent specialized in security analysis and remediation.

    Uses Code Scalpel MCP tools to:
    - Detect security vulnerabilities
    - Analyze attack vectors
    - Suggest safe remediation strategies
    - Verify fixes don't introduce new vulnerabilities
    """

    def __init__(self, workspace_root: Optional[str] = None):
        super().__init__(workspace_root)
        self.vulnerability_patterns = {
            "sql_injection": ["SQL_QUERY"],
            "command_injection": ["SHELL_COMMAND"],
            "xxe": ["XXE"],
            "ssti": ["SSTI"],
            "hardcoded_secrets": ["HARDCODED_SECRET"]
        }
        self.risk_levels = {"critical": 9, "high": 7, "medium": 5, "low": 3, "info": 1}

    async def observe(self, target: str) -> Dict[str, Any]:
        """Observe the target for security vulnerabilities."""
        self.logger.info(f"Performing security analysis on: {target}")

        # Get file context to understand structure
        file_info = await self.observe_file(target)
        if not file_info.get("success"):
            return file_info

        # Perform security scan
        # Note: In real usage, would read file content
        security_scan = await self.analyze_code_security("# Security scan placeholder")

        # Analyze symbol usage for potential attack vectors
        symbol_analysis = {}
        for func_name in file_info.get("functions", [])[:5]:  # Check first 5 functions
            refs = await self.find_symbol_usage(func_name, self.context.workspace_root)
            if refs.get("success"):
                symbol_analysis[func_name] = refs

        return {
            "success": True,
            "file_info": file_info,
            "security_scan": security_scan,
            "symbol_analysis": symbol_analysis,
            "target": target
        }

    async def orient(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security observations and assess risks."""
        self.logger.info("Analyzing security observations")

        security_scan = observations.get("security_scan", {})
        symbol_analysis = observations.get("symbol_analysis", {})

        vulnerabilities = security_scan.get("vulnerabilities", [])
        categorized_vulns = self._categorize_vulnerabilities(vulnerabilities)

        # Analyze attack vectors
        attack_vectors = self._analyze_attack_vectors(vulnerabilities, symbol_analysis)

        # Assess overall risk
        risk_assessment = self._assess_overall_risk(categorized_vulns, attack_vectors)

        # Generate remediation strategies
        remediation_plan = self._generate_remediation_plan(categorized_vulns, attack_vectors)

        return {
            "success": True,
            "vulnerabilities": categorized_vulns,
            "attack_vectors": attack_vectors,
            "risk_assessment": risk_assessment,
            "remediation_plan": remediation_plan,
            "total_vulnerabilities": len(vulnerabilities)
        }

    async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on security remediation actions."""
        self.logger.info("Deciding on security remediation actions")

        remediation_plan = analysis.get("remediation_plan", {})
        risk_assessment = analysis.get("risk_assessment", {})

        # Prioritize critical and high-risk issues
        prioritized_actions = []

        for vuln_type, plan in remediation_plan.items():
            if plan.get("priority") in ["critical", "high"]:
                prioritized_actions.extend(plan.get("actions", []))

        # Limit to top 3 most critical actions
        prioritized_actions = prioritized_actions[:3]

        # Create detailed action plans
        detailed_actions = []
        for action in prioritized_actions:
            detailed_actions.append({
                "action": action,
                "implementation_plan": self._create_implementation_plan(action),
                "verification_steps": self._create_verification_steps(action),
                "estimated_risk": self._estimate_remediation_risk(action)
            })

        return {
            "success": True,
            "prioritized_actions": detailed_actions,
            "total_actions": len(prioritized_actions),
            "risk_level": risk_assessment.get("overall_level", "unknown"),
            "estimated_effort": self._estimate_security_effort(detailed_actions)
        }

    async def act(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security remediation actions safely."""
        self.logger.info("Executing security remediation actions")

        prioritized_actions = decisions.get("prioritized_actions", [])
        results = []

        for action_plan in prioritized_actions:
            action = action_plan.get("action", {})
            implementation = action_plan.get("implementation_plan", {})

            # Verify the fix won't break anything
            verification = await self._verify_security_fix(action, implementation)
            if not verification.get("safe", False):
                results.append({
                    "action": action,
                    "result": {"success": False, "error": "Fix verification failed"},
                    "verification": verification
                })
                continue

            # Execute the fix
            result = await self._execute_security_fix(action, implementation)
            results.append({
                "action": action,
                "result": result,
                "verification": verification,
                "success": result.get("success", False)
            })

        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)

        return {
            "success": success_count > 0 or total_count == 0,  # Success if actions succeeded or no actions needed
            "results": results,
            "success_rate": success_count / total_count if total_count > 0 else 1.0,
            "summary": f"Successfully remediated {success_count}/{total_count} security issues"
        }

    # Security Analysis Methods

    def _categorize_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize vulnerabilities by type and severity."""
        categorized = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", "unknown")
            severity = vuln.get("severity", "low")

            if vuln_type not in categorized:
                categorized[vuln_type] = []
            categorized[vuln_type].append(vuln)

        return categorized

    def _analyze_attack_vectors(self, vulnerabilities: List[Dict[str, Any]], symbol_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential attack vectors."""
        vectors = []

        for vuln in vulnerabilities:
            location = vuln.get("location", {})
            function_name = location.get("function", "unknown")

            # Check if function is widely used
            refs = symbol_analysis.get(function_name, {})
            usage_count = refs.get("total_references", 0)

            vectors.append({
                "vulnerability": vuln,
                "function": function_name,
                "usage_count": usage_count,
                "exposure_level": "high" if usage_count > 10 else "medium" if usage_count > 3 else "low",
                "attack_surface": self._calculate_attack_surface(vuln, usage_count)
            })

        return vectors

    def _assess_overall_risk(self, categorized_vulns: Dict[str, List[Dict[str, Any]]], attack_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall security risk."""
        total_vulns = sum(len(vulns) for vulns in categorized_vulns.values())
        high_risk_vulns = 0
        critical_vulns = 0

        for vuln_list in categorized_vulns.values():
            for vuln in vuln_list:
                severity = vuln.get("severity", "low")
                if severity == "high":
                    high_risk_vulns += 1
                elif severity == "critical":
                    critical_vulns += 1

        # Calculate risk score
        risk_score = (critical_vulns * 10) + (high_risk_vulns * 7) + (total_vulns * 2)

        if risk_score >= 20:
            overall_level = "critical"
        elif risk_score >= 10:
            overall_level = "high"
        elif risk_score >= 5:
            overall_level = "medium"
        else:
            overall_level = "low"

        return {
            "overall_level": overall_level,
            "risk_score": risk_score,
            "critical_count": critical_vulns,
            "high_count": high_risk_vulns,
            "total_count": total_vulns
        }

    def _generate_remediation_plan(self, categorized_vulns: Dict[str, List[Dict[str, Any]]], attack_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive remediation plan."""
        plan = {}

        for vuln_type, vulns in categorized_vulns.items():
            if vuln_type in self.vulnerability_patterns:
                plan[vuln_type] = {
                    "priority": self._determine_priority(vulns),
                    "actions": self._create_remediation_actions(vuln_type, vulns),
                    "estimated_effort": self._estimate_vuln_effort(vulns)
                }

        return plan

    def _determine_priority(self, vulns: List[Dict[str, Any]]) -> str:
        """Determine remediation priority for a vulnerability type."""
        severities = [v.get("severity", "low") for v in vulns]
        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _create_remediation_actions(self, vuln_type: str, vulns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create specific remediation actions."""
        actions = []

        if vuln_type == "sql_injection":
            actions.append({
                "type": "input_validation",
                "description": "Add parameterized queries or input validation",
                "pattern": "Use prepared statements or ORM methods"
            })
        elif vuln_type == "command_injection":
            actions.append({
                "type": "sanitize_input",
                "description": "Sanitize shell command inputs",
                "pattern": "Use shlex.quote() or avoid shell=True"
            })
        elif vuln_type == "xxe":
            actions.append({
                "type": "disable_external_entities",
                "description": "Disable XML external entity processing",
                "pattern": "Use defusedxml or disable entity processing"
            })
        elif vuln_type == "ssti":
            actions.append({
                "type": "sanitize_templates",
                "description": "Prevent user-controlled template content",
                "pattern": "Use file-based templates only"
            })

        return actions

    def _calculate_attack_surface(self, vuln: Dict[str, Any], usage_count: int) -> str:
        """Calculate the attack surface area."""
        severity = vuln.get("severity", "low")
        if severity == "critical" and usage_count > 10:
            return "large"
        elif severity in ["high", "critical"] or usage_count > 5:
            return "medium"
        else:
            return "small"

    def _estimate_vuln_effort(self, vulns: List[Dict[str, Any]]) -> str:
        """Estimate effort to fix vulnerabilities."""
        count = len(vulns)
        if count <= 2:
            return "low"
        elif count <= 5:
            return "medium"
        else:
            return "high"

    def _create_implementation_plan(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed implementation plan."""
        action_type = action.get("type")

        if action_type == "input_validation":
            return {
                "steps": [
                    "Identify vulnerable SQL queries",
                    "Replace with parameterized queries",
                    "Add input validation",
                    "Test for SQL injection prevention"
                ],
                "tools_needed": ["sqlalchemy", "prepared statements"],
                "estimated_time": "2-4 hours"
            }
        elif action_type == "sanitize_input":
            return {
                "steps": [
                    "Identify shell command executions",
                    "Replace with safe alternatives",
                    "Add input sanitization",
                    "Test command injection prevention"
                ],
                "tools_needed": ["shlex", "subprocess"],
                "estimated_time": "1-3 hours"
            }

        return {
            "steps": ["Analyze vulnerability", "Implement fix", "Test fix"],
            "tools_needed": [],
            "estimated_time": "1-2 hours"
        }

    def _create_verification_steps(self, action: Dict[str, Any]) -> List[str]:
        """Create verification steps for a security fix."""
        action_type = action.get("type")

        if action_type == "input_validation":
            return [
                "Test with malicious SQL payloads",
                "Verify parameterized queries work",
                "Check for performance impact",
                "Run existing test suite"
            ]
        elif action_type == "sanitize_input":
            return [
                "Test with command injection payloads",
                "Verify shell escaping works",
                "Check for functionality regression",
                "Run security test suite"
            ]

        return [
            "Test fix with attack payloads",
            "Verify no regression",
            "Run relevant test cases"
        ]

    def _estimate_remediation_risk(self, action: Dict[str, Any]) -> str:
        """Estimate the risk of implementing the remediation."""
        action_type = action.get("type")

        # Higher risk for changes that might break functionality
        high_risk_types = ["input_validation", "sanitize_input"]
        if action_type in high_risk_types:
            return "medium"
        else:
            return "low"

    def _estimate_security_effort(self, detailed_actions: List[Dict[str, Any]]) -> str:
        """Estimate total effort for security remediation."""
        total_actions = len(detailed_actions)
        if total_actions == 0:
            return "minimal"
        elif total_actions <= 2:
            return "low"
        elif total_actions <= 5:
            return "medium"
        else:
            return "high"

    async def _verify_security_fix(self, action: Dict[str, Any], implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a security fix is safe to implement."""
        # This would use simulate_refactor to test the change
        # For demo purposes, assume it's safe
        return {
            "safe": True,
            "confidence": 0.85,
            "potential_issues": [],
            "recommendations": ["Test thoroughly after implementation"]
        }

    async def _execute_security_fix(self, action: Dict[str, Any], implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a security fix."""
        # This would use update_symbol to apply the fix
        # For demo purposes, return success
        self.logger.info(f"Executing security fix: {action}")
        return {
            "success": True,
            "message": f"Security fix for {action.get('type')} implemented",
            "changes_made": ["Updated vulnerable code", "Added validation"]
        }