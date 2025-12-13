"""
Tests for Code Scalpel Agents.

These tests verify that the agent framework works correctly and demonstrate
how AI agents can use Code Scalpel's MCP tools.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from code_scalpel.agents.base_agent import BaseCodeAnalysisAgent, AgentContext
from code_scalpel.agents.code_review_agent import CodeReviewAgent
from code_scalpel.agents.security_agent import SecurityAgent
from code_scalpel.agents.optimazation_agent import OptimizationAgent


class TestBaseCodeAnalysisAgent:
    """Test the base agent framework."""

    class TestAgent(BaseCodeAnalysisAgent):
        """Concrete test agent implementation."""
        
        async def observe(self, target: str) -> Dict[str, Any]:
            return {"success": True, "data": "test"}
        
        async def orient(self, observations: Dict[str, Any]) -> Dict[str, Any]:
            return {"success": True, "analysis": "test"}
        
        async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
            return {"success": True, "decisions": "test"}
        
        async def act(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
            return {"success": True, "actions": "test"}

    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return self.TestAgent(workspace_root="/test/workspace")

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.context.workspace_root == "/test/workspace"
        assert isinstance(agent.context, AgentContext)
        assert len(agent.context.recent_operations) == 0

    def test_context_summary(self, agent):
        """Test context summary generation."""
        summary = agent.get_context_summary()
        assert summary["workspace_root"] == "/test/workspace"
        assert summary["recent_operations_count"] == 0
        assert summary["knowledge_base_keys"] == []

    @pytest.mark.asyncio
    async def test_observe_file_success(self, agent):
        """Test successful file observation."""
        # Mock the get_file_context function
        import code_scalpel.agents.base_agent as base_module
        original_get_file_context = base_module.get_file_context
        base_module.get_file_context = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "file_path": "/test/file.py",
                "complexity_score": 5
            })
        ))

        try:
            result = await agent.observe_file("/test/file.py")
            assert result["success"] is True
            assert result["file_path"] == "/test/file.py"
            assert len(agent.context.recent_operations) == 1
        finally:
            base_module.get_file_context = original_get_file_context

    @pytest.mark.asyncio
    async def test_observe_file_failure(self, agent):
        """Test file observation failure."""
        import code_scalpel.agents.base_agent as base_module
        original_get_file_context = base_module.get_file_context
        base_module.get_file_context = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": False,
                "error": "File not found"
            })
        ))

        try:
            result = await agent.observe_file("/nonexistent/file.py")
            assert result["success"] is False
            assert "error" in result
        finally:
            base_module.get_file_context = original_get_file_context

    @pytest.mark.asyncio
    async def test_find_symbol_usage_success(self, agent):
        """Test successful symbol reference finding."""
        import code_scalpel.agents.base_agent as base_module
        original_get_symbol_references = base_module.get_symbol_references
        base_module.get_symbol_references = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "symbol_name": "test_function",
                "total_references": 3
            })
        ))

        try:
            result = await agent.find_symbol_usage("test_function")
            assert result["success"] is True
            assert result["symbol_name"] == "test_function"
        finally:
            base_module.get_symbol_references = original_get_symbol_references

    @pytest.mark.asyncio
    async def test_analyze_security_success(self, agent):
        """Test successful security analysis."""
        import code_scalpel.agents.base_agent as base_module
        original_security_scan = base_module.security_scan
        base_module.security_scan = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "vulnerabilities": []
            })
        ))

        try:
            result = await agent.analyze_code_security("test code")
            assert result["success"] is True
            assert "vulnerabilities" in result
        finally:
            base_module.security_scan = original_security_scan

    @pytest.mark.asyncio
    async def test_extract_function_success(self, agent):
        """Test successful function extraction."""
        import code_scalpel.agents.base_agent as base_module
        original_extract_code = base_module.extract_code
        base_module.extract_code = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "code": "def test():\n    pass"
            })
        ))

        try:
            result = await agent.extract_function("/test/file.py", "test_function")
            assert result["success"] is True
            assert "code" in result
        finally:
            base_module.extract_code = original_extract_code

    @pytest.mark.asyncio
    async def test_simulate_change_success(self, agent):
        """Test successful change simulation."""
        import code_scalpel.agents.base_agent as base_module
        original_simulate_refactor = base_module.simulate_refactor
        base_module.simulate_refactor = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "safe": True
            })
        ))

        try:
            result = await agent.simulate_code_change("old code", "new code")
            assert result["success"] is True
            assert result["safe"] is True
        finally:
            base_module.simulate_refactor = original_simulate_refactor

    @pytest.mark.asyncio
    async def test_apply_safe_change_success(self, agent):
        """Test successful safe change application."""
        import code_scalpel.agents.base_agent as base_module
        original_update_symbol = base_module.update_symbol
        base_module.update_symbol = AsyncMock(return_value=MagicMock(
            model_dump=MagicMock(return_value={
                "success": True,
                "message": "Change applied"
            })
        ))

        try:
            result = await agent.apply_safe_change("/test/file.py", "function", "test_func", "new code")
            assert result["success"] is True
            assert "message" in result
        finally:
            base_module.update_symbol = original_update_symbol


class TestCodeReviewAgent:
    """Test the code review agent."""

    @pytest.fixture
    def agent(self):
        """Create a test code review agent."""
        return CodeReviewAgent(workspace_root="/test/workspace")

    def test_initialization(self, agent):
        """Test code review agent initialization."""
        assert isinstance(agent, BaseCodeAnalysisAgent)
        assert agent.quality_thresholds["max_complexity_score"] == 10
        assert agent.quality_thresholds["max_function_length"] == 30

    @pytest.mark.asyncio
    async def test_observe_success(self, agent):
        """Test successful observation phase."""
        # Mock dependencies
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "functions": ["func1", "func2"],
            "complexity_score": 8
        })
        agent.analyze_code_security = AsyncMock(return_value={"vulnerabilities": []})
        agent.find_symbol_usage = AsyncMock(return_value={"success": True, "total_references": 2})

        result = await agent.observe("/test/file.py")

        assert result["success"] is True
        assert "file_info" in result
        assert "security_scan" in result
        assert "symbol_analysis" in result

    def test_analyze_file_structure(self, agent):
        """Test file structure analysis."""
        file_info = {
            "complexity_score": 15,
            "functions": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]
        }

        issues = agent._analyze_file_structure(file_info)

        assert len(issues) >= 1  # Should find complexity issue
        complexity_issue = next((i for i in issues if i["type"] == "complexity"), None)
        assert complexity_issue is not None
        assert complexity_issue["severity"] == "medium"

    def test_analyze_security_issues(self, agent):
        """Test security analysis."""
        security_info = {
            "vulnerabilities": [
                {"type": "sql_injection", "severity": "high", "description": "SQL injection detected"}
            ]
        }

        issues = agent._analyze_security(security_info)

        assert len(issues) == 1
        assert issues[0]["type"] == "security"
        assert issues[0]["severity"] == "high"

    def test_calculate_quality_score(self, agent):
        """Test quality score calculation."""
        issues = [
            {"severity": "high", "type": "security"},
            {"severity": "medium", "type": "complexity"}
        ]
        file_info = {"has_security_issues": False}

        score = agent._calculate_quality_score(issues, file_info)

        # Should be less than 100 due to issues
        assert score < 100
        assert score >= 0

    def test_categorize_issues(self, agent):
        """Test issue categorization."""
        issues = [
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"},
            {"severity": "high"}
        ]

        breakdown = agent._categorize_issues(issues)

        assert breakdown["high"] == 2
        assert breakdown["medium"] == 1
        assert breakdown["low"] == 1

    @pytest.mark.asyncio
    async def test_orient_success(self, agent):
        """Test successful orientation phase."""
        observations = {
            "file_info": {"complexity_score": 12, "functions": ["f1"]},
            "security_scan": {"vulnerabilities": []},
            "symbol_analysis": {}
        }

        result = await agent.orient(observations)

        assert result["success"] is True
        assert "issues" in result
        assert "suggestions" in result
        assert "quality_score" in result

    @pytest.mark.asyncio
    async def test_decide_success(self, agent):
        """Test successful decision phase."""
        analysis = {
            "issues": [
                {"type": "security", "severity": "high", "actionable": True, "confidence": 0.9}
            ],
            "suggestions": ["Fix security issues"]
        }

        result = await agent.decide(analysis)

        assert result["success"] is True
        assert "actionable_items" in result
        assert "total_actions" in result

    @pytest.mark.asyncio
    async def test_act_success(self, agent):
        """Test successful action phase."""
        decisions = {
            "actionable_items": [
                {
                    "action": {"type": "refactor_function"},
                    "implementation_plan": {},
                    "verification_steps": [],
                    "estimated_impact": {}
                }
            ]
        }

        # Mock the execution methods
        agent._execute_function_refactor = AsyncMock(return_value={"success": True})

        result = await agent.act(decisions)

        assert result["success"] is True
        assert "results" in result
        assert result["success_rate"] == 1.0


class TestSecurityAgent:
    """Test the security agent."""

    @pytest.fixture
    def agent(self):
        """Create a test security agent."""
        return SecurityAgent(workspace_root="/test/workspace")

    def test_initialization(self, agent):
        """Test security agent initialization."""
        assert isinstance(agent, BaseCodeAnalysisAgent)
        assert "sql_injection" in agent.vulnerability_patterns
        assert "xxe" in agent.vulnerability_patterns
        assert agent.risk_levels["critical"] == 9

    def test_categorize_vulnerabilities(self, agent):
        """Test vulnerability categorization."""
        vulnerabilities = [
            {"type": "sql_injection", "severity": "high"},
            {"type": "xxe", "severity": "medium"},
            {"type": "sql_injection", "severity": "low"}
        ]

        categorized = agent._categorize_vulnerabilities(vulnerabilities)

        assert "sql_injection" in categorized
        assert len(categorized["sql_injection"]) == 2
        assert "xxe" in categorized
        assert len(categorized["xxe"]) == 1

    def test_assess_overall_risk(self, agent):
        """Test overall risk assessment."""
        categorized_vulns = {
            "sql_injection": [{"severity": "critical"}, {"severity": "high"}],
            "xxe": [{"severity": "medium"}]
        }
        attack_vectors = []

        assessment = agent._assess_overall_risk(categorized_vulns, attack_vectors)

        assert assessment["overall_level"] == "critical"
        assert assessment["critical_count"] == 1
        assert assessment["high_count"] == 1
        assert assessment["risk_score"] >= 10

    def test_determine_priority(self, agent):
        """Test remediation priority determination."""
        vulns = [{"severity": "critical"}, {"severity": "high"}]
        priority = agent._determine_priority(vulns)
        assert priority == "critical"

        vulns = [{"severity": "medium"}]
        priority = agent._determine_priority(vulns)
        assert priority == "medium"

    def test_create_remediation_actions(self, agent):
        """Test remediation action creation."""
        vulns = [{"severity": "high"}]
        actions = agent._create_remediation_actions("sql_injection", vulns)

        assert len(actions) > 0
        assert actions[0]["type"] == "input_validation"

    @pytest.mark.asyncio
    async def test_observe_success(self, agent):
        """Test successful security observation."""
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "functions": ["func1"]
        })
        agent.analyze_code_security = AsyncMock(return_value={"vulnerabilities": []})
        agent.find_symbol_usage = AsyncMock(return_value={"success": True})

        result = await agent.observe("/test/file.py")

        assert result["success"] is True
        assert "file_info" in result
        assert "security_scan" in result

    @pytest.mark.asyncio
    async def test_orient_success(self, agent):
        """Test successful security orientation."""
        observations = {
            "security_scan": {"vulnerabilities": []},
            "symbol_analysis": {}
        }

        result = await agent.orient(observations)

        assert result["success"] is True
        assert "vulnerabilities" in result
        assert "risk_assessment" in result

    @pytest.mark.asyncio
    async def test_decide_success(self, agent):
        """Test successful security decision."""
        analysis = {
            "remediation_plan": {
                "sql_injection": {
                    "priority": "high",
                    "actions": [{"type": "input_validation"}]
                }
            }
        }

        result = await agent.decide(analysis)

        assert result["success"] is True
        assert "prioritized_actions" in result

    @pytest.mark.asyncio
    async def test_act_success(self, agent):
        """Test successful security action."""
        decisions = {
            "prioritized_actions": [
                {
                    "action": {"type": "input_validation"},
                    "implementation_plan": {},
                    "verification_steps": [],
                    "estimated_impact": {}
                }
            ]
        }

        agent._verify_security_fix = AsyncMock(return_value={"safe": True})
        agent._execute_security_fix = AsyncMock(return_value={"success": True})

        result = await agent.act(decisions)

        assert result["success"] is True
        assert result["success_rate"] == 1.0


class TestOptimizationAgent:
    """Test the optimization agent."""

    @pytest.fixture
    def agent(self):
        """Create a test optimization agent."""
        return OptimizationAgent(workspace_root="/test/workspace")

    def test_initialization(self, agent):
        """Test optimization agent initialization."""
        assert isinstance(agent, BaseCodeAnalysisAgent)
        assert agent.performance_thresholds["max_complexity"] == 15
        assert "algorithmic" in agent.optimization_patterns

    def test_analyze_complexity(self, agent):
        """Test complexity analysis."""
        file_info = {
            "complexity_score": 20,
            "functions": ["f1", "f2", "f3", "f4", "f5", "f6"]
        }

        analysis = agent._analyze_complexity(file_info)

        assert analysis["overall_complexity"] == 20
        assert analysis["function_count"] == 6
        assert len(analysis["issues"]) > 0  # Should find high complexity issue

    def test_identify_bottlenecks(self, agent):
        """Test bottleneck identification."""
        complexity_analysis = {
            "issues": [{"type": "high_complexity", "severity": "high", "description": "High complexity"}]
        }
        symbol_analysis = {
            "func1": {"total_references": 25}
        }

        bottlenecks = agent._identify_bottlenecks(complexity_analysis, symbol_analysis)

        assert len(bottlenecks) >= 2  # Complexity and high usage
        assert any(b["type"] == "complexity" for b in bottlenecks)
        assert any(b["type"] == "high_usage" for b in bottlenecks)

    def test_calculate_performance_score(self, agent):
        """Test performance score calculation."""
        bottlenecks = [
            {"severity": "high"},
            {"severity": "medium"}
        ]
        opportunities = [{"confidence": 0.8}]

        score = agent._calculate_performance_score(bottlenecks, opportunities)

        assert score < 100  # Should be reduced due to bottlenecks
        assert score >= 0

    def test_generate_optimization_recommendations(self, agent):
        """Test optimization recommendation generation."""
        opportunities = [
            {"type": "algorithmic", "target": "func1", "confidence": 0.7},
            {"type": "caching", "target": "func2", "confidence": 0.8}
        ]

        recommendations = agent._generate_optimization_recommendations(opportunities)

        assert len(recommendations) == 2
        assert recommendations[0]["title"] == "Algorithm Optimization"
        assert recommendations[1]["title"] == "Add Caching"

    @pytest.mark.asyncio
    async def test_observe_success(self, agent):
        """Test successful optimization observation."""
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "complexity_score": 10,
            "functions": ["func1"]
        })
        agent.find_symbol_usage = AsyncMock(return_value={"success": True})

        result = await agent.observe("/test/file.py")

        assert result["success"] is True
        assert "complexity_analysis" in result

    @pytest.mark.asyncio
    async def test_orient_success(self, agent):
        """Test successful optimization orientation."""
        observations = {
            "complexity_analysis": {"overall_complexity": 15, "issues": []},
            "symbol_analysis": {}
        }

        result = await agent.orient(observations)

        assert result["success"] is True
        assert "bottlenecks" in result
        assert "performance_score" in result

    @pytest.mark.asyncio
    async def test_decide_success(self, agent):
        """Test successful optimization decision."""
        analysis = {
            "opportunities": [{"type": "caching"}],
            "recommendations": [{
                "title": "Add Caching",
                "impact": "medium",
                "risk": "low",
                "category": "memory"
            }]
        }

        result = await agent.decide(analysis)

        assert result["success"] is True
        assert "prioritized_actions" in result

    @pytest.mark.asyncio
    async def test_act_success(self, agent):
        """Test successful optimization action."""
        decisions = {
            "prioritized_actions": [
                {
                    "action": {"category": "memory"},
                    "implementation_plan": {},
                    "verification_steps": [],
                    "estimated_impact": {}
                }
            ]
        }

        agent._verify_optimization = AsyncMock(return_value={"safe": True})
        agent._execute_optimization = AsyncMock(return_value={"success": True})

        result = await agent.act(decisions)

        assert result["success"] is True
        assert result["success_rate"] == 1.0


class TestAgentIntegration:
    """Test agent integration and OODA loop."""

    @pytest.mark.asyncio
    async def test_code_review_agent_ooda_loop(self):
        """Test complete OODA loop for code review agent."""
        agent = CodeReviewAgent()

        # Mock all the MCP tool calls
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "complexity_score": 5,
            "functions": ["test_func"],
            "classes": [],
            "imports": ["os"],
            "has_security_issues": False
        })
        agent.analyze_code_security = AsyncMock(return_value={
            "vulnerabilities": []
        })
        agent.find_symbol_usage = AsyncMock(return_value={
            "success": True,
            "total_references": 1
        })

        result = await agent.execute_ooda_loop("/test/file.py")

        assert result["success"] is True
        assert "phases" in result
        assert "observe" in result["phases"]
        assert "orient" in result["phases"]
        assert "decide" in result["phases"]
        assert "act" in result["phases"]

    @pytest.mark.asyncio
    async def test_security_agent_ooda_loop(self):
        """Test complete OODA loop for security agent."""
        agent = SecurityAgent()

        # Mock MCP tool calls
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "functions": ["vulnerable_func"]
        })
        agent.analyze_code_security = AsyncMock(return_value={
            "vulnerabilities": []
        })
        agent.find_symbol_usage = AsyncMock(return_value={
            "success": True,
            "total_references": 1
        })

        result = await agent.execute_ooda_loop("/test/file.py")

        assert result["success"] is True
        assert "phases" in result

    @pytest.mark.asyncio
    async def test_optimization_agent_ooda_loop(self):
        """Test complete OODA loop for optimization agent."""
        agent = OptimizationAgent()

        # Mock MCP tool calls
        agent.observe_file = AsyncMock(return_value={
            "success": True,
            "complexity_score": 8,
            "functions": ["slow_func"]
        })
        agent.find_symbol_usage = AsyncMock(return_value={
            "success": True,
            "total_references": 1
        })

        result = await agent.execute_ooda_loop("/test/file.py")

        assert result["success"] is True
        assert "phases" in result