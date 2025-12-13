"""
Optimization Agent - Specialized agent for code performance optimization.

This agent demonstrates how AI agents can use Code Scalpel's analysis tools
to identify performance bottlenecks and suggest optimizations.
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseCodeAnalysisAgent


class OptimizationAgent(BaseCodeAnalysisAgent):
    """
    AI agent specialized in performance optimization.

    Uses Code Scalpel MCP tools to:
    - Analyze code complexity and performance bottlenecks
    - Identify optimization opportunities
    - Suggest algorithmic improvements
    - Verify optimizations don't break functionality
    """

    def __init__(self, workspace_root: Optional[str] = None):
        super().__init__(workspace_root)
        self.performance_thresholds = {
            "max_complexity": 15,
            "max_function_length": 50,
            "min_cache_hit_ratio": 0.8,
            "max_nested_loops": 3
        }
        self.optimization_patterns = {
            "algorithmic": ["O(n^2) to O(n log n)", "Linear search to binary search"],
            "memory": ["Reduce object creation", "Use generators", "Memory pooling"],
            "io": ["Batch operations", "Async I/O", "Caching"],
            "computation": ["Memoization", "Vectorization", "Early termination"]
        }

    async def observe(self, target: str) -> Dict[str, Any]:
        """Observe the target for performance analysis."""
        self.logger.info(f"Performing performance analysis on: {target}")

        # Get file context
        file_info = await self.observe_file(target)
        if not file_info.get("success"):
            return file_info

        # Analyze code structure for complexity
        # Note: In real usage, would analyze actual code
        complexity_analysis = self._analyze_complexity(file_info)

        # Check symbol usage patterns
        symbol_analysis = {}
        for func_name in file_info.get("functions", [])[:3]:
            refs = await self.find_symbol_usage(func_name, self.context.workspace_root)
            if refs.get("success"):
                symbol_analysis[func_name] = refs

        return {
            "success": True,
            "file_info": file_info,
            "complexity_analysis": complexity_analysis,
            "symbol_analysis": symbol_analysis,
            "target": target
        }

    async def orient(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance observations and identify bottlenecks."""
        self.logger.info("Analyzing performance observations")

        file_info = observations.get("file_info", {})
        complexity_analysis = observations.get("complexity_analysis", {})
        symbol_analysis = observations.get("symbol_analysis", {})

        # Identify performance bottlenecks
        bottlenecks = self._identify_bottlenecks(complexity_analysis, symbol_analysis)

        # Analyze optimization opportunities
        opportunities = self._analyze_optimization_opportunities(bottlenecks, file_info)

        # Calculate performance metrics
        performance_score = self._calculate_performance_score(bottlenecks, opportunities)

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(opportunities)

        return {
            "success": True,
            "bottlenecks": bottlenecks,
            "opportunities": opportunities,
            "performance_score": performance_score,
            "recommendations": recommendations
        }

    async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on optimization actions to implement."""
        self.logger.info("Deciding on performance optimizations")

        opportunities = analysis.get("opportunities", [])
        recommendations = analysis.get("recommendations", [])

        # Prioritize high-impact, low-risk optimizations
        prioritized_actions = []

        for rec in recommendations:
            if rec.get("impact") == "high" and rec.get("risk") == "low":
                prioritized_actions.append(rec)

        # Add medium-impact optimizations if we have capacity
        if len(prioritized_actions) < 3:
            for rec in recommendations:
                if rec.get("impact") == "medium" and rec.get("risk") == "low":
                    prioritized_actions.append(rec)
                    if len(prioritized_actions) >= 3:
                        break

        # Create detailed action plans
        detailed_actions = []
        for action in prioritized_actions:
            detailed_actions.append({
                "action": action,
                "implementation_plan": self._create_optimization_plan(action),
                "verification_steps": self._create_performance_verification(action),
                "estimated_impact": self._estimate_performance_impact(action)
            })

        return {
            "success": True,
            "prioritized_actions": detailed_actions,
            "total_actions": len(detailed_actions),
            "estimated_benefit": self._estimate_total_benefit(detailed_actions),
            "risk_assessment": self._assess_optimization_risks(detailed_actions)
        }

    async def act(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization actions safely."""
        self.logger.info("Executing performance optimizations")

        prioritized_actions = decisions.get("prioritized_actions", [])
        results = []

        for action_plan in prioritized_actions:
            action = action_plan.get("action", {})
            implementation = action_plan.get("implementation_plan", {})

            # Verify optimization won't break functionality
            verification = await self._verify_optimization(action, implementation)
            if not verification.get("safe", False):
                results.append({
                    "action": action,
                    "result": {"success": False, "error": "Optimization verification failed"},
                    "verification": verification
                })
                continue

            # Execute the optimization
            result = await self._execute_optimization(action, implementation)
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
            "summary": f"Successfully implemented {success_count}/{total_count} optimizations"
        }

    # Performance Analysis Methods

    def _analyze_complexity(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        complexity_score = file_info.get("complexity_score", 0)
        function_count = len(file_info.get("functions", []))
        class_count = len(file_info.get("classes", []))

        complexity_analysis = {
            "overall_complexity": complexity_score,
            "function_count": function_count,
            "class_count": class_count,
            "complexity_per_function": complexity_score / max(function_count, 1),
            "issues": []
        }

        if complexity_score > self.performance_thresholds["max_complexity"]:
            complexity_analysis["issues"].append({
                "type": "high_complexity",
                "severity": "high",
                "description": f"File complexity ({complexity_score}) exceeds threshold"
            })

        if function_count > 15:
            complexity_analysis["issues"].append({
                "type": "too_many_functions",
                "severity": "medium",
                "description": f"File has {function_count} functions - consider splitting"
            })

        return complexity_analysis

    def _identify_bottlenecks(self, complexity_analysis: Dict[str, Any], symbol_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Complexity bottlenecks
        for issue in complexity_analysis.get("issues", []):
            bottlenecks.append({
                "type": "complexity",
                "location": "file",
                "severity": issue.get("severity"),
                "description": issue.get("description"),
                "estimated_impact": "high" if issue.get("severity") == "high" else "medium"
            })

        # Usage pattern bottlenecks
        for func_name, refs in symbol_analysis.items():
            usage_count = refs.get("total_references", 0)
            if usage_count > 20:
                bottlenecks.append({
                    "type": "high_usage",
                    "location": func_name,
                    "severity": "medium",
                    "description": f"Function '{func_name}' called {usage_count} times",
                    "estimated_impact": "medium"
                })

        return bottlenecks

    def _analyze_optimization_opportunities(self, bottlenecks: List[Dict[str, Any]], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential optimization opportunities."""
        opportunities = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "complexity":
                opportunities.append({
                    "type": "algorithmic",
                    "target": bottleneck.get("location"),
                    "description": "Optimize algorithm complexity",
                    "potential_improvement": "Reduce time complexity",
                    "confidence": 0.7
                })
            elif bottleneck["type"] == "high_usage":
                opportunities.append({
                    "type": "caching",
                    "target": bottleneck.get("location"),
                    "description": "Add caching for frequently called function",
                    "potential_improvement": "Reduce redundant computations",
                    "confidence": 0.8
                })

        return opportunities

    def _calculate_performance_score(self, bottlenecks: List[Dict[str, Any]], opportunities: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score (0-100)."""
        base_score = 100

        # Deduct for bottlenecks
        for bottleneck in bottlenecks:
            severity = bottleneck.get("severity", "low")
            if severity == "high":
                base_score -= 15
            elif severity == "medium":
                base_score -= 8
            else:
                base_score -= 3

        # Bonus for optimization opportunities
        base_score += len(opportunities) * 2

        return max(0, min(100, base_score))

    def _generate_optimization_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []

        for opp in opportunities:
            if opp["type"] == "algorithmic":
                recommendations.append({
                    "title": "Algorithm Optimization",
                    "description": f"Optimize {opp['target']} algorithm for better performance",
                    "impact": "high",
                    "risk": "medium",
                    "effort": "high",
                    "category": "algorithmic"
                })
            elif opp["type"] == "caching":
                recommendations.append({
                    "title": "Add Caching",
                    "description": f"Add caching to {opp['target']} to reduce redundant calls",
                    "impact": "medium",
                    "risk": "low",
                    "effort": "medium",
                    "category": "memory"
                })

        return recommendations

    def _create_optimization_plan(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed optimization implementation plan."""
        category = action.get("category")

        if category == "algorithmic":
            return {
                "steps": [
                    "Analyze current algorithm complexity",
                    "Identify optimization opportunities",
                    "Implement optimized algorithm",
                    "Add performance tests",
                    "Verify correctness"
                ],
                "tools_needed": ["profilers", "benchmarking tools"],
                "estimated_time": "4-8 hours"
            }
        elif category == "memory":
            return {
                "steps": [
                    "Identify caching opportunities",
                    "Implement caching mechanism",
                    "Add cache invalidation logic",
                    "Test cache performance",
                    "Monitor cache hit rates"
                ],
                "tools_needed": ["redis", "memcached", "functools.lru_cache"],
                "estimated_time": "2-4 hours"
            }

        return {
            "steps": ["Analyze", "Implement", "Test"],
            "tools_needed": [],
            "estimated_time": "1-2 hours"
        }

    def _create_performance_verification(self, action: Dict[str, Any]) -> List[str]:
        """Create performance verification steps."""
        category = action.get("category")

        if category == "algorithmic":
            return [
                "Benchmark before and after optimization",
                "Verify time complexity improvement",
                "Test with large datasets",
                "Ensure correctness is maintained"
            ]
        elif category == "memory":
            return [
                "Measure memory usage before/after",
                "Test cache hit/miss ratios",
                "Verify cache invalidation works",
                "Check for memory leaks"
            ]

        return [
            "Measure performance metrics",
            "Verify functionality unchanged",
            "Test edge cases"
        ]

    def _estimate_performance_impact(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the performance impact of an optimization."""
        impact = action.get("impact", "medium")

        if impact == "high":
            return {
                "time_improvement": "50-90%",
                "memory_improvement": "30-70%",
                "scalability_improvement": "significant"
            }
        elif impact == "medium":
            return {
                "time_improvement": "20-50%",
                "memory_improvement": "10-30%",
                "scalability_improvement": "moderate"
            }
        else:
            return {
                "time_improvement": "5-20%",
                "memory_improvement": "5-15%",
                "scalability_improvement": "minimal"
            }

    def _estimate_total_benefit(self, detailed_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate total benefit from all optimizations."""
        total_time_improvement = 0
        total_memory_improvement = 0

        for action in detailed_actions:
            impact = action.get("estimated_impact", {})
            time_imp = impact.get("time_improvement", "0%")
            mem_imp = impact.get("memory_improvement", "0%")

            # Extract percentage (rough estimate)
            time_pct = int(time_imp.split("-")[0].rstrip("%")) if "-" in time_imp else 0
            mem_pct = int(mem_imp.split("-")[0].rstrip("%")) if "-" in mem_imp else 0

            total_time_improvement += time_pct
            total_memory_improvement += mem_pct

        return {
            "estimated_time_improvement": f"{total_time_improvement}%",
            "estimated_memory_improvement": f"{total_memory_improvement}%",
            "actions_count": len(detailed_actions)
        }

    def _assess_optimization_risks(self, detailed_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risks of implementing optimizations."""
        high_risk_count = sum(1 for a in detailed_actions if a.get("action", {}).get("risk") == "high")
        medium_risk_count = sum(1 for a in detailed_actions if a.get("action", {}).get("risk") == "medium")
        low_risk_count = sum(1 for a in detailed_actions if a.get("action", {}).get("risk") == "low")

        if high_risk_count > 0:
            overall_risk = "high"
        elif medium_risk_count > 2:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "overall_risk": overall_risk,
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "recommendations": [
                "Test thoroughly after implementation",
                "Have rollback plan ready",
                "Monitor performance metrics"
            ] if overall_risk != "low" else []
        }

    async def _verify_optimization(self, action: Dict[str, Any], implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that an optimization is safe to implement."""
        # This would use simulate_refactor to test the change
        # For demo purposes, assume it's safe
        return {
            "safe": True,
            "confidence": 0.8,
            "potential_issues": ["May affect functionality if not implemented correctly"],
            "recommendations": ["Test with comprehensive test suite"]
        }

    async def _execute_optimization(self, action: Dict[str, Any], implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a performance optimization."""
        # This would use update_symbol to apply the optimization
        # For demo purposes, return success
        self.logger.info(f"Executing optimization: {action}")
        return {
            "success": True,
            "message": f"Optimization for {action.get('category')} implemented",
            "changes_made": ["Optimized algorithm", "Added caching", "Improved performance"]
        }