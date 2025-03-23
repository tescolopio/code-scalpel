import ast
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import z3
from collections import defaultdict
import logging
from pathlib import Path
import json
import networkx as nx
import time

class IssueType(Enum):
    """Types of issues that can be detected."""
    DIVISION_BY_ZERO = 'division_by_zero'
    NULL_POINTER = 'null_pointer'
    BUFFER_OVERFLOW = 'buffer_overflow'
    INTEGER_OVERFLOW = 'integer_overflow'
    MEMORY_LEAK = 'memory_leak'
    RACE_CONDITION = 'race_condition'
    DEADLOCK = 'deadlock'
    ASSERTION_VIOLATION = 'assertion_violation'
    UNREACHABLE_CODE = 'unreachable_code'
    INFINITE_LOOP = 'infinite_loop'

class IssueSeverity(Enum):
    """Severity levels for issues."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

@dataclass
class Issue:
    """Represents an issue found during analysis."""
    type: IssueType
    severity: IssueSeverity
    location: str
    message: str
    path_conditions: List[Any]
    variable_values: Dict[str, Any]
    trace: List[str]
    remediation: Optional[str] = None

@dataclass
class TestCase:
    """Represents a generated test case."""
    inputs: Dict[str, Any]
    expected_output: Any
    path_conditions: List[Any]
    coverage_info: Dict[str, Set[str]]
    description: str

@dataclass
class AnalysisResult:
    """Complete results of analysis."""
    issues: List[Issue]
    test_cases: List[TestCase]
    coverage: float
    execution_time: float
    path_statistics: Dict[str, Any]
    memory_usage: Dict[str, Any]
    recommendations: List[str]

class ResultAnalyzer:
    """Advanced analyzer for symbolic execution results."""
    
    def __init__(self, engine):
        self.engine = engine
        self.analysis_history = []
        self.path_cache = {}
        self._setup_logging()

    def analyze(self, code: str = None) -> AnalysisResult:
        """
        Perform comprehensive analysis of execution results.
        
        Args:
            code: Optional source code for additional analysis
        
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        try:
            # Collect base information
            issues = self._find_all_issues()
            test_cases = self._generate_test_cases()
            coverage = self._calculate_coverage()
            
            # Perform advanced analysis
            path_stats = self._analyze_paths()
            memory_stats = self._analyze_memory_usage()
            recommendations = self._generate_recommendations()
            
            result = AnalysisResult(
                issues=issues,
                test_cases=test_cases,
                coverage=coverage,
                execution_time=time.time() - start_time,
                path_statistics=path_stats,
                memory_usage=memory_stats,
                recommendations=recommendations
            )
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            raise

    def _find_all_issues(self) -> List[Issue]:
        """Find all types of issues."""
        issues = []
        
        # Check for arithmetic issues
        issues.extend(self._find_arithmetic_issues())
        
        # Check for memory issues
        issues.extend(self._find_memory_issues())
        
        # Check for concurrency issues
        issues.extend(self._find_concurrency_issues())
        
        # Check for logical issues
        issues.extend(self._find_logical_issues())
        
        # Check for security issues
        issues.extend(self._find_security_issues())
        
        return sorted(issues, key=lambda x: x.severity.value, reverse=True)

    def _find_arithmetic_issues(self) -> List[Issue]:
        """Find arithmetic-related issues."""
        issues = []
        
        # Check division operations
        for node, state in self._get_arithmetic_operations():
            if self._could_be_zero(state.get_divisor()):
                issues.append(
                    Issue(
                        type=IssueType.DIVISION_BY_ZERO,
                        severity=IssueSeverity.ERROR,
                        location=self._get_location(node),
                        message="Possible division by zero",
                        path_conditions=state.path_conditions,
                        variable_values=state.variable_values,
                        trace=self._get_execution_trace(state),
                        remediation="Add check for zero divisor"
                    )
                )
        
        # Check integer operations
        for node, state in self._get_integer_operations():
            if self._could_overflow(state):
                issues.append(
                    Issue(
                        type=IssueType.INTEGER_OVERFLOW,
                        severity=IssueSeverity.WARNING,
                        location=self._get_location(node),
                        message="Possible integer overflow",
                        path_conditions=state.path_conditions,
                        variable_values=state.variable_values,
                        trace=self._get_execution_trace(state),
                        remediation="Use bounds checking or wider integer type"
                    )
                )
                
        return issues

    def _find_memory_issues(self) -> List[Issue]:
        """Find memory-related issues."""
        issues = []
        
        # Check null pointer dereferences
        for node, state in self._get_pointer_operations():
            if self._could_be_null(state):
                issues.append(
                    Issue(
                        type=IssueType.NULL_POINTER,
                        severity=IssueSeverity.ERROR,
                        location=self._get_location(node),
                        message="Possible null pointer dereference",
                        path_conditions=state.path_conditions,
                        variable_values=state.variable_values,
                        trace=self._get_execution_trace(state),
                        remediation="Add null check"
                    )
                )
        
        # Check array bounds
        for node, state in self._get_array_operations():
            if self._could_exceed_bounds(state):
                issues.append(
                    Issue(
                        type=IssueType.BUFFER_OVERFLOW,
                        severity=IssueSeverity.CRITICAL,
                        location=self._get_location(node),
                        message="Possible buffer overflow",
                        path_conditions=state.path_conditions,
                        variable_values=state.variable_values,
                        trace=self._get_execution_trace(state),
                        remediation="Add bounds checking"
                    )
                )
                
        return issues

    def _find_logical_issues(self) -> List[Issue]:
        """Find logical issues in the code."""
        issues = []
        
        # Check for unreachable code
        for node in self._find_unreachable_nodes():
            issues.append(
                Issue(
                    type=IssueType.UNREACHABLE_CODE,
                    severity=IssueSeverity.WARNING,
                    location=self._get_location(node),
                    message="Unreachable code detected",
                    path_conditions=[],
                    variable_values={},
                    trace=[],
                    remediation="Remove or fix conditions leading to this code"
                )
            )
        
        # Check for infinite loops
        for node, state in self._get_loop_conditions():
            if self._could_be_infinite(state):
                issues.append(
                    Issue(
                        type=IssueType.INFINITE_LOOP,
                        severity=IssueSeverity.ERROR,
                        location=self._get_location(node),
                        message="Possible infinite loop",
                        path_conditions=state.path_conditions,
                        variable_values=state.variable_values,
                        trace=self._get_execution_trace(state),
                        remediation="Add or fix loop termination condition"
                    )
                )
                
        return issues

    def _generate_test_cases(self) -> List[TestCase]:
        """Generate comprehensive test cases."""
        test_cases = []
        
        # Generate tests for each path
        for path_conditions in self._get_all_path_conditions():
            inputs = self._solve_path_conditions(path_conditions)
            if inputs:
                test_cases.append(
                    TestCase(
                        inputs=inputs,
                        expected_output=self._compute_expected_output(inputs),
                        path_conditions=path_conditions,
                        coverage_info=self._get_coverage_for_inputs(inputs),
                        description=self._generate_test_description(
                            inputs, path_conditions
                        )
                    )
                )
        
        # Add boundary test cases
        test_cases.extend(self._generate_boundary_tests())
        
        # Add error-triggering test cases
        test_cases.extend(self._generate_error_tests())
        
        return test_cases

    def _analyze_paths(self) -> Dict[str, Any]:
        """Analyze execution paths."""
        return {
            'total_paths': len(self._get_all_paths()),
            'feasible_paths': len(self._get_feasible_paths()),
            'infeasible_paths': len(self._get_infeasible_paths()),
            'path_lengths': self._get_path_length_stats(),
            'branch_coverage': self._calculate_branch_coverage(),
            'path_complexity': self._calculate_path_complexity()
        }

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        return {
            'peak_memory': self._get_peak_memory(),
            'average_memory': self._get_average_memory(),
            'memory_leaks': self._find_memory_leaks(),
            'allocation_sites': self._get_allocation_sites(),
            'deallocation_patterns': self._analyze_deallocations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization and improvement recommendations."""
        recommendations = []
        
        # Performance recommendations
        if self._has_performance_issues():
            recommendations.extend(self._get_performance_recommendations())
        
        # Security recommendations
        if self._has_security_issues():
            recommendations.extend(self._get_security_recommendations())
        
        # Code quality recommendations
        recommendations.extend(self._get_code_quality_recommendations())
        
        return recommendations

    def generate_report(self, format: str = 'json') -> Union[str, Dict]:
        """Generate analysis report in specified format."""
        result = self.analysis_history[-1] if self.analysis_history else None
        
        if not result:
            return "No analysis results available"
            
        if format == 'json':
            return self._generate_json_report(result)
        elif format == 'html':
            return self._generate_html_report(result)
        elif format == 'markdown':
            return self._generate_markdown_report(result)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def visualize_results(self, output_file: str):
        """Create visualization of analysis results."""
        graph = nx.DiGraph()
        
        # Add issues as nodes
        for issue in self.analysis_history[-1].issues:
            graph.add_node(
                issue.type.value,
                severity=issue.severity.value,
                count=1
            )
        
        # Add relationships
        self._add_issue_relationships(graph)
        
        # Generate visualization
        self._save_visualization(graph, output_file)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ResultAnalyzer')

def create_analyzer(engine) -> ResultAnalyzer:
    """Create a new result analyzer instance."""
    return ResultAnalyzer(engine)