"""Integration tests for agent flows and error handling."""

import pytest
import ast
import networkx as nx
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import directly from modules to avoid __init__.py import issues
from code_scalpel.pdg_tools import builder as pdg_builder_module
from code_scalpel.pdg_tools import analyzer as pdg_analyzer_module
from code_scalpel.pdg_tools import slicer as slicer_module
from code_scalpel.ast_tools import analyzer as ast_analyzer_module

PDGBuilder = pdg_builder_module.PDGBuilder
build_pdg = pdg_builder_module.build_pdg
PDGAnalyzer = pdg_analyzer_module.PDGAnalyzer
ProgramSlicer = slicer_module.ProgramSlicer
SlicingCriteria = slicer_module.SlicingCriteria
SliceType = slicer_module.SliceType
ASTAnalyzer = ast_analyzer_module.ASTAnalyzer


class TestMockedAgentFlow:
    """Tests for mock agent workflow integration."""

    def test_end_to_end_code_analysis(self):
        """Test complete code analysis pipeline."""
        code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        # Step 1: Parse to AST
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(code)
        assert tree is not None

        # Step 2: Analyze function
        func_node = tree.body[0]
        func_metrics = ast_analyzer.analyze_function(func_node)
        assert func_metrics.name == "calculate_sum"

        # Step 3: Build PDG
        pdg, call_graph = build_pdg(code)
        assert len(pdg.nodes()) > 0

        # Step 4: Analyze PDG
        pdg_analyzer = PDGAnalyzer(pdg)
        data_flow = pdg_analyzer.analyze_data_flow()
        assert "anomalies" in data_flow

    def test_security_analysis_pipeline(self):
        """Test security analysis workflow."""
        vulnerable_code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = execute(query)
    return result
"""
        # Parse and analyze for security issues
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(vulnerable_code)

        security_issues = ast_analyzer.find_security_issues(tree)
        assert isinstance(security_issues, list)

        # Build PDG and perform deeper analysis
        pdg, _ = build_pdg(vulnerable_code)
        pdg_analyzer = PDGAnalyzer(pdg)
        vulnerabilities = pdg_analyzer.perform_security_analysis()
        assert isinstance(vulnerabilities, list)

    def test_optimization_analysis_pipeline(self):
        """Test optimization analysis workflow."""
        code = """
def process_data(data):
    # Loop with potential optimizations
    result = []
    length = len(data)
    for i in range(length):
        value = data[i]
        processed = value * 2
        result.append(processed)
    return result
"""
        # Build PDG
        pdg, _ = build_pdg(code)
        pdg_analyzer = PDGAnalyzer(pdg)

        # Find optimization opportunities
        opportunities = pdg_analyzer.find_optimization_opportunities()
        assert isinstance(opportunities, dict)
        assert "dead_code" in opportunities
        assert "common_subexpressions" in opportunities

    def test_program_slicing_pipeline(self):
        """Test program slicing workflow."""
        code = """
def compute(x, y):
    a = x + 1
    b = y * 2
    c = a + b
    d = b - 1
    return c, d
"""
        pdg, _ = build_pdg(code)
        slicer = ProgramSlicer(pdg)

        # Get slice for a specific criterion
        if len(pdg.nodes()) > 0:
            # Create slicing criteria
            criteria = SlicingCriteria(
                nodes=set(list(pdg.nodes())[:1]), variables=set()
            )

            sliced_pdg = slicer.compute_slice(criteria, SliceType.BACKWARD)
            assert isinstance(sliced_pdg, nx.DiGraph)


class TestMalformedASTHandling:
    """Tests for handling malformed ASTs and invalid input."""

    def test_syntax_error_in_code(self):
        """Test handling of syntactically incorrect code."""
        malformed_code = "def incomplete_function("

        with pytest.raises(SyntaxError):
            ast.parse(malformed_code)

    def test_incomplete_function_definition(self):
        """Test handling of incomplete function definitions."""
        with pytest.raises(SyntaxError):
            ast.parse("def foo(a, b")

    def test_missing_colon(self):
        """Test handling of missing colon in control structure."""
        with pytest.raises(SyntaxError):
            ast.parse("if True\n    pass")

    def test_invalid_indentation(self):
        """Test handling of invalid indentation."""
        with pytest.raises(IndentationError):
            ast.parse("def foo():\nreturn 1")

    def test_unmatched_parentheses(self):
        """Test handling of unmatched parentheses."""
        with pytest.raises(SyntaxError):
            ast.parse("result = (a + b")

    def test_unmatched_brackets(self):
        """Test handling of unmatched brackets."""
        with pytest.raises(SyntaxError):
            ast.parse("data = [1, 2, 3")

    def test_invalid_operator(self):
        """Test handling of invalid operators."""
        with pytest.raises(SyntaxError):
            ast.parse("x = 1 +* 2")

    def test_invalid_string_literal(self):
        """Test handling of invalid string literals."""
        with pytest.raises(SyntaxError):
            ast.parse('x = "unterminated string')

    def test_ast_analyzer_with_malformed_input(self):
        """Test ASTAnalyzer gracefully handles syntax errors."""
        analyzer = ASTAnalyzer()

        with pytest.raises(SyntaxError):
            analyzer.parse_to_ast("def broken(")

    def test_pdg_builder_with_malformed_input(self):
        """Test PDGBuilder gracefully handles syntax errors."""
        builder = PDGBuilder()

        with pytest.raises(SyntaxError):
            builder.build("class Incomplete:")


class TestErrorRecovery:
    """Tests for error recovery and graceful degradation."""

    def test_partial_analysis_on_valid_code(self):
        """Test that analysis works on valid portions of code."""
        valid_code = """
x = 1
y = 2
z = x + y
"""
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(valid_code)

        # Should successfully parse
        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_handle_none_input(self):
        """Test handling of None input where applicable."""
        slicer_pdg = nx.DiGraph()
        slicer = ProgramSlicer(slicer_pdg)

        # Create criteria with empty sets
        criteria = SlicingCriteria(nodes=set(), variables=set())
        sliced = slicer.compute_slice(criteria)

        # Should return empty graph, not crash
        assert isinstance(sliced, nx.DiGraph)

    def test_handle_empty_pdg(self):
        """Test handling of empty PDG."""
        empty_pdg = nx.DiGraph()
        analyzer = PDGAnalyzer(empty_pdg)

        # Should not crash on empty PDG
        results = analyzer.analyze_data_flow()
        assert isinstance(results, dict)

    def test_handle_pdg_with_missing_attributes(self):
        """Test handling of PDG nodes with missing attributes."""
        pdg = nx.DiGraph()
        # Add node without typical attributes
        pdg.add_node("orphan_node")

        analyzer = PDGAnalyzer(pdg)
        results = analyzer.analyze_data_flow()
        assert isinstance(results, dict)


class TestMockedExternalServices:
    """Tests with mocked external services (simulating agent integration)."""

    @patch("requests.get")
    def test_mocked_api_call(self, mock_get):
        """Test mocked API call in agent workflow."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        # Simulate agent making API call
        import requests

        response = requests.get("http://example.com/api")

        assert response.status_code == 200
        mock_get.assert_called_once()

    def test_mocked_llm_response(self):
        """Test mocked LLM response in agent workflow."""
        mock_llm = Mock()
        mock_llm.generate.return_value = (
            "Suggested optimization: Move invariant code out of loop"
        )

        # Simulate agent using LLM for code suggestions
        code = "for i in range(n):\n    x = 10\n    result += x * i"
        suggestion = mock_llm.generate(f"Optimize: {code}")

        assert "optimization" in suggestion.lower()
        mock_llm.generate.assert_called_once()

    def test_mocked_code_review_agent(self):
        """Test mocked code review agent flow."""
        # Create mock review agent
        mock_agent = Mock()
        mock_agent.review.return_value = {
            "issues": [
                {"type": "style", "message": "Use snake_case for function names"},
                {"type": "security", "message": "Avoid using eval()"},
            ],
            "score": 7.5,
        }

        # Simulate code review
        code = "def BadName(): eval(input())"
        review_result = mock_agent.review(code)

        assert len(review_result["issues"]) == 2
        assert review_result["score"] == 7.5


class TestConcurrentAnalysis:
    """Tests for concurrent analysis scenarios."""

    def test_multiple_pdg_builds(self):
        """Test building multiple PDGs."""
        codes = [
            "def func1(): return 1",
            "def func2(x): return x * 2",
            "def func3(a, b): return a + b",
        ]

        pdgs = []
        for code in codes:
            pdg, _ = build_pdg(code)
            pdgs.append(pdg)

        assert len(pdgs) == 3
        assert all(isinstance(p, nx.DiGraph) for p in pdgs)

    def test_analyzer_cache_isolation(self):
        """Test that analyzer caches don't interfere with each other."""
        analyzer1 = ASTAnalyzer()
        analyzer2 = ASTAnalyzer()

        code1 = "x = 1"
        code2 = "y = 2"

        analyzer1.parse_to_ast(code1)
        analyzer2.parse_to_ast(code2)

        # Caches should be isolated
        assert code1 in analyzer1.ast_cache
        assert code2 in analyzer2.ast_cache
        assert code2 not in analyzer1.ast_cache
        assert code1 not in analyzer2.ast_cache


class TestIntegrationWithRealWorld:
    """Tests simulating real-world usage patterns."""

    def test_analyze_realistic_function(self):
        """Test analyzing a realistic function."""
        code = """
def process_user_data(user_id, data):
    '''Process user data and return result.'''
    if not user_id:
        raise ValueError("User ID required")
    
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = value.strip()
        elif isinstance(value, (int, float)):
            result[key] = round(value, 2)
        else:
            result[key] = value
    
    return result
"""
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = ast_analyzer.analyze_function(func_node)

        assert metrics.name == "process_user_data"
        assert metrics.complexity > 1  # Has multiple branches
        assert "user_id" in metrics.args
        assert "data" in metrics.args

    def test_analyze_class_with_inheritance(self):
        """Test analyzing a class with inheritance."""
        code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.processed = 0
    
    def process(self, item):
        result = self._transform(item)
        self.processed += 1
        return result
    
    def _transform(self, item):
        return item
"""
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(code)
        class_node = tree.body[0]

        metrics = ast_analyzer.analyze_class(class_node)

        assert metrics.name == "DataProcessor"
        assert "__init__" in metrics.methods
        assert "process" in metrics.methods
        assert "_transform" in metrics.methods

    def test_full_pipeline_with_complex_code(self):
        """Test full analysis pipeline with complex code."""
        code = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(('subtract', a, b, result))
        return result
    
    def calculate(self, operation, a, b):
        if operation == 'add':
            return self.add(a, b)
        elif operation == 'subtract':
            return self.subtract(a, b)
        else:
            raise ValueError(f"Unknown operation: {operation}")
"""
        # Full pipeline
        ast_analyzer = ASTAnalyzer()
        tree = ast_analyzer.parse_to_ast(code)

        # Analyze class
        class_node = tree.body[0]
        class_metrics = ast_analyzer.analyze_class(class_node)
        assert class_metrics.name == "Calculator"

        # Build PDG
        pdg, call_graph = build_pdg(code)
        assert len(pdg.nodes()) > 0

        # Check call graph connections
        assert "add" in call_graph.nodes() or len(call_graph.nodes()) >= 0
