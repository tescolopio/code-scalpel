"""Unit tests for PDG analyzer functionality."""

import os
import sys

import networkx as nx
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import directly from the module to avoid __init__.py import issues
from code_scalpel.pdg_tools import analyzer as pdg_analyzer_module

PDGAnalyzer = pdg_analyzer_module.PDGAnalyzer
DependencyType = pdg_analyzer_module.DependencyType
DataFlowAnomaly = pdg_analyzer_module.DataFlowAnomaly
SecurityVulnerability = pdg_analyzer_module.SecurityVulnerability


class TestDependencyType:
    """Tests for the DependencyType enum."""

    def test_dependency_type_values(self):
        """Test DependencyType enum values."""
        assert DependencyType.DATA.value == "data_dependency"
        assert DependencyType.CONTROL.value == "control_dependency"
        assert DependencyType.CALL.value == "call_dependency"
        assert DependencyType.PARAMETER.value == "parameter_dependency"
        assert DependencyType.RETURN.value == "return_dependency"


class TestDataFlowAnomaly:
    """Tests for the DataFlowAnomaly dataclass."""

    def test_create_anomaly(self):
        """Test creating a DataFlowAnomaly."""
        anomaly = DataFlowAnomaly(
            type="undefined",
            variable="x",
            location=(10, 5),
            severity="error",
            message="Variable 'x' used before definition",
        )
        assert anomaly.type == "undefined"
        assert anomaly.variable == "x"
        assert anomaly.location == (10, 5)
        assert anomaly.severity == "error"
        assert "Variable 'x'" in anomaly.message


class TestSecurityVulnerability:
    """Tests for the SecurityVulnerability dataclass."""

    def test_create_vulnerability(self):
        """Test creating a SecurityVulnerability."""
        vuln = SecurityVulnerability(
            type="sql_injection",
            source="input_node",
            sink="execute_node",
            path=["input_node", "process_node", "execute_node"],
            severity="high",
            description="SQL injection vulnerability",
        )
        assert vuln.type == "sql_injection"
        assert vuln.source == "input_node"
        assert vuln.sink == "execute_node"
        assert len(vuln.path) == 3
        assert vuln.severity == "high"


class TestPDGAnalyzer:
    """Tests for the PDGAnalyzer class."""

    @pytest.fixture
    def simple_pdg(self):
        """Create a simple PDG for testing."""
        g = nx.DiGraph()
        g.add_node("assign_x", type="assign", defines=["x"], uses=[])
        g.add_node("assign_y", type="assign", defines=["y"], uses=["x"])
        g.add_edge("assign_x", "assign_y", type="data_dependency")
        return g

    @pytest.fixture
    def pdg_with_control_flow(self):
        """Create a PDG with control flow for testing."""
        g = nx.DiGraph()
        g.add_node("if_cond", type="if", condition="x > 0")
        g.add_node("assign_then", type="assign", defines=["y"], uses=["x"])
        g.add_node("assign_else", type="assign", defines=["y"], uses=["x"])
        g.add_edge("if_cond", "assign_then", type="control_dependency")
        g.add_edge("if_cond", "assign_else", type="control_dependency")
        return g

    @pytest.fixture
    def pdg_with_anomalies(self):
        """Create a PDG with data flow anomalies."""
        g = nx.DiGraph()
        # Undefined variable use
        g.add_node(
            "use_undefined", type="assign", defines=["y"], uses=["undefined_var"]
        )
        # Unused variable definition
        g.add_node("unused_def", type="assign", defines=["unused_var"], uses=[])
        return g

    def test_init(self, simple_pdg):
        """Test PDGAnalyzer initialization."""
        analyzer = PDGAnalyzer(simple_pdg)
        assert analyzer.pdg is simple_pdg
        assert analyzer.cached_results == {}

    def test_analyze_data_flow(self, simple_pdg):
        """Test data flow analysis."""
        analyzer = PDGAnalyzer(simple_pdg)
        results = analyzer.analyze_data_flow()

        assert "anomalies" in results
        assert "def_use_chains" in results
        assert "live_variables" in results
        assert "reaching_definitions" in results
        assert "value_ranges" in results

    def test_analyze_data_flow_caching(self, simple_pdg):
        """Test that data flow analysis results are cached."""
        analyzer = PDGAnalyzer(simple_pdg)
        results1 = analyzer.analyze_data_flow()
        results2 = analyzer.analyze_data_flow()
        assert results1 is results2

    def test_analyze_control_flow(self, pdg_with_control_flow):
        """Test control flow analysis."""
        analyzer = PDGAnalyzer(pdg_with_control_flow)
        results = analyzer.analyze_control_flow()

        assert "cyclomatic_complexity" in results
        assert "control_dependencies" in results
        assert "unreachable_code" in results
        assert "loop_info" in results
        assert "dominators" in results

    def test_perform_security_analysis(self, simple_pdg):
        """Test security analysis returns list."""
        analyzer = PDGAnalyzer(simple_pdg)
        vulnerabilities = analyzer.perform_security_analysis()
        assert isinstance(vulnerabilities, list)

    def test_find_optimization_opportunities(self, simple_pdg):
        """Test optimization opportunity detection."""
        analyzer = PDGAnalyzer(simple_pdg)
        opportunities = analyzer.find_optimization_opportunities()

        assert "loop_invariant" in opportunities
        assert "common_subexpressions" in opportunities
        assert "dead_code" in opportunities
        assert "redundant_computations" in opportunities

    def test_compute_program_slice_backward(self, simple_pdg):
        """Test backward program slicing."""
        analyzer = PDGAnalyzer(simple_pdg)
        slice_graph = analyzer.compute_program_slice("assign_y", direction="backward")

        assert isinstance(slice_graph, nx.DiGraph)
        assert "assign_y" in slice_graph.nodes()

    def test_compute_program_slice_forward(self, simple_pdg):
        """Test forward program slicing."""
        analyzer = PDGAnalyzer(simple_pdg)
        slice_graph = analyzer.compute_program_slice("assign_x", direction="forward")

        assert isinstance(slice_graph, nx.DiGraph)
        assert "assign_x" in slice_graph.nodes()

    def test_find_data_flow_anomalies_undefined(self, pdg_with_anomalies):
        """Test detection of undefined variable usage."""
        analyzer = PDGAnalyzer(pdg_with_anomalies)
        anomalies = analyzer._find_data_flow_anomalies()

        undefined_anomalies = [a for a in anomalies if a.type == "undefined"]
        assert len(undefined_anomalies) >= 1
        assert any(a.variable == "undefined_var" for a in undefined_anomalies)

    def test_find_data_flow_anomalies_unused(self, pdg_with_anomalies):
        """Test detection of unused variable definitions."""
        analyzer = PDGAnalyzer(pdg_with_anomalies)
        anomalies = analyzer._find_data_flow_anomalies()

        unused_anomalies = [a for a in anomalies if a.type == "unused"]
        assert len(unused_anomalies) >= 1
        assert any(a.variable == "unused_var" for a in unused_anomalies)

    def test_identify_taint_sources(self):
        """Test taint source identification."""
        g = nx.DiGraph()
        g.add_node("user_input", type="call", call_target="input")
        g.add_node("safe_call", type="call", call_target="print")

        analyzer = PDGAnalyzer(g)
        sources = analyzer._identify_taint_sources()

        assert "user_input" in sources

    def test_identify_taint_sinks(self):
        """Test taint sink identification."""
        g = nx.DiGraph()
        g.add_node("sql_exec", type="call", call_target="execute")
        g.add_node("eval_call", type="call", call_target="eval")
        g.add_node("safe_call", type="call", call_target="print")

        analyzer = PDGAnalyzer(g)
        sinks = analyzer._identify_taint_sinks()

        assert "sql_exec" in sinks
        assert "eval_call" in sinks
        assert "safe_call" not in sinks

    def test_determine_sink_type(self, simple_pdg):
        """Test sink type determination."""
        analyzer = PDGAnalyzer(simple_pdg)

        assert analyzer._determine_sink_type("execute") == "sql_injection"
        assert analyzer._determine_sink_type("eval") == "code_injection"
        assert analyzer._determine_sink_type("subprocess.run") == "command_injection"
        assert analyzer._determine_sink_type("render_template") == "xss"
        assert analyzer._determine_sink_type("other") == "unknown"

    def test_is_path_sanitized(self, simple_pdg):
        """Test path sanitization detection."""
        g = nx.DiGraph()
        g.add_node("input", type="call", call_target="input")
        g.add_node("sanitize", type="call", call_target="html.escape")
        g.add_node("output", type="call", call_target="print")

        analyzer = PDGAnalyzer(g)

        # Path with sanitization
        assert analyzer._is_path_sanitized(["input", "sanitize", "output"]) is True

        # Path without sanitization
        assert analyzer._is_path_sanitized(["input", "output"]) is False

    def test_generate_vulnerability_description(self, simple_pdg):
        """Test vulnerability description generation."""
        desc = PDGAnalyzer._generate_vulnerability_description(
            "user_input", "sql_injection"
        )

        assert "user_input" in desc
        assert "sql_injection" in desc

    def test_hash_expression_ast(self, simple_pdg):
        """Test AST expression hashing."""
        import ast

        analyzer = PDGAnalyzer(simple_pdg)

        expr1 = ast.parse("x + 1", mode="eval").body
        expr2 = ast.parse("x + 1", mode="eval").body

        hash1 = analyzer._hash_expression(expr1)
        hash2 = analyzer._hash_expression(expr2)

        assert hash1 == hash2

    def test_hash_expression_string(self, simple_pdg):
        """Test string expression hashing."""
        analyzer = PDGAnalyzer(simple_pdg)

        hash1 = analyzer._hash_expression("x + 1")
        hash2 = analyzer._hash_expression("x + 1")

        assert hash1 == hash2


class TestPDGAnalyzerEdgeCases:
    """Tests for edge cases in PDG analysis."""

    def test_empty_pdg(self):
        """Test analyzing an empty PDG."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)

        results = analyzer.analyze_data_flow()
        assert results["anomalies"] == []

    def test_disconnected_pdg(self):
        """Test analyzing a disconnected PDG."""
        g = nx.DiGraph()
        g.add_node("node1", type="assign", defines=["x"])
        g.add_node("node2", type="assign", defines=["y"])
        # No edges

        analyzer = PDGAnalyzer(g)
        results = analyzer.analyze_data_flow()
        assert isinstance(results, dict)

    def test_cyclic_pdg(self):
        """Test analyzing a PDG with cycles (e.g., loop back-edges)."""
        g = nx.DiGraph()
        g.add_node("loop_start", type="while", condition="x > 0")
        g.add_node("loop_body", type="assign", defines=["x"], uses=["x"])
        g.add_edge("loop_start", "loop_body", type="control_dependency")
        g.add_edge("loop_body", "loop_start", type="data_dependency")  # Back edge

        analyzer = PDGAnalyzer(g)
        # Should not raise any errors
        results = analyzer.analyze_control_flow()
        assert isinstance(results, dict)

    def test_slice_nonexistent_node(self):
        """Test slicing with a non-existent criterion node."""
        g = nx.DiGraph()
        g.add_node("node1", type="assign")

        analyzer = PDGAnalyzer(g)
        # The compute_program_slice expects the node to exist - test the error case
        with pytest.raises(KeyError):
            analyzer.compute_program_slice("nonexistent_node")
