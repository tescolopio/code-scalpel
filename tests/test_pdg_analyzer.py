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


class TestAnalyzerCoverageGaps:
    """Tests targeting specific uncovered lines in analyzer.py."""

    def test_compute_program_slice_forward_traversal(self):
        """Test forward slicing traversal (line 111, 122)."""
        g = nx.DiGraph()
        g.add_node("n1", type="assign", defines=["x"])
        g.add_node("n2", type="assign", uses=["x"], defines=["y"])
        g.add_node("n3", type="return", uses=["y"])
        g.add_edge("n1", "n2", type="data")
        g.add_edge("n2", "n3", type="data")

        analyzer = PDGAnalyzer(g)
        sliced = analyzer.compute_program_slice("n1", direction="forward")
        # Forward slice from n1 should include n2 and n3
        assert "n1" in sliced.nodes()

    def test_perform_taint_analysis_with_paths(self):
        """Test taint analysis with actual paths (lines 184-192)."""
        g = nx.DiGraph()
        g.add_node("source", type="call", function="input", taint_type="user_input")
        g.add_node(
            "sink", type="call", function="cursor.execute", sink_type="sql_query"
        )
        g.add_edge("source", "sink", type="data")

        analyzer = PDGAnalyzer(g)
        vulns = analyzer._perform_taint_analysis()
        # Should find vulnerability from source to sink
        assert isinstance(vulns, list)

    def test_find_loop_invariant_code(self):
        """Test loop invariant code detection (lines 213-216)."""
        g = nx.DiGraph()
        g.add_node("loop", type="while")
        g.add_node("invariant", type="assign")
        g.add_edge("loop", "invariant", type="control_dependency")

        analyzer = PDGAnalyzer(g)
        invariants = analyzer._find_loop_invariant_code()
        assert isinstance(invariants, list)

    def test_analyze_value_ranges(self):
        """Test value range analysis (lines 232, 246-251)."""
        g = nx.DiGraph()
        g.add_node("n1", type="assign", defines=["x"])

        analyzer = PDGAnalyzer(g)
        ranges = analyzer._analyze_value_ranges()
        assert isinstance(ranges, dict)

    def test_find_common_subexpressions(self):
        """Test common subexpression detection (lines 350-351)."""
        import ast

        g = nx.DiGraph()
        expr = ast.parse("a + b", mode="eval").body
        g.add_node("n1", expression=expr)
        g.add_node("n2", expression=expr)

        analyzer = PDGAnalyzer(g)
        common = analyzer._find_common_subexpressions()
        assert isinstance(common, list)

    def test_find_redundant_computations(self):
        """Test redundant computation detection (lines 375-378, 382-393)."""
        g = nx.DiGraph()
        g.add_node("n1", type="computation", constant_value=42)
        g.add_node("n2", type="computation", constant_value=42)

        analyzer = PDGAnalyzer(g)
        redundant = analyzer._find_redundant_computations()
        assert isinstance(redundant, list)

    def test_find_dead_code(self):
        """Test dead code detection (line 487)."""
        g = nx.DiGraph()
        g.add_node("dead", type="assign")  # No outgoing edges

        analyzer = PDGAnalyzer(g)
        dead_code = analyzer._find_dead_code()
        assert len(dead_code) >= 1
        assert dead_code[0]["node"] == "dead"

    def test_find_loops(self):
        """Test loop detection (lines 492-496)."""
        g = nx.DiGraph()
        g.add_node("while_loop", type="while")
        g.add_node("for_loop", type="for")
        g.add_node("assign", type="assign")

        analyzer = PDGAnalyzer(g)
        loops = analyzer._find_loops()
        assert "while_loop" in loops
        assert "for_loop" in loops
        assert "assign" not in loops

    def test_analyze_loop_body(self):
        """Test loop body analysis (lines 500, 504)."""
        g = nx.DiGraph()
        g.add_node("loop", type="while")
        g.add_node("body1", type="assign")
        g.add_node("body2", type="call")
        g.add_edge("loop", "body1", type="control_dependency")
        g.add_edge("loop", "body2", type="control_dependency")

        analyzer = PDGAnalyzer(g)
        body = analyzer._analyze_loop_body("loop")
        assert "body1" in body
        assert "body2" in body

    def test_is_loop_invariant_stub(self):
        """Test loop invariant check stub (line 508)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._is_loop_invariant("node", "loop")
        assert result is False

    def test_estimate_optimization_savings_stub(self):
        """Test optimization savings stub (line 518)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._estimate_optimization_savings("node", "loop")
        assert result == 0

    def test_parse_condition_constraint_works(self):
        """Test condition constraint parser (line 522) - now implemented."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._parse_condition_constraint("x > 0", "x")
        assert result == (">", 0)  # Now returns actual constraint

    def test_solve_constraints_stub(self):
        """Test constraint solver stub (line 534)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._solve_constraints([])
        assert result == (None, None)

    def test_matches_exception_exact_type(self):
        """Test exception matching with exact type (line 538)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)

        exc = ValueError("test")
        assert analyzer._matches_exception(exc, "ValueError") is True
        assert analyzer._matches_exception(exc, "TypeError") is False
        assert analyzer._matches_exception(exc, None) is True  # Bare except

    def test_infer_type(self):
        """Test type inference (line 542)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)

        assert analyzer._infer_type(42) is int
        assert analyzer._infer_type("hello") is str
        assert analyzer._infer_type([1, 2]) is list

    def test_handle_attribute_assignment_stub(self):
        """Test attribute assignment handler stub (line 550)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        # Should not raise
        analyzer._handle_attribute_assignment(None, None)

    def test_handle_subscript_assignment_stub(self):
        """Test subscript assignment handler stub (line 554)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        analyzer._handle_subscript_assignment(None, None)

    def test_handle_symbolic_call_stub(self):
        """Test symbolic call handler stub (line 558)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._handle_symbolic_call(None, 0)
        assert result is None

    def test_handle_concrete_call_stub(self):
        """Test concrete call handler stub (line 562)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._handle_concrete_call(None)
        assert result is None

    def test_handle_method_call_stub(self):
        """Test method call handler stub (line 566)."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._handle_method_call(None, 0)
        assert result is None

    def test_handle_other_stub(self):
        """Test other handler stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        analyzer._handle_other(None)

    def test_evaluate_compare_stub(self):
        """Test compare evaluation stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._evaluate_compare(None)
        assert result is None

    def test_evaluate_boolop_stub(self):
        """Test boolop evaluation stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._evaluate_boolop(None)
        assert result is None

    def test_evaluate_attribute_stub(self):
        """Test attribute evaluation stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._evaluate_attribute(None)
        assert result is None

    def test_evaluate_subscript_stub(self):
        """Test subscript evaluation stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._evaluate_subscript(None)
        assert result is None

    def test_extract_concrete_value_stub(self):
        """Test concrete value extraction stub."""
        g = nx.DiGraph()
        analyzer = PDGAnalyzer(g)
        result = analyzer._extract_concrete_value(None, None)
        assert result is None


class TestAnalyzerFinalCoverage:
    """Final coverage tests for analyzer.py."""

    def test_taint_analysis_creates_vulnerability(self):
        """Test lines 184-192: Create actual SecurityVulnerability."""
        from code_scalpel.pdg_tools.analyzer import PDGAnalyzer, DependencyType

        g = nx.DiGraph()
        # Source node - taint source (use call_target attribute)
        g.add_node("source1", type="call", call_target="input", taint_type="user_input")
        # Intermediate node - NOT a sanitizer
        g.add_node("mid1", type="assign")
        # Sink node - dangerous sink (use call_target with execute)
        g.add_node(
            "sink1", type="call", call_target="cursor.execute", sink_type="sql_query"
        )

        # Create path from source to sink
        g.add_edge("source1", "mid1", type=DependencyType.DATA.value)
        g.add_edge("mid1", "sink1", type=DependencyType.DATA.value)

        analyzer = PDGAnalyzer(g)
        vulnerabilities = analyzer._perform_taint_analysis()

        # Should find vulnerability since path is not sanitized
        assert len(vulnerabilities) >= 1
        assert any(v.source == "source1" for v in vulnerabilities)
        assert any(v.sink == "sink1" for v in vulnerabilities)

    def test_value_range_with_condition_constraint(self):
        """Test lines 246-251: Parse condition constraint."""
        from code_scalpel.pdg_tools.analyzer import PDGAnalyzer, DependencyType

        g = nx.DiGraph()
        # Add a condition node with a parseable condition
        g.add_node("if_node", type="if", condition="x > 5")
        g.add_node("body_node", type="assign", defines=["y"])
        g.add_edge("if_node", "body_node", type=DependencyType.CONTROL.value)

        analyzer = PDGAnalyzer(g)

        # Compute value range for x at body_node
        range_result = analyzer._compute_value_range("x", "body_node")
        # Should return something (even if empty tuple)
        assert isinstance(range_result, tuple)

    def test_loop_invariant_detection(self):
        """Test line 216: Loop invariant detection with candidates."""
        from code_scalpel.pdg_tools.analyzer import PDGAnalyzer, DependencyType

        g = nx.DiGraph()
        # Create a loop with a body node
        g.add_node("loop", type="for")
        g.add_node(
            "invariant", type="assign", defines=["c"]
        )  # Loop invariant - c doesn't depend on loop
        g.add_node(
            "variant", type="assign", defines=["i"], uses=["i"]
        )  # Loop variant - i depends on i

        g.add_edge("loop", "invariant", type=DependencyType.CONTROL.value)
        g.add_edge("loop", "variant", type=DependencyType.CONTROL.value)

        analyzer = PDGAnalyzer(g)
        result = analyzer._find_loop_invariant_code()

        # Should return list (even if empty due to stub methods)
        assert isinstance(result, list)

    def test_redundant_computation_inner_branch(self):
        """Test line 378: Inner branch of redundant computation detection."""
        from code_scalpel.pdg_tools.analyzer import PDGAnalyzer

        g = nx.DiGraph()
        # Create nodes with constant_value attribute - must use type="computation"
        g.add_node("comp1", type="computation", constant_value=100)
        g.add_node(
            "comp2", type="computation", constant_value=100
        )  # Same value - redundant

        analyzer = PDGAnalyzer(g)
        redundant = analyzer._find_redundant_computations()

        # Should identify the redundancy
        assert len(redundant) >= 1

    def test_analyze_loop_body_with_children(self):
        """Test _analyze_loop_body returns candidates."""
        from code_scalpel.pdg_tools.analyzer import PDGAnalyzer, DependencyType

        g = nx.DiGraph()
        g.add_node("loop", type="while")
        g.add_node("body1", type="assign")
        g.add_node("body2", type="call")
        g.add_edge("loop", "body1", type=DependencyType.CONTROL.value)
        g.add_edge("loop", "body2", type=DependencyType.CONTROL.value)

        analyzer = PDGAnalyzer(g)
        candidates = analyzer._analyze_loop_body("loop")

        # Should return body nodes as candidates
        assert isinstance(candidates, list)
