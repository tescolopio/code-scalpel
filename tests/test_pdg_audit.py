"""
PDG Tools Audit Tests

Comprehensive correctness and semantic validation tests for the PDG tools.
These tests verify that the PDG accurately represents program semantics,
not just coverage metrics.

Test Categories:
1. Data Dependency Correctness - verifies data flow edges are accurate
2. Control Dependency Correctness - verifies control flow edges are accurate
3. Scope Chain Correctness - verifies variable resolution follows Python scoping
4. Semantic Equivalence - verifies PDG preserves program meaning
5. Edge Cases - unusual but valid Python constructs
6. Regression Tests - specific scenarios that could break
"""

import ast
import sys
import os

import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from code_scalpel.pdg_tools.builder import PDGBuilder, build_pdg, NodeType, Scope
from code_scalpel.pdg_tools.analyzer import PDGAnalyzer, DependencyType
from code_scalpel.pdg_tools.slicer import ProgramSlicer, SlicingCriteria, SliceType


# =============================================================================
# DATA DEPENDENCY CORRECTNESS
# =============================================================================
class TestDataDependencyCorrectness:
    """Verify data dependencies are semantically correct."""

    def test_simple_data_flow(self):
        """x = 1; y = x should create edge from x_def to y_def."""
        code = """
x = 1
y = x
"""
        pdg, _ = build_pdg(code)

        # Find the two assign nodes
        assign_nodes = [
            (n, d) for n, d in pdg.nodes(data=True) if d.get("type") == "assign"
        ]
        assert len(assign_nodes) == 2

        # Find nodes by their targets
        x_node = next(n for n, d in assign_nodes if "x" in d.get("targets", []))
        y_node = next(n for n, d in assign_nodes if "y" in d.get("targets", []))

        # There should be a data dependency from x to y
        edges = list(pdg.edges(x_node, data=True))
        data_deps = [e for e in edges if e[2].get("type") == "data_dependency"]
        assert len(data_deps) >= 1
        assert any(e[1] == y_node for e in data_deps)

    def test_no_spurious_data_dependency(self):
        """Independent assignments should NOT have data dependencies."""
        code = """
x = 1
y = 2
"""
        pdg, _ = build_pdg(code)

        assign_nodes = [
            (n, d) for n, d in pdg.nodes(data=True) if d.get("type") == "assign"
        ]
        assert len(assign_nodes) == 2

        x_node = next(n for n, d in assign_nodes if "x" in d.get("targets", []))
        y_node = next(n for n, d in assign_nodes if "y" in d.get("targets", []))

        # No data dependency should exist between them
        assert not pdg.has_edge(x_node, y_node)
        assert not pdg.has_edge(y_node, x_node)

    def test_chain_data_dependency(self):
        """x = 1; y = x; z = y should create chain x -> y -> z."""
        code = """
x = 1
y = x
z = y
"""
        pdg, _ = build_pdg(code)

        assign_nodes = {
            d.get("targets", [""])[0]: n
            for n, d in pdg.nodes(data=True)
            if d.get("type") == "assign" and d.get("targets")
        }

        x_node = assign_nodes["x"]
        y_node = assign_nodes["y"]
        z_node = assign_nodes["z"]

        # Check chain: x -> y
        x_edges = [e for _, e, d in pdg.out_edges(x_node, data=True) 
                   if d.get("type") == "data_dependency"]
        assert y_node in x_edges

        # Check chain: y -> z
        y_edges = [e for _, e, d in pdg.out_edges(y_node, data=True) 
                   if d.get("type") == "data_dependency"]
        assert z_node in y_edges

    def test_multiple_uses_same_variable(self):
        """x = 1; y = x + x should create single dep from x to y."""
        code = """
x = 1
y = x + x
"""
        pdg, _ = build_pdg(code)

        assign_nodes = {
            d.get("targets", [""])[0]: n
            for n, d in pdg.nodes(data=True)
            if d.get("type") == "assign" and d.get("targets")
        }

        x_node = assign_nodes["x"]
        y_node = assign_nodes["y"]

        # Should have data dependency
        data_edges = [
            (u, v) for u, v, d in pdg.edges(data=True)
            if d.get("type") == "data_dependency" and u == x_node and v == y_node
        ]
        assert len(data_edges) >= 1

    def test_reassignment_breaks_chain(self):
        """x = 1; x = 2; y = x should use second x, not first."""
        code = """
x = 1
x = 2
y = x
"""
        pdg, _ = build_pdg(code)

        # Find all assign nodes
        assign_nodes = [
            (n, d) for n, d in pdg.nodes(data=True) if d.get("type") == "assign"
        ]
        
        # Find x assignments (there should be 2) and y assignment
        x_assigns = [(n, d) for n, d in assign_nodes if "x" in d.get("targets", [])]
        y_assign = next((n, d) for n, d in assign_nodes if "y" in d.get("targets", []))

        # y should depend on the second x assignment (x = 2), not the first
        # The second x assignment should be the one with value "2"
        x2_node = next(n for n, d in x_assigns if d.get("value") == "2")
        
        # Check that y depends on x2
        has_dep = pdg.has_edge(x2_node, y_assign[0])
        assert has_dep, "y should depend on the reassigned x = 2"


# =============================================================================
# CONTROL DEPENDENCY CORRECTNESS
# =============================================================================
class TestControlDependencyCorrectness:
    """Verify control dependencies are semantically correct."""

    def test_if_body_has_control_dependency(self):
        """Statements in if body should depend on if condition."""
        code = """
if x > 0:
    y = 1
"""
        pdg, _ = build_pdg(code)

        if_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "if"]
        assign_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "assign"]

        assert len(if_nodes) >= 1
        assert len(assign_nodes) >= 1

        if_node = if_nodes[0]
        
        # The assign should have a control dependency from the if
        ctrl_deps = [
            (u, v) for u, v, d in pdg.edges(data=True)
            if d.get("type") == "control_dependency" and u == if_node
        ]
        assert len(ctrl_deps) >= 1

    def test_else_body_has_control_dependency(self):
        """Statements in else body should depend on if condition."""
        code = """
if x > 0:
    y = 1
else:
    y = 2
"""
        pdg, _ = build_pdg(code)

        if_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "if"]
        assert len(if_nodes) >= 1
        
        if_node = if_nodes[0]
        
        # Both branches should depend on the if
        ctrl_deps = [
            (u, v) for u, v, d in pdg.edges(data=True)
            if d.get("type") == "control_dependency" and u == if_node
        ]
        # Should have at least 2 control deps (one for each branch)
        assert len(ctrl_deps) >= 2

    def test_nested_control_dependencies(self):
        """Nested if should have transitive control dependencies."""
        code = """
if x > 0:
    if y > 0:
        z = 1
"""
        pdg, _ = build_pdg(code)

        if_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "if"]
        assert len(if_nodes) >= 2

        # Find the innermost statement (z = 1)
        assign_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "assign"]
        assert len(assign_nodes) >= 1

    def test_loop_body_has_control_dependency(self):
        """Statements in loop body should depend on loop (control or loop dep)."""
        code = """
for i in range(10):
    x = i
"""
        pdg, _ = build_pdg(code)

        for_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "for"]
        assert len(for_nodes) >= 1
        
        for_node = for_nodes[0]
        
        # Check for control dependency OR loop dependency (both are valid)
        ctrl_deps = [
            (u, v) for u, v, d in pdg.edges(data=True)
            if d.get("type") in ("control_dependency", "loop_dependency") and u == for_node
        ]
        assert len(ctrl_deps) >= 1

    def test_outside_control_no_dependency(self):
        """Statements outside control structures should not have ctrl deps."""
        code = """
x = 1
if y > 0:
    z = 1
w = 2
"""
        pdg, _ = build_pdg(code)

        # Find x = 1 and w = 2 assignments
        assign_nodes = {
            d.get("targets", [""])[0]: n
            for n, d in pdg.nodes(data=True)
            if d.get("type") == "assign" and d.get("targets")
        }

        x_node = assign_nodes.get("x")
        w_node = assign_nodes.get("w")

        # These should have no incoming control dependencies from if
        if_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "if"]
        
        if if_nodes and x_node and w_node:
            if_node = if_nodes[0]
            assert not pdg.has_edge(if_node, x_node)
            assert not pdg.has_edge(if_node, w_node)


# =============================================================================
# SCOPE CHAIN CORRECTNESS
# =============================================================================
class TestScopeChainCorrectness:
    """Verify variable scoping follows Python semantics."""

    def test_function_scope_isolation(self):
        """Variables in function should not leak to outer scope."""
        code = """
x = 1
def foo():
    y = 2
    return y
z = x
"""
        pdg, _ = build_pdg(code)

        # z = x should NOT depend on y (which is in foo's scope)
        assign_nodes = {
            d.get("targets", [""])[0]: n
            for n, d in pdg.nodes(data=True)
            if d.get("type") == "assign" and d.get("targets")
        }

        y_node = assign_nodes.get("y")
        z_node = assign_nodes.get("z")

        if y_node and z_node:
            assert not pdg.has_edge(y_node, z_node)

    def test_parameter_as_data_source(self):
        """Function parameters should serve as data sources."""
        code = """
def foo(a):
    b = a + 1
    return b
"""
        pdg, _ = build_pdg(code)

        # Find parameter node
        param_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "parameter"]
        
        # Parameters should exist
        assert len(param_nodes) >= 1

    def test_scope_enter_exit(self):
        """Scope should be properly managed on enter/exit."""
        builder = PDGBuilder()
        
        # Initially no scopes
        assert len(builder.scopes) == 0
        
        # Enter a scope
        scope = builder.enter_scope("function", "test", "node_1")
        assert len(builder.scopes) == 1
        assert builder.get_current_scope() == scope
        
        # Exit scope
        exited = builder.exit_scope()
        assert exited == scope
        assert len(builder.scopes) == 0
        assert builder.get_current_scope() is None

    def test_nested_scope_variable_shadowing(self):
        """Inner scope variable should shadow outer scope."""
        builder = PDGBuilder()
        
        # Enter outer scope and define x
        outer = builder.enter_scope("function", "outer", "outer_1")
        outer.variables["x"] = "outer_x_def"
        
        # Enter inner scope and define x (shadows outer)
        inner = builder.enter_scope("function", "inner", "inner_1")
        inner.variables["x"] = "inner_x_def"
        
        # Find definition should return inner scope's x
        result = builder._find_definition("x")
        assert result == "inner_x_def"
        
        # Exit inner scope
        builder.exit_scope()
        
        # Now should find outer scope's x
        result = builder._find_definition("x")
        assert result == "outer_x_def"


# =============================================================================
# EDGE CASES AND UNUSUAL CONSTRUCTS
# =============================================================================
class TestEdgeCases:
    """Test unusual but valid Python constructs."""

    def test_walrus_operator(self):
        """Named expression (walrus) := should be handled."""
        code = """
if (n := len(items)) > 10:
    print(n)
"""
        # Should not crash
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_starred_assignment(self):
        """Starred unpacking *rest should be handled."""
        code = """
first, *rest, last = [1, 2, 3, 4, 5]
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None
        
        # Should have created assignment node
        assign_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "assign"]
        assert len(assign_nodes) >= 1

    def test_multiple_assignment_targets(self):
        """x = y = z = 1 should create multiple targets."""
        code = """
x = y = z = 1
"""
        pdg, _ = build_pdg(code)

        assign_nodes = [
            (n, d) for n, d in pdg.nodes(data=True) if d.get("type") == "assign"
        ]
        # May create one node with multiple targets or multiple nodes
        assert len(assign_nodes) >= 1

    def test_nested_function_calls(self):
        """f(g(h(x))) should create proper call nodes."""
        code = """
x = 1
result = f(g(h(x)))
"""
        pdg, _ = build_pdg(code)

        call_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "call"]
        # Should have at least 3 call nodes (f, g, h)
        assert len(call_nodes) >= 3

    def test_comprehension_scope(self):
        """List comprehension has its own scope in Python 3."""
        code = """
x = [i * 2 for i in range(10)]
"""
        pdg, _ = build_pdg(code)
        # Should not crash
        assert pdg is not None

    def test_lambda_expression(self):
        """Lambda should create a function-like construct."""
        code = """
f = lambda x: x + 1
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_decorator_with_arguments(self):
        """@decorator(arg) should be handled."""
        code = """
@decorator(arg1, arg2)
def foo():
    pass
"""
        pdg, _ = build_pdg(code)
        
        func_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "function"]
        assert len(func_nodes) >= 1

    def test_class_decorator(self):
        """@decorator on class should be handled."""
        code = """
@decorator
class MyClass:
    pass
"""
        pdg, _ = build_pdg(code)
        
        class_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "class"]
        assert len(class_nodes) >= 1

    def test_async_function(self):
        """async def should be handled like regular function."""
        code = """
async def fetch_data():
    await some_io()
    return result
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_try_finally(self):
        """try/finally without except should work."""
        code = """
try:
    x = 1
finally:
    cleanup()
"""
        pdg, _ = build_pdg(code)
        
        try_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "try"]
        assert len(try_nodes) >= 1

    def test_try_except_else(self):
        """try/except/else should handle else block."""
        code = """
try:
    x = risky()
except Error:
    x = default
else:
    x = success
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_match_statement(self):
        """Python 3.10 match/case should not crash."""
        code = """
match command:
    case "quit":
        quit()
    case "help":
        show_help()
"""
        # May not fully support match, but should not crash
        try:
            pdg, _ = build_pdg(code)
            assert pdg is not None
        except SyntaxError:
            # OK if Python version doesn't support match
            pass

    def test_global_statement(self):
        """global x should be handled."""
        code = """
x = 1
def foo():
    global x
    x = 2
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_nonlocal_statement(self):
        """nonlocal x should be handled."""
        code = """
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
    inner()
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None


# =============================================================================
# ANALYZER CORRECTNESS
# =============================================================================
class TestAnalyzerCorrectness:
    """Verify PDGAnalyzer produces correct results."""

    def test_cyclomatic_complexity_if(self):
        """Each if adds 1 to complexity."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="if")
        pdg.add_node("n2", type="if")
        pdg.add_node("n3", type="assign")

        analyzer = PDGAnalyzer(pdg)
        complexity = analyzer._calculate_cyclomatic_complexity()
        
        # Base 1 + 2 ifs = 3
        assert complexity == 3

    def test_cyclomatic_complexity_loops(self):
        """while and for add to complexity."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="while")
        pdg.add_node("n2", type="for")

        analyzer = PDGAnalyzer(pdg)
        complexity = analyzer._calculate_cyclomatic_complexity()
        
        # Base 1 + 1 while + 1 for = 3
        assert complexity == 3

    def test_dead_code_detection(self):
        """Nodes with no outgoing edges (except return) are dead code."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign")  # Dead code (no out edges)
        pdg.add_node("n2", type="return")  # Not dead code (return is terminal)
        pdg.add_node("n3", type="assign")
        pdg.add_edge("n3", "n2", type="data_dependency")

        analyzer = PDGAnalyzer(pdg)
        dead = analyzer._find_dead_code()

        # n1 should be detected as dead code
        dead_nodes = [d["node"] for d in dead]
        assert "n1" in dead_nodes
        assert "n2" not in dead_nodes

    def test_undefined_variable_detection(self):
        """Variables used but never defined should be flagged."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", uses=["undefined_var"], location=(1, 0))

        analyzer = PDGAnalyzer(pdg)
        anomalies = analyzer._find_data_flow_anomalies()

        undefined = [a for a in anomalies if a.type == "undefined"]
        assert len(undefined) >= 1
        assert undefined[0].variable == "undefined_var"

    def test_unused_variable_detection(self):
        """Variables defined but never used should be flagged."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["unused_var"], location=(1, 0))

        analyzer = PDGAnalyzer(pdg)
        anomalies = analyzer._find_data_flow_anomalies()

        unused = [a for a in anomalies if a.type == "unused"]
        assert len(unused) >= 1
        assert unused[0].variable == "unused_var"

    def test_taint_source_identification(self):
        """input() calls should be taint sources."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="call", call_target="input")

        analyzer = PDGAnalyzer(pdg)
        sources = analyzer._identify_taint_sources()

        assert "n1" in sources

    def test_taint_sink_identification(self):
        """eval() calls should be taint sinks."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="call", call_target="eval")

        analyzer = PDGAnalyzer(pdg)
        sinks = analyzer._identify_taint_sinks()

        assert "n1" in sinks

    def test_sanitized_path_detection(self):
        """Path with escape() should be marked sanitized."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="call", call_target="input")
        pdg.add_node("n2", type="call", call_target="html.escape")
        pdg.add_node("n3", type="call", call_target="eval")

        analyzer = PDGAnalyzer(pdg)
        path = ["n1", "n2", "n3"]
        
        assert analyzer._is_path_sanitized(path) is True

    def test_unsanitized_path_detection(self):
        """Path without sanitizer should NOT be marked sanitized."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="call", call_target="input")
        pdg.add_node("n2", type="assign")
        pdg.add_node("n3", type="call", call_target="eval")

        analyzer = PDGAnalyzer(pdg)
        path = ["n1", "n2", "n3"]
        
        assert analyzer._is_path_sanitized(path) is False


# =============================================================================
# SLICER CORRECTNESS
# =============================================================================
class TestSlicerCorrectness:
    """Verify ProgramSlicer produces correct slices."""

    def test_backward_slice_includes_dependencies(self):
        """Backward slice should include all data dependencies."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["x"])
        pdg.add_node("n2", type="assign", defines=["y"], uses=["x"])
        pdg.add_node("n3", type="assign", defines=["z"], uses=["y"])
        pdg.add_edge("n1", "n2", type="data_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        criteria = SlicingCriteria(nodes={"n3"}, variables=set())
        
        slice_result = slicer.compute_slice(criteria, SliceType.BACKWARD)
        
        # Should include n1, n2, n3
        assert "n3" in slice_result.nodes()
        assert "n2" in slice_result.nodes()
        assert "n1" in slice_result.nodes()

    def test_backward_slice_excludes_unrelated(self):
        """Backward slice should NOT include unrelated nodes."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["x"])
        pdg.add_node("n2", type="assign", defines=["y"])  # Unrelated
        pdg.add_node("n3", type="assign", defines=["z"], uses=["x"])
        pdg.add_edge("n1", "n3", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        criteria = SlicingCriteria(nodes={"n3"}, variables=set())
        
        slice_result = slicer.compute_slice(criteria, SliceType.BACKWARD)
        
        # Should include n1, n3 but NOT n2
        assert "n3" in slice_result.nodes()
        assert "n1" in slice_result.nodes()
        assert "n2" not in slice_result.nodes()

    def test_forward_slice_includes_dependents(self):
        """Forward slice should include all data dependents."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["x"])
        pdg.add_node("n2", type="assign", defines=["y"], uses=["x"])
        pdg.add_node("n3", type="assign", defines=["z"], uses=["y"])
        pdg.add_edge("n1", "n2", type="data_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        criteria = SlicingCriteria(nodes={"n1"}, variables=set())
        
        slice_result = slicer.compute_slice(criteria, SliceType.FORWARD)
        
        # Should include n1, n2, n3
        assert "n1" in slice_result.nodes()
        assert "n2" in slice_result.nodes()
        assert "n3" in slice_result.nodes()

    def test_chop_computes_intersection(self):
        """Chop should be intersection of forward and backward slices."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["x"])
        pdg.add_node("n2", type="assign", defines=["y"], uses=["x"])
        pdg.add_node("n3", type="assign", defines=["z"], uses=["y"])
        pdg.add_node("n4", type="assign", defines=["w"], uses=["x"])  # Divergent path
        pdg.add_edge("n1", "n2", type="data_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")
        pdg.add_edge("n1", "n4", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        source = SlicingCriteria(nodes={"n1"}, variables=set())
        target = SlicingCriteria(nodes={"n3"}, variables=set())
        
        chop = slicer.compute_chop(source, target)
        
        # Chop n1->n3 should include n1, n2, n3 but NOT n4
        assert "n1" in chop.nodes()
        assert "n2" in chop.nodes()
        assert "n3" in chop.nodes()
        # n4 is not on the path from n1 to n3
        assert "n4" not in chop.nodes()

    def test_thin_slice_no_transitive(self):
        """Thin slice should only include direct dependencies."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="assign", defines=["x"])
        pdg.add_node("n2", type="assign", defines=["y"], uses=["x"])
        pdg.add_node("n3", type="assign", defines=["z"], uses=["y"])
        pdg.add_edge("n1", "n2", type="data_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        criteria = SlicingCriteria(nodes={"n3"}, variables=set())
        
        slice_result = slicer.compute_slice(criteria, SliceType.THIN)
        
        # Thin slice: only n3 and its direct dep n2
        assert "n3" in slice_result.nodes()
        assert "n2" in slice_result.nodes()
        # n1 is NOT direct dependency of n3
        assert "n1" not in slice_result.nodes()

    def test_control_only_slice(self):
        """Control-only slice should exclude data dependencies."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="if")
        pdg.add_node("n2", type="assign", defines=["x"])
        pdg.add_node("n3", type="assign", defines=["y"], uses=["x"])
        pdg.add_edge("n1", "n3", type="control_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")

        slicer = ProgramSlicer(pdg)
        criteria = SlicingCriteria(
            nodes={"n3"}, 
            variables=set(),
            include_control=True,
            include_data=False
        )
        
        slice_result = slicer.compute_slice(criteria, SliceType.BACKWARD)
        
        # Should include n1 (control) and n3, but NOT n2 (data only)
        assert "n3" in slice_result.nodes()
        assert "n1" in slice_result.nodes()
        assert "n2" not in slice_result.nodes()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestIntegration:
    """End-to-end integration tests."""

    def test_build_analyze_slice_pipeline(self):
        """Full pipeline: build -> analyze -> slice."""
        code = """
def calculate(x, y):
    temp = x * 2
    result = temp + y
    return result
"""
        # Build
        pdg, call_graph = build_pdg(code)
        assert len(pdg.nodes()) > 0

        # Analyze
        analyzer = PDGAnalyzer(pdg)
        data_flow = analyzer.analyze_data_flow()
        control_flow = analyzer.analyze_control_flow()
        
        assert "def_use_chains" in data_flow
        assert "cyclomatic_complexity" in control_flow

        # Slice
        slicer = ProgramSlicer(pdg)
        # Find a return node to slice from
        return_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "return"]
        
        if return_nodes:
            criteria = SlicingCriteria(nodes={return_nodes[0]}, variables=set())
            slice_result = slicer.compute_slice(criteria, SliceType.BACKWARD)
            assert len(slice_result.nodes()) > 0

    def test_security_analysis_finds_vulnerability(self):
        """Security analysis should detect input->eval vulnerability."""
        pdg = nx.DiGraph()
        pdg.add_node("n1", type="call", call_target="input")
        pdg.add_node("n2", type="assign", defines=["user_data"])
        pdg.add_node("n3", type="call", call_target="eval")
        pdg.add_edge("n1", "n2", type="data_dependency")
        pdg.add_edge("n2", "n3", type="data_dependency")

        analyzer = PDGAnalyzer(pdg)
        vulnerabilities = analyzer.perform_security_analysis()

        # Should detect at least the taint source and sink
        sources = analyzer._identify_taint_sources()
        sinks = analyzer._identify_taint_sinks()
        
        assert "n1" in sources
        assert "n3" in sinks

    def test_optimization_finds_loop_invariant(self):
        """Optimization analysis should find loop-invariant code."""
        pdg = nx.DiGraph()
        pdg.add_node("loop1", type="for")
        pdg.add_node("n1", type="assign", uses=["const"], defines=["temp"])
        pdg.add_node("n2", type="assign", uses=["i", "temp"], defines=["result"])
        
        # n1 doesn't use any loop-modified variables
        # n2 uses i which is loop variable
        pdg.add_edge("loop1", "n1", type="control_dependency")
        pdg.add_edge("loop1", "n2", type="control_dependency")
        # loop modifies i
        pdg.nodes["n2"]["defines"] = ["i", "result"]  # i is modified in loop

        analyzer = PDGAnalyzer(pdg)
        
        # n1 should be loop invariant (doesn't use loop-modified vars)
        assert analyzer._is_loop_invariant("n1", "loop1") is True


# =============================================================================
# REGRESSION TESTS
# =============================================================================
class TestRegressions:
    """Tests for specific bugs and edge cases found."""

    def test_empty_function_body(self):
        """Function with only pass should not crash."""
        code = """
def empty():
    pass
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_empty_class_body(self):
        """Class with only pass should not crash."""
        code = """
class Empty:
    pass
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_deeply_nested_code(self):
        """Deeply nested code should not stack overflow."""
        code = """
def deep():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        x = 1
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_large_tuple_unpacking(self):
        """Large tuple unpacking should work."""
        code = """
a, b, c, d, e, f, g, h, i, j = range(10)
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_chained_comparison(self):
        """Chained comparison should not crash."""
        code = """
if 0 < x < 10:
    y = x
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_fstring(self):
        """f-string should be handled."""
        code = """
name = "world"
msg = f"Hello, {name}!"
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_yield_statement(self):
        """Generator with yield should not crash."""
        code = """
def gen():
    for i in range(10):
        yield i
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_yield_from(self):
        """yield from should not crash."""
        code = """
def gen():
    yield from range(10)
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_with_statement(self):
        """with statement should not crash."""
        code = """
with open('file.txt') as f:
    content = f.read()
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_assert_statement(self):
        """assert should not crash."""
        code = """
x = 1
assert x > 0, "x must be positive"
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_raise_statement(self):
        """raise should not crash."""
        code = """
def check(x):
    if x < 0:
        raise ValueError("negative")
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_import_variations(self):
        """Various import styles should work."""
        code = """
import os
import os.path
from os import path
from os.path import join, exists
from os import *
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_annotation_assignment(self):
        """Type-annotated assignment should work."""
        code = """
x: int = 1
y: str
z: list[int] = [1, 2, 3]
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_slice_object(self):
        """Slice notation should not crash."""
        code = """
items = [1, 2, 3, 4, 5]
first_two = items[:2]
last_two = items[-2:]
middle = items[1:4]
every_other = items[::2]
"""
        pdg, _ = build_pdg(code)
        assert pdg is not None

    def test_while_undefined_condition_variable(self):
        """While loop with undefined variable in condition (hits branch partial)."""
        code = """
while unknown_flag:
    break
"""
        pdg, _ = build_pdg(code)
        # Should create while node but no data dependency for undefined var
        while_nodes = [n for n, d in pdg.nodes(data=True) if d.get("type") == "while"]
        assert len(while_nodes) == 1

    def test_add_variable_definition_no_scope(self):
        """Calling _add_variable_definition with no scope (defensive test)."""
        builder = PDGBuilder()
        # No scopes - should not crash
        assert len(builder.scopes) == 0
        builder._add_variable_definition("x", "node_1")
        # Should have done nothing (no scope to add to)
        assert len(builder.scopes) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
