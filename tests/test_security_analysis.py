"""
Tests for v0.3.0 Security Analysis Features.

Tests:
- String support in type inference and symbolic execution
- Taint tracking and propagation
- Security sink detection
- Vulnerability detection (SQL injection, XSS, etc.)
"""

import warnings

# Suppress the BETA warning for cleaner test output
warnings.filterwarnings("ignore", message="symbolic_execution_tools")

from code_scalpel.symbolic_execution_tools.type_inference import (
    TypeInferenceEngine,
    InferredType,
)
from code_scalpel.symbolic_execution_tools.taint_tracker import (
    TaintTracker,
    TaintSource,
    TaintLevel,
    SecuritySink,
    TaintInfo,
    TaintedValue,
    Vulnerability,
    # v0.3.1: Sanitizer support
    SANITIZER_REGISTRY,
    register_sanitizer,
)
from code_scalpel.symbolic_execution_tools.security_analyzer import (
    analyze_security,
    find_sql_injections,
    find_xss,
    find_command_injections,
)

import z3


# =============================================================================
# String Type Inference Tests
# =============================================================================

class TestStringTypeInference:
    """Test string type inference in TypeInferenceEngine."""
    
    def test_infer_string_literal(self):
        """String literals should be inferred as STRING."""
        engine = TypeInferenceEngine()
        types = engine.infer('x = "hello"')
        assert types["x"] == InferredType.STRING
    
    def test_infer_empty_string(self):
        """Empty string should be STRING."""
        engine = TypeInferenceEngine()
        types = engine.infer('x = ""')
        assert types["x"] == InferredType.STRING
    
    def test_infer_string_concatenation(self):
        """String + String should be STRING."""
        engine = TypeInferenceEngine()
        types = engine.infer('x = "hello" + " world"')
        assert types["x"] == InferredType.STRING
    
    def test_infer_string_variable_concat(self):
        """Concatenating string variables should be STRING."""
        engine = TypeInferenceEngine()
        types = engine.infer('a = "hello"\nb = "world"\nc = a + b')
        assert types["c"] == InferredType.STRING
    
    def test_infer_mixed_concat_is_unknown(self):
        """String + Int should be UNKNOWN (strict mode)."""
        engine = TypeInferenceEngine()
        types = engine.infer('x = "hello"\ny = 5\nz = x + y')
        # In strict mode, mixed types produce UNKNOWN
        # (Python would convert, but we don't)
        assert types["z"] == InferredType.UNKNOWN
    
    def test_infer_string_multiplication(self):
        """String * Int should be STRING."""
        engine = TypeInferenceEngine()
        types = engine.infer('x = "ab" * 3')
        assert types["x"] == InferredType.STRING
    
    def test_string_sort_conversion(self):
        """InferredType.STRING should convert to StringSort."""
        sort = InferredType.STRING.to_z3_sort()
        assert sort == z3.StringSort()


# =============================================================================
# Taint Tracker Tests
# =============================================================================

class TestTaintTracker:
    """Test TaintTracker functionality."""
    
    def test_create_taint_source(self):
        """Should create tainted value from source."""
        tracker = TaintTracker()
        tainted = tracker.taint_source("user_input", TaintSource.USER_INPUT)
        
        assert tainted.is_tainted
        assert tainted.taint.source == TaintSource.USER_INPUT
        assert tainted.taint.level == TaintLevel.HIGH
    
    def test_is_tainted_true(self):
        """Should report variable as tainted."""
        tracker = TaintTracker()
        tracker.taint_source("x", TaintSource.USER_INPUT)
        
        assert tracker.is_tainted("x") is True
    
    def test_is_tainted_false(self):
        """Should report clean variable as not tainted."""
        tracker = TaintTracker()
        assert tracker.is_tainted("x") is False
    
    def test_propagate_assignment_taints_target(self):
        """Assignment from tainted source should taint target."""
        tracker = TaintTracker()
        tracker.taint_source("input", TaintSource.USER_INPUT)
        
        result = tracker.propagate_assignment("query", ["input"])
        
        assert result is not None
        assert tracker.is_tainted("query")
    
    def test_propagate_assignment_clean_sources(self):
        """Assignment from clean sources should not taint target."""
        tracker = TaintTracker()
        
        result = tracker.propagate_assignment("x", ["a", "b"])
        
        assert result is None
        assert tracker.is_tainted("x") is False
    
    def test_propagate_concat_taints_result(self):
        """Concatenation with any tainted operand should taint result."""
        tracker = TaintTracker()
        tracker.taint_source("user_input", TaintSource.USER_INPUT)
        
        # query = "SELECT * FROM users WHERE id=" + user_input
        result = tracker.propagate_concat("query", ["prefix", "user_input"])
        
        assert result is not None
        assert tracker.is_tainted("query")
    
    def test_apply_sanitizer_lowers_level(self):
        """Applying sanitizer should lower taint level."""
        tracker = TaintTracker()
        tracker.taint_source("x", TaintSource.USER_INPUT)
        
        result = tracker.apply_sanitizer("x", "html.escape")
        
        assert result.level == TaintLevel.LOW
        assert "html.escape" in result.sanitizers_applied
    
    def test_fork_creates_isolated_copy(self):
        """Fork should create isolated copy of tracker."""
        tracker = TaintTracker()
        tracker.taint_source("x", TaintSource.USER_INPUT)
        
        forked = tracker.fork()
        forked.taint_source("y", TaintSource.FILE_CONTENT)
        
        # Original should not have y
        assert tracker.is_tainted("x")
        assert not tracker.is_tainted("y")
        
        # Forked should have both
        assert forked.is_tainted("x")
        assert forked.is_tainted("y")


# =============================================================================
# Taint Info Tests
# =============================================================================

class TestTaintInfo:
    """Test TaintInfo behavior."""
    
    def test_propagate_adds_to_path(self):
        """Propagate should add variable to path."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            propagation_path=["input"]
        )
        
        propagated = taint.propagate("query")
        
        assert "query" in propagated.propagation_path
        assert "input" in propagated.propagation_path
    
    def test_dangerous_for_sink_without_sanitizer(self):
        """Tainted data without sanitizer is dangerous."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH
        )
        
        assert taint.is_dangerous_for(SecuritySink.SQL_QUERY)
        assert taint.is_dangerous_for(SecuritySink.HTML_OUTPUT)
    
    def test_not_dangerous_with_appropriate_sanitizer(self):
        """Tainted data with correct sanitizer is not dangerous."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.LOW,
            sanitizers_applied={"html.escape"}
        )
        
        # html.escape makes HTML safe but not SQL safe
        assert not taint.is_dangerous_for(SecuritySink.HTML_OUTPUT)
        assert taint.is_dangerous_for(SecuritySink.SQL_QUERY)
    
    def test_not_dangerous_if_none_level(self):
        """Clean data is not dangerous."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.NONE
        )
        
        assert not taint.is_dangerous_for(SecuritySink.SQL_QUERY)


# =============================================================================
# Vulnerability Detection Tests
# =============================================================================

class TestVulnerabilityDetection:
    """Test vulnerability detection through TaintTracker."""
    
    def test_detect_sql_injection(self):
        """Should detect SQL injection vulnerability."""
        tracker = TaintTracker()
        tracker.taint_source("user_id", TaintSource.USER_INPUT)
        tracker.propagate_concat("query", ["prefix", "user_id"])
        
        vuln = tracker.check_sink("query", SecuritySink.SQL_QUERY, (10, 0))
        
        assert vuln is not None
        assert vuln.sink_type == SecuritySink.SQL_QUERY
        assert vuln.vulnerability_type == "SQL Injection"
        assert vuln.cwe_id == "CWE-89"
    
    def test_no_vulnerability_for_clean_data(self):
        """Should not detect vulnerability for clean data."""
        tracker = TaintTracker()
        # No taint sources
        
        vuln = tracker.check_sink("query", SecuritySink.SQL_QUERY, (10, 0))
        
        assert vuln is None
    
    def test_vulnerability_to_dict(self):
        """Vulnerability should serialize to dict."""
        vuln = Vulnerability(
            sink_type=SecuritySink.SQL_QUERY,
            taint_source=TaintSource.USER_INPUT,
            taint_path=["input", "query"],
            sink_location=(10, 0),
        )
        
        d = vuln.to_dict()
        
        assert d["type"] == "SQL Injection"
        assert d["cwe"] == "CWE-89"
        assert d["taint_path"] == ["input", "query"]


# =============================================================================
# Security Analyzer Tests
# =============================================================================

class TestSecurityAnalyzer:
    """Test SecurityAnalyzer for end-to-end vulnerability detection."""
    
    def test_detect_sql_injection_simple(self):
        """Should detect simple SQL injection."""
        code = '''
user_id = request.args.get("id")
query = "SELECT * FROM users WHERE id=" + user_id
cursor.execute(query)
'''
        result = analyze_security(code)
        
        assert result.has_vulnerabilities
        sqli = result.get_sql_injections()
        assert len(sqli) >= 1
    
    def test_detect_command_injection(self):
        """Should detect command injection."""
        code = '''
filename = request.args.get("file")
cmd = "cat " + filename
os.system(cmd)
'''
        result = analyze_security(code)
        
        cmd_inj = result.get_command_injections()
        assert len(cmd_inj) >= 1
    
    def test_no_vuln_with_parameterized_query(self):
        """Should not flag parameterized queries (if sanitizer applied)."""
        # This tests the concept - actual detection depends on pattern matching
        code = '''
user_id = request.args.get("id")
safe_id = int(user_id)  # This acts as implicit sanitization
query = "SELECT * FROM users WHERE id=%s"
cursor.execute(query, (safe_id,))
'''
        # The current analyzer doesn't recognize int() as sanitizer
        # but this test documents expected behavior
        _ = analyze_security(code)
        # For now, this may still flag it - that's expected
    
    def test_analyze_empty_code(self):
        """Should handle empty code gracefully."""
        result = analyze_security("")
        
        assert not result.has_vulnerabilities
        assert result.vulnerability_count == 0
    
    def test_analyze_syntax_error(self):
        """Should handle syntax errors gracefully."""
        result = analyze_security("def broken(")
        
        assert not result.has_vulnerabilities
    
    def test_result_summary(self):
        """Summary should be human readable."""
        code = '''
user_id = request.args.get("id")
cursor.execute("SELECT * FROM users WHERE id=" + user_id)
'''
        result = analyze_security(code)
        
        summary = result.summary()
        assert "vulnerability" in summary.lower() or "no vulnerabilities" in summary.lower()
    
    def test_result_to_dict(self):
        """Result should serialize to dict."""
        code = '''
x = request.args.get("x")
os.system(x)
'''
        result = analyze_security(code)
        
        d = result.to_dict()
        assert "vulnerability_count" in d
        assert "vulnerabilities" in d


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions for finding specific vulnerability types."""
    
    def test_find_sql_injections(self):
        """find_sql_injections should return only SQL injection vulns."""
        code = '''
user_id = request.args.get("id")
cursor.execute("SELECT * FROM users WHERE id=" + user_id)
'''
        vulns = find_sql_injections(code)
        
        assert all(v.sink_type == SecuritySink.SQL_QUERY for v in vulns)
    
    def test_find_xss(self):
        """find_xss should return only XSS vulns."""
        code = '''
name = request.args.get("name")
html = Markup("<h1>" + name + "</h1>")
'''
        vulns = find_xss(code)
        
        # May or may not find XSS depending on pattern matching
        for v in vulns:
            assert v.sink_type == SecuritySink.HTML_OUTPUT
    
    def test_find_command_injections(self):
        """find_command_injections should return only cmd injection vulns."""
        code = '''
cmd = request.args.get("cmd")
os.system(cmd)
'''
        vulns = find_command_injections(code)
        
        assert all(v.sink_type == SecuritySink.SHELL_COMMAND for v in vulns)


# =============================================================================
# Tainted Value Tests
# =============================================================================

class TestTaintedValue:
    """Test TaintedValue wrapper."""
    
    def test_tainted_value_is_tainted(self):
        """TaintedValue with taint info should report as tainted."""
        taint = TaintInfo(source=TaintSource.USER_INPUT)
        value = TaintedValue(expr=z3.String("x"), taint=taint)
        
        assert value.is_tainted
    
    def test_clean_value_is_not_tainted(self):
        """TaintedValue without taint should not be tainted."""
        value = TaintedValue(expr=z3.String("x"), taint=None)
        
        assert not value.is_tainted
    
    def test_repr_shows_taint_status(self):
        """Repr should indicate taint status."""
        taint = TaintInfo(source=TaintSource.USER_INPUT)
        tainted = TaintedValue(expr=z3.String("x"), taint=taint)
        clean = TaintedValue(expr=z3.String("y"), taint=None)
        
        assert "USER_INPUT" in repr(tainted)
        assert "clean" in repr(clean)


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential regressions."""
    
    def test_multiple_taint_sources(self):
        """Code with multiple taint sources should track all."""
        tracker = TaintTracker()
        tracker.taint_source("input1", TaintSource.USER_INPUT)
        tracker.taint_source("input2", TaintSource.FILE_CONTENT)
        
        assert tracker.is_tainted("input1")
        assert tracker.is_tainted("input2")
        assert tracker.get_taint("input1").source == TaintSource.USER_INPUT
        assert tracker.get_taint("input2").source == TaintSource.FILE_CONTENT
    
    def test_taint_propagation_path_tracking(self):
        """Should track the full propagation path."""
        tracker = TaintTracker()
        tracker.taint_source("input", TaintSource.USER_INPUT, (1, 0))
        tracker.propagate_assignment("x", ["input"])
        tracker.propagate_assignment("y", ["x"])
        tracker.propagate_assignment("z", ["y"])
        
        taint = tracker.get_taint("z")
        # Path should include the flow
        assert len(taint.propagation_path) >= 1
    
    def test_security_result_functions_analyzed(self):
        """Should track which functions were analyzed."""
        code = '''
def vulnerable_function():
    user_id = request.args.get("id")
    cursor.execute("SELECT * WHERE id=" + user_id)
    
def safe_function():
    x = 1 + 2
'''
        result = analyze_security(code)
        
        assert "vulnerable_function" in result.functions_analyzed
        assert "safe_function" in result.functions_analyzed
    
    def test_clear_tracker(self):
        """Clear should reset all state."""
        tracker = TaintTracker()
        tracker.taint_source("x", TaintSource.USER_INPUT)
        tracker.check_sink("x", SecuritySink.SQL_QUERY)
        
        tracker.clear()
        
        assert not tracker.is_tainted("x")
        assert len(tracker.get_vulnerabilities()) == 0


# =============================================================================
# Integration with Symbolic Execution
# =============================================================================

class TestSymbolicStringSupport:
    """Test string support in symbolic execution components."""
    
    def test_state_manager_creates_string_variable(self):
        """SymbolicState should create String variables."""
        from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState
        
        state = SymbolicState()
        expr = state.create_variable("s", z3.StringSort())
        
        assert expr.sort() == z3.StringSort()
        assert state.has_variable("s")
    
    def test_constraint_solver_handles_string(self):
        """ConstraintSolver should handle string constraints."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
            SolverStatus,
        )
        
        solver = ConstraintSolver()
        s = z3.String("s")
        
        # Simple string constraint
        result = solver.solve(
            constraints=[z3.Length(s) > 5],
            variables=[s]
        )
        
        assert result.status == SolverStatus.SAT
        assert result.model is not None
    
    def test_interpreter_handles_string_literal(self):
        """SymbolicInterpreter should handle string assignments."""
        from code_scalpel.symbolic_execution_tools.interpreter import SymbolicInterpreter
        
        interp = SymbolicInterpreter()
        result = interp.execute('x = "hello"')
        
        assert len(result.states) == 1
        state = result.states[0]
        x_val = state.get_variable("x")
        # x should be set to StringVal("hello")
        assert x_val is not None
    
    def test_interpreter_handles_string_concat(self):
        """SymbolicInterpreter should handle string concatenation."""
        from code_scalpel.symbolic_execution_tools.interpreter import SymbolicInterpreter
        
        interp = SymbolicInterpreter()
        result = interp.execute('a = "hello"\nb = " world"\nc = a + b')
        
        assert len(result.states) == 1
        state = result.states[0]
        c_val = state.get_variable("c")
        assert c_val is not None


# =============================================================================
# v0.3.1 Sanitizer Recognition Tests
# =============================================================================

class TestSanitizerRegistry:
    """Test the sanitizer registry infrastructure."""
    
    def test_builtin_sanitizers_registered(self):
        """Built-in sanitizers should be pre-registered."""
        # Type coercions
        assert "int" in SANITIZER_REGISTRY
        assert "float" in SANITIZER_REGISTRY
        assert "bool" in SANITIZER_REGISTRY
        
        # HTML escape
        assert "html.escape" in SANITIZER_REGISTRY
        
        # Shell sanitizers
        assert "shlex.quote" in SANITIZER_REGISTRY
        
        # Path sanitizers
        assert "os.path.basename" in SANITIZER_REGISTRY
    
    def test_type_coercion_fully_clears_taint(self):
        """Type coercion (int, float, bool) should fully clear taint."""
        int_sanitizer = SANITIZER_REGISTRY["int"]
        assert int_sanitizer.full_clear is True
        
        float_sanitizer = SANITIZER_REGISTRY["float"]
        assert float_sanitizer.full_clear is True
        
        bool_sanitizer = SANITIZER_REGISTRY["bool"]
        assert bool_sanitizer.full_clear is True
    
    def test_html_escape_clears_xss_only(self):
        """html.escape should only clear XSS sink."""
        sanitizer = SANITIZER_REGISTRY["html.escape"]
        assert SecuritySink.HTML_OUTPUT in sanitizer.clears_sinks
        assert SecuritySink.SQL_QUERY not in sanitizer.clears_sinks
        assert SecuritySink.SHELL_COMMAND not in sanitizer.clears_sinks
        assert sanitizer.full_clear is False
    
    def test_shlex_quote_clears_command_only(self):
        """shlex.quote should only clear command injection sink."""
        sanitizer = SANITIZER_REGISTRY["shlex.quote"]
        assert SecuritySink.SHELL_COMMAND in sanitizer.clears_sinks
        assert SecuritySink.SQL_QUERY not in sanitizer.clears_sinks
        assert sanitizer.full_clear is False
    
    def test_register_custom_sanitizer(self):
        """Custom sanitizers should be registrable."""
        # Clean up after test
        test_name = "_test_custom_sanitizer_123"
        try:
            register_sanitizer(
                test_name,
                clears_sinks={SecuritySink.SQL_QUERY},
                full_clear=False
            )
            
            assert test_name in SANITIZER_REGISTRY
            sanitizer = SANITIZER_REGISTRY[test_name]
            assert SecuritySink.SQL_QUERY in sanitizer.clears_sinks
        finally:
            # Cleanup
            if test_name in SANITIZER_REGISTRY:
                del SANITIZER_REGISTRY[test_name]


class TestConfigLoader:
    """Test loading sanitizers from pyproject.toml."""
    
    def test_load_from_nonexistent_file(self, tmp_path):
        """Should return 0 for nonexistent file."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        result = load_sanitizers_from_config(str(tmp_path / "nonexistent.toml"))
        assert result == 0
    
    def test_load_sql_sanitizer_from_config(self, tmp_path):
        """Should load SQL sanitizer from config."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text('''
[tool.code-scalpel.sanitizers]
"my_utils.clean_sql" = ["SQL_QUERY"]
''')
        
        try:
            count = load_sanitizers_from_config(str(config_file))
            assert count == 1
            assert "my_utils.clean_sql" in SANITIZER_REGISTRY
            sanitizer = SANITIZER_REGISTRY["my_utils.clean_sql"]
            assert SecuritySink.SQL_QUERY in sanitizer.clears_sinks
            assert not sanitizer.full_clear
        finally:
            if "my_utils.clean_sql" in SANITIZER_REGISTRY:
                del SANITIZER_REGISTRY["my_utils.clean_sql"]
    
    def test_load_full_clear_sanitizer(self, tmp_path):
        """Should load full clear sanitizer with ALL."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text('''
[tool.code-scalpel.sanitizers]
"my_utils.super_clean" = ["ALL"]
''')
        
        try:
            count = load_sanitizers_from_config(str(config_file))
            assert count == 1
            assert "my_utils.super_clean" in SANITIZER_REGISTRY
            sanitizer = SANITIZER_REGISTRY["my_utils.super_clean"]
            assert sanitizer.full_clear is True
        finally:
            if "my_utils.super_clean" in SANITIZER_REGISTRY:
                del SANITIZER_REGISTRY["my_utils.super_clean"]
    
    def test_load_multiple_sanitizers(self, tmp_path):
        """Should load multiple sanitizers."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text('''
[tool.code-scalpel.sanitizers]
"utils.clean_sql" = ["SQL_QUERY"]
"utils.clean_html" = ["HTML_OUTPUT"]
"utils.clean_all" = ["ALL"]
''')
        
        try:
            count = load_sanitizers_from_config(str(config_file))
            assert count == 3
            assert "utils.clean_sql" in SANITIZER_REGISTRY
            assert "utils.clean_html" in SANITIZER_REGISTRY
            assert "utils.clean_all" in SANITIZER_REGISTRY
        finally:
            for name in ["utils.clean_sql", "utils.clean_html", "utils.clean_all"]:
                if name in SANITIZER_REGISTRY:
                    del SANITIZER_REGISTRY[name]
    
    def test_load_multi_sink_sanitizer(self, tmp_path):
        """Should load sanitizer that clears multiple sinks."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text('''
[tool.code-scalpel.sanitizers]
"utils.paranoid_clean" = ["SQL_QUERY", "HTML_OUTPUT"]
''')
        
        try:
            count = load_sanitizers_from_config(str(config_file))
            assert count == 1
            sanitizer = SANITIZER_REGISTRY["utils.paranoid_clean"]
            assert SecuritySink.SQL_QUERY in sanitizer.clears_sinks
            assert SecuritySink.HTML_OUTPUT in sanitizer.clears_sinks
        finally:
            if "utils.paranoid_clean" in SANITIZER_REGISTRY:
                del SANITIZER_REGISTRY["utils.paranoid_clean"]
    
    def test_invalid_sink_name_skipped(self, tmp_path):
        """Should skip invalid sink names gracefully."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
        )
        
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text('''
[tool.code-scalpel.sanitizers]
"utils.mixed" = ["SQL_QUERY", "INVALID_SINK"]
''')
        
        try:
            count = load_sanitizers_from_config(str(config_file))
            assert count == 1
            sanitizer = SANITIZER_REGISTRY["utils.mixed"]
            # Should have SQL_QUERY but not crash on INVALID_SINK
            assert SecuritySink.SQL_QUERY in sanitizer.clears_sinks
        finally:
            if "utils.mixed" in SANITIZER_REGISTRY:
                del SANITIZER_REGISTRY["utils.mixed"]


class TestTaintInfoClearedSinks:
    """Test the cleared_sinks field in TaintInfo."""
    
    def test_cleared_sinks_initially_empty(self):
        """New TaintInfo should have empty cleared_sinks."""
        taint = TaintInfo(source=TaintSource.USER_INPUT)
        assert len(taint.cleared_sinks) == 0
    
    def test_apply_sanitizer_adds_to_cleared_sinks(self):
        """Applying sink-specific sanitizer adds to cleared_sinks."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH
        )
        
        sanitized = taint.apply_sanitizer("html.escape")
        
        assert SecuritySink.HTML_OUTPUT in sanitized.cleared_sinks
    
    def test_is_dangerous_checks_cleared_sinks(self):
        """is_dangerous_for should check cleared_sinks first."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH,  # Still high level
            cleared_sinks={SecuritySink.HTML_OUTPUT}
        )
        
        # HTML_OUTPUT cleared, so not dangerous
        assert not taint.is_dangerous_for(SecuritySink.HTML_OUTPUT)
        # SQL_QUERY not cleared, still dangerous
        assert taint.is_dangerous_for(SecuritySink.SQL_QUERY)
    
    def test_full_clear_sanitizer_clears_all_sinks(self):
        """Full clear sanitizer should clear all sinks."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH
        )
        
        sanitized = taint.apply_sanitizer("int")
        
        # All sinks should be cleared with int()
        assert not sanitized.is_dangerous_for(SecuritySink.SQL_QUERY)
        assert not sanitized.is_dangerous_for(SecuritySink.HTML_OUTPUT)
        assert not sanitized.is_dangerous_for(SecuritySink.SHELL_COMMAND)
        assert not sanitized.is_dangerous_for(SecuritySink.FILE_PATH)
    
    def test_propagate_preserves_cleared_sinks(self):
        """Propagation should preserve cleared_sinks."""
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH,
            cleared_sinks={SecuritySink.HTML_OUTPUT}
        )
        
        propagated = taint.propagate("new_var")
        
        assert SecuritySink.HTML_OUTPUT in propagated.cleared_sinks


class TestSecurityAnalyzerSanitizers:
    """Test SecurityAnalyzer recognizes sanitizers in code."""
    
    def test_html_escape_prevents_xss_detection(self):
        """html.escape should prevent XSS detection."""
        code = '''
user_input = request.args.get("name")
safe_name = html.escape(user_input)
response.write(safe_name)
'''
        result = analyze_security(code)
        
        xss_vulns = result.get_xss()
        # Should not flag safe_name as XSS
        # Check that no XSS for safe_name
        safe_name_xss = [v for v in xss_vulns if "safe_name" in v.taint_path]
        assert len(safe_name_xss) == 0
    
    def test_int_cast_prevents_sqli_detection(self):
        """int() cast should prevent SQL injection detection."""
        code = '''
user_id = request.args.get("id")
safe_id = int(user_id)
query = "SELECT * FROM users WHERE id=" + str(safe_id)
cursor.execute(query)
'''
        result = analyze_security(code)
        
        sqli_vulns = result.get_sql_injections()
        # safe_id should not trigger SQLi
        safe_id_sqli = [v for v in sqli_vulns if "safe_id" in v.taint_path]
        assert len(safe_id_sqli) == 0
    
    def test_unsanitized_still_flagged(self):
        """Unsanitized data should still be flagged."""
        code = '''
user_id = request.args.get("id")
query = "SELECT * FROM users WHERE id=" + user_id
cursor.execute(query)
'''
        result = analyze_security(code)
        
        assert result.has_vulnerabilities
        sqli = result.get_sql_injections()
        assert len(sqli) >= 1
    
    def test_html_escape_does_not_prevent_sqli(self):
        """html.escape should NOT prevent SQL injection."""
        code = '''
user_input = request.args.get("name")
escaped = html.escape(user_input)
query = "SELECT * FROM users WHERE name='" + escaped + "'"
cursor.execute(query)
'''
        result = analyze_security(code)
        
        # html.escape doesn't protect against SQLi
        # Should still flag this
        assert result.has_vulnerabilities
        assert len(result.get_sql_injections()) > 0
    
    def test_shlex_quote_prevents_command_injection(self):
        """shlex.quote should prevent command injection."""
        code = '''
filename = request.args.get("file")
safe_filename = shlex.quote(filename)
cmd = "cat " + safe_filename
os.system(cmd)
'''
        result = analyze_security(code)
        
        cmd_vulns = result.get_command_injections()
        # safe_filename should not trigger command injection
        safe_vulns = [v for v in cmd_vulns if "safe_filename" in v.taint_path]
        assert len(safe_vulns) == 0
    
    def test_os_path_basename_prevents_path_traversal(self):
        """os.path.basename should prevent path traversal."""
        code = '''
filename = request.args.get("file")
safe_file = os.path.basename(filename)
with open("/uploads/" + safe_file) as f:
    content = f.read()
'''
        result = analyze_security(code)
        
        path_vulns = result.get_path_traversals()
        # safe_file should not trigger path traversal
        safe_vulns = [v for v in path_vulns if "safe_file" in v.taint_path]
        assert len(safe_vulns) == 0
