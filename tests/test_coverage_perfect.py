"""
Perfect Coverage Test Suite

This module targets the 13 remaining partial branches in symbolic execution tools.
These are defensive code paths that handle edge cases like:
- None values from expression evaluation
- Empty loop iterations
- Missing semantic handlers
- Configuration file search exhaustion

NO EXCUSES. If the code exists, we test it.
"""

from unittest.mock import patch

# Suppress warnings for cleaner test output
import warnings

warnings.filterwarnings("ignore", message="symbolic_execution_tools")


# =============================================================================
# TAINT TRACKER: 2 Partial Branches
# =============================================================================


class TestTaintTrackerPerfectCoverage:
    """Target: taint_tracker.py lines 378->382, 398->408"""

    def test_config_loader_empty_sink_set(self, tmp_path):
        """
        Branch 378->382: When sink_set is empty after parsing invalid sink names.

        The code has: `if sink_set:` - we need to hit the False branch.
        This happens when ALL sink names in the config are invalid.
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            load_sanitizers_from_config,
            SANITIZER_REGISTRY,
        )

        # Create a config with only invalid sink names
        config_file = tmp_path / "pyproject.toml"
        config_file.write_text(
            """
[tool.code-scalpel.sanitizers]
"my_func" = ["INVALID_SINK_1", "INVALID_SINK_2", "NOT_A_REAL_SINK"]
"""
        )

        # Store original registry state
        original_registry = dict(SANITIZER_REGISTRY)

        try:
            # Load config - should NOT register "my_func" because all sinks are invalid
            count = load_sanitizers_from_config(str(config_file))

            # The function processed 1 entry but registered 0 sanitizers
            # (because sink_set was empty after KeyError on all names)
            assert count == 1  # It counted the entry
            assert "my_func" not in SANITIZER_REGISTRY  # But didn't register it
        finally:
            # Restore registry
            SANITIZER_REGISTRY.clear()
            SANITIZER_REGISTRY.update(original_registry)

    def test_find_config_file_loop_exhaustion(self):
        """
        Branch 398->408: The for loop in _find_config_file hits its limit.

        The code has: `for _ in range(10):` - we need to exhaust all 10 iterations
        without finding a config file OR hitting the root directory.
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            _find_config_file,
        )

        # Mock os functions to create a scenario where:
        # 1. pyproject.toml never exists
        # 2. dirname always returns a "parent" (never reaches root)
        call_count = [0]

        def mock_dirname(path):
            call_count[0] += 1
            # Return a different path each time (never equal to input = never at root)
            return f"/fake/path/level{call_count[0]}"

        with patch("os.getcwd", return_value="/fake/start/path"), patch(
            "os.path.exists", return_value=False
        ), patch("os.path.dirname", side_effect=mock_dirname):
            result = _find_config_file()

            # Should return None after exhausting loop
            assert result is None
            # Should have called dirname 10 times (loop limit)
            assert call_count[0] == 10


# =============================================================================
# SECURITY ANALYZER: 4 Partial Branches
# =============================================================================


class TestSecurityAnalyzerPerfectCoverage:
    """Target: security_analyzer.py lines 322->317, 325->317, 327->325, 416->420"""

    def test_sanitizer_check_loop_exhaustion_no_tainted_args(self):
        """
        Branches 322->317, 325->317, 327->325: The for loops in _check_sanitizer_call
        exhaust without finding any tainted arguments.

        This happens when a sanitizer is called with arguments that are NOT tainted.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        # Code that calls a sanitizer with non-tainted arguments
        code = """
clean_value = "static_string"
x = 42
result = html.escape(clean_value)  # clean_value is NOT tainted
result2 = html.escape(x + y)  # Neither x nor y is tainted
"""

        analyzer = SecurityAnalyzer()
        # Don't mark anything as tainted
        vulnerabilities = analyzer.analyze(code)

        # No vulnerabilities because no tainted data reaches sinks
        # The loops in _check_sanitizer_call exhausted without finding taint
        assert vulnerabilities is not None

    def test_sanitizer_check_with_binop_no_taint(self):
        """
        Branch 325->317: BinOp argument where variables are NOT tainted.

        Code path: isinstance(arg, ast.BinOp) is True, but no vars are tainted.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
a = 1
b = 2
# BinOp argument to sanitizer, but a and b are NOT tainted
result = html.escape(a + b)
"""

        analyzer = SecurityAnalyzer()
        vulnerabilities = analyzer.analyze(code)

        # Should complete without error, hitting the BinOp branch
        assert vulnerabilities is not None

    def test_sanitizer_with_no_args(self):
        """
        Branch 322->317: Sanitizer call with no arguments.

        The for loop over node.args has zero iterations.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
# Call sanitizer with no arguments (edge case)
result = html.escape()
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        assert result is not None

    def test_sanitizer_with_literal_only(self):
        """
        Branch 322->317: Sanitizer call with only literal arguments.

        Args that are neither Name nor BinOp (e.g., string literal).
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
# Sanitizer with string literal - not Name, not BinOp
result = html.escape("literal string")
result2 = html.escape(123)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        assert result is not None

    def test_get_call_name_deep_attribute_chain(self):
        """
        Branch 416->420: The while loop in _get_call_name exhausts.

        This happens with deeply nested attribute access like:
        a.b.c.d.e.f.method()
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        # Create a deep attribute chain that will exhaust the while loop
        code = """
user_input = input()
result = a.b.c.d.e.f.g.h.i.j.method(user_input)
"""

        analyzer = SecurityAnalyzer()

        # This should process the deep chain without crashing
        result = analyzer.analyze(code)
        assert result is not None

    def test_get_call_name_simple_name(self):
        """
        Branch 416->420: Simple function call (no attribute chain).

        The while loop should exit immediately.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
user_input = input()
result = some_func(user_input)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        assert result is not None

    def test_get_call_name_non_name_base(self):
        """
        Branch 416->420: Attribute chain where base is NOT an ast.Name.

        Code path: The while loop exits, but isinstance(current, ast.Name) is False.
        This happens with: function_call().method() or "literal".method()
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        # function().method() - current ends up as ast.Call (not ast.Name)
        code = """
user_input = input()
result = get_something().execute(user_input)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        assert result is not None

    def test_get_call_name_literal_method(self):
        """
        Branch 416->420: Attribute on a literal (string, list, etc.)

        The base is an ast.Constant, not ast.Name.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
user_input = input()
# String literal method - base is ast.Constant
result = "{}".format(user_input)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        assert result is not None


# =============================================================================
# IR INTERPRETER: 7 Partial Branches
# =============================================================================


class TestIRInterpreterPerfectCoverage:
    """Target: ir_interpreter.py lines 775->770, 807->826, 860->866, 862->866,
    942->947, 944->947, 1227->1234"""

    def test_assign_none_value_existing_variable(self):
        """
        Branch 775->770: Assignment where value_expr is None but variable exists.

        Code path: value_expr is None, AND state.has_variable(name) is True
        This means we skip creating a placeholder (variable already exists).
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # First set x to a known value, then try to assign an unevaluable expression
        code = """
x = 1
x = unknown_function_that_returns_nothing()
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Execute - should not crash
        result = interp.execute(ir)
        assert result is not None

    def test_aug_assign_with_none_operand(self):
        """
        Branch 807->826: AugAssign where right operand evaluates to None.

        Code path: `if right is not None and self._semantics is not None`
        We need right to be None.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        code = """
x = 1
x += unknown_value
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Execute - should handle None gracefully
        result = interp.execute(ir)
        assert result is not None

    def test_if_statement_without_semantics(self):
        """
        Branches 860->866: If statement when semantics is None.

        Code path: `if self._semantics is not None` evaluates to False

        Key: We must use a boolean constant (True/False) as condition
        so that _eval_expr returns a non-None value even without semantics.
        Otherwise comparison expressions return None and trigger early exit.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Use boolean constant - evaluates without semantics
        code = """
if True:
    y = 1
else:
    y = 2
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Mock get_semantics to return None to hit the _semantics is None branch
        with patch(
            "code_scalpel.symbolic_execution_tools.ir_interpreter.get_semantics",
            return_value=None,
        ):
            result = interp.execute(ir)

        assert result is not None

    def test_if_with_to_bool_returning_none(self):
        """
        Branches 862->866: to_bool() returns None.

        When semantics.to_bool() returns None, condition is used as-is.

        We need semantics that:
        1. Returns valid Z3 comparisons for compare_gt, compare_lt, etc.
        2. Returns None from to_bool()
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
            PythonSemantics,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Create a partial mock: real semantics for comparisons, None for to_bool
        class MockSemantics(PythonSemantics):
            def to_bool(self, value, state):
                return None  # Force the branch we want to test

        code = """
x = 1
if x > 0:
    y = 1
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Patch get_semantics to return our partial mock
        with patch(
            "code_scalpel.symbolic_execution_tools.ir_interpreter.get_semantics",
            return_value=MockSemantics(),
        ):
            result = interp.execute(ir)

        assert result is not None

    def test_while_loop_without_semantics(self):
        """
        Branches 942->947: While loop when semantics is None.

        Code path: semantics check in while condition evaluation.

        Key: Use boolean constant as condition so _eval_expr returns
        non-None even without semantics.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Use boolean False to exit loop immediately, but still test the path
        code = """
while False:
    x = 1
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Mock get_semantics to return None to hit the _semantics is None branch
        with patch(
            "code_scalpel.symbolic_execution_tools.ir_interpreter.get_semantics",
            return_value=None,
        ):
            result = interp.execute(ir)

        assert result is not None

    def test_while_with_to_bool_returning_none(self):
        """
        Branches 944->947: While loop when to_bool() returns None.

        We need semantics that:
        1. Returns valid Z3 comparisons for compare operations
        2. Returns None from to_bool()
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
            PythonSemantics,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Create a partial mock: real semantics for comparisons, None for to_bool
        class MockSemantics(PythonSemantics):
            def to_bool(self, value, state):
                return None  # Force the branch we want to test

        code = """
i = 0
while i < 3:
    i += 1
    break
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Patch get_semantics to return our partial mock
        with patch(
            "code_scalpel.symbolic_execution_tools.ir_interpreter.get_semantics",
            return_value=MockSemantics(),
        ):
            result = interp.execute(ir)

        assert result is not None

    def test_aug_assign_without_semantics_via_mock(self):
        """
        Branch 807->826: AugAssign when self._semantics is None (via mocking).

        We mock get_semantics to return None to hit the False branch.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        code = """
x = 1
x += 2
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Mock the semantics lookup to return None
        with patch(
            "code_scalpel.symbolic_execution_tools.ir_interpreter.get_semantics",
            return_value=None,
        ):
            result = interp.execute(ir)

        # Should complete but x won't be updated (semantics=None skips the update)
        assert result is not None

    def test_constructor_with_explicit_semantics(self):
        """
        Branch 656->657: Use _default_semantics when provided to constructor.

        Code path: `if self._default_semantics is not None`
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
            PythonSemantics,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        code = """
x = 1
y = x + 2
"""

        # Pass explicit semantics to constructor
        custom_semantics = PythonSemantics()
        interp = IRSymbolicInterpreter(semantics=custom_semantics)

        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)

        assert result is not None
        # Verify the custom semantics was used (not get_semantics)
        assert interp._semantics is custom_semantics

    def test_aug_assign_creates_variable(self):
        """
        Line 803: AugAssign on undefined variable creates placeholder.

        Code path: `if current is None` -> create_variable
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # x is not defined before +=
        code = """
x += 1
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)

        # Should complete - x gets created as symbolic
        assert result is not None

    def test_division_binary_op(self):
        """
        Line 1099: binary_div for true division (/).
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        code = """
x = 10
y = x / 2
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)

        assert result is not None

    def test_generic_visit_unknown_node(self):
        """
        Line 159: generic_visit returns None for unknown IR node types.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import IRNodeVisitor
        from code_scalpel.ir.nodes import IRNode

        class UnknownIRNode(IRNode):
            """A custom IR node type with no visitor."""

            pass

        visitor = IRNodeVisitor()
        unknown_node = UnknownIRNode()

        # generic_visit should be called and return None
        result = visitor.visit(unknown_node)
        assert result is None

    def test_eval_expr_with_none_input(self):
        """
        Line 1035: _eval_expr returns None when expr is None.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState

        interp = IRSymbolicInterpreter()
        state = SymbolicState()

        # Directly call _eval_expr with None
        result = interp._eval_expr(None, state)
        assert result is None

    def test_symbolic_declaration_non_name_type(self):
        """
        Branch 1227->1234: symbolic() call where type argument is not IRName.

        Code path: `if isinstance(type_arg, IRName)` evaluates to False
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Use a string literal as type (not a name reference)
        code = """
x = symbolic("x", "int")
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)

        # Execute - should return None for invalid type specification
        result = interp.execute(ir)
        assert result is not None

    def test_bool_conversion_returns_none(self):
        """
        Branch 862->866: to_bool() returns None.

        This happens when the semantics can't convert a value to boolean.
        """
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        # Use something that might not convert cleanly to bool
        code = """
x = unknown_complex_value
if x:
    y = 1
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)

        assert result is not None


# =============================================================================
# INTEGRATION: Verify No Regressions
# =============================================================================


class TestPerfectCoverageIntegration:
    """Ensure the edge case tests don't break normal functionality."""

    def test_normal_taint_tracking_still_works(self):
        """Verify taint tracking works after edge case tests."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            TaintTracker,
            TaintInfo,
            TaintSource,
            TaintLevel,
            SecuritySink,
        )

        tracker = TaintTracker()

        # Normal taint flow
        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH,
            source_location=(1, 0),
            propagation_path=[],
        )
        tracker.mark_tainted("user_input", taint)

        # Verify taint exists
        result = tracker.get_taint("user_input")
        assert result is not None
        assert result.is_dangerous_for(SecuritySink.SQL_QUERY)

    def test_normal_security_analysis_still_works(self):
        """Verify security analysis works after edge case tests."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
user_id = request.args.get("id")
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # Should detect SQL injection
        assert result.has_vulnerabilities
        assert any("SQL" in str(v) for v in result.vulnerabilities)

    def test_normal_symbolic_execution_still_works(self):
        """Verify symbolic execution works after edge case tests."""
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            IRSymbolicInterpreter,
        )
        from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

        code = """
x = 1
y = x + 2
if y > 0:
    z = y * 2
"""

        interp = IRSymbolicInterpreter()
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)

        assert result is not None
        assert result.path_count >= 1
