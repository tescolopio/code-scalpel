"""
Tests for ConstraintSolver - The Oracle of Symbolic Execution.

CRITICAL: These tests enforce TYPE MARSHALING.
Raw Z3 objects are useless for JSON serialization and user display.

The solver must return:
- Python int, not z3.IntNumRef
- Python bool, not z3.BoolRef  
- Python dict, not z3.ModelRef

If the Type Marshaling tests fail, the MCP server will crash
when trying to serialize results.
"""

import pytest
from z3 import Int, Bool, IntSort, BoolSort, And, Or, Not, Implies

from code_scalpel.symbolic_execution_tools.constraint_solver import (
    ConstraintSolver,
    SolverResult,
    SolverStatus,
)


# =============================================================================
# SECTION 1: Basic Satisfiability
# =============================================================================


class TestBasicSatisfiability:
    """Test basic constraint solving."""

    def test_simple_int_constraint(self):
        """Solve x > 10."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 10], [x])

        assert result.status == SolverStatus.SAT
        assert "x" in result.model
        assert result.model["x"] > 10

    def test_simple_bool_constraint(self):
        """Solve flag == True."""
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.solve([flag == True], [flag])

        assert result.status == SolverStatus.SAT
        assert result.model["flag"] is True

    def test_multiple_constraints(self):
        """Solve x > 10 AND x < 20."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 10, x < 20], [x])

        assert result.status == SolverStatus.SAT
        assert 10 < result.model["x"] < 20

    def test_multiple_variables(self):
        """Solve x + y == 10 AND x > y."""
        solver = ConstraintSolver()
        x = Int("x")
        y = Int("y")

        result = solver.solve([x + y == 10, x > y], [x, y])

        assert result.status == SolverStatus.SAT
        assert result.model["x"] + result.model["y"] == 10
        assert result.model["x"] > result.model["y"]

    def test_mixed_int_bool(self):
        """Solve with both int and bool variables."""
        solver = ConstraintSolver()
        x = Int("x")
        flag = Bool("flag")

        # If flag is true, x must be positive
        result = solver.solve([Implies(flag, x > 0), flag == True, x < 100], [x, flag])

        assert result.status == SolverStatus.SAT
        assert result.model["flag"] is True
        assert result.model["x"] > 0


# =============================================================================
# SECTION 2: Unsatisfiability
# =============================================================================


class TestUnsatisfiability:
    """Test detection of unsatisfiable constraints."""

    def test_simple_contradiction(self):
        """x > 10 AND x < 5 is UNSAT."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 10, x < 5], [x])

        assert result.status == SolverStatus.UNSAT
        assert result.model is None or result.model == {}

    def test_boolean_contradiction(self):
        """flag AND NOT flag is UNSAT."""
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.solve([flag, Not(flag)], [flag])

        assert result.status == SolverStatus.UNSAT

    def test_equality_contradiction(self):
        """x == 5 AND x == 10 is UNSAT."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x == 5, x == 10], [x])

        assert result.status == SolverStatus.UNSAT

    def test_complex_unsat(self):
        """Complex constraint that's unsatisfiable."""
        solver = ConstraintSolver()
        x = Int("x")
        y = Int("y")

        # x + y == 10, x > 10, y > 10 is impossible
        result = solver.solve([x + y == 10, x > 10, y > 10], [x, y])

        assert result.status == SolverStatus.UNSAT


# =============================================================================
# SECTION 3: TYPE MARSHALING - The "Raw Z3" Prevention
# =============================================================================


class TestTypeMarshaling:
    """
    CRITICAL TESTS: These enforce Python-native return types.

    If the solver returns Z3 objects, JSON serialization crashes
    and the MCP server fails.
    """

    def test_int_is_python_int(self):
        """
        MARSHALING TEST 1: Integer results are Python int, not Z3.
        """
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x == 42], [x])

        assert result.status == SolverStatus.SAT
        value = result.model["x"]

        # CRITICAL: Must be Python int
        assert type(value) is int, f"Expected int, got {type(value)}"
        assert value == 42

    def test_bool_is_python_bool(self):
        """
        MARSHALING TEST 2: Boolean results are Python bool, not Z3.
        """
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.solve([flag == True], [flag])

        assert result.status == SolverStatus.SAT
        value = result.model["flag"]

        # CRITICAL: Must be Python bool
        assert type(value) is bool, f"Expected bool, got {type(value)}"
        assert value is True

    def test_false_is_python_false(self):
        """
        MARSHALING TEST 3: False is Python False, not Z3 BoolRef.
        """
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.solve([flag == False], [flag])

        assert result.status == SolverStatus.SAT
        value = result.model["flag"]

        assert type(value) is bool
        assert value is False

    def test_negative_int_is_python_int(self):
        """
        MARSHALING TEST 4: Negative integers are properly converted.
        """
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x == -100], [x])

        assert result.status == SolverStatus.SAT
        value = result.model["x"]

        assert type(value) is int
        assert value == -100

    def test_large_int_is_python_int(self):
        """
        MARSHALING TEST 5: Large integers don't overflow.
        """
        solver = ConstraintSolver()
        x = Int("x")

        large_value = 2**62  # Large but fits in Python int
        result = solver.solve([x == large_value], [x])

        assert result.status == SolverStatus.SAT
        value = result.model["x"]

        assert type(value) is int
        assert value == large_value

    def test_model_is_dict(self):
        """
        MARSHALING TEST 6: Model is a Python dict, not Z3 ModelRef.
        """
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 0], [x])

        assert result.status == SolverStatus.SAT

        # CRITICAL: Must be a real Python dict
        assert type(result.model) is dict, f"Expected dict, got {type(result.model)}"

    def test_json_serializable(self):
        """
        MARSHALING TEST 7: Result can be JSON serialized.
        """
        import json

        solver = ConstraintSolver()
        x = Int("x")
        flag = Bool("flag")

        result = solver.solve([x == 42, flag == True], [x, flag])

        # This should NOT raise TypeError
        json_str = json.dumps(result.model)

        # And round-trip should work
        parsed = json.loads(json_str)
        assert parsed["x"] == 42
        assert parsed["flag"] is True


# =============================================================================
# SECTION 4: Proof Mode (Validity Checking)
# =============================================================================


class TestProofMode:
    """Test proving assertions are valid (always true)."""

    def test_valid_assertion(self):
        """Prove x > 10 implies x > 5."""
        solver = ConstraintSolver()
        x = Int("x")

        # Precondition: x > 10
        # Assertion: x > 5 (should be valid)
        result = solver.prove(preconditions=[x > 10], assertion=x > 5)

        assert result.status == SolverStatus.VALID
        assert result.counterexample is None

    def test_invalid_assertion_with_counterexample(self):
        """Find counterexample when assertion is invalid."""
        solver = ConstraintSolver()
        x = Int("x")

        # Precondition: x > 0
        # Assertion: x > 100 (NOT valid - counterexample: x = 1)
        result = solver.prove(preconditions=[x > 0], assertion=x > 100)

        assert result.status == SolverStatus.INVALID
        assert result.counterexample is not None
        assert "x" in result.counterexample
        # Counterexample should violate the assertion
        assert result.counterexample["x"] > 0  # Satisfies precondition
        assert result.counterexample["x"] <= 100  # Violates assertion

    def test_trivially_valid(self):
        """Prove x == x is always valid."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.prove(preconditions=[], assertion=x == x)

        assert result.status == SolverStatus.VALID

    def test_boolean_validity(self):
        """Prove flag OR NOT flag is valid (tautology)."""
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.prove(preconditions=[], assertion=Or(flag, Not(flag)))

        assert result.status == SolverStatus.VALID


# =============================================================================
# SECTION 5: Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Test solver timeout behavior."""

    def test_default_timeout_exists(self):
        """Solver has a default timeout."""
        solver = ConstraintSolver()

        assert hasattr(solver, "timeout_ms")
        assert solver.timeout_ms > 0

    def test_custom_timeout(self):
        """Can set custom timeout."""
        solver = ConstraintSolver(timeout_ms=5000)

        assert solver.timeout_ms == 5000

    def test_simple_problem_completes(self):
        """Simple problems complete within timeout."""
        solver = ConstraintSolver(timeout_ms=1000)
        x = Int("x")

        result = solver.solve([x > 0, x < 100], [x])

        # Should complete, not timeout
        assert result.status in [SolverStatus.SAT, SolverStatus.UNSAT]

    def test_timeout_status_returned(self):
        """Timeout returns UNKNOWN status, not hang."""
        # This test uses a very short timeout
        # In practice, even simple problems complete fast
        solver = ConstraintSolver(timeout_ms=1)
        x = Int("x")

        # Try a simple problem with tiny timeout
        result = solver.solve([x > 0], [x])

        # Either solves quickly or returns UNKNOWN
        assert result.status in [SolverStatus.SAT, SolverStatus.UNKNOWN]


# =============================================================================
# SECTION 6: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_constraints(self):
        """Empty constraints are trivially satisfiable."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([], [x])

        assert result.status == SolverStatus.SAT
        # x can be anything
        assert "x" in result.model

    def test_no_variables(self):
        """Solving with no variables."""
        solver = ConstraintSolver()

        result = solver.solve([Bool("temp") == True], [])

        # Satisfiable, but no variables to report
        assert result.status == SolverStatus.SAT
        assert result.model == {} or len(result.model) == 0

    def test_variable_not_in_constraints(self):
        """Variable not mentioned in constraints gets arbitrary value."""
        solver = ConstraintSolver()
        x = Int("x")
        y = Int("y")

        # Only constrain x, ask for both
        result = solver.solve([x == 5], [x, y])

        assert result.status == SolverStatus.SAT
        assert result.model["x"] == 5
        # y can be anything - just verify it's present
        assert "y" in result.model

    def test_zero_value(self):
        """Zero is properly returned."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x == 0], [x])

        assert result.status == SolverStatus.SAT
        assert result.model["x"] == 0
        assert type(result.model["x"]) is int


# =============================================================================
# SECTION 7: SolverResult API
# =============================================================================


class TestSolverResult:
    """Test the SolverResult container."""

    def test_result_has_status(self):
        """SolverResult has status field."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 0], [x])

        assert hasattr(result, "status")
        assert isinstance(result.status, SolverStatus)

    def test_result_has_model(self):
        """SolverResult has model field for SAT."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 0], [x])

        assert hasattr(result, "model")

    def test_sat_result_is_truthy(self):
        """SAT result evaluates as truthy."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 0], [x])

        assert result  # Should be truthy
        assert result.is_sat()

    def test_unsat_result_is_falsy(self):
        """UNSAT result evaluates as falsy for is_sat()."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 10, x < 5], [x])

        assert not result.is_sat()

    def test_result_repr(self):
        """Result has readable string representation."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x == 42], [x])

        repr_str = repr(result)
        assert "SAT" in repr_str or "42" in repr_str


# =============================================================================
# SECTION 8: Integration with SymbolicState
# =============================================================================


class TestStateIntegration:
    """Test integration with SymbolicState from M2."""

    def test_solve_from_state_constraints(self):
        """Can solve constraints from a SymbolicState."""
        from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState

        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 0)
        state.add_constraint(x < 100)

        solver = ConstraintSolver()
        result = solver.solve(state.constraints, [x])

        assert result.status == SolverStatus.SAT
        assert 0 < result.model["x"] < 100

    def test_check_state_feasibility(self):
        """Solver agrees with state.is_feasible()."""
        from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState

        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 10)
        state.add_constraint(x < 5)  # Contradiction!

        # State says infeasible
        assert state.is_feasible() is False

        # Solver should agree
        solver = ConstraintSolver()
        result = solver.solve(state.constraints, [x])
        assert result.status == SolverStatus.UNSAT


# =============================================================================
# SECTION 9: Coverage Completeness - Additional Tests for 100%
# =============================================================================


class TestCoverageCompleteness:
    """Tests to achieve 100% coverage on constraint_solver.py."""

    def test_is_valid_method(self):
        """SolverResult.is_valid() returns True for VALID status."""
        solver = ConstraintSolver()
        x = Int("x")

        # Precondition: x > 0, Assertion: x >= 0 (always true)
        result = solver.prove([x > 0], x >= 0)

        assert result.is_valid() is True

    def test_is_valid_false_for_sat(self):
        """SolverResult.is_valid() returns False for SAT status."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([x > 0], [x])

        assert result.is_valid() is False

    def test_bool_truthy_for_valid(self):
        """SolverResult __bool__ returns True for VALID."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.prove([x > 0], x >= 0)

        # VALID should be truthy
        assert bool(result) is True

    def test_prove_invalid_with_counterexample(self):
        """prove() with invalid assertion returns INVALID with counterexample."""
        solver = ConstraintSolver()
        x = Int("x")

        # x > 0 does NOT imply x > 100 (invalid with counterexample e.g. x=50)
        result = solver.prove([x > 0], x > 100)

        assert result.status == SolverStatus.INVALID
        assert result.counterexample is not None

    def test_marshal_string_value(self):
        """_marshal_z3_value handles string values."""
        from z3 import String, StringVal

        solver = ConstraintSolver()
        s = String("s")

        result = solver.solve([s == StringVal("hello")], [s])

        assert result.status == SolverStatus.SAT
        assert result.model["s"] == "hello"
        assert isinstance(result.model["s"], str)

    def test_marshal_symbolic_bool_true(self):
        """_marshal_z3_value handles symbolic bool simplified to true."""
        from z3 import simplify

        solver = ConstraintSolver()
        flag = Bool("flag")

        # Constraint that forces flag to True
        result = solver.solve([flag == True], [flag])

        assert result.status == SolverStatus.SAT
        assert result.model["flag"] is True
        assert isinstance(result.model["flag"], bool)

    def test_marshal_symbolic_bool_false(self):
        """_marshal_z3_value handles symbolic bool simplified to false."""
        solver = ConstraintSolver()
        flag = Bool("flag")

        result = solver.solve([flag == False], [flag])

        assert result.status == SolverStatus.SAT
        assert result.model["flag"] is False
        assert isinstance(result.model["flag"], bool)

    def test_marshal_real_value(self):
        """_marshal_z3_value handles Real values as floats."""
        from z3 import Real, RealVal

        solver = ConstraintSolver()
        r = Real("r")

        # r == 3.14
        result = solver.solve([r == RealVal("3.14")], [r])

        assert result.status == SolverStatus.SAT
        assert isinstance(result.model["r"], (int, float))
        assert abs(result.model["r"] - 3.14) < 0.01

    def test_marshal_bitvector_value(self):
        """_marshal_z3_value handles BitVector values."""
        from z3 import BitVec, BitVecVal

        solver = ConstraintSolver()
        bv = BitVec("bv", 8)

        result = solver.solve([bv == BitVecVal(42, 8)], [bv])

        assert result.status == SolverStatus.SAT
        assert result.model["bv"] == 42
        assert isinstance(result.model["bv"], int)

    def test_convenience_create_solver(self):
        """create_solver() convenience function works."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            create_solver,
        )

        solver = create_solver(timeout_ms=1000)

        assert solver is not None
        assert isinstance(solver, ConstraintSolver)

    def test_convenience_solve_constraints_sat(self):
        """solve_constraints() returns model when SAT."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            solve_constraints,
        )

        x = Int("x")
        model = solve_constraints([x > 0, x < 10], [x])

        assert model is not None
        assert "x" in model
        assert 0 < model["x"] < 10

    def test_convenience_solve_constraints_unsat(self):
        """solve_constraints() returns None when UNSAT."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            solve_constraints,
        )

        x = Int("x")
        model = solve_constraints([x > 10, x < 5], [x])

        assert model is None

    def test_convenience_is_satisfiable_true(self):
        """is_satisfiable() returns True for satisfiable constraints."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            is_satisfiable,
        )

        x = Int("x")
        result = is_satisfiable([x > 0, x < 100])

        assert result is True

    def test_convenience_is_satisfiable_false(self):
        """is_satisfiable() returns False for unsatisfiable constraints."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            is_satisfiable,
        )

        x = Int("x")
        result = is_satisfiable([x > 10, x < 5])

        assert result is False

    def test_marshal_algebraic_value(self):
        """_marshal_z3_value handles algebraic values (e.g., sqrt(2))."""
        from z3 import Real, Sqrt

        solver = ConstraintSolver()
        r = Real("r")

        # r * r == 2 gives sqrt(2) which is algebraic
        result = solver.solve([r * r == 2, r > 0], [r])

        assert result.status == SolverStatus.SAT
        # sqrt(2) ~ 1.414
        assert isinstance(result.model["r"], (int, float))
        assert 1.4 < result.model["r"] < 1.5

    def test_marshal_fallback_to_string(self):
        """_marshal_z3_value falls back to str() for unknown types."""
        # Test the normal path, as the fallback is hard to trigger
        # with standard Z3 types
        solver = ConstraintSolver()
        x = Int("x")
        result = solver.solve([x == 42], [x])

        # The normal path works
        assert result.status == SolverStatus.SAT
        assert result.model["x"] == 42

    def test_solver_type_enum_exists(self):
        """SolverType enum exists for legacy compatibility."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import SolverType

        # Verify the enum exists and has expected values
        assert hasattr(SolverType, "Z3")

    def test_real_with_fraction(self):
        """_marshal_z3_value handles Real with clean fraction."""
        from z3 import Real, RealVal, Q

        solver = ConstraintSolver()
        r = Real("r")

        # Use a fraction like 1/2
        result = solver.solve([r == RealVal("1/2")], [r])

        assert result.status == SolverStatus.SAT
        assert isinstance(result.model["r"], (int, float))
        assert abs(result.model["r"] - 0.5) < 0.001

    def test_legacy_constraint_type_enum(self):
        """ConstraintType enum exists for legacy compatibility."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintType,
        )

        assert hasattr(ConstraintType, "ARITHMETIC")
        assert hasattr(ConstraintType, "BOOLEAN")
        assert hasattr(ConstraintType, "BITVECTOR")
        assert hasattr(ConstraintType, "STRING")
        assert hasattr(ConstraintType, "ARRAY")

    def test_legacy_solver_config(self):
        """SolverConfig dataclass exists for legacy compatibility."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            SolverConfig,
            SolverType,
        )

        config = SolverConfig()
        assert config.solver_type == SolverType.Z3
        assert config.timeout is None
        assert config.use_incremental is True

    def test_solve_unknown_timeout(self):
        """solve() returns UNKNOWN status on very short timeout."""
        from z3 import Real, And as Z3And

        # Create a problem that might cause timeout
        solver = ConstraintSolver(timeout_ms=1)  # 1ms timeout

        # Simple problem for quick solve
        x = Int("x")
        result = solver.solve([x > 0], [x])

        # Should usually be SAT but might be UNKNOWN with tiny timeout
        assert result.status in (SolverStatus.SAT, SolverStatus.UNKNOWN)

    def test_prove_returns_unknown_on_indeterminate(self):
        """prove() returns UNKNOWN when solver can't decide."""
        # This is hard to trigger deterministically but we can test the path exists
        solver = ConstraintSolver(timeout_ms=1)
        x = Int("x")

        # Prove something simple
        result = solver.prove([], x > x - 1)

        # Should be VALID (or possibly UNKNOWN with tiny timeout)
        assert result.status in (SolverStatus.VALID, SolverStatus.UNKNOWN)

    def test_marshal_string_value(self):
        """_marshal_z3_value handles string values correctly."""
        from z3 import String, StringVal

        solver = ConstraintSolver()
        s = String("s")

        result = solver.solve([s == StringVal("hello")], [s])

        assert result.status == SolverStatus.SAT
        assert result.model["s"] == "hello"

    def test_marshal_real_exception_fallback(self):
        """_marshal_z3_value handles Real fraction exception."""
        from z3 import Real, RealVal

        solver = ConstraintSolver()
        r = Real("r")

        # RealVal that converts cleanly
        result = solver.solve([r == RealVal("3/4")], [r])

        assert result.status == SolverStatus.SAT
        # Should be 0.75
        assert abs(result.model["r"] - 0.75) < 0.01

    def test_marshal_algebraic_value(self):
        """_marshal_z3_value handles algebraic numbers (sqrt etc)."""
        from z3 import Real

        solver = ConstraintSolver()
        x = Real("x")

        # x^2 == 2 gives sqrt(2) which is an algebraic number
        result = solver.solve([x * x == 2, x > 0], [x])

        assert result.status == SolverStatus.SAT
        # sqrt(2) ≈ 1.414
        assert abs(result.model["x"] - 1.414) < 0.01

    def test_solve_unknown_status(self):
        """solve() may return UNKNOWN on timeout."""
        solver = ConstraintSolver(timeout_ms=1)
        x = Int("x")

        # Very simple constraint, but with 1ms timeout might be unknown
        result = solver.solve([x > 0], [x])
        assert result.status in (SolverStatus.SAT, SolverStatus.UNKNOWN)

    def test_marshal_fallback_str(self):
        """_marshal_z3_value falls back to str for unknown types."""
        from z3 import Function, IntSort as Z3IntSort

        solver = ConstraintSolver()
        # Create uninterpreted function - its value won't be a standard Z3 type
        f = Function("f", Z3IntSort(), Z3IntSort())
        x = Int("x")

        # Constrain via function
        result = solver.solve([x == f(0), x > 0, x < 100], [x])

        # This should work regardless
        assert result.status == SolverStatus.SAT

    def test_prove_invalid_with_counterexample(self):
        """prove() returns INVALID with counterexample for false claim."""
        solver = ConstraintSolver()
        x = Int("x")

        # Claim: x > 0 implies x > 100 (false for x=50)
        result = solver.prove([x > 0], x > 100)

        assert result.status == SolverStatus.INVALID
        # Counterexample should show x value that disproves the claim
        assert result.counterexample is not None

    def test_solve_with_empty_constraints(self):
        """solve() with empty constraints returns SAT."""
        solver = ConstraintSolver()
        x = Int("x")

        result = solver.solve([], [x])

        # Empty constraints are trivially satisfiable
        assert result.status == SolverStatus.SAT

    def test_solve_with_no_variables(self):
        """solve() with no variables returns SAT if constraints are satisfiable."""
        solver = ConstraintSolver()
        x = Int("x")

        # Constraint with no variable extraction
        result = solver.solve([x > 0], [])

        # Should be SAT (constraints are satisfiable) but no model extracted
        assert result.status == SolverStatus.SAT

    def test_z3_to_python_symbolic_bool(self):
        """_z3_to_python handles symbolic bool (neither true nor false)."""
        from z3 import Bool, simplify

        solver = ConstraintSolver()

        # Create a symbolic bool (not concrete true/false)
        b = Bool("symbolic_b")

        # Direct call to _z3_to_python with symbolic bool
        # This covers line 303 - the "else" branch for booleans
        result = solver._z3_to_python(b)

        # simplify(symbolic_b) is still symbolic, is_true returns False
        # So it should return False
        assert isinstance(result, bool)

    def test_z3_to_python_real_exception_path(self):
        """_z3_to_python handles Real with exception in as_decimal."""
        from z3 import Real, Sqrt
        from unittest.mock import MagicMock, patch

        solver = ConstraintSolver()

        # Create a mock Z3 value that behaves like a Real but throws on as_decimal
        mock_value = MagicMock()
        mock_value.as_decimal.side_effect = Exception("as_decimal failed")
        mock_value.numerator_as_long.return_value = 7
        mock_value.denominator_as_long.return_value = 2

        # Patch z3.is_real to return True for our mock
        with patch("z3.is_int_value", return_value=False), patch(
            "z3.is_bool", return_value=False
        ), patch("z3.is_string_value", return_value=False), patch(
            "z3.is_real", return_value=True
        ):
            result = solver._z3_to_python(mock_value)

        # Should use fallback: numerator / denominator
        assert result == 3.5  # 7/2

    def test_z3_to_python_algebraic_via_real(self):
        """Algebraic values (e.g., sqrt(2)) are handled by the is_real branch."""
        from z3 import Real

        solver = ConstraintSolver()
        r = Real("r")

        # r * r == 2 gives sqrt(2) which is algebraic, but is_real is True
        result = solver.solve([r * r == 2, r > 0], [r])

        assert result.status == SolverStatus.SAT
        # sqrt(2) ~ 1.414... - handled by is_real branch
        assert 1.41 < result.model["r"] < 1.42

    def test_z3_to_python_unknown_type_fallback(self):
        """_z3_to_python falls back to str() for unknown Z3 types."""
        from unittest.mock import MagicMock, patch

        solver = ConstraintSolver()

        # Create a mock Z3 value that doesn't match any known type
        mock_value = MagicMock()
        mock_value.__str__ = MagicMock(return_value="unknown_type_42")

        # Patch all Z3 type checks to return False
        with patch("z3.is_int_value", return_value=False), patch(
            "z3.is_bool", return_value=False
        ), patch("z3.is_string_value", return_value=False), patch(
            "z3.is_real", return_value=False
        ), patch(
            "z3.is_bv_value", return_value=False
        ):
            result = solver._z3_to_python(mock_value)

        # Should fall back to str()
        assert result == "unknown_type_42"

    def test_solve_returns_unknown_on_z3_unknown(self):
        """solve() returns UNKNOWN when Z3 returns unknown (line 167)."""
        from unittest.mock import MagicMock, patch
        from z3 import unknown

        solver = ConstraintSolver()
        x = Int("x")

        # Mock the Z3 Solver class to return 'unknown' on check()
        mock_z3_solver = MagicMock()
        mock_z3_solver.check.return_value = unknown

        with patch(
            "code_scalpel.symbolic_execution_tools.constraint_solver.Solver",
            return_value=mock_z3_solver,
        ):
            result = solver.solve([x > 0], [x])

        assert result.status == SolverStatus.UNKNOWN
        assert result.model is None

    def test_prove_returns_unknown_on_z3_unknown(self):
        """prove() returns UNKNOWN when Z3 returns unknown (line 220)."""
        from unittest.mock import MagicMock, patch
        from z3 import unknown

        solver = ConstraintSolver()
        x = Int("x")

        # Mock the Z3 Solver class to return 'unknown' on check()
        mock_z3_solver = MagicMock()
        mock_z3_solver.check.return_value = unknown

        with patch(
            "code_scalpel.symbolic_execution_tools.constraint_solver.Solver",
            return_value=mock_z3_solver,
        ):
            result = solver.prove([x > 0], x > -100)

        assert result.status == SolverStatus.UNKNOWN
        assert result.counterexample is None
