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
        result = solver.solve(
            [Implies(flag, x > 0), flag == True, x < 100],
            [x, flag]
        )
        
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
        result = solver.prove(
            preconditions=[x > 10],
            assertion=x > 5
        )
        
        assert result.status == SolverStatus.VALID
        assert result.counterexample is None

    def test_invalid_assertion_with_counterexample(self):
        """Find counterexample when assertion is invalid."""
        solver = ConstraintSolver()
        x = Int("x")
        
        # Precondition: x > 0
        # Assertion: x > 100 (NOT valid - counterexample: x = 1)
        result = solver.prove(
            preconditions=[x > 0],
            assertion=x > 100
        )
        
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
        
        result = solver.prove(
            preconditions=[],
            assertion=x == x
        )
        
        assert result.status == SolverStatus.VALID

    def test_boolean_validity(self):
        """Prove flag OR NOT flag is valid (tautology)."""
        solver = ConstraintSolver()
        flag = Bool("flag")
        
        result = solver.prove(
            preconditions=[],
            assertion=Or(flag, Not(flag))
        )
        
        assert result.status == SolverStatus.VALID


# =============================================================================
# SECTION 5: Timeout Handling
# =============================================================================

class TestTimeoutHandling:
    """Test solver timeout behavior."""

    def test_default_timeout_exists(self):
        """Solver has a default timeout."""
        solver = ConstraintSolver()
        
        assert hasattr(solver, 'timeout_ms')
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
        
        assert hasattr(result, 'status')
        assert isinstance(result.status, SolverStatus)

    def test_result_has_model(self):
        """SolverResult has model field for SAT."""
        solver = ConstraintSolver()
        x = Int("x")
        
        result = solver.solve([x > 0], [x])
        
        assert hasattr(result, 'model')

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
