"""
Tests for SymbolicInterpreter - The Nervous System of Symbolic Execution.

CRITICAL: These tests enforce SMART FORKING.
Blind forking creates zombie paths that waste CPU cycles.

The interpreter must:
1. Check feasibility of BOTH branches before forking
2. Prune dead paths (infeasible branches)
3. Only fork when BOTH paths are possible

If the Dead Path Pruning tests fail, the engine will waste resources
on impossible execution paths.
"""

import pytest
from z3 import Int, Bool, IntSort, BoolSort, And, Or, Not, Solver, sat, unsat

from code_scalpel.symbolic_execution_tools.interpreter import (
    SymbolicInterpreter,
    ExecutionResult,
)
from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState


# =============================================================================
# SECTION 1: Linear Execution (No Branches)
# =============================================================================

class TestLinearExecution:
    """Test straight-line code without branches."""

    def test_single_assignment_integer(self):
        """x = 42 creates variable x with value 42."""
        code = "x = 42"
        result = SymbolicInterpreter().execute(code)
        
        # Should produce exactly one final state
        assert len(result.states) == 1
        state = result.states[0]
        
        # Variable x should exist
        assert state.has_variable("x")

    def test_single_assignment_boolean(self):
        """flag = True creates boolean variable."""
        code = "flag = True"
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("flag")

    def test_sequential_assignments(self):
        """Multiple assignments in sequence."""
        code = """
x = 1
y = 2
z = 3
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")
        assert state.has_variable("y")
        assert state.has_variable("z")

    def test_dependent_assignment(self):
        """y = x + 1 where x was previously defined."""
        code = """
x = 10
y = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")
        assert state.has_variable("y")
        
        # The constraint y == x + 1 should be satisfiable
        assert state.is_feasible()

    def test_reassignment(self):
        """x = x + 1 updates variable binding."""
        code = """
x = 5
x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")

    def test_arithmetic_expression(self):
        """Complex arithmetic expression."""
        code = """
a = 10
b = 20
c = a + b * 2
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("c")

    def test_boolean_expression(self):
        """Boolean operations."""
        code = """
a = True
b = False
c = a and b
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("c")

    def test_comparison_expression(self):
        """Comparison creates boolean."""
        code = """
x = 10
is_positive = x > 0
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("is_positive")


# =============================================================================
# SECTION 2: Basic Branching (Both Paths Feasible)
# =============================================================================

class TestBasicBranching:
    """Test if/else with symbolic input where both paths are possible."""

    def test_symbolic_input_branches(self):
        """
        Symbolic input creates two feasible paths.
        
        x = symbolic_int("x")
        if x > 0:
            y = 1
        else:
            y = -1
        
        Should produce TWO states: one where x > 0, one where x <= 0.
        """
        # We need a way to declare symbolic inputs
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 0:
    y = 1
else:
    y = -1
"""
        result = interp.execute(code)
        
        # Two paths: x > 0 and x <= 0
        assert len(result.states) == 2

    def test_both_branches_have_correct_constraints(self):
        """Each branch has the correct path constraint."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 10:
    y = 1
else:
    y = 0
"""
        result = interp.execute(code)
        
        assert len(result.states) == 2
        
        # Both states should be feasible
        for state in result.states:
            assert state.is_feasible()

    def test_if_without_else(self):
        """If statement without else still creates two paths."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
y = 0
if x > 0:
    y = 1
"""
        result = interp.execute(code)
        
        # Two paths: one where x > 0 (y=1), one where x <= 0 (y=0)
        assert len(result.states) == 2

    def test_nested_if_creates_multiple_paths(self):
        """
        Nested if statements multiply paths.
        
        if a > 0:
            if b > 0:
                ...
            else:
                ...
        else:
            ...
        
        Should produce 3 paths (or 4 if else has nested if).
        """
        interp = SymbolicInterpreter()
        interp.declare_symbolic("a", IntSort())
        interp.declare_symbolic("b", IntSort())
        
        code = """
if a > 0:
    if b > 0:
        result = 1
    else:
        result = 2
else:
    result = 3
"""
        result = interp.execute(code)
        
        # 3 paths: (a>0, b>0), (a>0, b<=0), (a<=0)
        assert len(result.states) == 3


# =============================================================================
# SECTION 3: DEAD PATH PRUNING - The "Blind Fork" Prevention
# =============================================================================

class TestDeadPathPruning:
    """
    CRITICAL TESTS: These enforce SMART FORKING.
    
    If the interpreter blindly forks without checking feasibility,
    it will create zombie paths that waste resources.
    """

    def test_concrete_true_condition_no_fork(self):
        """
        PRUNING TEST 1: Concrete True condition should NOT fork.
        
        x = 5
        if x > 0:  # Always true! x is 5.
            y = 1
        else:
            y = 0
        
        Should produce ONE state (the true branch only).
        """
        code = """
x = 5
if x > 0:
    y = 1
else:
    y = 0
"""
        result = SymbolicInterpreter().execute(code)
        
        # Only ONE path - the true branch
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("y")

    def test_concrete_false_condition_no_fork(self):
        """
        PRUNING TEST 2: Concrete False condition should NOT fork.
        
        x = 5
        if x > 10:  # Always false! x is 5.
            y = 1
        else:
            y = 0
        
        Should produce ONE state (the else branch only).
        """
        code = """
x = 5
if x > 10:
    y = 1
else:
    y = 0
"""
        result = SymbolicInterpreter().execute(code)
        
        # Only ONE path - the else branch
        assert len(result.states) == 1

    def test_constrained_symbolic_prunes_infeasible(self):
        """
        PRUNING TEST 3: Prior constraints eliminate branches.
        
        x is symbolic, but constrained to x > 100 before the if.
        
        if x > 50:  # Always true given x > 100!
            y = 1
        else:
            y = 0
        
        Should produce ONE state.
        """
        interp = SymbolicInterpreter()
        x = interp.declare_symbolic("x", IntSort())
        interp.add_precondition(x > 100)  # x is always > 100
        
        code = """
if x > 50:
    y = 1
else:
    y = 0
"""
        result = interp.execute(code)
        
        # Only the true branch is feasible
        assert len(result.states) == 1

    def test_impossible_condition_prunes_completely(self):
        """
        PRUNING TEST 4: Impossible condition produces empty true branch.
        
        x = 5
        if x > 10 and x < 3:  # Impossible!
            y = 1
        else:
            y = 0
        
        Should produce ONE state (else only).
        """
        code = """
x = 5
if x > 10:
    y = 1
"""
        result = SymbolicInterpreter().execute(code)
        
        # The if body is never entered
        # We get one state where y doesn't exist (or exists from before)
        assert len(result.states) == 1

    def test_chained_pruning(self):
        """
        PRUNING TEST 5: Pruning propagates through nested branches.
        
        x = 5
        if x > 0:      # True, no fork
            if x > 10:  # False, no fork
                y = 1
            else:
                y = 2
        else:
            y = 3
        
        Should produce ONE state (y = 2).
        """
        code = """
x = 5
if x > 0:
    if x > 10:
        y = 1
    else:
        y = 2
else:
    y = 3
"""
        result = SymbolicInterpreter().execute(code)
        
        # Only one path survives: x=5, x>0 true, x>10 false, y=2
        assert len(result.states) == 1


# =============================================================================
# SECTION 4: Path Constraint Accumulation
# =============================================================================

class TestPathConstraints:
    """Test that path constraints accumulate correctly."""

    def test_constraints_accumulate_through_branches(self):
        """Each branch adds its condition to path constraints."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 0:
    y = 1
"""
        result = interp.execute(code)
        
        # True branch should have x > 0 constraint
        # False branch should have NOT(x > 0) constraint
        assert len(result.states) == 2
        
        for state in result.states:
            assert len(state.constraints) >= 1

    def test_nested_branches_accumulate_constraints(self):
        """Nested branches accumulate all parent conditions."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 0:
    if x > 10:
        y = 1
"""
        result = interp.execute(code)
        
        # Find the innermost true branch (x > 0 AND x > 10)
        deep_states = [s for s in result.states if len(s.constraints) >= 2]
        assert len(deep_states) >= 1


# =============================================================================
# SECTION 5: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_code(self):
        """Empty code produces single empty state."""
        result = SymbolicInterpreter().execute("")
        
        assert len(result.states) == 1

    def test_only_pass_statement(self):
        """Pass statement does nothing."""
        result = SymbolicInterpreter().execute("pass")
        
        assert len(result.states) == 1

    def test_empty_if_body(self):
        """If with pass body still forks correctly."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 0:
    pass
else:
    y = 1
"""
        result = interp.execute(code)
        
        assert len(result.states) == 2

    def test_multiple_sequential_ifs(self):
        """Sequential if statements multiply paths."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("a", IntSort())
        interp.declare_symbolic("b", IntSort())
        
        code = """
if a > 0:
    x = 1
else:
    x = 0

if b > 0:
    y = 1
else:
    y = 0
"""
        result = interp.execute(code)
        
        # 2 * 2 = 4 paths
        assert len(result.states) == 4

    def test_elif_chain(self):
        """Elif creates multiple exclusive branches."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = """
if x > 10:
    y = 2
elif x > 0:
    y = 1
else:
    y = 0
"""
        result = interp.execute(code)
        
        # 3 mutually exclusive paths
        assert len(result.states) == 3


# =============================================================================
# SECTION 6: ExecutionResult API
# =============================================================================

class TestExecutionResult:
    """Test the ExecutionResult container."""

    def test_result_contains_all_states(self):
        """ExecutionResult.states has all final states."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = "if x > 0: y = 1"
        result = interp.execute(code)
        
        assert hasattr(result, 'states')
        assert isinstance(result.states, list)

    def test_result_tracks_path_count(self):
        """ExecutionResult tracks number of branch points explored."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = "if x > 0: y = 1"
        result = interp.execute(code)
        
        assert hasattr(result, 'path_count')
        # path_count tracks branch points, not terminal states
        # 1 if statement = 1 branch point, but 2 terminal states
        assert result.path_count >= 1
        assert len(result.states) == 2  # Two terminal states

    def test_result_can_filter_feasible(self):
        """Can filter to only feasible states."""
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        
        code = "if x > 0: y = 1"
        result = interp.execute(code)
        
        feasible = result.feasible_states()
        for state in feasible:
            assert state.is_feasible()


# =============================================================================
# SECTION 7: Unsupported Constructs (Graceful Degradation)
# =============================================================================

class TestUnsupportedConstructs:
    """Test handling of unsupported Python constructs."""

    def test_function_def_skipped(self):
        """Function definitions are skipped (not executed)."""
        code = """
def foo():
    return 1

x = 5
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should still process x = 5
        assert len(result.states) == 1
        assert result.states[0].has_variable("x")

    def test_loop_raises_or_skips(self):
        """
        Loops are not yet supported (M5).
        Should either skip or raise a clear error.
        """
        code = """
x = 0
for i in range(10):
    x = x + 1
"""
        # Either raises UnsupportedConstruct or skips the loop
        try:
            result = SymbolicInterpreter().execute(code)
            # If it doesn't raise, it should have processed x = 0
            assert len(result.states) >= 1
        except Exception as e:
            # Acceptable: clear error about unsupported loop
            assert "loop" in str(e).lower() or "unsupported" in str(e).lower()

    def test_function_call_marks_unknown(self):
        """
        Function calls return UNKNOWN type.
        Per M1 spec, function results are not tracked.
        """
        code = """
x = some_function()
y = 5
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should still process y = 5
        assert len(result.states) >= 1
