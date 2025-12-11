"""
Tests for SymbolicState - The Memory Model for Symbolic Execution.

CRITICAL: These tests specifically verify FORK ISOLATION.
The "shallow copy suicide" is the #1 killer of symbolic execution engines.

If Path A and Path B share mutable state, you get:
- Silent corruption
- Non-deterministic results
- Impossible-to-debug failures

The fork() method MUST provide total isolation.
"""

import pytest
from z3 import Int, IntSort, BoolSort, Or, Not, Solver, sat, unsat

from code_scalpel.symbolic_execution_tools.state_manager import (
    SymbolicState,
    SymbolicVariable,
)


# =============================================================================
# SECTION 1: Basic Variable Storage
# =============================================================================


class TestVariableStorage:
    """Test basic variable operations."""

    def test_create_int_variable(self):
        """Can create an integer symbolic variable."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        assert x is not None
        assert x.sort() == IntSort()

    def test_create_bool_variable(self):
        """Can create a boolean symbolic variable."""
        state = SymbolicState()
        flag = state.create_variable("flag", BoolSort())

        assert flag is not None
        assert flag.sort() == BoolSort()

    def test_get_existing_variable(self):
        """Retrieving an existing variable returns the same reference."""
        state = SymbolicState()
        x1 = state.create_variable("x", IntSort())
        x2 = state.get_variable("x")

        # Should be the exact same Z3 object
        assert x1 is x2

    def test_get_nonexistent_variable_returns_none(self):
        """Retrieving a non-existent variable returns None."""
        state = SymbolicState()
        result = state.get_variable("does_not_exist")

        assert result is None

    def test_set_variable_concrete_value(self):
        """Can bind a variable to a concrete Z3 expression."""
        state = SymbolicState()
        state.create_variable("x", IntSort())

        # Bind x to the value 42
        state.set_variable("x", Int("x") + 0)  # Still symbolic but constrained

        retrieved = state.get_variable("x")
        assert retrieved is not None

    def test_has_variable(self):
        """Can check if a variable exists."""
        state = SymbolicState()
        state.create_variable("x", IntSort())

        assert state.has_variable("x") is True
        assert state.has_variable("y") is False

    def test_list_variables(self):
        """Can list all variable names."""
        state = SymbolicState()
        state.create_variable("a", IntSort())
        state.create_variable("b", BoolSort())
        state.create_variable("c", IntSort())

        names = state.variable_names()
        assert set(names) == {"a", "b", "c"}


# =============================================================================
# SECTION 2: Path Condition Management
# =============================================================================


class TestPathConditions:
    """Test path condition accumulation."""

    def test_empty_path_condition(self):
        """New state has empty (trivially true) path condition."""
        state = SymbolicState()

        # No constraints = satisfiable
        assert state.is_feasible() is True

    def test_add_single_constraint(self):
        """Can add a single path constraint."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        state.add_constraint(x > 0)

        assert len(state.constraints) == 1

    def test_add_multiple_constraints(self):
        """Can accumulate multiple path constraints."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        state.add_constraint(x > 0)
        state.add_constraint(x < 100)
        state.add_constraint(x != 50)

        assert len(state.constraints) == 3

    def test_feasible_path(self):
        """Feasibility check returns True for satisfiable constraints."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        state.add_constraint(x > 0)
        state.add_constraint(x < 10)

        assert state.is_feasible() is True

    def test_infeasible_path(self):
        """Feasibility check returns False for contradictory constraints."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        # x > 10 AND x < 5 is impossible
        state.add_constraint(x > 10)
        state.add_constraint(x < 5)

        assert state.is_feasible() is False

    def test_get_path_condition_as_conjunction(self):
        """Can get all constraints as a single Z3 And expression."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        state.add_constraint(x > 0)
        state.add_constraint(x < 100)

        pc = state.path_condition()

        # Verify it's a valid Z3 expression
        solver = Solver()
        solver.add(pc)
        assert solver.check() == sat


# =============================================================================
# SECTION 3: THE FORK TEST - The "Shallow Copy Suicide" Prevention
# =============================================================================


class TestForkIsolation:
    """
    CRITICAL TESTS: These verify that fork() provides TOTAL ISOLATION.

    If ANY of these tests fail, the symbolic execution engine is BROKEN.
    Path corruption will cause silent, impossible-to-debug failures.
    """

    def test_fork_creates_new_object(self):
        """Fork returns a different object instance."""
        state_a = SymbolicState()
        state_b = state_a.fork()

        assert state_a is not state_b

    def test_fork_preserves_variables(self):
        """Forked state has the same variables as parent."""
        state_a = SymbolicState()
        state_a.create_variable("x", IntSort())
        state_a.create_variable("flag", BoolSort())

        state_b = state_a.fork()

        assert state_b.has_variable("x")
        assert state_b.has_variable("flag")

    def test_fork_preserves_constraints(self):
        """Forked state has the same path constraints as parent."""
        state_a = SymbolicState()
        x = state_a.create_variable("x", IntSort())
        state_a.add_constraint(x > 0)
        state_a.add_constraint(x < 100)

        state_b = state_a.fork()

        assert len(state_b.constraints) == 2

    # =========================================================================
    # THE CORRUPTION TESTS - If these fail, you have the "shallow copy suicide"
    # =========================================================================

    def test_fork_variable_isolation_add_new(self):
        """
        CORRUPTION TEST 1: Adding a variable to fork does NOT affect parent.
        """
        state_a = SymbolicState()
        state_a.create_variable("x", IntSort())

        state_b = state_a.fork()
        state_b.create_variable("y", IntSort())  # Only in B

        # A must NOT have y
        assert state_a.has_variable("x") is True
        assert state_a.has_variable("y") is False  # CRITICAL

        # B has both
        assert state_b.has_variable("x") is True
        assert state_b.has_variable("y") is True

    def test_fork_variable_isolation_modify_existing(self):
        """
        CORRUPTION TEST 2: Modifying a variable in fork does NOT affect parent.
        """
        state_a = SymbolicState()
        x_original = state_a.create_variable("x", IntSort())

        state_b = state_a.fork()

        # In B, rebind x to a new expression
        new_x = Int("x_modified")
        state_b.set_variable("x", new_x)

        # A's x must be unchanged
        assert state_a.get_variable("x") is x_original  # CRITICAL

        # B's x is the new one
        assert state_b.get_variable("x") is new_x

    def test_fork_constraint_isolation_add_new(self):
        """
        CORRUPTION TEST 3: Adding constraints to fork does NOT affect parent.
        """
        state_a = SymbolicState()
        x = state_a.create_variable("x", IntSort())
        state_a.add_constraint(x > 0)

        state_b = state_a.fork()
        state_b.add_constraint(x < 10)  # Only in B

        # A has only 1 constraint
        assert len(state_a.constraints) == 1  # CRITICAL

        # B has 2 constraints
        assert len(state_b.constraints) == 2

    def test_fork_deep_isolation_chained_forks(self):
        """
        CORRUPTION TEST 4: Chained forks maintain isolation at all levels.

        Simulates: if (a) { if (b) { ... } }
        """
        # Root state
        root = SymbolicState()
        x = root.create_variable("x", IntSort())
        root.add_constraint(x > 0)

        # First branch: x > 10
        branch_a = root.fork()
        branch_a.add_constraint(x > 10)

        # Second branch from branch_a: x > 100
        branch_aa = branch_a.fork()
        branch_aa.add_constraint(x > 100)

        # Verify isolation at all levels
        assert len(root.constraints) == 1  # Only x > 0
        assert len(branch_a.constraints) == 2  # x > 0, x > 10
        assert len(branch_aa.constraints) == 3  # x > 0, x > 10, x > 100

    def test_fork_isolation_real_world_if_else(self):
        """
        CORRUPTION TEST 5: Simulate real if/else branching.

        Code being analyzed:
            x = input()  # symbolic
            if x > 0:
                y = 1
            else:
                y = -1

        Both paths must be independent.
        """
        # Before the if statement
        state = SymbolicState()
        x = state.create_variable("x", IntSort())

        # Fork for TRUE branch
        true_branch = state.fork()
        true_branch.add_constraint(x > 0)
        true_branch.create_variable("y", IntSort())
        true_branch.set_variable("y", Int("y_true"))

        # Fork for FALSE branch (from original, NOT from true_branch)
        false_branch = state.fork()
        false_branch.add_constraint(Not(x > 0))
        false_branch.create_variable("y", IntSort())
        false_branch.set_variable("y", Int("y_false"))

        # Original state is untouched
        assert state.has_variable("y") is False
        assert len(state.constraints) == 0

        # True branch has y and constraint x > 0
        assert true_branch.has_variable("y") is True
        assert len(true_branch.constraints) == 1

        # False branch has y and constraint NOT(x > 0)
        assert false_branch.has_variable("y") is True
        assert len(false_branch.constraints) == 1

        # The y variables are different objects
        assert true_branch.get_variable("y") is not false_branch.get_variable("y")


# =============================================================================
# SECTION 4: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_state_fork(self):
        """Can fork an empty state."""
        state = SymbolicState()
        forked = state.fork()

        assert forked is not state
        assert len(forked.variable_names()) == 0
        assert len(forked.constraints) == 0

    def test_fork_count_tracking(self):
        """State can optionally track fork depth/count."""
        state = SymbolicState()

        child = state.fork()
        grandchild = child.fork()

        # Optional: track depth for debugging
        assert state.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2

    def test_multiple_forks_from_same_parent(self):
        """Multiple forks from the same parent are all independent."""
        parent = SymbolicState()
        x = parent.create_variable("x", IntSort())

        # Simulate a 3-way branch (like a switch statement)
        fork1 = parent.fork()
        fork2 = parent.fork()
        fork3 = parent.fork()

        fork1.add_constraint(x == 1)
        fork2.add_constraint(x == 2)
        fork3.add_constraint(x == 3)

        # All forks are independent
        assert len(fork1.constraints) == 1
        assert len(fork2.constraints) == 1
        assert len(fork3.constraints) == 1

        # Parent is untouched
        assert len(parent.constraints) == 0

    def test_constraint_with_multiple_variables(self):
        """Constraints involving multiple variables work correctly."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        y = state.create_variable("y", IntSort())

        state.add_constraint(x + y == 10)
        state.add_constraint(x > y)

        assert state.is_feasible() is True

        # x=6, y=4 satisfies both constraints

    def test_boolean_variable_in_constraint(self):
        """Boolean variables can be used directly as constraints."""
        state = SymbolicState()
        flag = state.create_variable("flag", BoolSort())
        x = state.create_variable("x", IntSort())

        # flag implies x > 0
        state.add_constraint(Or(Not(flag), x > 0))

        assert state.is_feasible() is True


# =============================================================================
# SECTION 5: SymbolicVariable Helper (Optional but Clean)
# =============================================================================


class TestSymbolicVariable:
    """Test the SymbolicVariable wrapper if implemented."""

    def test_symbolic_variable_creation(self):
        """SymbolicVariable wraps Z3 expression with metadata."""
        var = SymbolicVariable("x", IntSort())

        assert var.name == "x"
        assert var.sort == IntSort()
        assert var.expr is not None

    def test_symbolic_variable_expr_access(self):
        """Can access the underlying Z3 expression."""
        var = SymbolicVariable("counter", IntSort())

        # Should be usable in Z3 operations
        constraint = var.expr > 0

        solver = Solver()
        solver.add(constraint)
        assert solver.check() == sat

    def test_symbolic_variable_bool_sort(self):
        """SymbolicVariable can create boolean variables."""
        var = SymbolicVariable("flag", BoolSort())

        assert var.name == "flag"
        assert var.sort == BoolSort()
        assert var.expr is not None

    def test_symbolic_variable_string_sort(self):
        """SymbolicVariable can create string variables."""
        from z3 import StringSort

        var = SymbolicVariable("name", StringSort())

        assert var.name == "name"
        assert var.sort == StringSort()
        assert var.expr is not None

    def test_symbolic_variable_unsupported_sort_raises(self):
        """SymbolicVariable raises ValueError for unsupported sorts."""
        from z3 import RealSort

        with pytest.raises(ValueError, match="Unsupported sort"):
            SymbolicVariable("x", RealSort())


# =============================================================================
# SECTION 6: Coverage Completeness - Edge Cases for 100%
# =============================================================================


class TestCoverageCompleteness:
    """Tests to achieve 100% coverage on state_manager.py."""

    def test_create_string_variable(self):
        """Can create a string symbolic variable via SymbolicState."""
        from z3 import StringSort

        state = SymbolicState()
        s = state.create_variable("text", StringSort())

        assert s is not None
        assert s.sort() == StringSort()

    def test_create_variable_already_exists_same_sort(self):
        """Creating same variable with same sort returns existing."""
        state = SymbolicState()
        x1 = state.create_variable("x", IntSort())
        x2 = state.create_variable("x", IntSort())

        assert x1 is x2

    def test_create_variable_already_exists_different_sort_raises(self):
        """Creating same variable with different sort raises ValueError."""
        state = SymbolicState()
        state.create_variable("x", IntSort())

        with pytest.raises(ValueError, match="already exists with sort"):
            state.create_variable("x", BoolSort())

    def test_create_variable_unsupported_sort_raises(self):
        """Creating variable with unsupported sort raises ValueError."""
        from z3 import RealSort

        state = SymbolicState()

        with pytest.raises(ValueError, match="Unsupported sort"):
            state.create_variable("r", RealSort())

    def test_variables_property_returns_copy(self):
        """The variables property returns a copy, not the internal dict."""
        state = SymbolicState()
        state.create_variable("x", IntSort())

        vars_copy = state.variables
        vars_copy["y"] = Int("y")  # Modify the copy

        # Original should be unaffected
        assert "y" not in state.variables
        assert state.has_variable("x")
        assert not state.has_variable("y")

    def test_path_condition_empty(self):
        """path_condition() with no constraints returns trivially true."""
        state = SymbolicState()
        pc = state.path_condition()

        # Should be satisfiable (trivially true)
        solver = Solver()
        solver.add(pc)
        assert solver.check() == sat

    def test_path_condition_single_constraint(self):
        """path_condition() with one constraint returns it directly."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 0)

        pc = state.path_condition()

        # Should be the constraint itself (or equivalent)
        solver = Solver()
        solver.add(pc)
        solver.add(x == 5)
        assert solver.check() == sat

        solver2 = Solver()
        solver2.add(pc)
        solver2.add(x == -1)
        assert solver2.check() == unsat

    def test_path_condition_multiple_constraints(self):
        """path_condition() with multiple constraints returns And."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 0)
        state.add_constraint(x < 10)

        pc = state.path_condition()

        solver = Solver()
        solver.add(pc)
        solver.add(x == 5)
        assert solver.check() == sat

    def test_summary_method(self):
        """summary() returns a dictionary with state information."""
        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 0)

        summary = state.summary()

        assert isinstance(summary, dict)
        assert summary["depth"] == 0
        assert "x" in summary["variables"]
        assert summary["constraint_count"] == 1
        assert summary["is_feasible"] is True

    def test_repr_method(self):
        """__repr__ returns a useful string representation."""
        state = SymbolicState()
        state.create_variable("x", IntSort())
        state.create_variable("flag", BoolSort())

        repr_str = repr(state)

        assert "SymbolicState" in repr_str
        assert "depth=0" in repr_str
        assert "x" in repr_str
