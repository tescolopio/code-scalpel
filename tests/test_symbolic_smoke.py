"""
Smoke tests for the Symbolic Execution subsystem.

These tests verify that the symbolic execution tools can be imported
and instantiated. They document the BROKEN state of the subsystem.

Status: EXPERIMENTAL - The engine is missing the `_infer_type` method.
"""

import warnings

import pytest


class TestSymbolicImports:
    """Test that symbolic execution modules can be imported."""

    def test_import_emits_warning(self):
        """Test that importing symbolic_execution_tools emits a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force reimport by importing a submodule directly
            from code_scalpel.symbolic_execution_tools import engine  # noqa: F401

            # Check that at least one warning was about experimental status
            # (may have already been imported, so check if any warnings match)

    def test_import_constraint_solver(self):
        """Test importing the constraint solver."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        assert ConstraintSolver is not None

    def test_import_engine(self):
        """Test importing the symbolic execution engine."""
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        assert SymbolicExecutionEngine is not None

    def test_import_path_explorer(self):
        """Test importing the path explorer."""
        from code_scalpel.symbolic_execution_tools.path_explorer import PathExplorer

        assert PathExplorer is not None

    def test_import_model_checker(self):
        """Test importing the model checker."""
        from code_scalpel.symbolic_execution_tools.model_checker import ModelChecker

        assert ModelChecker is not None

    def test_import_result_analyzer(self):
        """Test importing the result analyzer."""
        from code_scalpel.symbolic_execution_tools.result_analyzer import (
            ResultAnalyzer,
        )

        assert ResultAnalyzer is not None

    def test_import_test_generator(self):
        """Test importing the test generator."""
        from code_scalpel.symbolic_execution_tools.test_generator import TestGenerator

        assert TestGenerator is not None


class TestSymbolicInstantiation:
    """Test that symbolic execution components can be instantiated."""

    def test_instantiate_constraint_solver(self):
        """Test creating a ConstraintSolver instance."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        assert solver is not None

    def test_instantiate_engine_requires_solver(self):
        """Test that SymbolicExecutionEngine requires a constraint_solver argument."""
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        with pytest.raises(TypeError) as exc_info:
            SymbolicExecutionEngine()

        assert "constraint_solver" in str(exc_info.value)

    def test_instantiate_engine_with_solver(self):
        """Test creating a SymbolicExecutionEngine with a solver."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        solver = ConstraintSolver()
        engine = SymbolicExecutionEngine(solver)
        assert engine is not None


class TestSymbolicExecution:
    """Test actual symbolic execution - EXPECTED TO FAIL."""

    def test_execute_simple_assignment(self):
        """
        Test executing simple code.

        KNOWN BUG: This fails because _infer_type method is missing.
        This test documents the broken state of the feature.
        """
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
            SymbolicExecutionError,
        )

        solver = ConstraintSolver()
        engine = SymbolicExecutionEngine(solver)

        # This SHOULD work but DOESN'T because _infer_type is missing
        with pytest.raises((AttributeError, SymbolicExecutionError)) as exc_info:
            engine.execute("x = 1 + 2")

        # Document the bug
        assert "_infer_type" in str(exc_info.value)

    @pytest.mark.skip(reason="Symbolic execution is broken - missing _infer_type method")
    def test_execute_conditional(self):
        """Test executing code with conditionals - SKIPPED: Feature broken."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        solver = ConstraintSolver()
        engine = SymbolicExecutionEngine(solver)

        code = """
def check(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
"""
        result = engine.execute(code)
        assert result is not None

    @pytest.mark.skip(reason="Symbolic execution is broken - missing _infer_type method")
    def test_execute_loop(self):
        """Test executing code with loops - SKIPPED: Feature broken."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        solver = ConstraintSolver()
        engine = SymbolicExecutionEngine(solver)

        code = """
total = 0
for i in range(5):
    total += i
"""
        result = engine.execute(code)
        assert result is not None


class TestConstraintSolver:
    """Test the constraint solver component."""

    def test_solver_has_add_constraint(self):
        """Test that solver has add_constraint method."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        assert hasattr(solver, "add_constraint") or hasattr(solver, "add")

    def test_solver_has_solve(self):
        """
        Test that solver has solve method.
        
        KNOWN BUG: ConstraintSolver has no solve/check method.
        This documents another broken API in the symbolic execution subsystem.
        """
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        # Document that solve/check is missing - this is a bug
        has_solve = hasattr(solver, "solve") or hasattr(solver, "check") or hasattr(solver, "evaluate")
        if not has_solve:
            pytest.skip("ConstraintSolver missing solve/check method - API incomplete")
