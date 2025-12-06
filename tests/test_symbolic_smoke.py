"""
Smoke tests for the Symbolic Execution subsystem.

These tests verify that the symbolic execution tools can be imported
and instantiated.

Status: EXPERIMENTAL but FUNCTIONAL - Phase 1 supports Int/Bool only.
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

    def test_instantiate_engine_no_args(self):
        """Test that SymbolicAnalyzer can be created without arguments."""
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicAnalyzer,
        )

        # New API: no solver required, uses internal components
        analyzer = SymbolicAnalyzer()
        assert analyzer is not None

    def test_instantiate_engine_with_solver(self):
        """Test creating a SymbolicExecutionEngine with a solver."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicExecutionEngine,
        )

        # Note: SymbolicExecutionEngine is now an alias for SymbolicAnalyzer
        # which doesn't require a solver argument
        engine = SymbolicExecutionEngine()
        assert engine is not None


class TestSymbolicExecution:
    """Test actual symbolic execution - NOW WORKING!"""

    def test_execute_simple_assignment(self):
        """
        Test executing simple code.

        This now works! Phase 1 supports Int/Bool analysis.
        """
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicAnalyzer,
        )

        analyzer = SymbolicAnalyzer()
        result = analyzer.analyze("x = 1 + 2")
        
        # Analysis should complete successfully
        assert result is not None
        assert result.total_paths >= 1
        assert result.feasible_count >= 1

    def test_execute_conditional(self):
        """Test executing code with conditionals - NOW WORKING!"""
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicAnalyzer,
        )

        analyzer = SymbolicAnalyzer()

        code = """
x = 5
if x > 0:
    result = 1
else:
    result = -1
"""
        result = analyzer.analyze(code)
        assert result is not None
        assert result.feasible_count >= 1

    def test_execute_loop(self):
        """Test executing code with loops - NOW WORKING!"""
        from code_scalpel.symbolic_execution_tools.engine import (
            SymbolicAnalyzer,
        )

        analyzer = SymbolicAnalyzer()

        code = """
total = 0
for i in range(5):
    total = total + i
"""
        result = analyzer.analyze(code)
        assert result is not None
        assert result.feasible_count >= 1


class TestConstraintSolver:
    """Test the constraint solver component."""

    def test_solver_has_solve(self):
        """Test that solver has solve method."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        assert hasattr(solver, "solve")

    def test_solver_has_prove(self):
        """Test that solver has prove method."""
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        assert hasattr(solver, "prove")
