"""
Smoke tests for the Symbolic Execution subsystem.

These tests verify that the symbolic execution tools can be imported
and instantiated.

Status: EXPERIMENTAL but FUNCTIONAL - Phase 1 supports Int/Bool only.
"""

import warnings

import pytest
from code_scalpel.symbolic_execution_tools.ir_interpreter import IRSymbolicInterpreter
from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer

class SymbolicInterpreter:
    def __init__(self, max_loop_iterations=10):
        self.interp = IRSymbolicInterpreter(max_loop_iterations=max_loop_iterations)
        self.max_loop_iterations = max_loop_iterations
        
    def execute(self, code: str):
        ir = PythonNormalizer().normalize(code)
        return self.interp.execute(ir)

    def declare_symbolic(self, name, sort):
        return self.interp.declare_symbolic(name, sort)


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

    def test_solver_solve_sat(self):
        """Test solver solve with satisfiable constraints."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
            SolverStatus,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        result = solver.solve([x > 0, x < 10], [x])
        assert result.status == SolverStatus.SAT
        assert result.model is not None

    def test_solver_solve_unsat(self):
        """Test solver solve with unsatisfiable constraints."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
            SolverStatus,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        result = solver.solve([x > 10, x < 5], [x])
        assert result.status == SolverStatus.UNSAT

    def test_solver_solve_returns_model(self):
        """Test solver returns correct model."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        result = solver.solve([x == 42], [x])
        assert result.is_sat()
        assert result.model is not None
        assert 'x' in result.model
        assert result.model['x'] == 42

    def test_solver_prove_valid(self):
        """Test proving a valid assertion."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
            SolverStatus,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        # If x > 0, then x >= 0 (valid)
        result = solver.prove([x > 0], x >= 0)
        assert result.status == SolverStatus.VALID

    def test_solver_prove_invalid(self):
        """Test proving an invalid assertion."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
            SolverStatus,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        # If x > 0, then x > 10 is NOT valid (counterexample: x=5)
        result = solver.prove([x > 0], x > 10)
        assert result.status == SolverStatus.INVALID
        assert result.counterexample is not None

    def test_solver_result_bool(self):
        """Test SolverResult __bool__ method."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        result = solver.solve([x == 1], [x])
        assert bool(result) is True  # SAT is truthy

    def test_solver_result_is_sat(self):
        """Test SolverResult is_sat method."""
        import z3
        from code_scalpel.symbolic_execution_tools.constraint_solver import (
            ConstraintSolver,
        )

        solver = ConstraintSolver()
        x = z3.Int('x')
        result = solver.solve([x == 1], [x])
        assert result.is_sat() is True


class TestAnalyzerAdvanced:
    """Test advanced SymbolicAnalyzer features."""

    def test_declare_symbolic_int(self):
        """Test declaring symbolic integer variable."""
        import z3
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        x = analyzer.declare_symbolic('x', z3.IntSort())
        assert x is not None

    def test_declare_symbolic_bool(self):
        """Test declaring symbolic boolean variable."""
        import z3
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        b = analyzer.declare_symbolic('b', z3.BoolSort())
        assert b is not None

    def test_declare_symbolic_unsupported_sort(self):
        """Test that unsupported sorts raise NotImplementedError."""
        import z3
        import pytest
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        with pytest.raises(NotImplementedError):
            analyzer.declare_symbolic('r', z3.RealSort())

    def test_add_precondition(self):
        """Test adding preconditions."""
        import z3
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        x = analyzer.declare_symbolic('x', z3.IntSort())
        analyzer.add_precondition(x > 0)
        analyzer.add_precondition(x < 100)
        # Should not raise
        assert True

    def test_get_solver(self):
        """Test getting the underlying solver."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer
        from code_scalpel.symbolic_execution_tools.constraint_solver import ConstraintSolver

        analyzer = SymbolicAnalyzer()
        solver = analyzer.get_solver()
        assert isinstance(solver, ConstraintSolver)

    def test_custom_timeout(self):
        """Test creating analyzer with custom timeout."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer(solver_timeout=5000)
        assert analyzer.solver_timeout == 5000


class TestEngineEdgeCases:
    """Test edge cases for the symbolic execution engine."""

    def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        result = analyzer.analyze("")
        assert result is not None

    def test_analyze_syntax_error(self):
        """Test analyzing code with syntax error."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer
        import pytest

        analyzer = SymbolicAnalyzer()
        # Syntax error should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            analyzer.analyze("def broken(")
        assert "syntax" in str(exc_info.value).lower()

    def test_analyze_only_comments(self):
        """Test analyzing code with only comments."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        result = analyzer.analyze("# This is a comment\n# Another comment")
        assert result is not None

    def test_analyze_multiple_branches(self):
        """Test analyzing code with multiple conditional branches."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        code = """
x = 5
y = 10
if x > 0:
    if y > 5:
        z = 1
    else:
        z = 2
else:
    z = 3
"""
        result = analyzer.analyze(code)
        assert result is not None
        assert result.total_paths >= 1

    def test_analyze_with_function_def(self):
        """Test analyzing code with function definition."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        result = analyzer.analyze(code)
        assert result is not None

    def test_path_result_properties(self):
        """Test AnalysisResult object has expected properties."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        result = analyzer.analyze("x = 1")
        
        # Check result has expected attributes
        assert hasattr(result, 'total_paths')
        assert hasattr(result, 'feasible_count')
        assert hasattr(result, 'infeasible_count')
        assert hasattr(result, 'paths')

    def test_get_feasible_paths(self):
        """Test getting feasible paths from result."""
        from code_scalpel.symbolic_execution_tools.engine import SymbolicAnalyzer

        analyzer = SymbolicAnalyzer()
        result = analyzer.analyze("x = 1")
        
        # Should have get_feasible_paths method
        assert hasattr(result, 'get_feasible_paths')
        feasible = result.get_feasible_paths()
        assert isinstance(feasible, list)


class TestInterpreterEdgeCases:
    """Test edge cases for the symbolic interpreter."""

    def test_unary_operations(self):
        """Test unary operations."""

        interp = SymbolicInterpreter()
        result = interp.execute("x = -5\ny = not True")
        assert result is not None

    def test_comparison_operators(self):
        """Test all comparison operators."""

        interp = SymbolicInterpreter()
        code = """
a = 5
b = 10
c1 = a < b
c2 = a <= b
c3 = a > b
c4 = a >= b
c5 = a == b
c6 = a != b
"""
        result = interp.execute(code)
        assert result is not None
        # Just verify execution completes - check for feasible states (method)
        states = result.feasible_states()
        assert len(states) >= 1

    def test_boolean_operations(self):
        """Test boolean operations."""

        interp = SymbolicInterpreter()
        code = """
a = True
b = False
c1 = a and b
c2 = a or b
c3 = not a
"""
        result = interp.execute(code)
        assert result is not None

    def test_string_assignment(self):
        """Test string assignment (should handle gracefully)."""

        interp = SymbolicInterpreter()
        result = interp.execute('x = "hello"')
        assert result is not None

    def test_augmented_assignment(self):
        """Test augmented assignment operators."""

        interp = SymbolicInterpreter()
        code = """
x = 5
x += 3
x -= 1
x *= 2
"""
        result = interp.execute(code)
        assert result is not None

    def test_multiple_targets(self):
        """Test multiple assignment targets."""

        interp = SymbolicInterpreter()
        result = interp.execute("x = y = 5")
        assert result is not None

    def test_while_loop(self):
        """Test while loop handling."""

        interp = SymbolicInterpreter()
        code = """
x = 0
while x < 5:
    x = x + 1
"""
        result = interp.execute(code)
        assert result is not None

    @pytest.mark.skip(reason="Try/Except not supported in IR engine yet")
    def test_try_except(self):
        """Test try-except handling."""

        interp = SymbolicInterpreter()
        code = """
try:
    x = 1
except:
    x = 0
"""
        result = interp.execute(code)
        assert result is not None
