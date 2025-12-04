import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import z3


class SolverType(Enum):
    """Supported constraint solver types."""

    Z3 = "z3"
    CVC4 = "cvc4"
    YICES = "yices"
    MATHSAT = "mathsat"


class ConstraintType(Enum):
    """Types of constraints."""

    ARITHMETIC = "arithmetic"
    BOOLEAN = "boolean"
    BITVECTOR = "bitvector"
    STRING = "string"
    ARRAY = "array"


@dataclass
class SolverConfig:
    """Configuration for the constraint solver."""

    solver_type: SolverType = SolverType.Z3
    timeout: Optional[int] = None
    memory_limit: Optional[int] = None
    use_incremental: bool = True
    simplify_constraints: bool = True
    parallel_solving: bool = False
    track_unsat_core: bool = False
    optimization_level: int = 1


@dataclass
class SolverStatistics:
    """Statistics about constraint solving."""

    num_constraints: int = 0
    num_variables: int = 0
    solving_time: float = 0.0
    memory_used: int = 0
    num_sat_checks: int = 0
    num_unsat_results: int = 0
    num_unknown_results: int = 0


class ConstraintError(Exception):
    """Base class for constraint solver errors."""

    pass


class UnsatisfiableError(ConstraintError):
    """Raised when constraints are unsatisfiable."""

    pass


class SolverTimeoutError(ConstraintError):
    """Raised when solver times out."""

    pass


class ConstraintSolver:
    """Advanced constraint solver with multiple backend support."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.stats = SolverStatistics()
        self._init_solver()
        self.variables: dict[str, Any] = {}
        self.constraints: list[Any] = []
        self.assertions_stack: list[list[Any]] = [[]]
        self._setup_logging()

    def _init_solver(self):
        """Initialize the appropriate solver backend."""
        if self.config.solver_type == SolverType.Z3:
            self.solver = z3.Solver()
            if self.config.timeout:
                self.solver.set(timeout=self.config.timeout)
        else:
            raise NotImplementedError(
                f"Solver {self.config.solver_type} not yet supported"
            )

    def create_variable(
        self, name: str, var_type: str, bit_width: Optional[int] = None
    ) -> Any:
        """
        Create a new variable with specified type.

        Args:
            name: Variable name
            var_type: Type of variable (int, bool, bv, etc.)
            bit_width: Bit width for bitvector variables
        """
        if name in self.variables:
            raise ValueError(f"Variable {name} already exists")

        var = None
        if var_type == "int":
            var = z3.Int(name)
        elif var_type == "bool":
            var = z3.Bool(name)
        elif var_type == "real":
            var = z3.Real(name)
        elif var_type == "bv":
            if bit_width is None:
                raise ValueError("Bit width required for bitvector variables")
            var = z3.BitVec(name, bit_width)
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

        self.variables[name] = var
        self.stats.num_variables += 1
        return var

    def add_constraint(
        self, constraint: Any, track: bool = True, simplify: bool = None
    ):
        """
        Add a constraint to the solver.

        Args:
            constraint: The constraint to add
            track: Whether to track this constraint for unsat core
            simplify: Whether to simplify the constraint
        """
        if simplify is None:
            simplify = self.config.simplify_constraints

        if simplify:
            constraint = z3.simplify(constraint)

        if track and self.config.track_unsat_core:
            tracker = z3.Bool(f"constraint_{len(self.constraints)}")
            self.solver.assert_and_track(constraint, tracker)
        else:
            self.solver.add(constraint)

        self.constraints.append(constraint)
        self.assertions_stack[-1].append(constraint)
        self.stats.num_constraints += 1

    def push(self):
        """Create a new scope for assertions."""
        self.solver.push()
        self.assertions_stack.append([])

    def pop(self):
        """Pop the current scope of assertions."""
        self.solver.pop()
        popped_assertions = self.assertions_stack.pop()
        self.stats.num_constraints -= len(popped_assertions)

    def check_sat(self, timeout: Optional[int] = None) -> bool:
        """
        Check if current constraints are satisfiable.

        Args:
            timeout: Optional timeout in milliseconds

        Returns:
            True if satisfiable, False if unsatisfiable

        Raises:
            SolverTimeoutError: If solver times out
            ConstraintError: For other solver errors
        """
        start_time = time.time()
        self.stats.num_sat_checks += 1

        try:
            if timeout:
                self.solver.set(timeout=timeout)
            result = self.solver.check()

            if result == z3.sat:
                return True
            elif result == z3.unsat:
                self.stats.num_unsat_results += 1
                return False
            else:
                self.stats.num_unknown_results += 1
                raise ConstraintError("Solver returned unknown result")

        except z3.Z3Exception as e:
            if "timeout" in str(e):
                raise SolverTimeoutError("Solver timed out")
            raise ConstraintError(f"Solver error: {str(e)}")
        finally:
            self.stats.solving_time += time.time() - start_time

    def get_model(self, partial: bool = False) -> Optional[dict[str, Any]]:
        """
        Get a model (solution) satisfying the constraints.

        Args:
            partial: Whether to return partial solutions

        Returns:
            Dictionary mapping variable names to values
        """
        if not self.check_sat():
            return None

        model = self.solver.model()
        result = {}

        for var_name, var in self.variables.items():
            try:
                value = model.eval(var, model_completion=not partial)
                result[var_name] = self._convert_z3_value(value)
            except z3.Z3Exception:
                if not partial:
                    raise

        return result

    def get_unsat_core(self) -> list[Any]:
        """Get the unsatisfiable core if constraints are unsatisfiable."""
        if not self.config.track_unsat_core:
            raise ValueError("Unsat core tracking not enabled")

        if self.check_sat():
            return []

        return self.solver.unsat_core()

    def minimize(
        self, objective: Any, timeout: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        """
        Find a model that minimizes the objective.

        Args:
            objective: Expression to minimize
            timeout: Optional timeout in milliseconds

        Returns:
            Optimal model if found, None if unsatisfiable
        """
        optimizer = z3.Optimize()

        # Add all current constraints
        for constraint in self.constraints:
            optimizer.add(constraint)

        # Add minimization objective
        optimizer.minimize(objective)

        if timeout:
            optimizer.set(timeout=timeout)

        try:
            if optimizer.check() == z3.sat:
                model = optimizer.model()
                return {
                    var_name: self._convert_z3_value(model.eval(var))
                    for var_name, var in self.variables.items()
                }
        except z3.Z3Exception as e:
            logging.error(f"Optimization error: {str(e)}")

        return None

    def get_statistics(self) -> SolverStatistics:
        """Get solving statistics."""
        return self.stats

    def simplify_constraints(self) -> list[Any]:
        """Simplify current set of constraints."""
        simplified = []
        for constraint in self.constraints:
            simplified.append(z3.simplify(constraint))
        return simplified

    def to_smt_lib(self) -> str:
        """Convert constraints to SMT-LIB format."""
        return self.solver.to_smt_string()

    def _convert_z3_value(self, value: Any) -> Any:
        """Convert Z3 value to Python value."""
        if z3.is_int_value(value):
            return value.as_long()
        elif z3.is_bool_value(value):
            return z3.is_true(value)
        elif z3.is_real_value(value):
            return float(value.as_decimal(10))
        elif z3.is_bv_value(value):
            return value.as_long()
        return str(value)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ConstraintSolver")


# Convenience functions
def create_solver(config: Optional[SolverConfig] = None) -> ConstraintSolver:
    """Create a new constraint solver instance."""
    return ConstraintSolver(config)


def solve_constraints(
    constraints: list[Any], timeout: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """
    Solve a list of constraints.

    Args:
        constraints: List of constraints to solve
        timeout: Optional timeout in milliseconds

    Returns:
        Solution if found, None if unsatisfiable
    """
    solver = ConstraintSolver()
    for constraint in constraints:
        solver.add_constraint(constraint)
    return solver.get_model(timeout=timeout)


def is_satisfiable(constraints: list[Any]) -> bool:
    """Check if a list of constraints is satisfiable."""
    solver = ConstraintSolver()
    for constraint in constraints:
        solver.add_constraint(constraint)
    return solver.check_sat()
