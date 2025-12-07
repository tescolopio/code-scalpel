"""
Symbolic Analysis Engine - The Heart of code-scalpel's symbolic execution.

This module wires together the core components:
- TypeInferenceEngine: Infers Z3 types from Python AST
- SymbolicState: Tracks variables and path constraints with fork() isolation
- SymbolicInterpreter: Walks AST with smart forking and bounded loops
- ConstraintSolver: Marshals Z3 to Python natives for JSON/CLI consumption

PHASE 1 SCOPE (RFC-001): Integers and Booleans only.
"""
import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import z3

from .type_inference import InferredType, TypeInferenceEngine
from .state_manager import SymbolicState
from ..ir.normalizers.python_normalizer import PythonNormalizer
from ..ir.normalizers.javascript_normalizer import JavaScriptNormalizer
from ..ir.normalizers.java_normalizer import JavaNormalizer
from .ir_interpreter import IRSymbolicInterpreter
from .constraint_solver import ConstraintSolver, SolverStatus

logger = logging.getLogger(__name__)


# Emit warning on import - this is still experimental
warnings.warn(
    "symbolic_execution_tools is EXPERIMENTAL and incomplete. "
    "The engine will fail on most inputs. Use ast_tools or pdg_tools for production.",
    UserWarning,
    stacklevel=2,
)


class PathStatus(Enum):
    """Status of an explored execution path."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass
class PathResult:
    """Result of exploring a single execution path."""

    path_id: int
    status: PathStatus
    constraints: List[z3.BoolRef]
    variables: Dict[str, Any]  # Python native values (marshaled from Z3)
    model: Optional[Dict[str, Any]] = None  # Concrete satisfying assignment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to cache-serializable dictionary.

        Z3 constraints are converted to string representations.
        """
        return {
            "path_id": self.path_id,
            "status": self.status.value,
            "constraints": [str(c) for c in self.constraints],
            "variables": self.variables,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathResult":
        """Reconstruct from cached dictionary.

        Note: constraints are stored as strings (not executable Z3 objects).
        """
        return cls(
            path_id=data["path_id"],
            status=PathStatus(data["status"]),
            constraints=[],  # Z3 objects cannot be reconstructed from strings
            variables=data["variables"],
            model=data.get("model"),
        )


@dataclass
class AnalysisResult:
    """
    Complete result from symbolic analysis.

    This is the primary output format for CLI/MCP consumption.
    All values are Python natives (int, bool) - no raw Z3 objects.
    """

    paths: List[PathResult] = field(default_factory=list)
    all_variables: Dict[str, InferredType] = field(default_factory=dict)
    feasible_count: int = 0
    infeasible_count: int = 0
    total_paths: int = 0
    from_cache: bool = False  # True if result was retrieved from cache

    def get_feasible_paths(self) -> List[PathResult]:
        """Return only feasible paths."""
        return [p for p in self.paths if p.status == PathStatus.FEASIBLE]

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Return concrete models from all feasible paths."""
        return [p.model for p in self.paths if p.model is not None]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to cache-serializable dictionary."""
        return {
            "paths": [p.to_dict() for p in self.paths],
            "all_variables": {
                k: v.value if isinstance(v, InferredType) else str(v)
                for k, v in self.all_variables.items()
            },
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "total_paths": self.total_paths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Reconstruct from cached dictionary."""
        return cls(
            paths=[PathResult.from_dict(p) for p in data.get("paths", [])],
            all_variables={
                k: InferredType(v) if isinstance(v, str) else v
                for k, v in data.get("all_variables", {}).items()
            },
            feasible_count=data.get("feasible_count", 0),
            infeasible_count=data.get("infeasible_count", 0),
            total_paths=data.get("total_paths", 0),
            from_cache=True,
        )


class SymbolicAnalyzer:
    """
    High-level symbolic analysis interface.

    Wires together TypeInferenceEngine, SymbolicInterpreter, and ConstraintSolver
    to provide a clean API for symbolic execution.

    Example:
        >>> analyzer = SymbolicAnalyzer()
        >>> result = analyzer.analyze('''
        ... x = symbolic('x', int)
        ... if x > 10:
        ...     y = x + 5
        ... else:
        ...     y = x - 5
        ... ''')
        >>> print(result.feasible_count)
        2
        >>> print(result.get_all_models())
        [{'x': 11, 'y': 16}, {'x': 0, 'y': -5}]

    Note:
        PHASE 1: Only supports Int and Bool types. Float/String/List will raise errors.
    """

    def __init__(
        self,
        max_loop_iterations: int = 10,
        solver_timeout: int = 2000,
        enable_cache: bool = True,
    ):
        """
        Initialize the symbolic analyzer.

        Args:
            max_loop_iterations: Maximum iterations before terminating loops (default 10)
            solver_timeout: Z3 solver timeout in milliseconds (default 2000)
            enable_cache: Enable result caching for repeated analysis (default True)
        """
        self.max_loop_iterations = max_loop_iterations
        self.solver_timeout = solver_timeout
        self.enable_cache = enable_cache

        # Cache for expensive symbolic analysis
        self._cache = None
        if enable_cache:
            try:
                from ..utilities.cache import get_cache

                self._cache = get_cache()
            except ImportError:
                logger.warning("Cache module not available, caching disabled")

        # Core components - initialized fresh for each analysis
        self._type_engine: Optional[TypeInferenceEngine] = None
        self._interpreter: Optional[IRSymbolicInterpreter] = None
        self._solver: Optional[ConstraintSolver] = None

        # Manual symbolic declarations for advanced use
        self._preconditions: List[z3.BoolRef] = []
        self._declared_symbols: Dict[str, z3.ExprRef] = {}

    def _get_cache_config(self, language: str) -> Dict[str, Any]:
        """Generate cache configuration key components."""
        return {
            "language": language,
            "max_loop_iterations": self.max_loop_iterations,
            "solver_timeout": self.solver_timeout,
        }

    def analyze(self, code: str, language: str = "python") -> AnalysisResult:
        """
        Perform symbolic analysis on source code.

        Results are cached to avoid expensive Z3 solving on repeated analysis.
        Cache key: SHA256(code + tool_version + config)

        Args:
            code: Source code string
            language: Source language ("python", "javascript", or "java")

        Returns:
            AnalysisResult with all explored paths and their models

        Raises:
            SyntaxError: If code cannot be parsed
            NotImplementedError: If code uses unsupported constructs
            ValueError: If language is not supported
        """
        cache_config = self._get_cache_config(language)

        # Check cache first (hashing is cheap, Z3 solving is expensive)
        if self._cache:
            cached = self._cache.get(code, "symbolic", cache_config)
            if cached is not None:
                logger.debug("Cache hit for symbolic analysis")
                if isinstance(cached, dict):
                    return AnalysisResult.from_dict(cached)
                return cached

        # Cache miss - perform expensive symbolic analysis
        result = self._analyze_uncached(code, language)

        # Store in cache for future runs
        if self._cache:
            try:
                self._cache.set(code, "symbolic", result.to_dict(), cache_config)
                logger.debug("Cached symbolic analysis result")
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

        return result

    def _analyze_uncached(self, code: str, language: str) -> AnalysisResult:
        """Perform symbolic analysis without caching (internal method)."""
        # Fresh components for this analysis
        self._type_engine = TypeInferenceEngine()
        self._solver = ConstraintSolver(timeout_ms=self.solver_timeout)
        self._interpreter = IRSymbolicInterpreter(
            max_loop_iterations=self.max_loop_iterations
        )

        # Step 1: Type inference (Python only for now)
        inferred_types = {}
        if language == "python":
            inferred_types = self._type_engine.infer(code)

        # Step 2: Normalize to IR and execute symbolically
        try:
            if language == "python":
                ir_module = PythonNormalizer().normalize(code)
            elif language == "javascript":
                ir_module = JavaScriptNormalizer().normalize(code)
            elif language == "java":
                ir_module = JavaNormalizer().normalize(code)
            else:
                raise ValueError(f"Unsupported language: {language}")
        except SyntaxError as e:
            raise ValueError(f"Invalid {language} syntax: {e}")

        execution_result = self._interpreter.execute(ir_module)
        terminal_states = execution_result.states

        # Step 3: Process each path through solver
        result = AnalysisResult(
            all_variables=inferred_types,
            total_paths=len(terminal_states),
        )

        for i, state in enumerate(terminal_states):
            path_result = self._process_path(i, state)
            result.paths.append(path_result)

            if path_result.status == PathStatus.FEASIBLE:
                result.feasible_count += 1
            elif path_result.status == PathStatus.INFEASIBLE:
                result.infeasible_count += 1

        return result

    def _process_path(self, path_id: int, state: SymbolicState) -> PathResult:
        """Process a single execution path through the solver."""
        # Build list of Z3 constraints
        constraints = list(state.constraints)

        # Add any preconditions from manual declarations
        constraints.extend(self._preconditions)

        # Get variables from state
        state_vars = state.variables
        variables_list = list(state_vars.values())

        # Check satisfiability
        solver_result = self._solver.solve(constraints, variables_list)

        if solver_result.status == SolverStatus.SAT:
            # Extract variable values (already marshaled to Python natives)
            variables = {}
            for name in state_vars.keys():
                if solver_result.model and name in solver_result.model:
                    variables[name] = solver_result.model[name]
                else:
                    # Variable not in model - might be unconstrained
                    variables[name] = None

            return PathResult(
                path_id=path_id,
                status=PathStatus.FEASIBLE,
                constraints=constraints,
                variables=variables,
                model=solver_result.model,
            )
        elif solver_result.status == SolverStatus.UNSAT:
            return PathResult(
                path_id=path_id,
                status=PathStatus.INFEASIBLE,
                constraints=constraints,
                variables={},
            )
        else:
            # UNKNOWN or TIMEOUT
            return PathResult(
                path_id=path_id,
                status=PathStatus.UNKNOWN,
                constraints=constraints,
                variables={},
            )

    def declare_symbolic(self, name: str, sort: z3.SortRef) -> z3.ExprRef:
        """
        Manually declare a symbolic variable.

        This is for advanced use when you want to constrain inputs
        before calling analyze().

        Args:
            name: Variable name
            sort: Z3 sort (z3.IntSort(), z3.BoolSort(), z3.StringSort())

        Returns:
            Z3 expression reference for the symbolic variable

        Example:
            >>> analyzer = SymbolicAnalyzer()
            >>> x = analyzer.declare_symbolic('x', z3.IntSort())
            >>> analyzer.add_precondition(x > 0)
            >>> result = analyzer.analyze('y = x * 2')
        """
        if sort == z3.IntSort():
            var = z3.Int(name)
        elif sort == z3.BoolSort():
            var = z3.Bool(name)
        elif sort == z3.StringSort():
            var = z3.String(name)
        else:
            raise NotImplementedError(
                f"Only IntSort, BoolSort, and StringSort supported, got {sort}"
            )

        self._declared_symbols[name] = var
        return var

    def add_precondition(self, constraint: z3.BoolRef) -> None:
        """
        Add a precondition constraint.

        All preconditions are added to every path during analysis.

        Args:
            constraint: Z3 boolean constraint
        """
        self._preconditions.append(constraint)

    def find_inputs(self, target_condition: z3.BoolRef) -> Optional[Dict[str, Any]]:
        """
        Find input values that make a target condition true.

        This is the "reverse" query: given a target (e.g., error condition),
        find inputs that trigger it.

        Args:
            target_condition: Z3 boolean expression representing target

        Returns:
            Dictionary of variable names to concrete values, or None if impossible

        Example:
            >>> analyzer = SymbolicAnalyzer()
            >>> x = analyzer.declare_symbolic('x', z3.IntSort())
            >>> result = analyzer.find_inputs(x * x == 16)
            >>> print(result)  # {'x': 4} or {'x': -4}
        """
        if self._solver is None:
            self._solver = ConstraintSolver(timeout_ms=self.solver_timeout)

        constraints = list(self._preconditions) + [target_condition]
        solver_result = self._solver.check(constraints)

        if solver_result.status == SolverStatus.SAT:
            return solver_result.model
        return None

    def get_solver(self) -> ConstraintSolver:
        """Get the underlying constraint solver for advanced use."""
        if self._solver is None:
            self._solver = ConstraintSolver(timeout_ms=self.solver_timeout)
        return self._solver

    def reset(self) -> None:
        """Reset analyzer state for fresh analysis."""
        self._preconditions.clear()
        self._declared_symbols.clear()
        self._type_engine = None
        self._interpreter = None
        self._solver = None


# Legacy alias for backward compatibility
SymbolicExecutionEngine = SymbolicAnalyzer


def create_analyzer(
    max_loop_iterations: int = 10,
    solver_timeout: int = 2000,
) -> SymbolicAnalyzer:
    """
    Create a new symbolic analyzer.

    Factory function for creating analyzers with custom configuration.

    Args:
        max_loop_iterations: Maximum loop iterations (default 10)
        solver_timeout: Solver timeout in ms (default 2000)

    Returns:
        Configured SymbolicAnalyzer instance
    """
    return SymbolicAnalyzer(
        max_loop_iterations=max_loop_iterations,
        solver_timeout=solver_timeout,
    )
