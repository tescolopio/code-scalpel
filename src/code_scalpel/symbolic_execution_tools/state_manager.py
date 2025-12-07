"""
SymbolicState - The Memory Model for Symbolic Execution.

This module provides the state management for symbolic execution, handling:
- Symbolic variable creation and storage
- Path condition accumulation
- State forking with TOTAL ISOLATION

CRITICAL DESIGN DECISION: Copy-on-Write / Deep Copy
====================================================
When symbolic execution hits a branch (if/else), the universe splits.
Both paths must be COMPLETELY INDEPENDENT.

The WRONG way (causes "shallow copy suicide"):
    state_b = state_a  # Same reference!
    state_b.variables["x"] = 5  # Corrupts state_a!

The RIGHT way (this implementation):
    state_b = state_a.fork()  # Deep copy of mutable structures
    state_b.set_variable("x", 5)  # state_a is untouched

Z3 objects (ExprRef, BoolRef, Sort) are IMMUTABLE (C++ bindings).
They can be safely shared. Only the Python containers (dict, list) 
need to be copied.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from z3 import (
    ExprRef,
    BoolRef,
    Sort,
    IntSort,
    BoolSort,
    StringSort,
    Int,
    Bool,
    String,
    Solver,
    And,
    sat,
)


@dataclass
class SymbolicVariable:
    """
    Wrapper around a Z3 symbolic variable with metadata.

    Provides a clean interface for creating and accessing symbolic variables
    with their associated type information.

    Attributes:
        name: The variable name (e.g., "x", "counter")
        sort: The Z3 sort (IntSort(), BoolSort(), StringSort())
        expr: The underlying Z3 expression
    """

    name: str
    sort: Sort
    expr: ExprRef = field(init=False)

    def __post_init__(self):
        """Create the Z3 expression based on the sort."""
        if self.sort == IntSort():
            self.expr = Int(self.name)
        elif self.sort == BoolSort():
            self.expr = Bool(self.name)
        elif self.sort == StringSort():
            self.expr = String(self.name)
        else:
            raise ValueError(f"Unsupported sort: {self.sort}")


class SymbolicState:
    """
    Manages the symbolic state during execution.

    This is the memory model for symbolic execution. It tracks:
    - Symbolic variables and their current bindings
    - Path conditions (accumulated constraints)
    - Fork depth for debugging

    CRITICAL: The fork() method provides TOTAL ISOLATION between branches.

    Example:
        # Create initial state
        state = SymbolicState()
        x = state.create_variable("x", IntSort())
        state.add_constraint(x > 0)

        # Branch: if x > 10
        true_branch = state.fork()
        true_branch.add_constraint(x > 10)

        false_branch = state.fork()
        false_branch.add_constraint(Not(x > 10))

        # Original state is UNTOUCHED
        assert len(state.constraints) == 1  # Only x > 0
    """

    def __init__(self, depth: int = 0):
        """
        Initialize a new symbolic state.

        Args:
            depth: Fork depth (0 for root state, increments on fork)
        """
        self._variables: Dict[str, ExprRef] = {}
        self._constraints: List[BoolRef] = []
        self._depth: int = depth

    # =========================================================================
    # Variable Management
    # =========================================================================

    def create_variable(self, name: str, sort: Sort) -> ExprRef:
        """
        Create a new symbolic variable.

        Args:
            name: Variable name
            sort: Z3 sort (IntSort(), BoolSort(), or StringSort())

        Returns:
            The Z3 expression for the variable

        Raises:
            ValueError: If variable already exists with different sort
        """
        if name in self._variables:
            existing = self._variables[name]
            if existing.sort() != sort:
                raise ValueError(
                    f"Variable '{name}' already exists with sort {existing.sort()}, "
                    f"cannot create with sort {sort}"
                )
            return existing

        # Create the Z3 symbolic variable
        if sort == IntSort():
            expr = Int(name)
        elif sort == BoolSort():
            expr = Bool(name)
        elif sort == StringSort():
            expr = String(name)
        else:
            raise ValueError(
                f"Unsupported sort: {sort}. Only IntSort, BoolSort, and StringSort are supported."
            )

        self._variables[name] = expr
        return expr

    def get_variable(self, name: str) -> Optional[ExprRef]:
        """
        Get a variable by name.

        Args:
            name: Variable name

        Returns:
            The Z3 expression, or None if not found
        """
        return self._variables.get(name)

    def set_variable(self, name: str, expr: ExprRef) -> None:
        """
        Bind a variable to a new expression.

        This is used when a variable is reassigned:
            x = x + 1  -->  state.set_variable("x", x + 1)

        Args:
            name: Variable name
            expr: The new Z3 expression
        """
        self._variables[name] = expr

    def has_variable(self, name: str) -> bool:
        """
        Check if a variable exists.

        Args:
            name: Variable name

        Returns:
            True if the variable exists
        """
        return name in self._variables

    def variable_names(self) -> List[str]:
        """
        Get all variable names.

        Returns:
            List of variable names
        """
        return list(self._variables.keys())

    @property
    def variables(self) -> Dict[str, ExprRef]:
        """
        Get a copy of the variables dictionary.

        Returns:
            Dictionary mapping names to Z3 expressions

        Note:
            Returns a copy to prevent external mutation.
        """
        return self._variables.copy()

    # =========================================================================
    # Path Condition Management
    # =========================================================================

    @property
    def constraints(self) -> List[BoolRef]:
        """
        Get the list of path constraints.

        Returns:
            List of Z3 boolean expressions
        """
        return self._constraints

    def add_constraint(self, constraint: BoolRef) -> None:
        """
        Add a path constraint.

        Constraints accumulate as execution proceeds through branches.

        Args:
            constraint: A Z3 boolean expression
        """
        self._constraints.append(constraint)

    def path_condition(self) -> BoolRef:
        """
        Get the conjunction of all path constraints.

        Returns:
            A single Z3 And expression, or True if no constraints
        """
        if not self._constraints:
            return Bool("__true__") == Bool("__true__")  # Trivially true

        if len(self._constraints) == 1:
            return self._constraints[0]

        return And(*self._constraints)

    def is_feasible(self) -> bool:
        """
        Check if the current path is satisfiable.

        Uses Z3 solver to determine if there exists an assignment
        that satisfies all path constraints.

        Returns:
            True if the path is feasible (sat), False if infeasible (unsat)
        """
        if not self._constraints:
            return True  # No constraints = trivially satisfiable

        solver = Solver()
        solver.set("timeout", 5000)  # 5 second timeout to prevent hangs
        solver.add(*self._constraints)
        result = solver.check()
        # Treat unknown (timeout) as infeasible to fail safely
        return result == sat

    # =========================================================================
    # Fork - THE CRITICAL METHOD
    # =========================================================================

    @property
    def depth(self) -> int:
        """Get the fork depth."""
        return self._depth

    def fork(self) -> SymbolicState:
        """
        Create an isolated copy of this state.

        CRITICAL: This method provides TOTAL ISOLATION between branches.

        The forked state:
        - Has copies of all variables (not references)
        - Has a copy of the constraint list (not the same list)
        - Can be modified without affecting the parent

        Z3 objects (ExprRef, BoolRef) are immutable and safe to share.
        Only the Python containers (dict, list) are copied.

        Returns:
            A new SymbolicState that is completely independent
        """
        # Create new state with incremented depth
        forked = SymbolicState(depth=self._depth + 1)

        # CRITICAL: Copy the dictionary, not the reference
        # dict.copy() creates a shallow copy of the dict itself
        # The Z3 ExprRef values are immutable, so shallow copy is sufficient
        forked._variables = self._variables.copy()

        # CRITICAL: Copy the list, not the reference
        # list.copy() (or list[:]) creates a shallow copy of the list
        # The Z3 BoolRef values are immutable, so shallow copy is sufficient
        forked._constraints = self._constraints.copy()

        return forked

    # =========================================================================
    # Debug / Utility
    # =========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        var_str = ", ".join(f"{k}: {v.sort()}" for k, v in self._variables.items())
        return (
            f"SymbolicState(depth={self._depth}, "
            f"vars=[{var_str}], "
            f"constraints={len(self._constraints)})"
        )

    def summary(self) -> Dict:
        """
        Get a summary of the state for debugging.

        Returns:
            Dictionary with state information
        """
        return {
            "depth": self._depth,
            "variables": {
                name: str(expr.sort()) for name, expr in self._variables.items()
            },
            "constraint_count": len(self._constraints),
            "is_feasible": self.is_feasible(),
        }
