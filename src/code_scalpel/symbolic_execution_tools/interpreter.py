"""
SymbolicInterpreter - The Nervous System of Symbolic Execution.

This module provides the AST interpreter that:
- Walks Python code and updates symbolic state
- Handles branching with SMART FORKING (feasibility check before fork)
- Prunes dead paths to avoid zombie path explosion

CRITICAL DESIGN DECISION: Smart Forking
========================================
Naive symbolic execution forks blindly at every branch:
    if x > 10:  --> Fork! Create Path A and Path B

But what if we already know x = 5? Then x > 10 is ALWAYS FALSE.
Forking blindly creates a "zombie path" that wastes resources.

SMART FORKING:
1. Check: Is `condition` feasible given current constraints?
2. Check: Is `NOT condition` feasible given current constraints?
3. Fork ONLY if BOTH are feasible
4. Otherwise, continue down the single feasible path

This prevents exponential path explosion from dead branches.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from copy import copy

from z3 import (
    ExprRef,
    BoolRef,
    ArithRef,
    Sort,
    IntSort,
    BoolSort,
    Int,
    Bool,
    IntVal,
    BoolVal,
    Solver,
    And,
    Or,
    Not,
    sat,
    unsat,
    is_true,
    is_false,
    simplify,
)

from .state_manager import SymbolicState
from .type_inference import TypeInferenceEngine, InferredType


@dataclass
class ExecutionResult:
    """
    Result of symbolic execution.
    
    Contains all terminal states (paths that reached the end of execution)
    and metadata about the execution.
    
    Attributes:
        states: All terminal symbolic states
        path_count: Total number of paths explored (including pruned)
        pruned_count: Number of infeasible paths that were pruned
    """
    states: List[SymbolicState] = field(default_factory=list)
    path_count: int = 0
    pruned_count: int = 0
    
    def feasible_states(self) -> List[SymbolicState]:
        """Get only the feasible (satisfiable) terminal states."""
        return [s for s in self.states if s.is_feasible()]
    
    def __repr__(self) -> str:
        return (
            f"ExecutionResult(paths={self.path_count}, "
            f"terminal={len(self.states)}, pruned={self.pruned_count})"
        )


class SymbolicInterpreter(ast.NodeVisitor):
    """
    Symbolic interpreter that walks Python AST and maintains symbolic state.
    
    Supports:
    - Integer and boolean variables (per Phase 1 scope)
    - Arithmetic and boolean expressions
    - If/elif/else branches with SMART FORKING
    - Dead path pruning
    
    Does NOT support (yet):
    - Loops (M5)
    - Function calls
    - Classes/objects
    - Strings, floats, lists, dicts
    
    Example:
        interp = SymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        result = interp.execute("if x > 0: y = 1")
        print(result.states)  # Two states: x>0 and x<=0
    """
    
    def __init__(self):
        """Initialize the interpreter."""
        self._initial_state: SymbolicState = SymbolicState()
        self._type_engine: TypeInferenceEngine = TypeInferenceEngine()
        self._preconditions: List[BoolRef] = []
        
    # =========================================================================
    # Setup API
    # =========================================================================
    
    def declare_symbolic(self, name: str, sort: Sort) -> ExprRef:
        """
        Declare a symbolic input variable.
        
        Symbolic inputs are unconstrained variables that can take any value.
        They represent unknown inputs to the program.
        
        Args:
            name: Variable name
            sort: Z3 sort (IntSort() or BoolSort())
            
        Returns:
            The Z3 expression for the variable
        """
        return self._initial_state.create_variable(name, sort)
    
    def add_precondition(self, constraint: BoolRef) -> None:
        """
        Add a precondition that constrains symbolic inputs.
        
        Preconditions are added to the initial state before execution.
        They narrow the space of possible inputs.
        
        Args:
            constraint: A Z3 boolean expression
        """
        self._preconditions.append(constraint)
    
    # =========================================================================
    # Main Execution
    # =========================================================================
    
    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code symbolically.
        
        Args:
            code: Python source code
            
        Returns:
            ExecutionResult with all terminal states
        """
        # Handle empty code
        if not code or not code.strip():
            return ExecutionResult(states=[self._initial_state], path_count=1)
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        
        # Apply preconditions to initial state
        for pre in self._preconditions:
            self._initial_state.add_constraint(pre)
        
        # Execute with initial state, collect terminal states
        result = ExecutionResult()
        terminal_states = self._execute_block(tree.body, self._initial_state, result)
        result.states = terminal_states
        
        return result
    
    def _execute_block(
        self, 
        statements: List[ast.stmt], 
        state: SymbolicState,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """
        Execute a block of statements.
        
        Args:
            statements: List of AST statements
            state: Current symbolic state
            result: ExecutionResult to track path count
            
        Returns:
            List of terminal states after executing the block
        """
        current_states = [state]
        
        for stmt in statements:
            next_states = []
            for s in current_states:
                # Execute statement, may produce multiple states (branching)
                produced = self._execute_statement(stmt, s, result)
                next_states.extend(produced)
            current_states = next_states
            
            # Early exit if no states left (all paths infeasible)
            if not current_states:
                break
        
        return current_states
    
    def _execute_statement(
        self, 
        stmt: ast.stmt, 
        state: SymbolicState,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """
        Execute a single statement.
        
        Args:
            stmt: AST statement node
            state: Current symbolic state
            result: ExecutionResult to track path count
            
        Returns:
            List of states after execution (1 for linear, 2+ for branches)
        """
        if isinstance(stmt, ast.Assign):
            return self._execute_assign(stmt, state)
        elif isinstance(stmt, ast.AugAssign):
            return self._execute_aug_assign(stmt, state)
        elif isinstance(stmt, ast.If):
            return self._execute_if(stmt, state, result)
        elif isinstance(stmt, ast.Pass):
            return [state]
        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., function call as statement)
            # We evaluate for side effects but don't do anything with result
            return [state]
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip function/class definitions (not executed at module level)
            return [state]
        elif isinstance(stmt, (ast.For, ast.While)):
            # Loops not yet supported (M5)
            # For now, skip them with a warning
            return [state]
        elif isinstance(stmt, ast.Import) or isinstance(stmt, ast.ImportFrom):
            # Skip imports
            return [state]
        else:
            # Unsupported statement type - continue without error
            return [state]
    
    # =========================================================================
    # Assignment Handling
    # =========================================================================
    
    def _execute_assign(
        self, 
        stmt: ast.Assign, 
        state: SymbolicState
    ) -> List[SymbolicState]:
        """
        Execute an assignment statement: x = expr
        
        Args:
            stmt: AST Assign node
            state: Current symbolic state
            
        Returns:
            List with single updated state
        """
        # Get the value expression as Z3
        value_expr = self._eval_expr(stmt.value, state)
        
        # Handle each target
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                name = target.id
                
                if value_expr is not None:
                    # We have a concrete Z3 expression
                    state.set_variable(name, value_expr)
                else:
                    # Unknown type - create placeholder
                    # We still track the variable exists, but can't reason about it
                    if not state.has_variable(name):
                        # Create as Int by default (will be UNKNOWN in type inference)
                        state.create_variable(name, IntSort())
            elif isinstance(target, ast.Tuple):
                # Tuple unpacking - simplified handling
                # For now, just track that variables exist
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        if not state.has_variable(elt.id):
                            state.create_variable(elt.id, IntSort())
        
        return [state]
    
    def _execute_aug_assign(
        self, 
        stmt: ast.AugAssign, 
        state: SymbolicState
    ) -> List[SymbolicState]:
        """
        Execute augmented assignment: x += 1, x -= 1, etc.
        
        Args:
            stmt: AST AugAssign node
            state: Current symbolic state
            
        Returns:
            List with single updated state
        """
        if not isinstance(stmt.target, ast.Name):
            return [state]
        
        name = stmt.target.id
        current = state.get_variable(name)
        
        if current is None:
            # Variable doesn't exist - create it
            current = state.create_variable(name, IntSort())
        
        right = self._eval_expr(stmt.value, state)
        
        if right is not None and isinstance(current, ArithRef):
            # Compute new value based on operator
            if isinstance(stmt.op, ast.Add):
                new_value = current + right
            elif isinstance(stmt.op, ast.Sub):
                new_value = current - right
            elif isinstance(stmt.op, ast.Mult):
                new_value = current * right
            elif isinstance(stmt.op, ast.FloorDiv):
                new_value = current / right  # Z3 integer division
            elif isinstance(stmt.op, ast.Mod):
                new_value = current % right
            else:
                new_value = current  # Unsupported operator
            
            state.set_variable(name, new_value)
        
        return [state]
    
    # =========================================================================
    # Expression Evaluation
    # =========================================================================
    
    def _eval_expr(self, expr: ast.expr, state: SymbolicState) -> Optional[ExprRef]:
        """
        Evaluate an expression to a Z3 expression.
        
        Args:
            expr: AST expression node
            state: Current symbolic state
            
        Returns:
            Z3 expression, or None if unsupported
        """
        if isinstance(expr, ast.Constant):
            return self._eval_constant(expr)
        elif isinstance(expr, ast.Name):
            return state.get_variable(expr.id)
        elif isinstance(expr, ast.BinOp):
            return self._eval_binop(expr, state)
        elif isinstance(expr, ast.UnaryOp):
            return self._eval_unaryop(expr, state)
        elif isinstance(expr, ast.Compare):
            return self._eval_compare(expr, state)
        elif isinstance(expr, ast.BoolOp):
            return self._eval_boolop(expr, state)
        elif isinstance(expr, ast.IfExp):
            # Ternary: x if cond else y
            # For now, return None (complex to handle properly)
            return None
        elif isinstance(expr, ast.Call):
            # Function calls are unsupported
            return None
        elif isinstance(expr, ast.Attribute):
            # Attribute access unsupported
            return None
        elif isinstance(expr, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            # Collections unsupported
            return None
        else:
            return None
    
    def _eval_constant(self, expr: ast.Constant) -> Optional[ExprRef]:
        """Evaluate a constant literal."""
        value = expr.value
        
        if isinstance(value, bool):
            return BoolVal(value)
        elif isinstance(value, int):
            return IntVal(value)
        else:
            # Floats, strings, etc. not supported
            return None
    
    def _eval_binop(
        self, 
        expr: ast.BinOp, 
        state: SymbolicState
    ) -> Optional[ExprRef]:
        """Evaluate a binary operation."""
        left = self._eval_expr(expr.left, state)
        right = self._eval_expr(expr.right, state)
        
        if left is None or right is None:
            return None
        
        op = expr.op
        
        # Arithmetic operations
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.FloorDiv):
            return left / right  # Z3 integer division
        elif isinstance(op, ast.Mod):
            return left % right
        else:
            return None
    
    def _eval_unaryop(
        self, 
        expr: ast.UnaryOp, 
        state: SymbolicState
    ) -> Optional[ExprRef]:
        """Evaluate a unary operation."""
        operand = self._eval_expr(expr.operand, state)
        
        if operand is None:
            return None
        
        op = expr.op
        
        if isinstance(op, ast.USub):
            return -operand
        elif isinstance(op, ast.UAdd):
            return operand
        elif isinstance(op, ast.Not):
            return Not(operand)
        else:
            return None
    
    def _eval_compare(
        self, 
        expr: ast.Compare, 
        state: SymbolicState
    ) -> Optional[BoolRef]:
        """Evaluate a comparison expression."""
        left = self._eval_expr(expr.left, state)
        
        if left is None:
            return None
        
        # Handle chained comparisons: a < b < c
        result = None
        current_left = left
        
        for op, comparator in zip(expr.ops, expr.comparators):
            right = self._eval_expr(comparator, state)
            
            if right is None:
                return None
            
            # Evaluate single comparison
            if isinstance(op, ast.Lt):
                cmp_result = current_left < right
            elif isinstance(op, ast.LtE):
                cmp_result = current_left <= right
            elif isinstance(op, ast.Gt):
                cmp_result = current_left > right
            elif isinstance(op, ast.GtE):
                cmp_result = current_left >= right
            elif isinstance(op, ast.Eq):
                cmp_result = current_left == right
            elif isinstance(op, ast.NotEq):
                cmp_result = current_left != right
            else:
                return None
            
            # Chain with And
            if result is None:
                result = cmp_result
            else:
                result = And(result, cmp_result)
            
            current_left = right
        
        return result
    
    def _eval_boolop(
        self, 
        expr: ast.BoolOp, 
        state: SymbolicState
    ) -> Optional[BoolRef]:
        """Evaluate a boolean operation (and/or)."""
        values = [self._eval_expr(v, state) for v in expr.values]
        
        if any(v is None for v in values):
            return None
        
        if isinstance(expr.op, ast.And):
            return And(*values)
        elif isinstance(expr.op, ast.Or):
            return Or(*values)
        else:
            return None
    
    # =========================================================================
    # If Statement - SMART FORKING
    # =========================================================================
    
    def _execute_if(
        self, 
        stmt: ast.If, 
        state: SymbolicState,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """
        Execute an if statement with SMART FORKING.
        
        CRITICAL: We check feasibility BEFORE forking to avoid zombie paths.
        
        Args:
            stmt: AST If node
            state: Current symbolic state
            result: ExecutionResult to track path count
            
        Returns:
            List of states after branch execution
        """
        # Evaluate the condition
        condition = self._eval_expr(stmt.test, state)
        
        if condition is None:
            # Can't evaluate condition - conservatively fork
            # This handles cases like `if some_function():`
            return self._fork_blind(stmt, state, result)
        
        # SMART FORKING: Check feasibility of both branches
        true_feasible = self._is_feasible_with(state, condition)
        false_feasible = self._is_feasible_with(state, Not(condition))
        
        result.path_count += 1  # Count this branch point
        
        terminal_states = []
        
        if true_feasible and false_feasible:
            # BOTH branches are feasible - must fork
            terminal_states.extend(
                self._execute_true_branch(stmt, state, condition, result)
            )
            terminal_states.extend(
                self._execute_false_branch(stmt, state, condition, result)
            )
        elif true_feasible:
            # Only TRUE branch is feasible - no fork needed
            result.pruned_count += 1
            terminal_states.extend(
                self._execute_true_branch(stmt, state, condition, result)
            )
        elif false_feasible:
            # Only FALSE branch is feasible - no fork needed
            result.pruned_count += 1
            terminal_states.extend(
                self._execute_false_branch(stmt, state, condition, result)
            )
        else:
            # BOTH branches infeasible - this state is dead
            result.pruned_count += 2
            # Return empty list - this path terminates
        
        return terminal_states
    
    def _is_feasible_with(self, state: SymbolicState, extra_constraint: BoolRef) -> bool:
        """
        Check if adding a constraint keeps the path feasible.
        
        Args:
            state: Current symbolic state
            extra_constraint: Additional constraint to check
            
        Returns:
            True if path remains satisfiable
        """
        solver = Solver()
        
        # Add existing constraints
        for c in state.constraints:
            solver.add(c)
        
        # Add the new constraint
        solver.add(extra_constraint)
        
        return solver.check() == sat
    
    def _execute_true_branch(
        self, 
        stmt: ast.If, 
        state: SymbolicState, 
        condition: BoolRef,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """Execute the true branch of an if statement."""
        # Fork state and add true condition
        true_state = state.fork()
        true_state.add_constraint(condition)
        
        # Execute true body
        return self._execute_block(stmt.body, true_state, result)
    
    def _execute_false_branch(
        self, 
        stmt: ast.If, 
        state: SymbolicState, 
        condition: BoolRef,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """Execute the false/else branch of an if statement."""
        # Fork state and add negated condition
        false_state = state.fork()
        false_state.add_constraint(Not(condition))
        
        if stmt.orelse:
            # Has else/elif
            return self._execute_block(stmt.orelse, false_state, result)
        else:
            # No else - state continues without executing anything
            return [false_state]
    
    def _fork_blind(
        self, 
        stmt: ast.If, 
        state: SymbolicState,
        result: ExecutionResult
    ) -> List[SymbolicState]:
        """
        Fallback: Fork without feasibility check.
        
        Used when we can't evaluate the condition (e.g., function call).
        """
        terminal_states = []
        
        # Execute true branch
        true_state = state.fork()
        terminal_states.extend(
            self._execute_block(stmt.body, true_state, result)
        )
        
        # Execute false/else branch
        false_state = state.fork()
        if stmt.orelse:
            terminal_states.extend(
                self._execute_block(stmt.orelse, false_state, result)
            )
        else:
            terminal_states.append(false_state)
        
        result.path_count += 1
        return terminal_states
