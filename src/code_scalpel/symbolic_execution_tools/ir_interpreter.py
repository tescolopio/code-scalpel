"""
IR Symbolic Interpreter - Language-Agnostic Symbolic Execution.

This module provides a symbolic interpreter that operates on Unified IR nodes
instead of Python AST nodes. This enables the same symbolic execution logic
to analyze Python, JavaScript, and any other language with an IR normalizer.

Key Design Decisions:
=====================

1. UNIFIED IR: We visit IRNode subclasses, not ast.AST nodes
   - IRAssign, IRBinaryOp, IRName, IRConstant, etc.
   - Same execution logic for all source languages

2. SEMANTIC DELEGATION: Language-specific behavior is delegated
   - Python: "5" + 3 -> TypeError  
   - JavaScript: "5" + 3 -> "53"
   - The interpreter calls self.semantics.binary_add(left, right)
   - LanguageSemantics implementations handle the differences

3. SMART FORKING: Same as the AST interpreter
   - Check feasibility before forking
   - Prune dead paths to avoid explosion

4. BOUNDED LOOPS: Same as the AST interpreter
   - Maximum iterations before pruning
   - Guarantees termination

Usage:
    from code_scalpel.ir.normalizers import PythonNormalizer
    from code_scalpel.symbolic_execution_tools.ir_interpreter import IRSymbolicInterpreter
    
    code = "x = 1 + 2"
    ir = PythonNormalizer().normalize(code)
    interp = IRSymbolicInterpreter()
    result = interp.execute(ir)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from z3 import (
    ArithRef,
    BoolRef,
    BoolSort,
    BoolVal,
    ExprRef,
    Int,
    IntSort,
    IntVal,
    Not,
    Or,
    Solver,
    Sort,
    String,
    StringSort,
    StringVal,
    sat,
)

from ..ir.nodes import (
    AnyIRNode,
    IRAugAssign,
    IRAssign,
    IRBinaryOp,
    IRBoolOp,
    IRBreak,
    IRCall,
    IRClassDef,
    IRCompare,
    IRConstant,
    IRContinue,
    IRExpr,
    IRExprStmt,
    IRFor,
    IRFunctionDef,
    IRIf,
    IRList,
    IRModule,
    IRName,
    IRNode,
    IRPass,
    IRReturn,
    IRUnaryOp,
    IRWhile,
)
from ..ir.operators import BinaryOperator, BoolOperator, CompareOperator, UnaryOperator
from .state_manager import SymbolicState


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class IRExecutionResult:
    """
    Result of IR symbolic execution.

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
            f"IRExecutionResult(paths={self.path_count}, "
            f"terminal={len(self.states)}, pruned={self.pruned_count})"
        )


# =============================================================================
# IR Node Visitor Base Class
# =============================================================================


class IRNodeVisitor(ABC):
    """
    Base class for IR node visitors.

    Unlike ast.NodeVisitor, this uses explicit isinstance checks since
    IR nodes are dataclasses, not ast.AST subclasses.

    Subclasses should implement visit_* methods for each node type.
    The generic_visit method is called for unhandled node types.
    """

    def visit(self, node: IRNode) -> Any:
        """
        Visit a node by dispatching to the appropriate visit_* method.

        Args:
            node: IR node to visit

        Returns:
            Result from the visit method
        """
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: IRNode) -> Any:
        """
        Called when no specific visit_* method exists.

        Default implementation does nothing. Override to raise errors
        or handle unknown nodes.
        """
        return None


# =============================================================================
# Language Semantics Protocol
# =============================================================================


class LanguageSemantics(ABC):
    """
    Abstract base class for language-specific semantics.

    Each language (Python, JavaScript, etc.) provides an implementation
    that handles type coercion, operator behavior, and truthiness rules.

    The interpreter delegates to these methods for language-specific behavior.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name (e.g., 'python', 'javascript')."""
        ...

    # -------------------------------------------------------------------------
    # Binary Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def binary_add(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """
        Handle addition: left + right

        Python: int + int -> int, str + str -> str, int + str -> TypeError
        JavaScript: int + str -> str (coercion)
        """
        ...

    @abstractmethod
    def binary_sub(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Handle subtraction: left - right"""
        ...

    @abstractmethod
    def binary_mul(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Handle multiplication: left * right"""
        ...

    @abstractmethod
    def binary_div(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Handle division: left / right (or //)"""
        ...

    @abstractmethod
    def binary_mod(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Handle modulo: left % right"""
        ...

    # -------------------------------------------------------------------------
    # Comparison Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def compare_eq(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle equality: left == right"""
        ...

    @abstractmethod
    def compare_ne(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle inequality: left != right"""
        ...

    @abstractmethod
    def compare_lt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle less than: left < right"""
        ...

    @abstractmethod
    def compare_le(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle less than or equal: left <= right"""
        ...

    @abstractmethod
    def compare_gt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle greater than: left > right"""
        ...

    @abstractmethod
    def compare_ge(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle greater than or equal: left >= right"""
        ...

    # -------------------------------------------------------------------------
    # Unary Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def unary_neg(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Handle negation: -operand"""
        ...

    @abstractmethod
    def unary_not(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        """Handle logical not: not operand"""
        ...

    # -------------------------------------------------------------------------
    # Truthiness
    # -------------------------------------------------------------------------

    @abstractmethod
    def to_bool(self, value: ExprRef, state: SymbolicState) -> Optional[BoolRef]:
        """
        Convert a value to boolean for conditionals.

        Python: 0, "", [], None are falsy
        JavaScript: 0, "", null, undefined, NaN are falsy
        """
        ...


# =============================================================================
# Python Semantics Implementation
# =============================================================================


class PythonSemantics(LanguageSemantics):
    """
    Python language semantics for symbolic execution.

    Implements Python's strict typing and operator behavior:
    - No implicit type coercion in arithmetic
    - String + String is concatenation
    - Int + String raises TypeError (returns None in symbolic execution)
    """

    @property
    def name(self) -> str:
        return "python"

    def binary_add(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Python addition: requires matching types."""
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left + right
        # String + String not yet supported in Z3 symbolic execution
        return None

    def binary_sub(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left - right
        return None

    def binary_mul(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left * right
        return None

    def binary_div(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            # Z3 integer division
            return left / right
        return None

    def binary_mod(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left % right
        return None

    def compare_eq(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        return left == right

    def compare_ne(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        return left != right

    def compare_lt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left < right
        return None

    def compare_le(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left <= right
        return None

    def compare_gt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left > right
        return None

    def compare_ge(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left >= right
        return None

    def unary_neg(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(operand, ArithRef):
            return -operand
        return None

    def unary_not(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        bool_val = self.to_bool(operand, state)
        if bool_val is not None:
            return Not(bool_val)
        return None

    def to_bool(self, value: ExprRef, state: SymbolicState) -> Optional[BoolRef]:
        """Python truthiness: 0 and False are falsy."""
        if isinstance(value, BoolRef):
            return value
        if isinstance(value, ArithRef):
            return value != IntVal(0)
        return None


# =============================================================================
# JavaScript Semantics Implementation (Stub for now)
# =============================================================================


class JavaScriptSemantics(LanguageSemantics):
    """
    JavaScript language semantics for symbolic execution.

    Implements JavaScript's loose typing and coercion rules.
    NOTE: This is a stub - full JS coercion is complex.
    """

    @property
    def name(self) -> str:
        return "javascript"

    def binary_add(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        """JavaScript addition with coercion (simplified)."""
        # For now, same as Python - full coercion is complex
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left + right
        return None

    def binary_sub(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left - right
        return None

    def binary_mul(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left * right
        return None

    def binary_div(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left / right
        return None

    def binary_mod(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left % right
        return None

    def compare_eq(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        # JavaScript == has coercion, but we use strict equality for simplicity
        return left == right

    def compare_ne(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        return left != right

    def compare_lt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left < right
        return None

    def compare_le(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left <= right
        return None

    def compare_gt(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left > right
        return None

    def compare_ge(
        self, left: ExprRef, right: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        if isinstance(left, ArithRef) and isinstance(right, ArithRef):
            return left >= right
        return None

    def unary_neg(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[ExprRef]:
        if isinstance(operand, ArithRef):
            return -operand
        return None

    def unary_not(
        self, operand: ExprRef, state: SymbolicState
    ) -> Optional[BoolRef]:
        bool_val = self.to_bool(operand, state)
        if bool_val is not None:
            return Not(bool_val)
        return None

    def to_bool(self, value: ExprRef, state: SymbolicState) -> Optional[BoolRef]:
        """JavaScript truthiness: 0, "", null, undefined, NaN are falsy."""
        if isinstance(value, BoolRef):
            return value
        if isinstance(value, ArithRef):
            return value != IntVal(0)
        return None


# =============================================================================
# Semantics Registry
# =============================================================================

_SEMANTICS_REGISTRY: Dict[str, Type[LanguageSemantics]] = {
    "python": PythonSemantics,
    "javascript": JavaScriptSemantics,
    "js": JavaScriptSemantics,
    "unknown": PythonSemantics,  # Default to Python
}


def get_semantics(language: str) -> LanguageSemantics:
    """
    Get the semantics implementation for a language.

    Args:
        language: Language name (e.g., 'python', 'javascript')

    Returns:
        LanguageSemantics instance for the language
    """
    cls = _SEMANTICS_REGISTRY.get(language.lower(), PythonSemantics)
    return cls()


# =============================================================================
# IR Symbolic Interpreter
# =============================================================================


class IRSymbolicInterpreter(IRNodeVisitor):
    """
    Symbolic interpreter that operates on Unified IR nodes.

    This enables language-agnostic symbolic execution:
    - Python code -> PythonNormalizer -> IR -> IRSymbolicInterpreter
    - JavaScript code -> JavaScriptNormalizer -> IR -> IRSymbolicInterpreter

    The same interpreter handles both, with language-specific behavior
    delegated to LanguageSemantics implementations.

    Supports:
    - Integer and boolean variables (per Phase 1 scope)
    - Arithmetic and boolean expressions
    - If/elif/else branches with SMART FORKING
    - While loops with BOUNDED UNROLLING
    - For loops with BOUNDED UNROLLING
    - Dead path pruning

    Example:
        from code_scalpel.ir.normalizers import PythonNormalizer
        
        code = '''
        x = symbolic('x', int)
        if x > 10:
            y = x + 5
        else:
            y = x - 5
        '''
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)
    """

    DEFAULT_MAX_LOOP_ITERATIONS = 10

    def __init__(
        self,
        max_loop_iterations: int = DEFAULT_MAX_LOOP_ITERATIONS,
        semantics: Optional[LanguageSemantics] = None,
    ):
        """
        Initialize the IR interpreter.

        Args:
            max_loop_iterations: Maximum loop iterations before pruning
            semantics: Language semantics to use (auto-detected from IR if None)
        """
        self.max_loop_iterations = max_loop_iterations
        self._default_semantics = semantics
        self._semantics: Optional[LanguageSemantics] = None
        self._initial_state: SymbolicState = SymbolicState()
        self._preconditions: List[BoolRef] = []

    # =========================================================================
    # Setup API
    # =========================================================================

    def declare_symbolic(self, name: str, sort: Sort) -> ExprRef:
        """
        Declare a symbolic input variable.

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

        Args:
            constraint: A Z3 boolean expression
        """
        self._preconditions.append(constraint)

    # =========================================================================
    # Main Execution
    # =========================================================================

    def execute(self, ir: IRModule) -> IRExecutionResult:
        """
        Execute an IR module symbolically.

        Args:
            ir: IR module to execute

        Returns:
            IRExecutionResult with all terminal states
        """
        # Determine semantics from IR source_language
        if self._default_semantics is not None:
            self._semantics = self._default_semantics
        else:
            self._semantics = get_semantics(ir.source_language)

        # Apply preconditions
        for pre in self._preconditions:
            self._initial_state.add_constraint(pre)

        # Execute
        result = IRExecutionResult()
        terminal_states = self._execute_block(ir.body, self._initial_state, result)
        result.states = terminal_states

        return result

    def _execute_block(
        self,
        statements: List[IRNode],
        state: SymbolicState,
        result: IRExecutionResult,
    ) -> List[SymbolicState]:
        """
        Execute a block of IR statements.

        Args:
            statements: List of IR statement nodes
            state: Current symbolic state
            result: IRExecutionResult to track path count

        Returns:
            List of terminal states after executing the block
        """
        current_states = [state]

        for stmt in statements:
            next_states = []
            for s in current_states:
                produced = self._execute_statement(stmt, s, result)
                next_states.extend(produced)
            current_states = next_states

            if not current_states:
                break

        return current_states

    def _execute_statement(
        self,
        stmt: IRNode,
        state: SymbolicState,
        result: IRExecutionResult,
    ) -> List[SymbolicState]:
        """
        Execute a single IR statement.

        Args:
            stmt: IR statement node
            state: Current symbolic state
            result: IRExecutionResult to track path count

        Returns:
            List of states after execution
        """
        if isinstance(stmt, IRAssign):
            return self._execute_assign(stmt, state)
        elif isinstance(stmt, IRAugAssign):
            return self._execute_aug_assign(stmt, state)
        elif isinstance(stmt, IRIf):
            return self._execute_if(stmt, state, result)
        elif isinstance(stmt, IRWhile):
            return self._execute_while(stmt, state, result)
        elif isinstance(stmt, IRFor):
            return self._execute_for(stmt, state, result)
        elif isinstance(stmt, IRPass):
            return [state]
        elif isinstance(stmt, IRExprStmt):
            # Expression statement - evaluate for side effects
            self._eval_expr(stmt.value, state)
            return [state]
        elif isinstance(stmt, (IRFunctionDef, IRClassDef)):
            # Skip definitions (not executed at module level)
            return [state]
        elif isinstance(stmt, IRReturn):
            # For now, just stop this path
            return [state]
        elif isinstance(stmt, (IRBreak, IRContinue)):
            # Loop control - handled by loop executors
            return [state]
        else:
            # Unknown statement - continue
            return [state]

    # =========================================================================
    # Assignment Handling
    # =========================================================================

    def _execute_assign(
        self,
        stmt: IRAssign,
        state: SymbolicState,
    ) -> List[SymbolicState]:
        """
        Execute an assignment: x = expr

        Args:
            stmt: IRAssign node
            state: Current symbolic state

        Returns:
            List with single updated state
        """
        value_expr = self._eval_expr(stmt.value, state)

        for target in stmt.targets:
            if isinstance(target, IRName):
                name = target.id
                if value_expr is not None:
                    state.set_variable(name, value_expr)
                elif not state.has_variable(name):
                    # Unknown type - create placeholder
                    state.create_variable(name, IntSort())

        return [state]

    def _execute_aug_assign(
        self,
        stmt: IRAugAssign,
        state: SymbolicState,
    ) -> List[SymbolicState]:
        """
        Execute augmented assignment: x += 1

        Args:
            stmt: IRAugAssign node
            state: Current symbolic state

        Returns:
            List with single updated state
        """
        if not isinstance(stmt.target, IRName):
            return [state]

        name = stmt.target.id
        current = state.get_variable(name)

        if current is None:
            current = state.create_variable(name, IntSort())

        right = self._eval_expr(stmt.value, state)

        if right is not None and self._semantics is not None:
            from ..ir.operators import AugAssignOperator

            if stmt.op == AugAssignOperator.ADD:
                new_value = self._semantics.binary_add(current, right, state)
            elif stmt.op == AugAssignOperator.SUB:
                new_value = self._semantics.binary_sub(current, right, state)
            elif stmt.op == AugAssignOperator.MUL:
                new_value = self._semantics.binary_mul(current, right, state)
            elif stmt.op == AugAssignOperator.DIV:
                new_value = self._semantics.binary_div(current, right, state)
            elif stmt.op == AugAssignOperator.MOD:
                new_value = self._semantics.binary_mod(current, right, state)
            else:
                new_value = None

            if new_value is not None:
                state.set_variable(name, new_value)

        return [state]

    # =========================================================================
    # Control Flow
    # =========================================================================

    def _execute_if(
        self,
        stmt: IRIf,
        state: SymbolicState,
        result: IRExecutionResult,
    ) -> List[SymbolicState]:
        """
        Execute an if statement with SMART FORKING.

        Only forks if both branches are feasible.

        Args:
            stmt: IRIf node
            state: Current symbolic state
            result: IRExecutionResult to track path count

        Returns:
            List of terminal states from both branches
        """
        condition = self._eval_expr(stmt.test, state)
        if condition is None:
            # Can't evaluate condition - take both branches blindly
            result.path_count += 2
            true_states = self._execute_block(stmt.body, state.fork(), result)
            false_states = self._execute_block(stmt.orelse, state.fork(), result)
            return true_states + false_states

        # Convert to boolean if needed
        if self._semantics is not None:
            bool_cond = self._semantics.to_bool(condition, state)
            if bool_cond is not None:
                condition = bool_cond

        # SMART FORKING: Check feasibility before forking
        true_feasible = self._is_feasible(state, condition)
        false_feasible = self._is_feasible(state, Not(condition))

        terminal_states = []

        if true_feasible and false_feasible:
            # Both branches feasible - fork
            result.path_count += 2

            true_state = state.fork()
            true_state.add_constraint(condition)
            true_states = self._execute_block(stmt.body, true_state, result)
            terminal_states.extend(true_states)

            false_state = state.fork()
            false_state.add_constraint(Not(condition))
            if stmt.orelse:
                false_states = self._execute_block(stmt.orelse, false_state, result)
            else:
                false_states = [false_state]
            terminal_states.extend(false_states)

        elif true_feasible:
            # Only true branch feasible
            result.path_count += 1
            result.pruned_count += 1
            state.add_constraint(condition)
            terminal_states = self._execute_block(stmt.body, state, result)

        elif false_feasible:
            # Only false branch feasible
            result.path_count += 1
            result.pruned_count += 1
            state.add_constraint(Not(condition))
            if stmt.orelse:
                terminal_states = self._execute_block(stmt.orelse, state, result)
            else:
                terminal_states = [state]

        else:
            # Neither branch feasible - dead path
            result.pruned_count += 2

        return terminal_states

    def _execute_while(
        self,
        stmt: IRWhile,
        state: SymbolicState,
        result: IRExecutionResult,
    ) -> List[SymbolicState]:
        """
        Execute a while loop with BOUNDED UNROLLING.

        Args:
            stmt: IRWhile node
            state: Current symbolic state
            result: IRExecutionResult to track path count

        Returns:
            List of terminal states
        """
        current_states = [state]

        for _ in range(self.max_loop_iterations):
            next_states = []

            for s in current_states:
                condition = self._eval_expr(stmt.test, s)
                if condition is None:
                    # Can't evaluate - assume one iteration and exit
                    result.path_count += 1
                    next_states.append(s)
                    continue

                if self._semantics is not None:
                    bool_cond = self._semantics.to_bool(condition, s)
                    if bool_cond is not None:
                        condition = bool_cond

                true_feasible = self._is_feasible(s, condition)
                false_feasible = self._is_feasible(s, Not(condition))

                if true_feasible:
                    # Continue loop
                    loop_state = s.fork()
                    loop_state.add_constraint(condition)
                    body_states = self._execute_block(stmt.body, loop_state, result)
                    next_states.extend(body_states)

                if false_feasible:
                    # Exit loop
                    exit_state = s.fork()
                    exit_state.add_constraint(Not(condition))
                    # Execute else clause if present
                    if stmt.orelse:
                        else_states = self._execute_block(
                            stmt.orelse, exit_state, result
                        )
                        # These are terminal - don't add to next_states
                        result.states.extend(else_states)
                    else:
                        result.states.append(exit_state)

            current_states = next_states
            if not current_states:
                break

        # Remaining states hit max iterations - treat as terminal
        result.states.extend(current_states)
        return result.states

    def _execute_for(
        self,
        stmt: IRFor,
        state: SymbolicState,
        result: IRExecutionResult,
    ) -> List[SymbolicState]:
        """
        Execute a for loop with BOUNDED UNROLLING.

        Currently only supports range() iteration.

        Args:
            stmt: IRFor node
            state: Current symbolic state
            result: IRExecutionResult to track path count

        Returns:
            List of terminal states
        """
        # For now, just unroll up to max_iterations
        # Full range() analysis would require evaluating the iterator
        current_states = [state]

        for i in range(self.max_loop_iterations):
            next_states = []
            for s in current_states:
                # Set loop variable to iteration count
                if isinstance(stmt.target, IRName):
                    s.set_variable(stmt.target.id, IntVal(i))

                body_states = self._execute_block(stmt.body, s, result)
                next_states.extend(body_states)

            current_states = next_states
            if not current_states:
                break

        return current_states

    # =========================================================================
    # Expression Evaluation
    # =========================================================================

    def _eval_expr(
        self, expr: Optional[IRExpr], state: SymbolicState
    ) -> Optional[ExprRef]:
        """
        Evaluate an IR expression to a Z3 expression.

        Args:
            expr: IR expression node
            state: Current symbolic state

        Returns:
            Z3 expression, or None if unsupported
        """
        if expr is None:
            return None

        if isinstance(expr, IRConstant):
            return self._eval_constant(expr)
        elif isinstance(expr, IRName):
            return self._eval_name(expr, state)
        elif isinstance(expr, IRBinaryOp):
            return self._eval_binary_op(expr, state)
        elif isinstance(expr, IRUnaryOp):
            return self._eval_unary_op(expr, state)
        elif isinstance(expr, IRCompare):
            return self._eval_compare(expr, state)
        elif isinstance(expr, IRBoolOp):
            return self._eval_bool_op(expr, state)
        elif isinstance(expr, IRCall):
            return self._eval_call(expr, state)
        else:
            return None

    def _eval_constant(self, expr: IRConstant) -> Optional[ExprRef]:
        """Evaluate a constant literal."""
        value = expr.value
        if isinstance(value, bool):
            return BoolVal(value)
        elif isinstance(value, int):
            return IntVal(value)
        elif value is None:
            return None
        else:
            return None

    def _eval_name(self, expr: IRName, state: SymbolicState) -> Optional[ExprRef]:
        """Evaluate a variable reference."""
        name = expr.id

        # Check for symbolic declaration
        if name == "symbolic":
            return None  # Handled at call site

        var = state.get_variable(name)
        if var is not None:
            return var

        # Variable doesn't exist - create it as unknown int
        return state.create_variable(name, IntSort())

    def _eval_binary_op(
        self, expr: IRBinaryOp, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Evaluate a binary operation."""
        left = self._eval_expr(expr.left, state)
        right = self._eval_expr(expr.right, state)

        if left is None or right is None or self._semantics is None:
            return None

        op = expr.op
        if op == BinaryOperator.ADD:
            return self._semantics.binary_add(left, right, state)
        elif op == BinaryOperator.SUB:
            return self._semantics.binary_sub(left, right, state)
        elif op == BinaryOperator.MUL:
            return self._semantics.binary_mul(left, right, state)
        elif op == BinaryOperator.DIV:
            return self._semantics.binary_div(left, right, state)
        elif op == BinaryOperator.FLOOR_DIV:
            return self._semantics.binary_div(left, right, state)
        elif op == BinaryOperator.MOD:
            return self._semantics.binary_mod(left, right, state)
        else:
            return None

    def _eval_unary_op(
        self, expr: IRUnaryOp, state: SymbolicState
    ) -> Optional[ExprRef]:
        """Evaluate a unary operation."""
        operand = self._eval_expr(expr.operand, state)
        if operand is None or self._semantics is None:
            return None

        op = expr.op
        if op == UnaryOperator.NEG:
            return self._semantics.unary_neg(operand, state)
        elif op == UnaryOperator.NOT:
            return self._semantics.unary_not(operand, state)
        else:
            return None

    def _eval_compare(self, expr: IRCompare, state: SymbolicState) -> Optional[BoolRef]:
        """Evaluate a comparison."""
        left = self._eval_expr(expr.left, state)
        if left is None or self._semantics is None:
            return None

        # Build chained comparison: a < b < c -> (a < b) AND (b < c)
        result: Optional[BoolRef] = None

        for op, comparator in zip(expr.ops, expr.comparators):
            right = self._eval_expr(comparator, state)
            if right is None:
                return None

            if op == CompareOperator.EQ:
                cmp_result = self._semantics.compare_eq(left, right, state)
            elif op == CompareOperator.NE:
                cmp_result = self._semantics.compare_ne(left, right, state)
            elif op == CompareOperator.LT:
                cmp_result = self._semantics.compare_lt(left, right, state)
            elif op == CompareOperator.LE:
                cmp_result = self._semantics.compare_le(left, right, state)
            elif op == CompareOperator.GT:
                cmp_result = self._semantics.compare_gt(left, right, state)
            elif op == CompareOperator.GE:
                cmp_result = self._semantics.compare_ge(left, right, state)
            else:
                return None

            if cmp_result is None:
                return None

            if result is None:
                result = cmp_result
            else:
                from z3 import And

                result = And(result, cmp_result)

            left = right

        return result

    def _eval_bool_op(self, expr: IRBoolOp, state: SymbolicState) -> Optional[BoolRef]:
        """Evaluate a boolean operation (and/or)."""
        if not expr.values or self._semantics is None:
            return None

        results = []
        for value in expr.values:
            evaluated = self._eval_expr(value, state)
            if evaluated is None:
                return None
            bool_val = self._semantics.to_bool(evaluated, state)
            if bool_val is None:
                return None
            results.append(bool_val)

        from z3 import And, Or

        if expr.op == BoolOperator.AND:
            return And(*results)
        elif expr.op == BoolOperator.OR:
            return Or(*results)
        else:
            return None

    def _eval_call(self, expr: IRCall, state: SymbolicState) -> Optional[ExprRef]:
        """
        Evaluate a function call.

        Special handling for symbolic() declarations.
        """
        # Check for symbolic() call
        if isinstance(expr.func, IRName) and expr.func.id == "symbolic":
            return self._handle_symbolic_call(expr, state)

        # Other calls not supported yet
        return None

    def _handle_symbolic_call(
        self, expr: IRCall, state: SymbolicState
    ) -> Optional[ExprRef]:
        """
        Handle symbolic('name', type) call.

        Args:
            expr: IRCall node for symbolic()
            state: Current symbolic state

        Returns:
            Newly created symbolic variable
        """
        if len(expr.args) < 2:
            return None

        # Get name from first argument
        name_arg = expr.args[0]
        if not isinstance(name_arg, IRConstant) or not isinstance(
            name_arg.value, str
        ):
            return None
        name = name_arg.value

        # Get type from second argument
        type_arg = expr.args[1]
        if isinstance(type_arg, IRName):
            type_name = type_arg.id
            if type_name == "int":
                return state.create_variable(name, IntSort())
            elif type_name == "bool":
                return state.create_variable(name, BoolSort())

        return None

    # =========================================================================
    # Feasibility Checking
    # =========================================================================

    def _is_feasible(self, state: SymbolicState, condition: BoolRef) -> bool:
        """
        Check if a condition is feasible given current constraints.

        Args:
            state: Current symbolic state
            condition: Condition to check

        Returns:
            True if condition can be satisfied
        """
        solver = Solver()
        solver.add(*state.constraints)
        solver.add(condition)
        return solver.check() == sat
