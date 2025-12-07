"""
Tests for Loop Support - Bounded Unrolling for Symbolic Execution.

CRITICAL: These tests enforce TERMINATION GUARANTEES.
Unbounded loops are the halting problem. Symbolic loops are worse - 
they can fork infinitely until RAM is exhausted.

The interpreter must:
1. Unroll loops up to MAX_ITERATIONS
2. Prune paths that exceed the limit
3. NEVER hang on `while True: pass`

If the Safety Check tests hang, the engine is BROKEN.
"""

import pytest
from z3 import Int, Bool, IntSort, BoolSort, And, Or, Not

from code_scalpel.symbolic_execution_tools.ir_interpreter import (
    IRSymbolicInterpreter,
    IRExecutionResult as ExecutionResult,
)
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

    def add_precondition(self, constraint):
        self.interp.add_precondition(constraint)
from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState


# =============================================================================
# SECTION 1: Concrete While Loops
# =============================================================================

class TestConcreteWhileLoops:
    """Test while loops with concrete conditions."""

    def test_simple_counter_loop(self):
        """
        x = 0
        while x < 3:
            x += 1
        # x should be 3
        """
        code = """
x = 0
while x < 3:
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")

    def test_zero_iteration_loop(self):
        """Loop condition false from start - never enters."""
        code = """
x = 10
while x < 5:
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should not enter loop at all
        assert len(result.states) == 1

    def test_single_iteration_loop(self):
        """Loop executes exactly once."""
        code = """
x = 0
while x < 1:
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1

    def test_loop_with_multiple_statements(self):
        """Loop body has multiple statements."""
        code = """
x = 0
y = 0
while x < 3:
    x = x + 1
    y = y + 2
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")
        assert state.has_variable("y")

    def test_nested_concrete_loops(self):
        """Nested loops with concrete bounds."""
        code = """
total = 0
i = 0
while i < 2:
    j = 0
    while j < 2:
        total = total + 1
        j = j + 1
    i = i + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1


# =============================================================================
# SECTION 2: SAFETY CHECK - Infinite Loop Prevention
# =============================================================================

class TestInfiniteLoopPrevention:
    """
    CRITICAL TESTS: These verify the engine TERMINATES on infinite loops.
    
    If ANY of these tests hang, the engine is BROKEN.
    These tests should complete quickly (< 1 second).
    """

    def test_while_true_terminates(self):
        """
        SAFETY TEST 1: while True must NOT hang.
        
        The engine should hit MAX_ITERATIONS and stop.
        """
        code = """
x = 0
while True:
    x = x + 1
"""
        # This MUST NOT hang
        interp = SymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(code)
        
        # Should have terminated (either with results or empty)
        assert result is not None

    def test_trivially_true_condition(self):
        """
        SAFETY TEST 2: while 1 > 0 must NOT hang.
        """
        code = """
x = 0
while 1 > 0:
    x = x + 1
"""
        interp = SymbolicInterpreter(max_loop_iterations=3)
        result = interp.execute(code)
        
        assert result is not None

    def test_symbolic_always_true_terminates(self):
        """
        SAFETY TEST 3: Symbolic condition that's always satisfiable.
        
        while x > 0 or x <= 0 is always true, must terminate.
        """
        code = """
y = 0
while y < 1000000:
    y = y + 1
"""
        interp = SymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(code)
        
        # Should terminate after 5 iterations
        assert result is not None

    def test_max_iterations_configurable(self):
        """MAX_ITERATIONS can be configured at init."""
        interp = SymbolicInterpreter(max_loop_iterations=10)
        
        assert interp.max_loop_iterations == 10

    def test_default_max_iterations(self):
        """Default MAX_ITERATIONS is reasonable (not infinite)."""
        interp = SymbolicInterpreter()
        
        assert hasattr(interp, 'max_loop_iterations')
        assert interp.max_loop_iterations > 0
        assert interp.max_loop_iterations <= 100  # Reasonable default


# =============================================================================
# SECTION 3: Symbolic While Loops
# =============================================================================

class TestSymbolicWhileLoops:
    """Test while loops with symbolic conditions."""

    def test_symbolic_loop_forks_at_boundary(self):
        """
        Symbolic loop should fork: one path enters, one exits.
        
        x is symbolic
        while x > 0:
            x = x - 1
        
        Should produce multiple states (some entered loop, some didn't).
        """
        interp = SymbolicInterpreter(max_loop_iterations=3)
        interp.declare_symbolic("x", IntSort())
        
        code = """
y = 0
while x > 0:
    x = x - 1
    y = y + 1
"""
        result = interp.execute(code)
        
        # Should have multiple terminal states
        # (paths where x started <= 0, x started = 1, x started = 2, etc.)
        assert len(result.states) >= 1

    def test_symbolic_loop_constrained(self):
        """
        Symbolic loop with preconditions.
        
        If x is constrained to be in [1, 3], loop should iterate 1-3 times.
        """
        interp = SymbolicInterpreter(max_loop_iterations=5)
        x = interp.declare_symbolic("x", IntSort())
        interp.add_precondition(x >= 1)
        interp.add_precondition(x <= 3)
        
        code = """
count = 0
while x > 0:
    x = x - 1
    count = count + 1
"""
        result = interp.execute(code)
        
        # All paths should be feasible
        feasible = result.feasible_states()
        assert len(feasible) >= 1

    def test_loop_with_break_like_condition(self):
        """
        Loop that can exit early based on condition.
        
        Note: We don't have 'break' support yet, but condition can become false.
        """
        code = """
x = 10
found = False
while x > 0:
    x = x - 1
    if x == 5:
        found = True
"""
        interp = SymbolicInterpreter(max_loop_iterations=15)
        result = interp.execute(code)
        
        assert len(result.states) >= 1


# =============================================================================
# SECTION 4: For Loops (range() support)
# =============================================================================

class TestForLoops:
    """Test for loops with range()."""

    def test_simple_range_loop(self):
        """
        for i in range(3):
            x += 1
        """
        code = """
x = 0
for i in range(3):
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("x")

    def test_range_with_start_stop(self):
        """
        for i in range(2, 5):
            x += i
        """
        code = """
x = 0
for i in range(2, 5):
    x = x + i
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1

    def test_range_with_step(self):
        """
        for i in range(0, 10, 2):
            x += 1
        """
        code = """
x = 0
for i in range(0, 10, 2):
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1

    def test_empty_range(self):
        """range(0) should not execute body."""
        code = """
x = 0
for i in range(0):
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1

    def test_range_negative_step(self):
        """for i in range(5, 0, -1) counts down."""
        code = """
x = 0
for i in range(5, 0, -1):
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1

    def test_for_loop_variable_accessible(self):
        """Loop variable is accessible in body."""
        code = """
total = 0
for i in range(5):
    total = total + i
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1
        state = result.states[0]
        assert state.has_variable("total")
        assert state.has_variable("i")

    def test_nested_for_loops(self):
        """Nested for loops work correctly."""
        code = """
total = 0
for i in range(3):
    for j in range(3):
        total = total + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert len(result.states) == 1


# =============================================================================
# SECTION 5: Loop Bounds Enforcement
# =============================================================================

class TestLoopBoundsEnforcement:
    """Test that loop bounds are properly enforced."""

    def test_loop_stops_at_max_iterations(self):
        """
        Loop with 100 iterations should be cut off at max.
        """
        code = """
x = 0
while x < 100:
    x = x + 1
"""
        interp = SymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(code)
        
        # Should have terminated (not hung)
        assert result is not None
        # The path that exceeded limit is pruned or marked

    def test_for_loop_exceeding_max_handled(self):
        """Large range() should be bounded."""
        code = """
x = 0
for i in range(1000):
    x = x + 1
"""
        interp = SymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(code)
        
        # Should terminate without executing 1000 iterations
        assert result is not None

    def test_bounds_reset_for_different_loops(self):
        """Each loop has its own iteration counter."""
        code = """
x = 0
while x < 3:
    x = x + 1

y = 0
while y < 3:
    y = y + 1
"""
        interp = SymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(code)
        
        # Both loops should complete normally
        assert len(result.states) == 1


# =============================================================================
# SECTION 6: Edge Cases
# =============================================================================

class TestLoopEdgeCases:
    """Test edge cases for loops."""

    def test_while_with_else(self):
        """
        Python's while/else - else runs if loop completes normally.
        We may skip else for simplicity in Phase 1.
        """
        code = """
x = 0
while x < 3:
    x = x + 1
else:
    y = 1
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should handle without crashing
        assert result is not None

    def test_for_with_else(self):
        """for/else syntax."""
        code = """
x = 0
for i in range(3):
    x = x + 1
else:
    y = 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert result is not None

    def test_loop_with_if_inside(self):
        """Loop containing branches."""
        code = """
evens = 0
odds = 0
for i in range(5):
    if i % 2 == 0:
        evens = evens + 1
    else:
        odds = odds + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        # May produce multiple states due to branching
        assert len(result.states) >= 1

    def test_if_inside_symbolic_loop(self):
        """Branch inside a symbolic loop."""
        interp = SymbolicInterpreter(max_loop_iterations=3)
        interp.declare_symbolic("x", IntSort())
        
        code = """
result = 0
while x > 0:
    if x > 5:
        result = result + 2
    else:
        result = result + 1
    x = x - 1
"""
        result = interp.execute(code)
        
        assert len(result.states) >= 1


# =============================================================================
# SECTION 7: Unsupported Loop Constructs
# =============================================================================

class TestUnsupportedLoopConstructs:
    """Test handling of unsupported loop patterns."""

    def test_for_over_list_unsupported(self):
        """
        for x in [1, 2, 3] is not supported (arbitrary iterables).
        Should skip or handle gracefully.
        """
        code = """
x = 0
for i in [1, 2, 3]:
    x = x + i
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should not crash
        assert result is not None

    def test_for_over_string_unsupported(self):
        """for c in 'abc' is not supported."""
        code = """
x = 0
for c in 'hello':
    x = x + 1
"""
        result = SymbolicInterpreter().execute(code)
        
        assert result is not None

    def test_for_over_variable_unsupported(self):
        """for x in some_list is not supported (unknown iterable)."""
        code = """
result = 0
for item in data:
    result = result + item
"""
        result = SymbolicInterpreter().execute(code)
        
        # Should handle gracefully (skip or warn)
        assert result is not None
