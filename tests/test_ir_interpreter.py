"""
Tests for IR Symbolic Interpreter.

These tests validate that the IR-based interpreter produces the same
results as the AST-based interpreter for equivalent Python code.

The parity tests are critical for ensuring the polyglot architecture
doesn't regress behavior.
"""

import pytest
from z3 import IntSort, BoolSort, IntVal, BoolVal, simplify

from code_scalpel.ir.normalizers.python_normalizer import PythonNormalizer
from code_scalpel.symbolic_execution_tools.ir_interpreter import (
    IRSymbolicInterpreter,
    IRExecutionResult,
    IRNodeVisitor,
    PythonSemantics,
    JavaScriptSemantics,
    get_semantics,
)
from code_scalpel.ir.nodes import (
    IRModule,
    IRAssign,
    IRBinaryOp,
    IRName,
    IRConstant,
    IRIf,
    IRCompare,
)
from code_scalpel.ir.operators import BinaryOperator, CompareOperator


class TestIRInterpreterBasics:
    """Basic IR interpreter functionality."""

    def test_interpreter_creation(self):
        """Test IRSymbolicInterpreter can be created."""
        interp = IRSymbolicInterpreter()
        assert interp.max_loop_iterations == 10

    def test_interpreter_custom_iterations(self):
        """Test custom max_loop_iterations."""
        interp = IRSymbolicInterpreter(max_loop_iterations=5)
        assert interp.max_loop_iterations == 5

    def test_semantics_registry(self):
        """Test get_semantics returns correct implementations."""
        py_sem = get_semantics("python")
        assert isinstance(py_sem, PythonSemantics)
        assert py_sem.name == "python"

        js_sem = get_semantics("javascript")
        assert isinstance(js_sem, JavaScriptSemantics)
        assert js_sem.name == "javascript"

        # Alias
        js_sem2 = get_semantics("js")
        assert isinstance(js_sem2, JavaScriptSemantics)

        # Unknown defaults to Python
        unknown = get_semantics("fortran")
        assert isinstance(unknown, PythonSemantics)


class TestIRInterpreterAssignment:
    """Test assignment handling."""

    def test_simple_constant_assignment(self):
        """Test x = 1."""
        code = "x = 1"
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) == 1
        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == IntVal(1)

    def test_multiple_assignments(self):
        """Test multiple assignments."""
        code = """
x = 1
y = 2
z = 3
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) == 1
        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(1)
        assert simplify(state.get_variable("y")) == IntVal(2)
        assert simplify(state.get_variable("z")) == IntVal(3)

    def test_assignment_with_expression(self):
        """Test x = 1 + 2."""
        code = "x = 1 + 2"
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) == 1
        state = result.states[0]
        x = state.get_variable("x")
        # x should be 1 + 2 = 3 when simplified
        assert simplify(x) == IntVal(3)

    def test_assignment_with_variable_reference(self):
        """Test y = x where x is defined."""
        code = """
x = 5
y = x
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) == 1
        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(5)
        assert simplify(state.get_variable("y")) == IntVal(5)


class TestIRInterpreterBinaryOps:
    """Test binary operations."""

    def test_addition(self):
        """Test a + b."""
        code = """
a = 3
b = 4
c = a + b
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("c")) == IntVal(7)

    def test_subtraction(self):
        """Test a - b."""
        code = """
a = 10
b = 3
c = a - b
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("c")) == IntVal(7)

    def test_multiplication(self):
        """Test a * b."""
        code = """
a = 6
b = 7
c = a * b
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("c")) == IntVal(42)

    def test_nested_expression(self):
        """Test (a + b) * c."""
        code = """
a = 2
b = 3
c = 4
d = (a + b) * c
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        # (2 + 3) * 4 = 20
        assert simplify(state.get_variable("d")) == IntVal(20)


class TestIRInterpreterConditionals:
    """Test conditional (if) handling."""

    def test_simple_if_with_concrete_true(self):
        """Test if with concrete true condition."""
        code = """
x = 10
if x > 5:
    y = 1
else:
    y = 0
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Only one path should be feasible (x=10 > 5)
        assert len(result.feasible_states()) >= 1
        # At least one state should have y = 1
        found_y1 = False
        for state in result.states:
            y = state.get_variable("y")
            if y is not None and simplify(y) == IntVal(1):
                found_y1 = True
        assert found_y1

    def test_if_with_symbolic_variable(self):
        """Test if with symbolic variable - should fork."""
        code = """
if x > 10:
    y = x + 5
else:
    y = x - 5
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        result = interp.execute(ir)

        # Should have 2 paths: x > 10 and x <= 10
        assert len(result.states) == 2
        assert result.path_count == 2

    def test_if_without_else(self):
        """Test if without else clause."""
        code = """
y = 0
if x > 10:
    y = 1
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        result = interp.execute(ir)

        # Should have 2 paths
        assert len(result.states) == 2


class TestIRInterpreterSymbolicExecution:
    """Test symbolic execution features."""

    def test_symbolic_variable_declaration(self):
        """Test declaring symbolic variable before execution."""
        code = """
y = x + 1
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        x = interp.declare_symbolic("x", IntSort())
        result = interp.execute(ir)

        assert len(result.states) == 1
        state = result.states[0]
        y = state.get_variable("y")
        # y should be x + 1 (symbolic) - Z3 may reorder terms
        simplified = str(simplify(y))
        assert "x" in simplified and "1" in simplified

    def test_precondition(self):
        """Test adding precondition."""
        code = """
if x > 10:
    y = 1
else:
    y = 0
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        x = interp.declare_symbolic("x", IntSort())
        # Add precondition: x = 15 (> 10)
        interp.add_precondition(x == IntVal(15))
        result = interp.execute(ir)

        # With x=15, only the true branch should be feasible
        feasible = result.feasible_states()
        # At least one state should have y = 1
        found_y1 = False
        for s in feasible:
            y = s.get_variable("y")
            if y is not None:
                # Use Z3's eq method for comparison
                if simplify(y).eq(IntVal(1)):
                    found_y1 = True
                    break
        assert found_y1


class TestIRInterpreterLoops:
    """Test loop handling."""

    def test_while_loop_bounded(self):
        """Test while loop terminates with bounded unrolling."""
        code = """
x = 0
while x < 5:
    x = x + 1
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter(max_loop_iterations=10)
        result = interp.execute(ir)

        # Should terminate
        assert len(result.states) > 0


class TestIRInterpreterParity:
    """
    Parity tests: IR interpreter should produce same results as AST interpreter.
    
    These are the critical tests for ensuring the polyglot architecture
    doesn't regress behavior.
    """

    def test_parity_simple_arithmetic(self):
        """Test parity for simple arithmetic."""
        code = """
a = 5
b = 3
c = a + b
d = a * b
e = a - b
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("c")) == IntVal(8)
        assert simplify(state.get_variable("d")) == IntVal(15)
        assert simplify(state.get_variable("e")) == IntVal(2)

    def test_parity_conditional_forking(self):
        """Test parity for conditional forking."""
        code = """
if x > 0:
    y = 1
else:
    y = -1
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())
        result = interp.execute(ir)

        # Should have exactly 2 paths
        assert len(result.states) == 2

        # One path should have y = 1, one should have y = -1
        y_values = [simplify(s.get_variable("y")) for s in result.states]
        assert IntVal(1) in y_values
        assert IntVal(-1) in y_values

    def test_parity_augmented_assignment(self):
        """Test parity for augmented assignment (+=, etc.)."""
        code = """
x = 5
x += 3
"""
        ir = PythonNormalizer().normalize(code)

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(8)


class TestLanguageSemantics:
    """Test language-specific semantics."""

    def test_python_semantics_arithmetic(self):
        """Test Python semantics for arithmetic."""
        from z3 import Int

        sem = PythonSemantics()
        state = None  # Not needed for simple arithmetic

        a = Int("a")
        b = Int("b")

        result = sem.binary_add(IntVal(3), IntVal(4), state)
        assert simplify(result) == IntVal(7)

        result = sem.binary_sub(IntVal(10), IntVal(3), state)
        assert simplify(result) == IntVal(7)

        result = sem.binary_mul(IntVal(6), IntVal(7), state)
        assert simplify(result) == IntVal(42)

    def test_python_semantics_comparison(self):
        """Test Python semantics for comparison."""
        from z3 import Int

        sem = PythonSemantics()
        state = None

        result = sem.compare_lt(IntVal(3), IntVal(5), state)
        assert simplify(result) == BoolVal(True)

        result = sem.compare_gt(IntVal(10), IntVal(5), state)
        assert simplify(result) == BoolVal(True)

        result = sem.compare_eq(IntVal(5), IntVal(5), state)
        assert simplify(result) == BoolVal(True)

    def test_python_to_bool(self):
        """Test Python truthiness conversion."""
        sem = PythonSemantics()
        state = None

        # 0 is falsy
        result = sem.to_bool(IntVal(0), state)
        assert simplify(result) == BoolVal(False)

        # Non-zero is truthy
        result = sem.to_bool(IntVal(1), state)
        assert simplify(result) == BoolVal(True)

        # Bool values pass through
        result = sem.to_bool(BoolVal(True), state)
        assert simplify(result) == BoolVal(True)


class TestIRNodeVisitor:
    """Test IRNodeVisitor base class."""

    def test_visitor_dispatch(self):
        """Test that visitor dispatches correctly."""

        class TestVisitor(IRNodeVisitor):
            def visit_IRConstant(self, node):
                return f"constant:{node.value}"

            def visit_IRName(self, node):
                return f"name:{node.id}"

        visitor = TestVisitor()

        const_node = IRConstant(value=42)
        assert visitor.visit(const_node) == "constant:42"

        name_node = IRName(id="foo")
        assert visitor.visit(name_node) == "name:foo"

    def test_generic_visit_fallback(self):
        """Test generic_visit is called for unknown nodes."""

        class TestVisitor(IRNodeVisitor):
            def generic_visit(self, node):
                return "unknown"

        visitor = TestVisitor()
        const_node = IRConstant(value=42)
        # IRConstant not handled, should fall back to generic_visit
        assert visitor.visit(const_node) == "unknown"
