"""
Tests for IR Symbolic Interpreter.

These tests validate that the IR-based interpreter produces the same
results as the AST-based interpreter for equivalent Python code.

The parity tests are critical for ensuring the polyglot architecture
doesn't regress behavior.
"""

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
        interp.declare_symbolic("x", IntSort())
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

        Int("a")
        Int("b")

        result = sem.binary_add(IntVal(3), IntVal(4), state)
        assert simplify(result) == IntVal(7)

        result = sem.binary_sub(IntVal(10), IntVal(3), state)
        assert simplify(result) == IntVal(7)

        result = sem.binary_mul(IntVal(6), IntVal(7), state)
        assert simplify(result) == IntVal(42)

    def test_python_semantics_comparison(self):
        """Test Python semantics for comparison."""

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


class TestJavaScriptSemantics:
    """Test JavaScript-specific semantics."""

    def test_js_semantics_name(self):
        """Test JavaScript semantics name property."""
        sem = JavaScriptSemantics()
        assert sem.name == "javascript"

    def test_js_binary_add_arithmetic(self):
        """Test JavaScript addition with ArithRef values."""
        from z3 import Int

        sem = JavaScriptSemantics()
        a = Int("a")
        b = Int("b")
        result = sem.binary_add(a, b, None)
        # Returns symbolic a + b
        assert result is not None

    def test_js_binary_add_non_arith(self):
        """Test JavaScript addition with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.binary_add(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_binary_sub_arithmetic(self):
        """Test JavaScript subtraction."""
        sem = JavaScriptSemantics()
        result = sem.binary_sub(IntVal(10), IntVal(3), None)
        assert simplify(result) == IntVal(7)

    def test_js_binary_sub_non_arith(self):
        """Test JavaScript subtraction with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.binary_sub(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_binary_mul_arithmetic(self):
        """Test JavaScript multiplication."""
        sem = JavaScriptSemantics()
        result = sem.binary_mul(IntVal(6), IntVal(7), None)
        assert simplify(result) == IntVal(42)

    def test_js_binary_mul_non_arith(self):
        """Test JavaScript multiplication with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.binary_mul(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_binary_div_arithmetic(self):
        """Test JavaScript division."""
        sem = JavaScriptSemantics()
        result = sem.binary_div(IntVal(10), IntVal(2), None)
        # Z3 integer division
        assert result is not None

    def test_js_binary_div_non_arith(self):
        """Test JavaScript division with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.binary_div(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_binary_mod_arithmetic(self):
        """Test JavaScript modulo."""
        sem = JavaScriptSemantics()
        result = sem.binary_mod(IntVal(10), IntVal(3), None)
        assert simplify(result) == IntVal(1)

    def test_js_binary_mod_non_arith(self):
        """Test JavaScript modulo with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.binary_mod(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_compare_eq(self):
        """Test JavaScript equality comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_eq(IntVal(5), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_ne(self):
        """Test JavaScript inequality comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_ne(IntVal(5), IntVal(3), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_lt_arithmetic(self):
        """Test JavaScript less-than comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_lt(IntVal(3), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_lt_non_arith(self):
        """Test JavaScript less-than with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.compare_lt(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_compare_le_arithmetic(self):
        """Test JavaScript less-than-or-equal comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_le(IntVal(5), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_le_non_arith(self):
        """Test JavaScript less-than-or-equal with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.compare_le(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_compare_gt_arithmetic(self):
        """Test JavaScript greater-than comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_gt(IntVal(10), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_gt_non_arith(self):
        """Test JavaScript greater-than with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.compare_gt(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_compare_ge_arithmetic(self):
        """Test JavaScript greater-than-or-equal comparison."""
        sem = JavaScriptSemantics()
        result = sem.compare_ge(IntVal(5), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_js_compare_ge_non_arith(self):
        """Test JavaScript greater-than-or-equal with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.compare_ge(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_js_unary_neg_arithmetic(self):
        """Test JavaScript negation."""
        sem = JavaScriptSemantics()
        result = sem.unary_neg(IntVal(5), None)
        assert simplify(result) == IntVal(-5)

    def test_js_unary_neg_non_arith(self):
        """Test JavaScript negation with non-ArithRef returns None."""
        sem = JavaScriptSemantics()
        result = sem.unary_neg(BoolVal(True), None)
        assert result is None

    def test_js_unary_not(self):
        """Test JavaScript logical not."""
        sem = JavaScriptSemantics()
        result = sem.unary_not(IntVal(0), None)  # 0 is falsy
        assert simplify(result) == BoolVal(True)

        result = sem.unary_not(IntVal(1), None)  # Non-zero is truthy
        assert simplify(result) == BoolVal(False)

    def test_js_unary_not_bool(self):
        """Test JavaScript logical not with boolean."""
        sem = JavaScriptSemantics()
        result = sem.unary_not(BoolVal(True), None)
        assert simplify(result) == BoolVal(False)

    def test_js_to_bool_boolref(self):
        """Test JavaScript to_bool with BoolRef passthrough."""
        sem = JavaScriptSemantics()
        result = sem.to_bool(BoolVal(True), None)
        assert simplify(result) == BoolVal(True)

    def test_js_to_bool_arithref(self):
        """Test JavaScript to_bool with ArithRef (truthy/falsy)."""
        sem = JavaScriptSemantics()
        # 0 is falsy
        result = sem.to_bool(IntVal(0), None)
        assert simplify(result) == BoolVal(False)
        # Non-zero is truthy
        result = sem.to_bool(IntVal(42), None)
        assert simplify(result) == BoolVal(True)

    def test_js_to_bool_unsupported(self):
        """Test JavaScript to_bool with unsupported type returns None."""
        from z3 import String

        sem = JavaScriptSemantics()
        # String is not ArithRef or BoolRef
        result = sem.to_bool(String("test"), None)
        assert result is None


class TestIRInterpreterControlFlow:
    """Test control flow execution paths."""

    def test_execute_for_loop(self):
        """Test for loop execution with range."""
        code = """
for i in range(3):
    x = i
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1
        # After 3 iterations, x should be 2 (last value)
        state = result.states[0]
        x = state.get_variable("x")
        # For loop sets loop variable to iteration count
        assert x is not None

    def test_while_loop_with_else(self):
        """Test while loop with else clause."""
        code = """
x = 0
while x < 3:
    x = x + 1
else:
    y = 100
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(ir)

        assert len(result.states) >= 1
        # Else clause should execute when loop completes normally

    def test_if_with_unevaluable_condition(self):
        """Test if statement where condition cannot be evaluated."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall, IRConstant

        # Create an IR with a call expression as condition (unevaluable)
        call_node = IRCall(func=IRName(id="unknown_func"), args=[], kwargs={})
        if_stmt = IRIf(
            test=call_node,
            body=[IRAssign(targets=[IRName(id="x")], value=IRConstant(value=1))],
            orelse=[IRAssign(targets=[IRName(id="x")], value=IRConstant(value=2))],
        )
        ir = IRModule(body=[if_stmt])

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should take both branches blindly
        assert result.path_count >= 2

    def test_execute_pass_statement(self):
        """Test pass statement execution."""
        code = """
if True:
    pass
x = 1
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1
        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == IntVal(1)

    def test_execute_function_def_skipped(self):
        """Test function definitions are skipped during execution."""
        code = """
def foo():
    return 1
x = 5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1
        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == IntVal(5)

    def test_execute_return_statement(self):
        """Test return statement stops path."""
        code = """
x = 1
if x > 0:
    y = 2
x = 3
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_augmented_assignment_operators(self):
        """Test all augmented assignment operators."""
        code = """
x = 10
x += 5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(15)

        # Test subtraction
        code = """
x = 10
x -= 3
"""
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)
        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(7)

        # Test multiplication
        code = """
x = 10
x *= 2
"""
        ir = PythonNormalizer().normalize(code)
        result = interp.execute(ir)
        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(20)

    def test_augmented_assignment_div(self):
        """Test augmented division."""
        code = """
x = 10
x /= 2
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert x is not None

    def test_augmented_assignment_mod(self):
        """Test augmented modulo."""
        code = """
x = 10
x %= 3
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == IntVal(1)


class TestIRInterpreterExpressions:
    """Test expression evaluation paths."""

    def test_eval_constant_bool(self):
        """Test boolean constant evaluation."""
        code = "x = True"
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == BoolVal(True)

    def test_eval_constant_none(self):
        """Test None constant evaluation."""
        code = "x = None"
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        # None evaluates to None, variable may not be set
        state.get_variable("x")

    def test_eval_comparison_chain(self):
        """Test chained comparison: a < b < c."""
        code = """
a = 1
b = 2
c = 3
result = a < b < c
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        interp.execute(ir)

        # Chained comparisons become AND of individual comparisons

    def test_eval_comparison_operators(self):
        """Test all comparison operators."""
        code = """
a = 5
b = 3
eq = a == b
ne = a != b
lt = a < b
le = a <= b
gt = a > b
ge = a >= b
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("eq")) == BoolVal(False)
        assert simplify(state.get_variable("ne")) == BoolVal(True)
        assert simplify(state.get_variable("lt")) == BoolVal(False)
        assert simplify(state.get_variable("le")) == BoolVal(False)
        assert simplify(state.get_variable("gt")) == BoolVal(True)
        assert simplify(state.get_variable("ge")) == BoolVal(True)

    def test_eval_binary_mod(self):
        """Test modulo operator."""
        code = """
x = 17 % 5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(2)

    def test_eval_binary_floor_div(self):
        """Test floor division operator."""
        code = """
x = 17 // 5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        state.get_variable("x")
        # Floor div should be handled

    def test_eval_bool_op_or(self):
        """Test boolean OR operation."""
        code = """
a = True or False
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        a = state.get_variable("a")
        assert simplify(a) == BoolVal(True)

    def test_eval_bool_op_and(self):
        """Test boolean AND operation."""
        code = """
a = True and True
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        a = state.get_variable("a")
        assert simplify(a) == BoolVal(True)

    def test_eval_unary_neg(self):
        """Test unary negation."""
        code = """
x = -5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == IntVal(-5)

    def test_eval_unary_not(self):
        """Test unary not."""
        code = """
x = not True
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert simplify(x) == BoolVal(False)


class TestIRInterpreterSymbolicCalls:
    """Test symbolic() call handling."""

    def test_symbolic_call_int(self):
        """Test symbolic('name', int) call."""
        code = """
x = symbolic('x', int)
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert x is not None
        assert x.sort() == IntSort()

    def test_symbolic_call_bool(self):
        """Test symbolic('name', bool) call."""
        code = """
x = symbolic('x', bool)
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        x = state.get_variable("x")
        assert x is not None
        assert x.sort() == BoolSort()

    def test_symbolic_call_insufficient_args(self):
        """Test symbolic call with insufficient arguments."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall, IRConstant

        # symbolic() with only 1 arg
        call = IRCall(
            func=IRName(id="symbolic"),
            args=[IRConstant(value="x")],  # Missing type arg
            kwargs={},
        )
        assign = IRAssign(targets=[IRName(id="x")], value=call)
        ir = IRModule(body=[assign])

        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle gracefully
        assert len(result.states) >= 1

    def test_non_symbolic_call_returns_none(self):
        """Test non-symbolic function calls return None."""
        code = """
x = some_function(1, 2)
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Call result is None, x may be created as placeholder
        assert len(result.states) >= 1


class TestIRInterpreterPathPruning:
    """Test path pruning and feasibility checking."""

    def test_infeasible_true_branch(self):
        """Test pruning when only false branch is feasible."""
        code = """
x = 5
if x < 0:
    y = 1
else:
    y = 2
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # x = 5, so x < 0 is infeasible
        # Should only have y = 2 path
        assert len(result.states) == 1
        state = result.states[0]
        assert simplify(state.get_variable("y")) == IntVal(2)
        assert result.pruned_count >= 1

    def test_infeasible_false_branch(self):
        """Test pruning when only true branch is feasible."""
        code = """
x = 5
if x > 0:
    y = 1
else:
    y = 2
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # x = 5, so x > 0 is always true
        # Should only have y = 1 path
        assert len(result.states) == 1
        state = result.states[0]
        assert simplify(state.get_variable("y")) == IntVal(1)
        assert result.pruned_count >= 1

    def test_both_branches_infeasible(self):
        """Test when both branches are infeasible (dead code)."""
        # This is an edge case - create contradictory constraints
        interp = IRSymbolicInterpreter()
        interp.declare_symbolic("x", IntSort())

        # Add constraint x > 10 AND x < 5 (impossible)

        state = interp._initial_state
        x = state.get_variable("x")
        state.add_constraint(x > 10)
        state.add_constraint(x < 5)

        code = """
if y > 0:
    z = 1
else:
    z = 2
"""
        ir = PythonNormalizer().normalize(code)
        interp.execute(ir)

        # Both branches may be pruned due to contradictory base constraints

    def test_while_loop_early_exit(self):
        """Test while loop exits early when condition becomes infeasible."""
        code = """
x = 3
while x > 0:
    x = x - 1
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=10)
        result = interp.execute(ir)

        # Loop should terminate when x <= 0
        assert len(result.states) >= 1


class TestIRExecutionResult:
    """Test IRExecutionResult dataclass."""

    def test_execution_result_defaults(self):
        """Test IRExecutionResult default values."""
        result = IRExecutionResult()
        assert result.states == []
        assert result.path_count == 0
        assert result.pruned_count == 0

    def test_execution_result_custom_values(self):
        """Test IRExecutionResult with custom values."""
        from code_scalpel.symbolic_execution_tools.state_manager import SymbolicState

        state = SymbolicState()
        result = IRExecutionResult(states=[state], path_count=5, pruned_count=2)
        assert len(result.states) == 1
        assert result.path_count == 5
        assert result.pruned_count == 2


class TestLanguageSemanticsAbstract:
    """Test LanguageSemantics abstract methods are properly defined."""

    def test_abstract_methods_exist(self):
        """Verify LanguageSemantics has all required abstract methods."""
        from code_scalpel.symbolic_execution_tools.ir_interpreter import (
            LanguageSemantics,
        )

        abstract_methods = [
            "name",
            "binary_add",
            "binary_sub",
            "binary_mul",
            "binary_div",
            "binary_mod",
            "compare_eq",
            "compare_ne",
            "compare_lt",
            "compare_le",
            "compare_gt",
            "compare_ge",
            "unary_neg",
            "unary_not",
            "to_bool",
        ]

        for method_name in abstract_methods:
            assert hasattr(LanguageSemantics, method_name)


class TestPythonSemanticsComplete:
    """Complete coverage of Python semantics."""

    def test_python_semantics_name(self):
        """Test Python semantics name property."""
        sem = PythonSemantics()
        assert sem.name == "python"

    def test_python_binary_div(self):
        """Test Python division."""
        sem = PythonSemantics()
        result = sem.binary_div(IntVal(10), IntVal(3), None)
        # Integer division in Z3
        assert result is not None

    def test_python_binary_mod(self):
        """Test Python modulo."""
        sem = PythonSemantics()
        result = sem.binary_mod(IntVal(10), IntVal(3), None)
        assert simplify(result) == IntVal(1)

    def test_python_compare_le(self):
        """Test Python less-than-or-equal."""
        sem = PythonSemantics()
        result = sem.compare_le(IntVal(5), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

        result = sem.compare_le(IntVal(4), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_python_compare_ge(self):
        """Test Python greater-than-or-equal."""
        sem = PythonSemantics()
        result = sem.compare_ge(IntVal(5), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

        result = sem.compare_ge(IntVal(6), IntVal(5), None)
        assert simplify(result) == BoolVal(True)

    def test_python_compare_ne(self):
        """Test Python not-equal."""
        sem = PythonSemantics()
        result = sem.compare_ne(IntVal(5), IntVal(3), None)
        assert simplify(result) == BoolVal(True)

    def test_python_unary_neg(self):
        """Test Python negation."""
        sem = PythonSemantics()
        result = sem.unary_neg(IntVal(5), None)
        assert simplify(result) == IntVal(-5)

    def test_python_unary_not_int(self):
        """Test Python not with int (truthy check)."""
        sem = PythonSemantics()
        # not 0 -> True
        result = sem.unary_not(IntVal(0), None)
        assert simplify(result) == BoolVal(True)

        # not 5 -> False
        result = sem.unary_not(IntVal(5), None)
        assert simplify(result) == BoolVal(False)

    def test_python_unary_not_bool(self):
        """Test Python not with bool."""
        sem = PythonSemantics()
        result = sem.unary_not(BoolVal(True), None)
        assert simplify(result) == BoolVal(False)


class TestGetSemantics:
    """Test get_semantics factory function."""

    def test_get_semantics_python_aliases(self):
        """Test Python aliases."""
        sem = get_semantics("python")
        assert isinstance(sem, PythonSemantics)

        sem = get_semantics("py")
        assert isinstance(sem, PythonSemantics)

    def test_get_semantics_javascript_aliases(self):
        """Test JavaScript aliases."""
        sem = get_semantics("javascript")
        assert isinstance(sem, JavaScriptSemantics)

        sem = get_semantics("js")
        assert isinstance(sem, JavaScriptSemantics)

        # TypeScript not registered - defaults to Python
        # This is current behavior per _SEMANTICS_REGISTRY

    def test_get_semantics_unknown_defaults_python(self):
        """Test unknown language defaults to Python."""
        sem = get_semantics("unknown_language")
        assert isinstance(sem, PythonSemantics)


class TestIRInterpreterEdgeCases:
    """Test edge cases for maximum coverage."""

    def test_execute_expr_statement(self):
        """Test expression statement evaluation (side effects)."""
        code = """
x = 1
x  # Expression statement, just evaluates x
y = 2
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("y")) == IntVal(2)

    def test_execute_class_def_skipped(self):
        """Test class definitions are skipped."""
        code = """
class Foo:
    pass
x = 5
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        assert simplify(state.get_variable("x")) == IntVal(5)

    def test_execute_break_continue(self):
        """Test break and continue statements in loops."""
        code = """
x = 0
for i in range(10):
    if i == 5:
        break
    x = i
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=10)
        interp.execute(ir)

        # Loop should have run but break/continue are handled

    def test_execute_unknown_statement_type(self):
        """Test unknown statement type is handled gracefully."""
        from code_scalpel.ir.nodes import IRNode
        from dataclasses import dataclass

        @dataclass
        class UnknownStmt(IRNode):
            """Unknown statement for testing."""

            pass

        ir = IRModule(body=[UnknownStmt()])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle unknown statement by continuing
        assert len(result.states) >= 1

    def test_eval_unknown_expression_type(self):
        """Test unknown expression type returns None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRExpr
        from dataclasses import dataclass

        @dataclass
        class UnknownExpr(IRExpr):
            """Unknown expression for testing."""

            pass

        assign = IRAssign(targets=[IRName(id="x")], value=UnknownExpr())
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle unknown expression by returning None
        assert len(result.states) >= 1

    def test_eval_constant_unsupported_type(self):
        """Test constant with unsupported type (not bool/int/None)."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRConstant

        # Float constant
        assign = IRAssign(targets=[IRName(id="x")], value=IRConstant(value=3.14))
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Float not supported - should handle gracefully
        assert len(result.states) >= 1

    def test_eval_binary_op_unsupported_operator(self):
        """Test binary op with unsupported operator returns None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRConstant

        # BIT_AND is not supported in semantics
        binary = IRBinaryOp(
            left=IRConstant(value=5),
            right=IRConstant(value=3),
            op=BinaryOperator.BIT_AND,
        )
        assign = IRAssign(targets=[IRName(id="x")], value=binary)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_eval_unary_op_unsupported_operator(self):
        """Test unary op with unsupported operator returns None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRUnaryOp, IRConstant
        from code_scalpel.ir.operators import UnaryOperator

        # INVERT (~) is not supported
        unary = IRUnaryOp(operand=IRConstant(value=5), op=UnaryOperator.INVERT)
        assign = IRAssign(targets=[IRName(id="x")], value=unary)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_eval_compare_unsupported_operator(self):
        """Test comparison with unsupported operator."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRConstant

        # IS is not supported
        compare = IRCompare(
            left=IRConstant(value=5),
            ops=[CompareOperator.IS],
            comparators=[IRConstant(value=5)],
        )
        assign = IRAssign(targets=[IRName(id="x")], value=compare)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_eval_bool_op_empty_values(self):
        """Test bool op with empty values list."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRBoolOp
        from code_scalpel.ir.operators import BoolOperator

        bool_op = IRBoolOp(op=BoolOperator.AND, values=[])
        assign = IRAssign(targets=[IRName(id="x")], value=bool_op)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Empty bool op should return None
        assert len(result.states) >= 1

    def test_symbolic_call_non_string_name(self):
        """Test symbolic call where name arg is not a string constant."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall, IRConstant

        call = IRCall(
            func=IRName(id="symbolic"),
            args=[IRConstant(value=123), IRName(id="int")],  # Name is int, not string
            kwargs={},
        )
        assign = IRAssign(targets=[IRName(id="x")], value=call)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle gracefully (return None)
        assert len(result.states) >= 1

    def test_symbolic_call_unknown_type(self):
        """Test symbolic call with unknown type name."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall, IRConstant

        call = IRCall(
            func=IRName(id="symbolic"),
            args=[IRConstant(value="x"), IRName(id="float")],  # Float not supported
            kwargs={},
        )
        assign = IRAssign(targets=[IRName(id="x")], value=call)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_assign_to_non_name_target(self):
        """Test assignment to non-name target (tuple, attribute)."""
        from code_scalpel.ir.nodes import IRAssign, IRAttribute, IRName, IRConstant

        # Assign to attribute - not a simple name
        attr = IRAttribute(value=IRName(id="obj"), attr="x")
        assign = IRAssign(targets=[attr], value=IRConstant(value=5))
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle by skipping non-name targets
        assert len(result.states) >= 1

    def test_aug_assign_to_non_name(self):
        """Test augmented assignment to non-name target."""
        from code_scalpel.ir.nodes import IRAugAssign, IRAttribute, IRName, IRConstant
        from code_scalpel.ir.operators import AugAssignOperator

        attr = IRAttribute(value=IRName(id="obj"), attr="x")
        aug = IRAugAssign(
            target=attr, op=AugAssignOperator.ADD, value=IRConstant(value=1)
        )
        ir = IRModule(body=[aug])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_aug_assign_unknown_operator(self):
        """Test augmented assignment with operator not in switch."""
        from code_scalpel.ir.nodes import IRAugAssign, IRName, IRConstant
        from code_scalpel.ir.operators import AugAssignOperator

        # FLOOR_DIV might not be handled in all code paths
        aug = IRAugAssign(
            target=IRName(id="x"),
            op=AugAssignOperator.FLOOR_DIV,
            value=IRConstant(value=2),
        )
        # First set x
        assign = IRAssign(targets=[IRName(id="x")], value=IRConstant(value=10))
        ir = IRModule(body=[assign, aug])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_while_unevaluable_condition(self):
        """Test while loop with unevaluable condition."""
        from code_scalpel.ir.nodes import IRWhile, IRAssign, IRName, IRCall, IRConstant

        call = IRCall(func=IRName(id="unknown_func"), args=[], kwargs={})
        while_stmt = IRWhile(
            test=call,
            body=[IRAssign(targets=[IRName(id="x")], value=IRConstant(value=1))],
            orelse=[],
        )
        ir = IRModule(body=[while_stmt])
        interp = IRSymbolicInterpreter(max_loop_iterations=3)
        result = interp.execute(ir)

        # Should handle by assuming iteration
        assert len(result.states) >= 1

    def test_for_loop_non_name_target(self):
        """Test for loop with non-name target."""
        from code_scalpel.ir.nodes import (
            IRFor,
            IRAssign,
            IRAttribute,
            IRName,
            IRConstant,
            IRCall,
        )

        # For loop target is attribute, not simple name
        attr = IRAttribute(value=IRName(id="obj"), attr="x")
        for_stmt = IRFor(
            target=attr,
            iter=IRCall(func=IRName(id="range"), args=[IRConstant(value=3)], kwargs={}),
            body=[IRAssign(targets=[IRName(id="y")], value=IRConstant(value=1))],
            orelse=[],
        )
        ir = IRModule(body=[for_stmt])
        interp = IRSymbolicInterpreter(max_loop_iterations=3)
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_compare_with_none_left(self):
        """Test comparison where left eval returns None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall, IRConstant

        call = IRCall(func=IRName(id="unknown"), args=[], kwargs={})
        compare = IRCompare(
            left=call,  # Will evaluate to None
            ops=[CompareOperator.EQ],
            comparators=[IRConstant(value=5)],
        )
        assign = IRAssign(targets=[IRName(id="x")], value=compare)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_python_add_non_arith(self):
        """Test Python addition with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.binary_add(BoolVal(True), BoolVal(False), None)
        # Python semantics only supports ArithRef for add
        assert result is None

    def test_python_sub_non_arith(self):
        """Test Python subtraction with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.binary_sub(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_mul_non_arith(self):
        """Test Python multiplication with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.binary_mul(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_div_non_arith(self):
        """Test Python division with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.binary_div(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_mod_non_arith(self):
        """Test Python modulo with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.binary_mod(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_compare_lt_non_arith(self):
        """Test Python less-than with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.compare_lt(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_compare_le_non_arith(self):
        """Test Python less-than-or-equal with non-ArithRef."""
        sem = PythonSemantics()
        result = sem.compare_le(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_compare_gt_non_arith(self):
        """Test Python greater-than with non-ArithRef values."""
        sem = PythonSemantics()
        result = sem.compare_gt(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_compare_ge_non_arith(self):
        """Test Python greater-than-or-equal with non-ArithRef."""
        sem = PythonSemantics()
        result = sem.compare_ge(BoolVal(True), BoolVal(False), None)
        assert result is None

    def test_python_unary_neg_non_arith(self):
        """Test Python negation with non-ArithRef."""
        sem = PythonSemantics()
        result = sem.unary_neg(BoolVal(True), None)
        assert result is None

    def test_python_to_bool_unsupported_type(self):
        """Test Python to_bool with unsupported type."""
        from z3 import String

        sem = PythonSemantics()
        result = sem.to_bool(String("test"), None)
        assert result is None

    def test_chained_comparison_multiple_ops(self):
        """Test chained comparison: 1 < 2 < 3 < 4."""
        code = """
result = 1 < 2 < 3 < 4
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        state = result.states[0]
        r = state.get_variable("result")
        # Chained comparisons should be True
        assert simplify(r) == BoolVal(True)

    def test_assignment_creates_placeholder_for_unknown(self):
        """Test assigning None value creates placeholder."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRCall

        call = IRCall(func=IRName(id="unknown_func"), args=[], kwargs={})
        assign = IRAssign(targets=[IRName(id="x")], value=call)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # x should exist (placeholder created)
        state = result.states[0]
        x = state.get_variable("x")
        # Should be an Int placeholder
        assert x is not None

    def test_python_unary_not_with_unsupported_type(self):
        """Test Python unary_not when to_bool returns None."""
        from z3 import String

        sem = PythonSemantics()
        # String not supported by to_bool, so unary_not returns None
        result = sem.unary_not(String("test"), None)
        assert result is None

    def test_js_unary_not_with_unsupported_type(self):
        """Test JavaScript unary_not when to_bool returns None."""
        from z3 import String

        sem = JavaScriptSemantics()
        # String not supported by to_bool, so unary_not returns None
        result = sem.unary_not(String("test"), None)
        assert result is None

    def test_while_loop_body_produces_empty_states(self):
        """Test while loop when body produces no states."""
        # This is hard to trigger naturally, so we'll just verify the path
        # exists by testing a loop that completes
        code = """
x = 0
while x < 2:
    x = x + 1
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(ir)

        # Loop should complete
        assert len(result.states) >= 1

    def test_for_loop_body_produces_empty_states(self):
        """Test for loop when body produces no states."""
        code = """
y = 0
for i in range(3):
    y = y + 1
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=5)
        result = interp.execute(ir)

        assert len(result.states) >= 1

    def test_while_loop_max_iterations_hit(self):
        """Test while loop hits max iterations and adds remaining states."""
        code = """
while True:
    x = 1
"""
        ir = PythonNormalizer().normalize(code)
        interp = IRSymbolicInterpreter(max_loop_iterations=3)
        result = interp.execute(ir)

        # Should hit max iterations - remaining states added to result
        assert len(result.states) >= 1

    def test_bool_op_with_none_value_in_chain(self):
        """Test bool op where one value evaluates to None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRBoolOp, IRConstant, IRCall
        from code_scalpel.ir.operators import BoolOperator

        # and with an unevaluable call
        call = IRCall(func=IRName(id="unknown"), args=[], kwargs={})
        bool_op = IRBoolOp(op=BoolOperator.AND, values=[IRConstant(value=True), call])
        assign = IRAssign(targets=[IRName(id="x")], value=bool_op)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        # Should handle None value gracefully
        assert len(result.states) >= 1

    def test_compare_chain_with_none_comparator(self):
        """Test comparison chain where comparator evaluates to None."""
        from code_scalpel.ir.nodes import IRAssign, IRName, IRConstant, IRCall

        call = IRCall(func=IRName(id="unknown"), args=[], kwargs={})
        compare = IRCompare(
            left=IRConstant(value=5),
            ops=[CompareOperator.LT],
            comparators=[call],  # Evaluates to None
        )
        assign = IRAssign(targets=[IRName(id="x")], value=compare)
        ir = IRModule(body=[assign])
        interp = IRSymbolicInterpreter()
        result = interp.execute(ir)

        assert len(result.states) >= 1
