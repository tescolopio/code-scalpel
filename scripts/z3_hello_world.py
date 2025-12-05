#!/usr/bin/env python3
"""
Z3 Tracer Bullet - Milestone M0
================================

This script proves that:
1. z3-solver is installed correctly
2. We can create symbolic variables
3. We can add constraints
4. We can solve for satisfying assignments

If this script runs without error, we have a foundation for symbolic execution.

Usage:
    python scripts/z3_hello_world.py
"""

from z3 import (
    Int, Bool, Real, String,
    Solver, sat, unsat, unknown,
    And, Or, Not, Implies,
    If, ForAll, Exists
)


def test_basic_integer_constraints():
    """Test 1: Basic integer constraint solving."""
    print("=" * 60)
    print("TEST 1: Basic Integer Constraints")
    print("=" * 60)
    
    # Declare symbolic integer
    x = Int('x')
    
    # Create solver
    solver = Solver()
    
    # Add constraints: x > 10 AND x < 20 AND x is even
    solver.add(x > 10)
    solver.add(x < 20)
    solver.add(x % 2 == 0)
    
    # Check satisfiability
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        x_value = model[x].as_long()
        print(f"✅ SATISFIABLE")
        print(f"   Constraint: x > 10 ∧ x < 20 ∧ x % 2 == 0")
        print(f"   Solution: x = {x_value}")
        assert 10 < x_value < 20 and x_value % 2 == 0
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_unsatisfiable_constraints():
    """Test 2: Prove we can detect impossible constraints."""
    print("\n" + "=" * 60)
    print("TEST 2: Unsatisfiable Constraints")
    print("=" * 60)
    
    x = Int('x')
    solver = Solver()
    
    # Impossible: x > 10 AND x < 5
    solver.add(x > 10)
    solver.add(x < 5)
    
    result = solver.check()
    
    if result == unsat:
        print(f"✅ UNSATISFIABLE (as expected)")
        print(f"   Constraint: x > 10 ∧ x < 5")
        print(f"   No solution exists (contradiction)")
    else:
        print(f"❌ FAILED: Expected UNSAT, got {result}")
        return False
    
    return True


def test_boolean_logic():
    """Test 3: Boolean constraint solving."""
    print("\n" + "=" * 60)
    print("TEST 3: Boolean Logic")
    print("=" * 60)
    
    # Symbolic booleans
    a = Bool('a')
    b = Bool('b')
    c = Bool('c')
    
    solver = Solver()
    
    # (a OR b) AND (NOT a OR c) AND (NOT b OR NOT c)
    solver.add(Or(a, b))
    solver.add(Or(Not(a), c))
    solver.add(Or(Not(b), Not(c)))
    
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        print(f"✅ SATISFIABLE")
        print(f"   Constraint: (a ∨ b) ∧ (¬a ∨ c) ∧ (¬b ∨ ¬c)")
        print(f"   Solution: a={model[a]}, b={model[b]}, c={model[c]}")
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_multiple_variables():
    """Test 4: Multiple interrelated variables (simulating program state)."""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Variables (Program State)")
    print("=" * 60)
    
    # Simulate: y = x * 2; z = y + 5; assert z > 100
    x = Int('x')
    y = Int('y')
    z = Int('z')
    
    solver = Solver()
    
    # Program semantics as constraints
    solver.add(y == x * 2)      # y = x * 2
    solver.add(z == y + 5)      # z = y + 5
    solver.add(z > 100)         # Path condition: z > 100
    
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        x_val = model[x].as_long()
        y_val = model[y].as_long()
        z_val = model[z].as_long()
        print(f"✅ SATISFIABLE")
        print(f"   Program: y = x * 2; z = y + 5")
        print(f"   Path condition: z > 100")
        print(f"   Solution: x={x_val}, y={y_val}, z={z_val}")
        
        # Verify the solution
        assert y_val == x_val * 2
        assert z_val == y_val + 5
        assert z_val > 100
        print(f"   Verified: Constraints hold!")
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_if_branch_analysis():
    """Test 5: Simulate analyzing an if-statement branch."""
    print("\n" + "=" * 60)
    print("TEST 5: If-Branch Analysis")
    print("=" * 60)
    
    # Simulate:
    # def check(x):
    #     if x > 10:
    #         if x < 20:
    #             return "SECRET"  # What x reaches here?
    #     return "NORMAL"
    
    x = Int('x')
    solver = Solver()
    
    # Path condition to reach "SECRET"
    solver.add(x > 10)   # First if
    solver.add(x < 20)   # Second if
    
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        x_val = model[x].as_long()
        print(f"✅ SATISFIABLE")
        print(f"   Code: if x > 10: if x < 20: return 'SECRET'")
        print(f"   Path condition: x > 10 ∧ x < 20")
        print(f"   Trigger value: x = {x_val}")
        print(f"   Interpretation: Input x={x_val} reaches 'SECRET' branch")
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_password_cracker():
    """Test 6: The classic symbolic execution demo - password cracking."""
    print("\n" + "=" * 60)
    print("TEST 6: Password Cracker (Classic Demo)")
    print("=" * 60)
    
    # Simulate:
    # def login(password):
    #     if password == 42:
    #         return "ACCESS GRANTED"
    #     return "DENIED"
    
    password = Int('password')
    solver = Solver()
    
    # Path condition to reach "ACCESS GRANTED"
    solver.add(password == 42)
    
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        cracked = model[password].as_long()
        print(f"✅ PASSWORD CRACKED!")
        print(f"   Code: if password == 42: return 'ACCESS GRANTED'")
        print(f"   Discovered password: {cracked}")
        assert cracked == 42
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_complex_arithmetic():
    """Test 7: Complex arithmetic expressions."""
    print("\n" + "=" * 60)
    print("TEST 7: Complex Arithmetic")
    print("=" * 60)
    
    # Simulate finding inputs where result overflows a threshold
    a = Int('a')
    b = Int('b')
    
    solver = Solver()
    
    # Constraints: a and b are positive single digits
    solver.add(a > 0, a < 10)
    solver.add(b > 0, b < 10)
    
    # Find: a * b + a + b == 35
    solver.add(a * b + a + b == 35)
    
    result = solver.check()
    
    if result == sat:
        model = solver.model()
        a_val = model[a].as_long()
        b_val = model[b].as_long()
        computed = a_val * b_val + a_val + b_val
        print(f"✅ SATISFIABLE")
        print(f"   Constraint: a * b + a + b == 35 (with 0 < a,b < 10)")
        print(f"   Solution: a={a_val}, b={b_val}")
        print(f"   Verification: {a_val} * {b_val} + {a_val} + {b_val} = {computed}")
        assert computed == 35
    else:
        print(f"❌ FAILED: Expected SAT, got {result}")
        return False
    
    return True


def test_dead_code_detection():
    """Test 8: Detect unreachable code via unsatisfiable path conditions."""
    print("\n" + "=" * 60)
    print("TEST 8: Dead Code Detection")
    print("=" * 60)
    
    # Simulate:
    # def example(x):
    #     if x > 10:
    #         if x < 5:  # IMPOSSIBLE! Dead code follows
    #             unreachable()
    
    x = Int('x')
    solver = Solver()
    
    # Path condition to reach unreachable()
    solver.add(x > 10)
    solver.add(x < 5)
    
    result = solver.check()
    
    if result == unsat:
        print(f"✅ DEAD CODE DETECTED!")
        print(f"   Path condition: x > 10 ∧ x < 5")
        print(f"   Result: UNSATISFIABLE - code is unreachable")
    else:
        print(f"❌ FAILED: Expected UNSAT, got {result}")
        return False
    
    return True


def main():
    """Run all tracer bullet tests."""
    print("\n" + "=" * 60)
    print("      Z3 TRACER BULLET - Operation Redemption M0")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_integer_constraints,
        test_unsatisfiable_constraints,
        test_boolean_logic,
        test_multiple_variables,
        test_if_branch_analysis,
        test_password_cracker,
        test_complex_arithmetic,
        test_dead_code_detection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION in {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print(f"   Passed: {passed}/{len(tests)}")
    print(f"   Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print()
        print("   🎉 ALL TESTS PASSED!")
        print("   Z3 integration is working. Ready for Milestone M1.")
        print("=" * 60)
        return 0
    else:
        print()
        print("   ❌ SOME TESTS FAILED")
        print("   Fix issues before proceeding.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
