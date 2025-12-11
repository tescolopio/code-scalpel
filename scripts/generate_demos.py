#!/usr/bin/env python3
"""
Generate Demo Suite for Code Scalpel v1.0.x

Creates evidence-based demos proving three core value propositions:
1. Security: Taint tracking through variable indirection
2. Enterprise: Cross-file Java call graph analysis
3. QA: Symbolic execution test generation

Usage:
    python scripts/generate_demos.py
"""

import os


def create_demo_suite():
    """Generate the complete demo suite."""
    os.makedirs("demos/enterprise", exist_ok=True)

    # =========================================================================
    # 1. Security Demo: Hidden Taint Flow
    # Target: The Security Engineer
    # Proves: Taint tracking, not just keyword matching
    # =========================================================================
    vibe_code = '''"""
Vibe Check Demo: Hidden SQL Injection

This demo proves that Code Scalpel tracks TAINT FLOW, not just keywords.
A regex-based linter would miss this because the dangerous input is hidden
through multiple variable assignments.

Run:
    code-scalpel scan demos/vibe_check.py

Expected: Code Scalpel detects SQL Injection at cursor.execute() 
          by tracing: request.args -> user_id -> query_base -> final_query
"""
import sqlite3
from flask import request


def get_user_data():
    """Fetch user data - contains hidden SQL injection vulnerability."""
    # SOURCE: User input (Tainted)
    user_id = request.args.get('id')
    
    # PROPAGATION: Hiding the taint through variable indirection
    # A regex linter looking for "request.args" near "execute" would miss this
    part1 = "SELECT * FROM users WHERE id = '"
    part2 = "'"
    query_base = part1 + user_id  # Taint flows here
    
    # More indirection
    final_query = query_base + part2  # Taint flows here too
    
    # SINK: SQL Execution with tainted data
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    
    # Code Scalpel should detect this despite 3 levels of indirection
    # Taint path: request.args.get('id') -> user_id -> query_base -> final_query
    cursor.execute(final_query)  # VULNERABILITY: CWE-89 SQL Injection
    
    return cursor.fetchall()


def safe_version():
    """The correct way - parameterized queries."""
    user_id = request.args.get('id')
    
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    
    # Parameterized query - no taint flow to execute()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    
    return cursor.fetchall()
'''
    with open("demos/vibe_check.py", "w") as f:
        f.write(vibe_code)
    print("  ✓ Created demos/vibe_check.py (Security: Taint Flow)")

    # =========================================================================
    # 2. Enterprise Demo: Java Call Graph
    # Target: The Java Architect
    # Proves: Cross-file dependency analysis
    # =========================================================================
    auth_controller = """/*
 * Enterprise Demo: Cross-File Call Graph
 * 
 * This demo proves Code Scalpel can analyze Enterprise Java projects
 * and build call graphs across multiple files.
 * 
 * Run:
 *     code-scalpel analyze demos/enterprise/AuthController.java
 * 
 * Expected: Call graph shows AuthController.login() -> AuthService.validate()
 */
package com.demo;

public class AuthController {
    private AuthService service = new AuthService();
    
    /**
     * Handle login request.
     * Code Scalpel should detect the cross-file call to AuthService.validate()
     */
    public void login(String user, String pass) {
        // Cross-file dependency - calls method in AuthService.java
        if (service.validate(user, pass)) {
            System.out.println("Welcome, " + user);
            auditLog("LOGIN_SUCCESS", user);
        } else {
            System.out.println("Access Denied");
            auditLog("LOGIN_FAILURE", user);
        }
    }
    
    /**
     * Internal audit logging.
     * Code Scalpel should detect this as an internal call.
     */
    private void auditLog(String event, String user) {
        System.out.println("[AUDIT] " + event + " for " + user);
    }
    
    /**
     * Logout handler - no cross-file calls.
     */
    public void logout(String user) {
        System.out.println("Goodbye, " + user);
        auditLog("LOGOUT", user);
    }
}
"""
    auth_service = """/*
 * Enterprise Demo: Service Layer
 * 
 * This file is called BY AuthController.java.
 * Code Scalpel's call graph should show this dependency.
 */
package com.demo;

public class AuthService {
    
    /**
     * Validate user credentials.
     * Called from: AuthController.login()
     */
    public boolean validate(String username, String password) {
        // Simplified validation for demo
        if (username == null || password == null) {
            return false;
        }
        
        // In real code, this would check a database
        return password.length() >= 8;
    }
    
    /**
     * Check if user has admin privileges.
     * Not called in this demo - should NOT appear in call graph.
     */
    public boolean isAdmin(String username) {
        return "admin".equals(username);
    }
}
"""
    with open("demos/enterprise/AuthController.java", "w") as f:
        f.write(auth_controller)
    with open("demos/enterprise/AuthService.java", "w") as f:
        f.write(auth_service)
    print("  ✓ Created demos/enterprise/*.java (Enterprise: Call Graph)")

    # =========================================================================
    # 3. Logic Demo: Z3 Constraints for Test Generation
    # Target: The QA Lead
    # Proves: Symbolic execution generates concrete test inputs
    # =========================================================================
    logic_code = '''"""
Test Generation Demo: Loan Approval Algorithm

This demo proves Code Scalpel's symbolic execution engine can analyze
branching logic and generate concrete test inputs for EVERY path.

Run:
    code-scalpel analyze demos/test_gen_scenario.py
    
Then use the MCP server:
    generate_unit_tests(code, function_name="loan_approval")

Expected Output: pytest cases with exact values like:
    - test_reject: credit_score=599
    - test_instant_approve: income=100001, debt=4999, credit_score=700
    - test_manual_review: income=100001, debt=5001, credit_score=700
    - test_high_risk: income=50000, debt=30000, credit_score=700
    - test_standard_approve: income=50000, debt=20000, credit_score=700
"""


def loan_approval(income: int, debt: int, credit_score: int) -> str:
    """
    Determine loan approval status based on financial metrics.
    
    This function has 5 distinct paths that symbolic execution should find:
    
    Path 1: REJECT (credit_score < 600)
    Path 2: INSTANT_APPROVE (income > 100000 AND debt < 5000)
    Path 3: MANUAL_REVIEW (income > 100000 AND debt >= 5000)
    Path 4: HIGH_RISK (debt > income * 0.5)
    Path 5: STANDARD_APPROVE (default case)
    
    Args:
        income: Annual income in dollars
        debt: Total debt in dollars
        credit_score: FICO score (300-850)
    
    Returns:
        Approval status string
    """
    # Path 1: Immediate rejection for low credit
    if credit_score < 600:
        return "REJECT"
    
    # Path 2 & 3: High income branch
    if income > 100000:
        if debt < 5000:
            return "INSTANT_APPROVE"  # Path 2: Rich and debt-free
        else:
            return "MANUAL_REVIEW"    # Path 3: Rich but has debt
    
    # Path 4: Debt-to-income ratio check
    if debt > (income * 0.5):
        return "HIGH_RISK"
    
    # Path 5: Default approval
    return "STANDARD_APPROVE"


def calculate_interest_rate(credit_score: int, loan_amount: int) -> float:
    """
    Calculate interest rate based on credit score and loan amount.
    
    Another function for symbolic execution to analyze.
    Has 4 paths based on credit score tiers.
    """
    # Base rate
    base_rate = 5.0
    
    # Credit score adjustments
    if credit_score >= 800:
        rate_adjustment = -1.5  # Excellent credit discount
    elif credit_score >= 700:
        rate_adjustment = -0.5  # Good credit discount
    elif credit_score >= 600:
        rate_adjustment = 1.0   # Fair credit premium
    else:
        rate_adjustment = 3.0   # Poor credit premium
    
    # Large loan adjustment
    if loan_amount > 500000:
        rate_adjustment += 0.25
    
    return base_rate + rate_adjustment


# Self-test to verify the logic
if __name__ == "__main__":
    # These are the exact values Z3 should derive
    test_cases = [
        (50000, 10000, 599, "REJECT"),           # Path 1
        (100001, 4999, 700, "INSTANT_APPROVE"),  # Path 2
        (100001, 5001, 700, "MANUAL_REVIEW"),    # Path 3
        (50000, 30000, 700, "HIGH_RISK"),        # Path 4
        (50000, 20000, 700, "STANDARD_APPROVE"), # Path 5
    ]
    
    print("Loan Approval Logic Verification:")
    print("-" * 50)
    for income, debt, score, expected in test_cases:
        result = loan_approval(income, debt, score)
        status = "✓" if result == expected else "✗"
        print(f"  {status} income={income}, debt={debt}, score={score}")
        print(f"      Expected: {expected}, Got: {result}")
    print("-" * 50)
    print("If Code Scalpel's test generator works, it will derive these values automatically.")
'''
    with open("demos/test_gen_scenario.py", "w") as f:
        f.write(logic_code)
    print("  ✓ Created demos/test_gen_scenario.py (QA: Test Generation)")

    # =========================================================================
    # 4. README for the demos
    # =========================================================================
    readme = """# Code Scalpel Demo Suite

Evidence-based demonstrations proving Code Scalpel's core capabilities.

## Quick Start

```bash
pip install code-scalpel==1.0.1
```

## Demo 1: Security - Taint Flow Detection

**Target Audience:** Security Engineers

**The Challenge:** Regex-based linters miss SQL injection when user input is 
hidden through variable assignments.

```bash
code-scalpel scan demos/vibe_check.py
```

**Expected:** Detects SQL Injection at line 37 despite 3 levels of variable 
indirection from `request.args.get('id')` to `cursor.execute()`.

## Demo 2: Enterprise - Cross-File Call Graph

**Target Audience:** Java Architects

**The Challenge:** Understanding dependencies in large Java codebases.

```bash
code-scalpel analyze demos/enterprise/AuthController.java --json
```

**Expected:** Call graph shows:
- `AuthController.login()` → `AuthService.validate()`
- `AuthController.login()` → `AuthController.auditLog()`

## Demo 3: QA - Automatic Test Generation

**Target Audience:** QA Leads

**The Challenge:** AI writes code, but how do you test every branch?

```bash
# Using the MCP server or Python API:
from code_scalpel.mcp.server import _generate_tests_sync

with open("demos/test_gen_scenario.py") as f:
    code = f.read()

result = _generate_tests_sync(code, function_name="loan_approval")
print(result.pytest_code)
```

**Expected:** Generates pytest cases with concrete inputs:
- `credit_score=599` → REJECT
- `income=100001, debt=4999, credit_score=700` → INSTANT_APPROVE
- `income=100001, debt=5001, credit_score=700` → MANUAL_REVIEW
- `income=50000, debt=30000, credit_score=700` → HIGH_RISK
- `income=50000, debt=20000, credit_score=700` → STANDARD_APPROVE

## Running All Demos

```bash
python scripts/run_demos.py
```
"""
    with open("demos/README.md", "w") as f:
        f.write(readme)
    print("  ✓ Created demos/README.md")

    print("\n✅ Demo suite generated in demos/")
    print("\nNext steps:")
    print("  1. code-scalpel scan demos/vibe_check.py")
    print("  2. code-scalpel analyze demos/enterprise/AuthController.java")
    print("  3. code-scalpel analyze demos/test_gen_scenario.py")


if __name__ == "__main__":
    create_demo_suite()
