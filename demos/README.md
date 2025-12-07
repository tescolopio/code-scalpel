# Code Scalpel Demo Suite

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
