# Code Scalpel Demo Suite

Evidence-based demonstrations proving Code Scalpel's core capabilities.

## Quick Start

```bash
pip install code-scalpel==1.1.0
```

## Core Demos

### Demo 1: Security - Taint Flow Detection

**Target Audience:** Security Engineers  
**File:** `vibe_check.py`

**The Challenge:** Regex-based linters miss SQL injection when user input is 
hidden through variable assignments.

```bash
code-scalpel scan demos/vibe_check.py
```

**Expected:** Detects SQL Injection at line 38 despite 3 levels of variable 
indirection from `request.args.get('id')` to `cursor.execute()`.

### Demo 2: Enterprise - Cross-File Call Graph

**Target Audience:** Java Architects  
**Files:** `enterprise/AuthController.java`, `enterprise/AuthService.java`

**The Challenge:** Understanding dependencies in large Java codebases.

```bash
code-scalpel analyze demos/enterprise/AuthController.java --json
```

**Expected:** Call graph shows:
- `AuthController.login()` → `AuthService.validate()`
- `AuthController.login()` → `AuthController.auditLog()`

### Demo 3: QA - Automatic Test Generation

**Target Audience:** QA Leads  
**File:** `test_gen_scenario.py`

**The Challenge:** AI writes code, but how do you test every branch?

```python
from code_scalpel.mcp.server import _generate_tests_sync

with open("demos/test_gen_scenario.py") as f:
    code = f.read()

result = _generate_tests_sync(code, function_name="loan_approval")
print(result.pytest_code)
```

**Expected:** Generates pytest cases with concrete inputs for all 5 paths.

---

## Real-World Demos

### FastAPI Application

**File:** `real_world/fastapi_app.py`

A realistic REST API with 4 vulnerability types:
- SQL Injection in search endpoint
- Command Injection in export
- XSS in profile rendering
- Path Traversal in file download

```bash
code-scalpel scan demos/real_world/fastapi_app.py
```

### Django Views

**File:** `real_world/django_views.py`

Shows the difference between:
- Vulnerable raw SQL vs safe ORM
- Unsafe template rendering vs proper escaping
- Path traversal vs validated paths

```bash
code-scalpel scan demos/real_world/django_views.py
```

### React Components

**File:** `real_world/UserDashboard.jsx`

JavaScript/React security analysis:
- XSS via dangerouslySetInnerHTML
- Code injection via eval()
- Open redirect vulnerabilities

```bash
code-scalpel analyze demos/real_world/UserDashboard.jsx
```

---

## Directory Structure

```
demos/
├── README.md                          # This file
├── vibe_check.py                      # Security: Hidden SQLi
├── test_gen_scenario.py               # QA: Loan approval logic
├── enterprise/
│   ├── AuthController.java            # Java: Controller
│   └── AuthService.java               # Java: Service
└── real_world/
    ├── fastapi_app.py                 # FastAPI vulnerabilities
    ├── django_views.py                # Django safe vs unsafe
    └── UserDashboard.jsx              # React security
```

## Running All Demos

```bash
# Regenerate demo files
python scripts/generate_demos.py

# Run security scans
code-scalpel scan demos/vibe_check.py
code-scalpel scan demos/real_world/fastapi_app.py
code-scalpel scan demos/real_world/django_views.py

# Run analysis
code-scalpel analyze demos/test_gen_scenario.py --json
code-scalpel analyze demos/enterprise/AuthController.java
code-scalpel analyze demos/real_world/UserDashboard.jsx
```
