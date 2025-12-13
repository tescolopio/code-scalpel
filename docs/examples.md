# Code Scalpel Examples

Practical examples demonstrating Code Scalpel's capabilities.

## Table of Contents

- [Basic Analysis](#basic-analysis)
- [Security Scanning](#security-scanning)
- [Surgical Extraction](#surgical-extraction)
- [Surgical Modification](#surgical-modification)
- [Cross-File Dependencies](#cross-file-dependencies)
- [Project Crawling](#project-crawling)
- [Symbolic Execution](#symbolic-execution)
- [Test Generation](#test-generation)

---

## Basic Analysis

### Analyze a Function

```python
from code_scalpel import analyze_code

code = """
def calculate_discount(price, customer_type):
    if customer_type == "premium":
        return price * 0.8
    elif customer_type == "regular":
        return price * 0.9
    else:
        return price
"""

result = analyze_code(code)

print(f"Functions: {result.metrics.num_functions}")
print(f"Complexity: {result.metrics.cyclomatic_complexity}")
print(f"Dead code: {len(result.dead_code)}")
```

### Analyze a File

```python
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()

with open("mymodule.py") as f:
    code = f.read()

result = analyzer.analyze(code)

for issue in result.security_issues:
    print(f"Security: {issue['type']} at line {issue['line']}")
```

---

## Security Scanning

### Detect SQL Injection

```python
from code_scalpel.mcp.server import security_scan

# This code has a hidden SQL injection
vulnerable_code = """
def get_user(user_id):
    query_base = "SELECT * FROM users WHERE id = "
    final_query = query_base + user_id  # Taint flows here
    cursor.execute(final_query)
    return cursor.fetchone()
"""

result = security_scan(code=vulnerable_code)

for vuln in result.vulnerabilities:
    print(f"Type: {vuln['type']}")
    print(f"CWE: {vuln['cwe']}")
    print(f"Sink: line {vuln['sink_line']}")
```

### Detect NoSQL Injection (v1.3.0+)

```python
# MongoDB injection via PyMongo
nosql_vulnerable = """
from pymongo import MongoClient

def find_user(username):
    db = MongoClient().mydb
    # User input goes directly into query - NoSQL injection!
    return db.users.find_one({"username": username})
"""

result = security_scan(code=nosql_vulnerable)

for vuln in result.vulnerabilities:
    print(f"Type: {vuln['type']}")  # "NoSQL Injection"
    print(f"CWE: {vuln['cwe']}")    # "CWE-943"
```

### Detect LDAP Injection (v1.3.0+)

```python
# LDAP injection via python-ldap
ldap_vulnerable = """
import ldap

def find_user(username):
    conn = ldap.initialize("ldap://localhost")
    # User input in LDAP filter - injection risk!
    filter_str = f"(uid={username})"
    return conn.search_s("dc=example,dc=com", ldap.SCOPE_SUBTREE, filter_str)
"""

result = security_scan(code=ldap_vulnerable)

for vuln in result.vulnerabilities:
    print(f"Type: {vuln['type']}")  # "LDAP Injection"
    print(f"CWE: {vuln['cwe']}")    # "CWE-90"
```

### Detect Hardcoded Secrets

```python
code_with_secrets = '''
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
stripe_key = "sk_live_abcdefghijklmnop"
github_token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
'''

result = security_scan(code=code_with_secrets)

for secret in result.secrets:
    print(f"Found {secret['type']} at line {secret['line']}")
    # Detects: AWS Access Key, Stripe Live Key, GitHub Token
```

---

## Surgical Extraction

### Extract a Function (Token-Efficient)

```python
from code_scalpel import SurgicalExtractor

# Instead of reading entire file:
extractor = SurgicalExtractor.from_file("src/utils.py")

# Extract just the function you need:
result = extractor.get_function("calculate_tax")

print(f"Lines: {result.line_start}-{result.line_end}")
print(f"Token estimate: {result.token_estimate}")
print(f"Dependencies: {result.dependencies}")
print(result.code)
```

### Extract a Method

```python
extractor = SurgicalExtractor.from_file("src/models.py")

# Use "ClassName.method_name" format
result = extractor.get_method("User", "validate_email")

print(result.code)
```

### Extract a Class

```python
extractor = SurgicalExtractor.from_file("src/models.py")

result = extractor.get_class("User")

print(f"Class with {len(result.dependencies)} dependencies")
print(result.code)
```

### Extract with Dependencies

```python
extractor = SurgicalExtractor.from_file("src/utils.py")

# Get the function plus everything it depends on
result = extractor.get_function_with_context("calculate_total", max_depth=2)

print(f"Target: {result.target.name}")
print(f"Context items: {result.context_items}")
print(f"Total lines: {result.total_lines}")
print(result.full_code)  # Combined for LLM
```

---

## Surgical Modification

### Update a Function

```python
from code_scalpel import SurgicalPatcher

patcher = SurgicalPatcher.from_file("src/utils.py")

# Replace the function
success = patcher.update_function(
    "calculate_tax",
    """def calculate_tax(amount: float, rate: float = 0.1) -> float:
    \"\"\"Calculate tax with configurable rate.\"\"\"
    return round(amount * rate, 2)
"""
)

if success:
    # Save with backup
    backup_path = patcher.save(create_backup=True)
    print(f"Updated! Backup at: {backup_path}")
```

### Update a Method

```python
patcher = SurgicalPatcher.from_file("src/models.py")

success = patcher.update_method(
    "User",
    "validate_email",
    """def validate_email(self, email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""
)

patcher.save()
```

### Update a Class

```python
patcher = SurgicalPatcher.from_file("src/models.py")

success = patcher.update_class(
    "User",
    """class User:
    \"\"\"Updated User model with validation.\"\"\"
    
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = self._validate_email(email)
    
    def _validate_email(self, email: str) -> str:
        if "@" not in email:
            raise ValueError("Invalid email")
        return email
"""
)

patcher.save()
```

---

## Cross-File Dependencies

### Resolve Imports Automatically

```python
from code_scalpel import SurgicalExtractor

# When calculate_tax imports TaxRate from models.py:
extractor = SurgicalExtractor.from_file("src/billing/calculator.py")

result = extractor.resolve_cross_file_dependencies(
    target_name="calculate_tax",
    target_type="function",
    max_depth=1
)

print("External symbols resolved:")
for sym in result.external_symbols:
    print(f"  - {sym.name} from {sym.source_file}")

print("\nComplete code for LLM:")
print(result.full_code)
```

### Handle Unresolved Imports

```python
result = extractor.resolve_cross_file_dependencies(
    target_name="process_order",
    target_type="function"
)

if result.unresolved_imports:
    print("Could not resolve:")
    for unresolved in result.unresolved_imports:
        print(f"  - {unresolved}")
```

---

## Project Crawling

### Discover Project Structure

```python
from code_scalpel import crawl_project

result = crawl_project(
    root_path="/path/to/project",
    include_patterns=["*.py"],
    exclude_patterns=["**/test_*", "**/__pycache__/**", "**/venv/**"],
    max_files=100
)

print(f"Found {result.total_files} files")
print(f"Total lines: {result.summary.total_lines_of_code}")
print(f"Total functions: {result.summary.total_functions}")

for file in result.files[:10]:
    print(f"  {file.path}: {file.analysis.function_count} functions")
```

### Find Files with High Complexity

```python
result = crawl_project(root_path="/project", include_analysis=True)

complex_files = [
    f for f in result.files
    if f.analysis and f.analysis.complexity > 20
]

for f in complex_files:
    print(f"{f.path}: complexity={f.analysis.complexity}")
```

---

## Symbolic Execution

### Explore All Paths

```python
from code_scalpel.mcp.server import symbolic_execute

code = """
def classify(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
"""

result = symbolic_execute(code=code, function_name="classify")

print(f"Found {len(result.paths)} paths:")
for path in result.paths:
    print(f"  Input: score={path['inputs']['score']}")
    print(f"  Output: {path['output']}")
    print(f"  Constraints: {path['constraints']}")
```

### Find Edge Cases

```python
code = """
def divide(a, b):
    if b == 0:
        return None
    return a / b
"""

result = symbolic_execute(code=code, function_name="divide")

# Z3 will find: b=0 triggers the None path
for path in result.paths:
    if path['output'] is None:
        print(f"Division by zero: a={path['inputs']['a']}, b={path['inputs']['b']}")
```

---

## Test Generation

### Generate pytest Tests

```python
from code_scalpel.mcp.server import generate_unit_tests

code = """
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age < 18:
        return "minor"
    if age < 65:
        return "adult"
    return "senior"
"""

result = generate_unit_tests(
    code=code,
    function_name="validate_age",
    framework="pytest"
)

print(result.test_code)
# Generates tests for:
# - age=-1 (ValueError)
# - age=10 ("minor")
# - age=30 ("adult")
# - age=70 ("senior")
```

### Generate with Coverage Target

```python
result = generate_unit_tests(
    code=code,
    function_name="validate_age",
    framework="pytest",
    coverage_target=100  # Cover all branches
)

print(f"Generated {result.test_count} tests")
print(f"Expected coverage: {result.coverage_estimate}%")
```

---

## MCP Server Usage

### From Claude/Copilot

Once configured, simply ask:

```
"Extract the calculate_tax function from src/billing.py and show me its dependencies"

"Update the validate_email method in the User class to use regex validation"

"Scan src/ for SQL injection vulnerabilities"

"Generate tests for the login function"
```

### Programmatic MCP Calls

```python
import asyncio
from code_scalpel.mcp.server import extract_code, update_symbol

async def refactor_function():
    # Extract
    extract_result = await extract_code(
        file_path="src/utils.py",
        target_type="function",
        target_name="old_function",
        include_cross_file_deps=True
    )
    
    if not extract_result.success:
        print(f"Error: {extract_result.error}")
        return
    
    # Modify (your LLM logic here)
    new_code = modify_with_llm(extract_result.target_code)
    
    # Update
    update_result = await update_symbol(
        file_path="src/utils.py",
        target_type="function",
        target_name="old_function",
        new_code=new_code
    )
    
    print(f"Success: {update_result.success}")
    print(f"Backup: {update_result.backup_path}")

asyncio.run(refactor_function())
```

---

## Complete Workflow Example

### Refactoring with Full Context

```python
from code_scalpel import SurgicalExtractor, SurgicalPatcher, crawl_project

# 1. Understand the project
project = crawl_project("/myproject", include_patterns=["*.py"])
print(f"Project has {project.total_files} files")

# 2. Extract function with dependencies
extractor = SurgicalExtractor.from_file("/myproject/src/calculator.py")
result = extractor.resolve_cross_file_dependencies(
    "calculate_total",
    "function",
    max_depth=2
)

print(f"Function uses: {[s.name for s in result.external_symbols]}")
print(f"Token estimate: {result.token_estimate}")

# 3. Analyze for issues
from code_scalpel import analyze_code
analysis = analyze_code(result.full_code)
print(f"Complexity: {analysis.metrics.cyclomatic_complexity}")

# 4. Make surgical modification
patcher = SurgicalPatcher.from_file("/myproject/src/calculator.py")
patcher.update_function("calculate_total", improved_code)
patcher.save(create_backup=True)

# 5. Verify
from code_scalpel.mcp.server import security_scan
scan = security_scan(file_path="/myproject/src/calculator.py")
print(f"Security issues: {len(scan.vulnerabilities)}")
```
