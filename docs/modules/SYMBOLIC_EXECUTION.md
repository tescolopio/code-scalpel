# Symbolic Execution Tools - Complete Reference

The `symbolic_execution_tools` module provides Z3-powered symbolic execution for path exploration, constraint solving, and test generation.

**Status: Beta (v0.3.0)**

---

## Table of Contents

1. [Overview](#overview)
2. [Concepts](#concepts)
3. [SymbolicAnalyzer](#symbolicanalyzer)
4. [ConstraintSolver](#constraintsolver)
5. [SymbolicExecutionEngine](#symbolicexecutionengine)
6. [Test Generation](#test-generation)
7. [Security Analysis](#security-analysis)
8. [Taint Tracking](#taint-tracking)
9. [Limitations](#limitations)
10. [Examples](#examples)

---

## Overview

Symbolic execution explores program paths by treating inputs as symbolic variables and using constraint solvers to determine path feasibility.

### Capabilities

| Feature | Status |
|---------|--------|
| Integer arithmetic | ✅ Full |
| Boolean logic | ✅ Full |
| String operations | ✅ Full |
| Path exploration | ✅ Full |
| Test generation | ✅ Full |
| Security analysis | ✅ Full |
| Float arithmetic | ❌ Planned |
| Collections | ❌ Planned |

### Quick Start

```python
from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer

analyzer = SymbolicAnalyzer()

code = """
def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
"""

result = analyzer.analyze(code)
print(f"Paths found: {result.total_paths}")  # 3

for path in result.paths:
    print(f"  Conditions: {path.conditions}")
    print(f"  Result: {path.final_state}")
```

---

## Concepts

### Symbolic Variables

Instead of concrete values, symbolic variables represent any possible value:

```python
# Concrete: x = 5
# Symbolic: x = Int("x")  # x can be any integer

# The engine explores what happens for ALL possible values of x
```

### Path Conditions

Each execution path has constraints that must be satisfied:

```python
def example(x):
    if x > 0:      # Path 1: x > 0
        if x < 10:  # Path 1a: x > 0 AND x < 10
            return "small positive"
        else:       # Path 1b: x > 0 AND x >= 10
            return "large positive"
    else:          # Path 2: x <= 0
        return "non-positive"
```

### Z3 Solver

Code Scalpel uses Microsoft's Z3 theorem prover:

```python
from z3 import Int, Solver, sat

x = Int('x')
solver = Solver()
solver.add(x > 0)
solver.add(x < 10)

if solver.check() == sat:
    model = solver.model()
    print(f"x = {model[x]}")  # e.g., x = 5
```

---

## SymbolicAnalyzer

The main entry point for symbolic analysis.

### Constructor

```python
class SymbolicAnalyzer:
    def __init__(
        self,
        max_paths: int = 100,
        timeout_seconds: int = 5,
        loop_bound: int = 10
    ):
        """
        Args:
            max_paths: Maximum execution paths to explore
            timeout_seconds: Z3 solver timeout per query
            loop_bound: Maximum loop iterations (prevents infinite loops)
        """
```

### Methods

#### analyze

```python
def analyze(
    self,
    code: str,
    function_name: str = None
) -> SymbolicResult:
    """
    Perform symbolic execution on code.
    
    Args:
        code: Python source code
        function_name: Specific function to analyze (optional)
        
    Returns:
        SymbolicResult with paths, constraints, and states
    """
```

**Example:**

```python
from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer

analyzer = SymbolicAnalyzer(max_paths=50)

code = """
def absolute_value(x):
    if x >= 0:
        return x
    else:
        return -x
"""

result = analyzer.analyze(code, function_name="absolute_value")

print(f"Total paths: {result.total_paths}")
print(f"Feasible: {result.feasible_count}")
print(f"Infeasible: {result.infeasible_count}")

for path in result.paths:
    print(f"\nPath {path.id}:")
    print(f"  Conditions: {path.conditions}")
    print(f"  Satisfiable: {path.is_feasible}")
    if path.concrete_inputs:
        print(f"  Example input: {path.concrete_inputs}")
```

#### analyze_function

```python
def analyze_function(
    self,
    code: str,
    function_name: str,
    parameter_types: dict[str, str] = None
) -> FunctionAnalysisResult:
    """
    Analyze a specific function with type hints.
    
    Args:
        code: Source code containing the function
        function_name: Name of function to analyze
        parameter_types: Override parameter types (e.g., {"x": "Int"})
    """
```

**Example:**

```python
result = analyzer.analyze_function(
    code,
    "classify",
    parameter_types={"x": "Int"}  # Force x to be integer
)
```

### SymbolicResult

```python
@dataclass
class SymbolicResult:
    total_paths: int           # Total paths explored
    feasible_count: int        # Satisfiable paths
    infeasible_count: int      # Unsatisfiable paths
    paths: list[ExecutionPath] # All discovered paths
    symbolic_variables: list[str]  # Variables treated symbolically
    timeout_paths: int         # Paths that timed out
```

### ExecutionPath

```python
@dataclass
class ExecutionPath:
    id: int                    # Unique path ID
    conditions: list[str]      # Path conditions as strings
    final_state: dict[str, Any]  # Variable values at path end
    is_feasible: bool          # Whether path is satisfiable
    concrete_inputs: dict[str, Any] | None  # Example triggering inputs
    return_value: Any | None   # Return value (if function)
```

---

## ConstraintSolver

Direct interface to Z3 for custom constraint solving.

### Constructor

```python
class ConstraintSolver:
    def __init__(self, timeout_ms: int = 5000):
        """
        Args:
            timeout_ms: Solver timeout in milliseconds
        """
```

### Methods

#### add_constraint

```python
def add_constraint(self, constraint: str | z3.BoolRef):
    """
    Add a constraint to the solver.
    
    Args:
        constraint: String expression or Z3 constraint
    """
```

**Example:**

```python
from code_scalpel.symbolic_execution_tools import ConstraintSolver

solver = ConstraintSolver()

# Add constraints as strings
solver.add_constraint("x > 0")
solver.add_constraint("x < 100")
solver.add_constraint("x % 2 == 0")  # x is even

# Or as Z3 expressions
from z3 import Int
x = Int('x')
solver.add_constraint(x > 0)
```

#### check

```python
def check(self) -> bool:
    """
    Check if constraints are satisfiable.
    
    Returns:
        True if satisfiable, False otherwise
    """
```

#### get_model

```python
def get_model(self) -> dict[str, Any]:
    """
    Get a satisfying assignment.
    
    Returns:
        Dictionary mapping variable names to values
        
    Raises:
        ValueError: If constraints are unsatisfiable
    """
```

**Example:**

```python
solver = ConstraintSolver()
solver.add_constraint("x > 10")
solver.add_constraint("x < 20")
solver.add_constraint("y == x * 2")

if solver.check():
    model = solver.get_model()
    print(f"x = {model['x']}")  # e.g., 15
    print(f"y = {model['y']}")  # e.g., 30
else:
    print("No solution")
```

#### push / pop

```python
def push(self):
    """Save current solver state."""

def pop(self):
    """Restore previous solver state."""
```

**Example:**

```python
solver.add_constraint("x > 0")
solver.push()  # Save state

solver.add_constraint("x < 0")  # Contradicts!
assert not solver.check()

solver.pop()  # Restore: only "x > 0"
assert solver.check()
```

#### get_unsat_core

```python
def get_unsat_core(self) -> list[str]:
    """
    Get the minimal set of conflicting constraints.
    
    Returns:
        List of constraint strings causing unsatisfiability
    """
```

---

## SymbolicExecutionEngine

Lower-level engine for custom symbolic execution.

### Constructor

```python
class SymbolicExecutionEngine:
    def __init__(
        self,
        timeout_seconds: int = 5,
        loop_bound: int = 10,
        enable_strings: bool = True
    ):
        """
        Args:
            timeout_seconds: Z3 timeout per query
            loop_bound: Maximum loop unrolling
            enable_strings: Enable string constraint solving
        """
```

### Methods

#### execute

```python
def execute(
    self,
    code: str,
    symbolic_vars: list[str] = None,
    initial_state: dict[str, Any] = None
) -> ExecutionResult:
    """
    Execute code symbolically.
    
    Args:
        code: Python source code
        symbolic_vars: Variables to treat as symbolic
        initial_state: Initial variable values
    """
```

**Example:**

```python
from code_scalpel.symbolic_execution_tools import SymbolicExecutionEngine

engine = SymbolicExecutionEngine()

code = """
y = x * 2
if y > 10:
    z = y + 1
else:
    z = y - 1
"""

result = engine.execute(
    code,
    symbolic_vars=["x"],
    initial_state={}
)

for path in result.paths:
    print(f"Path conditions: {path.conditions}")
    print(f"Final z: {path.state['z']}")
```

#### get_concrete_inputs

```python
def get_concrete_inputs(self, path: ExecutionPath) -> dict[str, Any]:
    """
    Get concrete input values that trigger a specific path.
    """
```

---

## Test Generation

Generate test cases from symbolic execution paths.

### TestGenerator

```python
from code_scalpel.symbolic_execution_tools import TestGenerator

generator = TestGenerator()

code = """
def is_valid_age(age):
    if age < 0:
        return False
    if age > 150:
        return False
    return True
"""

analyzer = SymbolicAnalyzer()
result = analyzer.analyze_function(code, "is_valid_age")

# Generate test cases
test_result = generator.generate(
    result,
    function_name="is_valid_age",
    output_format="pytest"
)

print(test_result.pytest_code)
```

**Output:**

```python
import pytest
from module import is_valid_age

class TestIsValidAge:
    def test_negative_age(self):
        """Path: age < 0 -> False"""
        assert is_valid_age(-1) == False
    
    def test_age_over_150(self):
        """Path: age >= 0 AND age > 150 -> False"""
        assert is_valid_age(151) == False
    
    def test_valid_age(self):
        """Path: age >= 0 AND age <= 150 -> True"""
        assert is_valid_age(25) == True
```

### TestGenerationResult

```python
@dataclass
class TestGenerationResult:
    test_cases: list[TestCase]  # Individual test cases
    pytest_code: str            # Generated pytest code
    unittest_code: str          # Generated unittest code
    coverage_estimate: float    # Estimated path coverage
```

### TestCase

```python
@dataclass
class TestCase:
    path_id: int                # Associated execution path
    function_name: str          # Function being tested
    inputs: dict[str, Any]      # Input values
    expected_output: Any        # Expected return value
    description: str            # Human-readable description
    path_conditions: list[str]  # Conditions for this path
```

---

## Security Analysis

Detect vulnerabilities using symbolic execution and taint analysis.

### SecurityAnalyzer

```python
from code_scalpel.symbolic_execution_tools import SecurityAnalyzer

analyzer = SecurityAnalyzer()

code = """
user_id = request.args.get("id")
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
"""

result = analyzer.analyze_code(code)

if result.has_vulnerabilities:
    for vuln in result.vulnerabilities:
        print(f"[{vuln.severity}] {vuln.vulnerability_type}")
        print(f"  CWE: {vuln.cwe}")
        print(f"  Line: {vuln.line}")
        print(f"  Path: {vuln.taint_path}")
```

### analyze_security (Convenience Function)

```python
from code_scalpel.symbolic_execution_tools import analyze_security

result = analyze_security(code)

print(result.summary())
# Found 1 vulnerability:
#   - SQL Injection (CWE-89) at line 3
#     Taint path: request.args.get -> user_id -> query -> execute
```

### Vulnerability

```python
@dataclass
class Vulnerability:
    vulnerability_type: str    # e.g., "SQL Injection"
    cwe: str                   # CWE identifier
    severity: str              # CRITICAL, HIGH, MEDIUM, LOW
    line: int                  # Source line number
    description: str           # Human-readable description
    taint_path: list[str]      # Data flow from source to sink
    source: str                # Taint source
    sink: str                  # Dangerous sink
```

### Detected Vulnerability Types

| Type | CWE | Sinks |
|------|-----|-------|
| SQL Injection | CWE-89 | `execute`, `executemany`, `raw` |
| Command Injection | CWE-78 | `system`, `popen`, `subprocess.run` |
| XSS | CWE-79 | `render`, `HTMLResponse`, `innerHTML` |
| Path Traversal | CWE-22 | `open`, `read_file`, `send_file` |
| Code Injection | CWE-94 | `eval`, `exec`, `compile` |
| Hardcoded Secrets | CWE-798 | AWS Keys, Stripe Keys, Private Keys |

### Secret Scanning

The `SecretScanner` detects hardcoded credentials using high-entropy analysis and pattern matching.

```python
from code_scalpel.symbolic_execution_tools import SecretScanner
import ast

scanner = SecretScanner()
tree = ast.parse("AWS_KEY = 'AKIAIOSFODNN7EXAMPLE'")
secrets = scanner.scan(tree)

for secret in secrets:
    print(f"Found {secret.secret_type} at line {secret.line_number}")
```

**Supported Patterns:**
- AWS Access Keys
- Stripe Secret Keys
- Private Keys (RSA, DSA, EC)
- Generic High-Entropy Strings

---

## Taint Tracking

Track data flow for security analysis.

### TaintTracker

```python
from code_scalpel.symbolic_execution_tools import TaintTracker, TaintLevel

tracker = TaintTracker()

# Mark input as tainted
tracker.mark_tainted("user_input", TaintLevel.HIGH, source="request.get")

# Track through operations
tracker.propagate("validated", ["user_input"])  # validated is now tainted

# Check if variable reaches a sink
if tracker.reaches_sink("validated", "execute"):
    print("Potential SQL injection!")
```

### TaintLevel

```python
class TaintLevel:
    UNTAINTED = 0   # Clean data
    LOW = 1         # Partially validated
    MEDIUM = 2      # User input with some sanitization
    HIGH = 3        # Raw user input
    CRITICAL = 4    # Highly sensitive (passwords, tokens)
```

### TaintSource

```python
class TaintSource:
    """Known sources of tainted data."""
    USER_INPUT = "user_input"
    REQUEST_ARGS = "request.args"
    REQUEST_FORM = "request.form"
    REQUEST_JSON = "request.json"
    FILE_READ = "file.read"
    ENV_VAR = "os.environ"
    DATABASE = "database"
```

### Sanitizers

Register custom sanitizers:

```python
from code_scalpel.symbolic_execution_tools import (
    register_sanitizer, SecuritySink
)

# Register a custom SQL sanitizer
register_sanitizer(
    "my_escape_sql",
    clears_sinks={SecuritySink.SQL_QUERY},
    full_clear=False  # Only clears SQL, not other sinks
)

# Type coercions are automatic sanitizers
# int(), float(), bool() fully clear taint
```

Built-in sanitizers:

| Function | Clears |
|----------|--------|
| `html.escape` | XSS |
| `shlex.quote` | Command Injection |
| `os.path.basename` | Path Traversal |
| `int()`, `float()` | All (type coercion) |
| `re.escape` | Regex Injection |

---

## Limitations

### Current Limitations (v0.3.0)

1. **Float Types**: Not supported. Use integers.
   ```python
   # Won't work
   x = 3.14
   
   # Workaround: use integers
   x = 314  # Represent as cents, etc.
   ```

2. **Loop Bound**: Loops limited to 10 iterations.
   ```python
   # This will only explore first 10 iterations
   for i in range(1000):
       ...
   ```

3. **Function Calls**: External calls are stubbed.
   ```python
   # my_function() returns a fresh symbolic variable
   result = my_function(x)
   ```

4. **Collections**: Lists/dicts not symbolically modeled.
   ```python
   # This treats the list concretely
   items = [1, 2, 3]
   ```

5. **Recursion**: Not supported.
   ```python
   # Will not work correctly
   def factorial(n):
       if n <= 1:
           return 1
       return n * factorial(n - 1)
   ```

### Z3 Timeout

Default timeout is 5 seconds per query. Adjust if needed:

```python
analyzer = SymbolicAnalyzer(timeout_seconds=10)
```

---

## Examples

### Example 1: Path Coverage Analysis

```python
from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer

def analyze_coverage(code: str, function_name: str) -> dict:
    """Analyze path coverage for a function."""
    analyzer = SymbolicAnalyzer()
    result = analyzer.analyze_function(code, function_name)
    
    return {
        "total_paths": result.total_paths,
        "feasible_paths": result.feasible_count,
        "coverage": result.feasible_count / result.total_paths * 100,
        "paths": [
            {
                "id": p.id,
                "conditions": p.conditions,
                "example_inputs": p.concrete_inputs,
            }
            for p in result.paths if p.is_feasible
        ]
    }

code = """
def grade(score):
    if score < 0 or score > 100:
        return "invalid"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
"""

coverage = analyze_coverage(code, "grade")
print(f"Path coverage: {coverage['coverage']:.1f}%")
for path in coverage['paths']:
    print(f"  Path {path['id']}: {path['example_inputs']}")
```

### Example 2: Boundary Value Testing

```python
from code_scalpel.symbolic_execution_tools import (
    SymbolicAnalyzer, TestGenerator
)

def generate_boundary_tests(code: str, function_name: str) -> str:
    """Generate tests focusing on boundary conditions."""
    analyzer = SymbolicAnalyzer()
    result = analyzer.analyze_function(code, function_name)
    
    generator = TestGenerator()
    
    # Focus on boundary conditions
    boundary_paths = [
        p for p in result.paths
        if any("==" in c or "<=" in c or ">=" in c for c in p.conditions)
    ]
    
    test_result = generator.generate_for_paths(
        boundary_paths,
        function_name=function_name
    )
    
    return test_result.pytest_code

code = """
def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False
"""

tests = generate_boundary_tests(code, "is_leap_year")
print(tests)
```

### Example 3: Security Fuzzing

```python
from code_scalpel.symbolic_execution_tools import (
    SecurityAnalyzer, SymbolicExecutionEngine
)

def security_fuzz(code: str) -> list[dict]:
    """Generate inputs that trigger security vulnerabilities."""
    analyzer = SecurityAnalyzer()
    result = analyzer.analyze_code(code)
    
    fuzz_inputs = []
    for vuln in result.vulnerabilities:
        # Use symbolic execution to find triggering inputs
        engine = SymbolicExecutionEngine()
        
        # Find path to vulnerability
        for path in engine.find_paths_to_line(code, vuln.line):
            inputs = engine.get_concrete_inputs(path)
            fuzz_inputs.append({
                "vulnerability": vuln.vulnerability_type,
                "inputs": inputs,
                "path": path.conditions,
            })
    
    return fuzz_inputs

code = """
def process_request(user_id, action):
    if action == "delete":
        query = f"DELETE FROM users WHERE id = {user_id}"
        db.execute(query)
    elif action == "update":
        # Safe: parameterized
        db.execute("UPDATE users SET active=1 WHERE id=?", [user_id])
"""

fuzzing_results = security_fuzz(code)
for result in fuzzing_results:
    print(f"Trigger {result['vulnerability']}:")
    print(f"  Inputs: {result['inputs']}")
```

### Example 4: Contract Verification

```python
from code_scalpel.symbolic_execution_tools import (
    ConstraintSolver, SymbolicAnalyzer
)

def verify_contract(
    code: str,
    function_name: str,
    precondition: str,
    postcondition: str
) -> bool:
    """
    Verify that postcondition holds when precondition is met.
    
    Returns True if contract is satisfied, False otherwise.
    """
    analyzer = SymbolicAnalyzer()
    result = analyzer.analyze_function(code, function_name)
    
    for path in result.paths:
        solver = ConstraintSolver()
        
        # Add precondition
        solver.add_constraint(precondition)
        
        # Add path conditions
        for condition in path.conditions:
            solver.add_constraint(condition)
        
        # Check if postcondition can be violated
        solver.add_constraint(f"not ({postcondition})")
        
        if solver.check():
            # Found counterexample!
            model = solver.get_model()
            print(f"Contract violation with: {model}")
            return False
    
    return True

code = """
def absolute_value(x):
    if x >= 0:
        return x
    else:
        return -x
"""

# Verify: abs(x) >= 0
is_valid = verify_contract(
    code,
    "absolute_value",
    precondition="True",  # No precondition
    postcondition="result >= 0"
)
print(f"Contract valid: {is_valid}")
```

---

## Troubleshooting

### Z3 Not Installed

```
ImportError: No module named 'z3'
```

**Solution:**
```bash
pip install z3-solver
```

### Timeout Errors

If analysis times out frequently:

```python
# Increase timeout
analyzer = SymbolicAnalyzer(timeout_seconds=30)

# Reduce path limit
analyzer = SymbolicAnalyzer(max_paths=20)

# Reduce loop bound
analyzer = SymbolicAnalyzer(loop_bound=5)
```

### Memory Issues

For large codebases:

```python
# Analyze functions individually
for func in functions:
    result = analyzer.analyze_function(code, func)
    process(result)
    # Results are garbage collected between iterations
```

---

*Symbolic Execution Tools - Mathematically rigorous code analysis powered by Z3.*
