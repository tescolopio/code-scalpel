# PDG Tools - Complete Reference

The `pdg_tools` module provides Program Dependence Graph construction and analysis for data flow tracking and program slicing.

---

## Table of Contents

1. [Overview](#overview)
2. [Concepts](#concepts)
3. [PDGBuilder](#pdgbuilder)
4. [PDGAnalyzer](#pdganalyzer)
5. [ProgramSlicer](#programslicer)
6. [Data Flow Analysis](#data-flow-analysis)
7. [Examples](#examples)

---

## Overview

Program Dependence Graphs (PDGs) capture the relationships between program statements:

- **Data Dependencies**: Variable definitions and uses
- **Control Dependencies**: Conditional execution paths
- **Call Dependencies**: Function call relationships

### Why Use PDGs?

| Use Case | PDG Advantage |
|----------|---------------|
| Taint Analysis | Track data flow from source to sink |
| Dead Code Detection | Find unreachable statements |
| Program Slicing | Extract minimal code affecting a variable |
| Impact Analysis | Determine what a change affects |

### Quick Start

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer

code = """
def process(user_input):
    validated = validate(user_input)
    result = compute(validated)
    return result
"""

pdg = build_pdg(code)
analyzer = PDGAnalyzer(pdg)

# What does 'result' depend on?
deps = analyzer.get_dependencies("result")
print(deps)  # ['validated', 'compute']
```

---

## Concepts

### Node Types

```python
from code_scalpel.pdg_tools import NodeType

class NodeType:
    ENTRY = "entry"           # Function entry point
    EXIT = "exit"             # Function exit point
    ASSIGN = "assign"         # Variable assignment
    CALL = "call"             # Function call
    RETURN = "return"         # Return statement
    BRANCH = "branch"         # if/elif/else
    LOOP = "loop"             # for/while
    EXCEPT = "except"         # Exception handler
```

### Dependency Types

```python
from code_scalpel.pdg_tools import DependencyType

class DependencyType:
    DATA = "data"             # def-use relationship
    CONTROL = "control"       # conditional execution
    CALL = "call"             # function invocation
```

### PDG Structure

```
          [ENTRY]
             |
        (control)
             v
    [x = input()]  -----(data)-----> [y = x + 1]
             |                            |
        (control)                    (control)
             v                            v
    [if x > 0]                      [return y]
        /    \
   (True)  (False)
      v       v
  [z = x]  [z = 0]
```

---

## PDGBuilder

The `PDGBuilder` class constructs PDGs from source code.

### Constructor

```python
class PDGBuilder:
    def __init__(
        self,
        include_implicit: bool = True,
        track_attributes: bool = True
    ):
        """
        Args:
            include_implicit: Include implicit data flow (e.g., closures)
            track_attributes: Track object attribute dependencies
        """
```

### Methods

#### build

```python
def build(self, code: str) -> PDG:
    """
    Build a PDG from source code.
    
    Args:
        code: Python source code
        
    Returns:
        PDG: Constructed program dependence graph
    """
```

**Example:**

```python
from code_scalpel.pdg_tools import PDGBuilder

builder = PDGBuilder()

code = """
x = 10
y = x * 2
if y > 15:
    z = y + 1
else:
    z = y - 1
print(z)
"""

pdg = builder.build(code)

# Access nodes
for node in pdg.nodes:
    print(f"Node {node.id}: {node.type} - {node.code}")

# Access edges
for edge in pdg.edges:
    print(f"{edge.source} --({edge.type})--> {edge.target}")
```

#### build_from_file

```python
def build_from_file(self, filepath: str) -> PDG:
    """Build a PDG from a source file."""
```

### Convenience Function

```python
from code_scalpel.pdg_tools import build_pdg

# Quick PDG construction
pdg = build_pdg(code)
pdg = build_pdg(open("file.py").read())
```

---

## PDGAnalyzer

The `PDGAnalyzer` class provides queries over the PDG.

### Constructor

```python
class PDGAnalyzer:
    def __init__(self, pdg: PDG):
        """
        Args:
            pdg: Program Dependence Graph to analyze
        """
```

### Dependency Analysis

#### get_dependencies

```python
def get_dependencies(
    self,
    variable: str,
    dep_type: DependencyType = None
) -> list[str]:
    """
    Get all variables that a variable depends on.
    
    Args:
        variable: Variable name to analyze
        dep_type: Filter by dependency type (optional)
        
    Returns:
        List of variable names this variable depends on
    """
```

**Example:**

```python
code = """
a = 1
b = 2
c = a + b
d = c * 2
"""

pdg = build_pdg(code)
analyzer = PDGAnalyzer(pdg)

# Direct dependencies
print(analyzer.get_dependencies("c"))  # ['a', 'b']
print(analyzer.get_dependencies("d"))  # ['c']

# Only data dependencies
print(analyzer.get_dependencies("d", DependencyType.DATA))
```

#### get_dependents

```python
def get_dependents(
    self,
    variable: str,
    dep_type: DependencyType = None
) -> list[str]:
    """
    Get all variables that depend on a variable.
    
    Args:
        variable: Variable name to analyze
        dep_type: Filter by dependency type (optional)
        
    Returns:
        List of variable names that depend on this variable
    """
```

**Example:**

```python
# What depends on 'a'?
print(analyzer.get_dependents("a"))  # ['c']

# Transitive dependents (follow the chain)
# a -> c -> d
```

#### find_data_flow_paths

```python
def find_data_flow_paths(
    self,
    source: str,
    sink: str,
    max_depth: int = 10
) -> list[list[str]]:
    """
    Find all data flow paths from source to sink.
    
    Args:
        source: Starting variable
        sink: Target variable
        max_depth: Maximum path length
        
    Returns:
        List of paths (each path is a list of variable names)
    """
```

**Example:**

```python
code = """
user_input = request.get("id")
validated = sanitize(user_input)
query = f"SELECT * FROM users WHERE id={validated}"
cursor.execute(query)
"""

analyzer = PDGAnalyzer(build_pdg(code))

# Find how user_input flows to execute
paths = analyzer.find_data_flow_paths("user_input", "query")
for path in paths:
    print(" -> ".join(path))
# Output: user_input -> validated -> query
```

### Anomaly Detection

#### detect_anomalies

```python
def detect_anomalies(self) -> list[DataFlowAnomaly]:
    """
    Detect data flow anomalies.
    
    Detects:
        - Undefined variables (use before def)
        - Unused variables (def without use)
        - Redefinition without use
        
    Returns:
        List of detected anomalies
    """
```

**Example:**

```python
code = """
x = 10
y = z  # z is undefined!
x = 20  # x redefined without use
# y is never used
"""

analyzer = PDGAnalyzer(build_pdg(code))
anomalies = analyzer.detect_anomalies()

for anomaly in anomalies:
    print(f"[{anomaly.type}] {anomaly.variable} at line {anomaly.line}")
# [undefined] z at line 2
# [unused] y at line 2
# [redef_no_use] x at line 3
```

### Security Analysis

#### find_taint_flows

```python
def find_taint_flows(
    self,
    sources: list[str],
    sinks: list[str]
) -> list[SecurityVulnerability]:
    """
    Find potential security vulnerabilities via taint analysis.
    
    Args:
        sources: Taint source patterns (e.g., ["request.get", "input"])
        sinks: Dangerous sink patterns (e.g., ["execute", "system"])
        
    Returns:
        List of potential vulnerabilities
    """
```

**Example:**

```python
code = """
user_id = request.args.get("id")
query = "SELECT * FROM users WHERE id=" + user_id
db.execute(query)
"""

analyzer = PDGAnalyzer(build_pdg(code))

vulns = analyzer.find_taint_flows(
    sources=["request.args.get", "request.form.get"],
    sinks=["execute", "system", "eval"]
)

for vuln in vulns:
    print(f"[{vuln.type}] {vuln.source} -> {vuln.sink}")
    print(f"  Path: {' -> '.join(vuln.path)}")
    print(f"  Line: {vuln.line}")
```

---

## ProgramSlicer

Extract minimal code subsets based on slicing criteria.

### Slicing Types

```python
from code_scalpel.pdg_tools import SliceType

class SliceType:
    BACKWARD = "backward"   # What affects this variable?
    FORWARD = "forward"     # What does this variable affect?
    CHOP = "chop"           # Intersection of forward and backward
```

### SlicingCriteria

```python
from code_scalpel.pdg_tools import SlicingCriteria

@dataclass
class SlicingCriteria:
    variable: str              # Variable of interest
    line: int                  # Line number
    slice_type: SliceType      # Direction of slice
    include_control: bool = True  # Include control dependencies
```

### Constructor

```python
class ProgramSlicer:
    def __init__(self, pdg: PDG):
        """
        Args:
            pdg: Program Dependence Graph
        """
```

### Methods

#### slice

```python
def slice(self, criteria: SlicingCriteria) -> SliceInfo:
    """
    Compute a program slice.
    
    Args:
        criteria: Slicing criteria
        
    Returns:
        SliceInfo with extracted lines and code
    """
```

**Example:**

```python
from code_scalpel.pdg_tools import (
    ProgramSlicer, SlicingCriteria, SliceType, build_pdg
)

code = """
def compute(a, b, c):
    x = a + b
    y = b * c
    z = x + y
    w = c * 2
    return z
"""

pdg = build_pdg(code)
slicer = ProgramSlicer(pdg)

# Backward slice: What affects z?
criteria = SlicingCriteria(
    variable="z",
    line=5,
    slice_type=SliceType.BACKWARD
)
backward = slicer.slice(criteria)

print(f"Lines: {backward.lines}")  # [1, 2, 3, 4, 5]
print(backward.code)
# def compute(a, b, c):
#     x = a + b
#     y = b * c
#     z = x + y
#     return z
# Note: w = c * 2 is excluded (doesn't affect z)
```

#### Forward Slice Example

```python
# Forward slice: What does x affect?
criteria = SlicingCriteria(
    variable="x",
    line=3,
    slice_type=SliceType.FORWARD
)
forward = slicer.slice(criteria)

print(f"Lines: {forward.lines}")  # [3, 5, 6]
print(forward.code)
# x = a + b
# z = x + y
# return z
```

### SliceInfo

```python
@dataclass
class SliceInfo:
    lines: list[int]         # Line numbers in slice
    variables: list[str]     # Variables in slice
    code: str                # Extracted source code
    criteria: SlicingCriteria  # Original criteria
```

---

## Data Flow Analysis

### Reaching Definitions

```python
def get_reaching_definitions(
    self,
    variable: str,
    line: int
) -> list[tuple[str, int]]:
    """
    Find all definitions that reach a use.
    
    Returns:
        List of (variable, line) tuples for reaching definitions
    """
```

**Example:**

```python
code = """
x = 1
if condition:
    x = 2
y = x  # Which x reaches here?
"""

analyzer = PDGAnalyzer(build_pdg(code))
defs = analyzer.get_reaching_definitions("x", line=5)
# [(x, 1), (x, 3)]  - Both definitions may reach line 5
```

### Live Variables

```python
def get_live_variables(self, line: int) -> set[str]:
    """
    Get variables that are live (may be used later) at a line.
    """
```

### Def-Use Chains

```python
def get_def_use_chains(self, variable: str) -> list[tuple[int, list[int]]]:
    """
    Get all def-use chains for a variable.
    
    Returns:
        List of (def_line, [use_lines]) tuples
    """
```

**Example:**

```python
code = """
x = 10
y = x + 1
z = x * 2
x = 20
w = x + 5
"""

chains = analyzer.get_def_use_chains("x")
# [(1, [2, 3]), (4, [5])]
# x defined at line 1, used at lines 2, 3
# x defined at line 4, used at line 5
```

---

## Examples

### Example 1: Impact Analysis

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer

def analyze_change_impact(code: str, changed_variable: str) -> dict:
    """
    Analyze what a variable change might affect.
    """
    pdg = build_pdg(code)
    analyzer = PDGAnalyzer(pdg)
    
    # Forward analysis: what does this variable affect?
    affected = analyzer.get_dependents(changed_variable)
    
    # Transitive closure
    all_affected = set(affected)
    queue = list(affected)
    while queue:
        var = queue.pop(0)
        deps = analyzer.get_dependents(var)
        for dep in deps:
            if dep not in all_affected:
                all_affected.add(dep)
                queue.append(dep)
    
    return {
        "direct_impact": affected,
        "total_impact": list(all_affected),
        "impact_count": len(all_affected),
    }

# Usage
code = """
config = load_config()
db_url = config["database_url"]
connection = connect(db_url)
cursor = connection.cursor()
results = cursor.execute("SELECT * FROM users")
"""

impact = analyze_change_impact(code, "config")
print(f"Changing 'config' affects: {impact['total_impact']}")
# ['db_url', 'connection', 'cursor', 'results']
```

### Example 2: Dead Code Detection

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer

def find_dead_code(code: str) -> list[dict]:
    """Find unused variables and unreachable code."""
    pdg = build_pdg(code)
    analyzer = PDGAnalyzer(pdg)
    
    dead_code = []
    anomalies = analyzer.detect_anomalies()
    
    for anomaly in anomalies:
        if anomaly.type == "unused":
            dead_code.append({
                "type": "unused_variable",
                "variable": anomaly.variable,
                "line": anomaly.line,
                "suggestion": f"Remove unused variable '{anomaly.variable}'"
            })
    
    return dead_code

# Usage
code = """
def process(data):
    result = transform(data)
    temp = data.copy()  # temp is never used!
    debug_info = "test"  # debug_info is never used!
    return result
"""

dead = find_dead_code(code)
for item in dead:
    print(f"Line {item['line']}: {item['suggestion']}")
```

### Example 3: Security Taint Analysis

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer

TAINT_SOURCES = [
    "request.args.get",
    "request.form.get",
    "request.json",
    "input",
    "sys.argv",
]

DANGEROUS_SINKS = [
    "execute",    # SQL
    "system",     # Command injection
    "eval",       # Code injection
    "open",       # Path traversal
    "render_template_string",  # SSTI
]

def security_audit(code: str) -> list[dict]:
    """Perform security audit using taint analysis."""
    pdg = build_pdg(code)
    analyzer = PDGAnalyzer(pdg)
    
    vulnerabilities = analyzer.find_taint_flows(
        sources=TAINT_SOURCES,
        sinks=DANGEROUS_SINKS
    )
    
    return [
        {
            "type": vuln.type,
            "severity": "HIGH",
            "source": vuln.source,
            "sink": vuln.sink,
            "path": vuln.path,
            "line": vuln.line,
            "cwe": get_cwe(vuln.type),
        }
        for vuln in vulnerabilities
    ]

def get_cwe(vuln_type: str) -> str:
    """Map vulnerability type to CWE."""
    mapping = {
        "sql_injection": "CWE-89",
        "command_injection": "CWE-78",
        "xss": "CWE-79",
        "path_traversal": "CWE-22",
    }
    return mapping.get(vuln_type, "CWE-Unknown")
```

### Example 4: Extract Minimal Test Scope

```python
from code_scalpel.pdg_tools import (
    build_pdg, ProgramSlicer, SlicingCriteria, SliceType
)

def extract_test_scope(code: str, function_name: str) -> str:
    """
    Extract the minimal code needed to test a function's output.
    """
    pdg = build_pdg(code)
    slicer = ProgramSlicer(pdg)
    
    # Find the return statement
    # Backward slice from return to get minimal code
    criteria = SlicingCriteria(
        variable="return",
        line=find_return_line(code, function_name),
        slice_type=SliceType.BACKWARD
    )
    
    slice_info = slicer.slice(criteria)
    return slice_info.code

# This extracts only the code that affects the function's output,
# removing any dead code or side-effect-only statements.
```

---

## Performance Considerations

### Large Codebases

For large codebases:

```python
# Build PDG with limited scope
builder = PDGBuilder(
    include_implicit=False,  # Skip closure analysis
    track_attributes=False   # Skip object attribute tracking
)

# Limit path exploration depth
paths = analyzer.find_data_flow_paths(source, sink, max_depth=5)
```

### Caching

PDG construction is cached when using the MCP server:

```python
# First call: builds PDG
pdg1 = build_pdg(code)

# Same code: returns cached PDG
pdg2 = build_pdg(code)
```

---

## Error Handling

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer

try:
    pdg = build_pdg(code)
except SyntaxError as e:
    print(f"Invalid Python: {e}")
except Exception as e:
    print(f"PDG construction failed: {e}")

try:
    deps = analyzer.get_dependencies("nonexistent")
except KeyError:
    print("Variable not found in PDG")
```

---

*PDG Tools - Data flow analysis for security and program understanding.*
