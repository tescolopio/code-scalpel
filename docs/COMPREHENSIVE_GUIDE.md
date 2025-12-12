# Code Scalpel v1.2.3 - Comprehensive Documentation

**The AI Agent Toolkit for Precision Code Analysis**

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Modules](#core-modules)
   - [AST Tools](#ast-tools)
   - [PDG Tools](#pdg-tools)
   - [Symbolic Execution](#symbolic-execution)
   - [Security Analysis](#security-analysis)
5. [MCP Server](#mcp-server)
6. [Surgical Tools](#surgical-tools)
7. [AI Agent Integrations](#ai-agent-integrations)
8. [CLI Reference](#cli-reference)
9. [Real-World Examples](#real-world-examples)
10. [API Reference](#api-reference)
11. [Performance & Caching](#performance--caching)
12. [Troubleshooting](#troubleshooting)

---

## Overview

Code Scalpel is a precision toolkit for AI-driven code analysis. Unlike general-purpose linters, Code Scalpel provides **surgical precision** for AI agents to understand, analyze, and transform code programmatically.

### Key Capabilities

| Module | Purpose | Maturity |
|--------|---------|----------|
| **AST Tools** | Parse code into Abstract Syntax Trees | Stable (100% coverage) |
| **PDG Tools** | Build Program Dependence Graphs | Stable (100% coverage) |
| **Surgical Extractor** | Token-efficient code extraction with cross-file deps | Stable (94% coverage) |
| **Surgical Patcher** | Safe, atomic code modifications | Stable (96% coverage) |
| **Symbolic Execution** | Explore execution paths with Z3 | Stable (100% coverage) |
| **Security Analysis** | Taint-based vulnerability detection | Stable (100% coverage) |
| **MCP Server** | Model Context Protocol integration (8 tools) | Stable |

### Supported Languages

- **Python**: Full support (AST, PDG, Symbolic, Security, Surgical Tools)
- **JavaScript/JSX**: Structural parsing (via tree-sitter)
- **Java**: Structural parsing (via tree-sitter)

---

## Installation

### From PyPI (Recommended)

```bash
pip install code-scalpel
```

### From Source

```bash
git clone https://github.com/tescolopio/code-scalpel.git
cd code-scalpel
pip install -e ".[dev]"
```

### With Optional Dependencies

```bash
# Full installation with all parsers
pip install code-scalpel[full]

# With Z3 for symbolic execution
pip install code-scalpel[z3]
```

---

## Quick Start

### 30-Second Demo

```python
from code_scalpel import CodeAnalyzer, analyze_code

# One-liner analysis
result = analyze_code("""
def calculate_price(quantity, unit_price, discount=0):
    subtotal = quantity * unit_price
    if discount > 0:
        subtotal = subtotal * (1 - discount)
    return subtotal
""")

print(f"Functions: {result.metrics.num_functions}")
print(f"Complexity: {result.metrics.cyclomatic_complexity}")
print(f"Issues: {len(result.issues)}")
```

### Full Analysis Example

```python
from code_scalpel import CodeAnalyzer, AnalysisLevel

analyzer = CodeAnalyzer()

# Multi-level analysis
result = analyzer.analyze(code, level=AnalysisLevel.DEEP)

# Access results
print(f"Functions found: {result.function_count}")
print(f"Classes found: {result.class_count}")
print(f"Cyclomatic complexity: {result.metrics.cyclomatic_complexity}")

# Get refactoring suggestions
for suggestion in result.refactor_suggestions:
    print(f"  - {suggestion.type}: {suggestion.message}")
```

---

## Core Modules

### AST Tools

The `ast_tools` module provides foundational code parsing and analysis.

#### Building an AST

```python
from code_scalpel.ast_tools import build_ast, build_ast_from_file, ASTAnalyzer

# From string
code = """
def greet(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
"""
tree = build_ast(code)

# From file
tree = build_ast_from_file("my_module.py")

# Disable preprocessing for raw parsing
tree = build_ast(code, preprocess=False)
```

#### Analyzing the AST

```python
from code_scalpel.ast_tools import ASTAnalyzer, ASTBuilder

builder = ASTBuilder()
tree = builder.build_ast(code)
analyzer = ASTAnalyzer(tree)

# Get function metrics
metrics = analyzer.analyze_function("greet")
print(f"Parameters: {metrics.parameter_count}")
print(f"Lines: {metrics.lines_of_code}")
print(f"Complexity: {metrics.cyclomatic_complexity}")

# Get class metrics
class_metrics = analyzer.analyze_class("Person")
print(f"Methods: {class_metrics.method_count}")
print(f"Attributes: {class_metrics.attribute_count}")
```

#### AST Visualization

```python
from code_scalpel.ast_tools import visualize_ast

# Generate PNG visualization
visualize_ast(tree, output_file="my_ast", format="png")

# Generate DOT file for custom rendering
visualize_ast(tree, output_file="my_ast", format="dot", view=False)
```

#### Key Classes

| Class | Purpose |
|-------|---------|
| `ASTBuilder` | Parse code into AST with preprocessing |
| `ASTAnalyzer` | Extract metrics and structure from AST |
| `ASTTransformer` | Modify AST nodes programmatically |
| `ASTValidator` | Validate AST structure and semantics |
| `FunctionMetrics` | Dataclass with function-level metrics |
| `ClassMetrics` | Dataclass with class-level metrics |

---

### PDG Tools

Program Dependence Graphs capture data flow and control dependencies. **100% test coverage.**

#### Building a PDG

```python
from code_scalpel.pdg_tools import build_pdg, PDGBuilder, PDGAnalyzer

code = """
def process_order(user_id, amount):
    user = get_user(user_id)  # Data dependency: user_id -> user
    if user.is_premium:        # Control dependency
        discount = 0.1
    else:
        discount = 0
    final_amount = amount * (1 - discount)
    return final_amount
"""

# Quick build - returns PDG and call graph
pdg, call_graph = build_pdg(code)

# With builder for more control
builder = PDGBuilder(
    track_constants=True,    # Track constant values
    interprocedural=True     # Cross-function analysis
)
pdg, call_graph = builder.build(code)
```

#### PDG Node and Edge Types

**Node Types:** `assign`, `if`, `for`, `while`, `call`, `return`, `function`, `class`, `try`, `except`

**Edge Types:**
- `data_dependency` - Variable def-use chains
- `control_dependency` - Conditional execution
- `loop_dependency` - Loop body membership  
- `exception_dependency` - Try/except flow
- `parameter_dependency` - Function parameters

#### Analyzing Data Flow

```python
analyzer = PDGAnalyzer(pdg)

# Comprehensive data flow analysis
data_flow = analyzer.analyze_data_flow()
print(f"Def-use chains: {data_flow['def_use_chains']}")
print(f"Anomalies: {data_flow['anomalies']}")

# Control flow analysis  
control_flow = analyzer.analyze_control_flow()
print(f"Cyclomatic complexity: {control_flow['cyclomatic_complexity']}")

# Security analysis (taint tracking)
vulns = analyzer.perform_security_analysis()
for v in vulns:
    print(f"{v.type}: {v.source} -> {v.sink}")
```

#### Program Slicing

```python
from code_scalpel.pdg_tools import ProgramSlicer, SlicingCriteria, SliceType

slicer = ProgramSlicer(pdg)

# Backward slice: "What affects this node?"
result_nodes = [n for n, d in pdg.nodes(data=True) 
                if 'final_amount' in d.get('targets', [])]
criteria = SlicingCriteria(
    nodes=set(result_nodes),
    variables=set(),
    include_control=True,
    include_data=True
)
backward_slice = slicer.compute_slice(criteria, SliceType.BACKWARD)
print(f"Nodes affecting final_amount: {len(backward_slice.nodes())}")

# Forward slice: "What does this node affect?"
forward_slice = slicer.compute_slice(criteria, SliceType.FORWARD)

# Thin slice: Direct dependencies only (no transitive)
thin_slice = slicer.compute_slice(criteria, SliceType.THIN)

# Program chop: What's between two points?
source = SlicingCriteria(nodes={"input_node"}, variables=set())
target = SlicingCriteria(nodes={"output_node"}, variables=set())
chop = slicer.compute_chop(source, target)

# Get slice metadata
info = slicer.get_slice_info(backward_slice)
print(f"Slice size: {info.size}, complexity: {info.complexity}")
```

#### Key Classes

| Class | Purpose |
|-------|--------|
| `PDGBuilder` | Construct PDG from source code with `visit_*` methods |
| `PDGAnalyzer` | Data flow, control flow, security, and optimization analysis |
| `ProgramSlicer` | Extract code slices (backward, forward, thin, chop) |
| `SlicingCriteria` | Define slice criteria (nodes, variables, options) |
| `SliceType` | BACKWARD, FORWARD, THIN, UNION, INTERSECTION |
| `NodeType` | Enum: ASSIGN, IF, WHILE, FOR, CALL, FUNCTION, etc. |
| `DependencyType` | DATA, CONTROL, CALL, PARAMETER, RETURN |

---

### Symbolic Execution

**Stable (100% coverage)** - Explore execution paths using Z3 constraint solving.

#### Basic Symbolic Analysis

```python
from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer

analyzer = SymbolicAnalyzer()

code = """
def absolute_value(x):
    if x >= 0:
        return x
    else:
        return -x
"""

result = analyzer.analyze(code)
print(f"Paths explored: {result.total_paths}")
print(f"Feasible paths: {result.feasible_count}")

for path in result.paths:
    print(f"  Path {path.id}: {path.conditions}")
    print(f"    Final state: {path.final_state}")
```

#### Symbolic Execution Engine

```python
from code_scalpel.symbolic_execution_tools import SymbolicExecutionEngine

engine = SymbolicExecutionEngine()

# Execute with symbolic inputs
result = engine.execute(code, symbolic_vars=["x"])

# Get all possible execution paths
for path in result.paths:
    print(f"Conditions: {path.conditions}")
    print(f"Output: {path.output}")
    
    # Get concrete inputs that trigger this path
    inputs = engine.get_concrete_inputs(path)
    print(f"Example input: {inputs}")
```

#### Constraint Solving

```python
from code_scalpel.symbolic_execution_tools import ConstraintSolver
from z3 import Int

solver = ConstraintSolver()

# Create symbolic variables
x = Int('x')

# Solve with constraints
result = solver.solve(
    constraints=[x > 0, x < 100, x % 2 == 0],
    variables=[x]
)

# Check result
if result.is_sat():
    print(f"Solution: x = {result.model['x']}")
elif result.status.name == 'UNKNOWN':
    print("Solver timed out or constraint undecidable")
else:
    print("No solution exists (UNSAT)")
```

#### Supported Types

| Type | Status | Notes |
|------|--------|-------|
| `Int` | ✅ Full | Arbitrary precision |
| `Bool` | ✅ Full | |
| `String` | ✅ Full | With length constraints |
| `Float` | ❌ Planned | v1.3.0 |
| `List` | ❌ Planned | v1.4.0 |

#### Limitations

- **Loop bound**: 10 iterations maximum (configurable)
- **Function calls**: Stubbed (not symbolically executed)
- **Z3 timeout**: 5 seconds per query
- **Recursion**: Not supported

---

### Security Analysis

Taint-based vulnerability detection using symbolic execution.

#### Quick Security Scan

```python
from code_scalpel.symbolic_execution_tools import analyze_security

code = """
user_id = request.args.get("id")
query = "SELECT * FROM users WHERE id=" + user_id
cursor.execute(query)
"""

result = analyze_security(code)

if result.has_vulnerabilities:
    print(f"Found {len(result.vulnerabilities)} vulnerabilities:")
    for vuln in result.vulnerabilities:
        print(f"  [{vuln.severity}] {vuln.type}")
        print(f"    Line {vuln.line}: {vuln.description}")
        print(f"    CWE: {vuln.cwe}")
```

#### Security Analyzer

```python
from code_scalpel.symbolic_execution_tools import SecurityAnalyzer

analyzer = SecurityAnalyzer()

# Analyze code
result = analyzer.analyze_code(code)

# Check taint sources
print(f"Taint sources: {result.taint_sources}")
# Output: ['request.args.get']

# Check taint flow
for flow in result.taint_flows:
    print(f"  {flow.source} -> {flow.sink}")
    print(f"    Path: {' -> '.join(flow.variables)}")
```

#### Detected Vulnerabilities

| Vulnerability | CWE | Detection Method |
|--------------|-----|------------------|
| SQL Injection | CWE-89 | Taint: user input → SQL query |
| XSS | CWE-79 | Taint: user input → HTML output |
| Command Injection | CWE-78 | Taint: user input → subprocess |
| Path Traversal | CWE-22 | Taint: user input → file path |

#### Custom Sanitizers

```python
from code_scalpel.symbolic_execution_tools import (
    register_sanitizer, SecuritySink
)

# Register a custom sanitizer
register_sanitizer(
    "my_escape_sql",
    clears_sinks={SecuritySink.SQL_QUERY},
    full_clear=False  # Only clears SQL sink, not XSS
)

# Sanitizers are recognized during analysis
code = """
user_input = request.form["name"]
safe_input = my_escape_sql(user_input)
cursor.execute("SELECT * FROM users WHERE name='" + safe_input + "'")
"""
# No SQL injection reported due to sanitizer
```

#### Built-in Sanitizers

| Function | Clears |
|----------|--------|
| `html.escape()` | XSS |
| `shlex.quote()` | Command Injection |
| `int()`, `float()` | All (type coercion) |
| `os.path.basename()` | Path Traversal |

---

## MCP Server

The Model Context Protocol server exposes Code Scalpel to AI assistants.

### Starting the Server

```bash
# stdio transport (for Claude Desktop, Cursor)
python -m code_scalpel.mcp.server

# HTTP transport (for network deployment)
python -m code_scalpel.mcp.server --transport streamable-http --port 8593

# Using CLI
code-scalpel mcp --port 8593
```

### Available Tools

| Tool | Description |
|------|-------------|
| `analyze_code` | Parse code, extract structure (functions, classes, imports) |
| `security_scan` | Detect vulnerabilities using taint analysis |
| `symbolic_execute` | Explore execution paths with Z3 |
| `generate_unit_tests` | Generate pytest/unittest from paths |
| `simulate_refactor` | Verify a refactor is safe before applying |
| `extract_code` | **NEW** Token-efficient extraction from file path |
| `update_symbol` | **NEW** Surgical modification with validation |
| `crawl_project` | **NEW** Project structure discovery |

### Using with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "code-scalpel": {
      "command": "python",
      "args": ["-m", "code_scalpel.mcp.server"]
    }
  }
}
```

### Tool Examples

#### analyze_code

```json
{
  "code": "def hello(): return 42",
  "language": "python"
}
```

Response:
```json
{
  "success": true,
  "server_version": "1.1.0",
  "functions": ["hello"],
  "classes": [],
  "imports": [],
  "function_count": 1,
  "class_count": 0,
  "complexity": 1,
  "lines_of_code": 1
}
```

#### security_scan

```json
{
  "code": "cursor.execute('SELECT * FROM users WHERE id=' + user_input)"
}
```

Response:
```json
{
  "success": true,
  "has_vulnerabilities": true,
  "vulnerability_count": 1,
  "risk_level": "HIGH",
  "vulnerabilities": [
    {
      "type": "SQL Injection",
      "cwe": "CWE-89",
      "severity": "HIGH",
      "line": 1,
      "description": "User input concatenated into SQL query"
    }
  ]
}
```

#### generate_unit_tests

```json
{
  "code": "def is_adult(age): return age >= 18",
  "function_name": "is_adult"
}
```

Response includes generated pytest code:
```python
import pytest

def test_is_adult_path_0():
    """Test case for path where age >= 18"""
    assert is_adult(18) == True

def test_is_adult_path_1():
    """Test case for path where age < 18"""
    assert is_adult(17) == False
```

---

## Surgical Tools

The Surgical Tools provide token-efficient code extraction and modification for AI agents. These tools minimize context window usage while maximizing precision.

### Token Efficiency Comparison

| Approach | Tokens to Agent | Tokens from Agent |
|----------|-----------------|-------------------|
| Traditional (full file) | ~10,000 | ~10,000 |
| Surgical (file_path) | ~50 | ~200 |
| **Savings** | **99.5%** | **98%** |

### SurgicalExtractor

Extract specific code elements without reading entire files:

```python
from code_scalpel import SurgicalExtractor

# Create extractor from file path (not code string!)
extractor = SurgicalExtractor.from_file("/path/to/module.py")

# Extract just what you need
result = extractor.get_function("process_data")
# Returns: function code + docstring + dependencies

# Extract with full context (decorators, type hints)
result = extractor.get_function_with_context("validate_user")

# Get class with all methods
result = extractor.get_class("UserService")
```

### Cross-File Dependency Resolution

Automatically resolve imports and dependencies:

```python
from code_scalpel import SurgicalExtractor

# Resolve all dependencies for a symbol
resolution = SurgicalExtractor.resolve_cross_file_dependencies(
    file_path="/path/to/service.py",
    symbol_name="UserService",
    project_root="/path/to/project"
)

# Get dependency tree
for dep in resolution.dependencies:
    print(f"{dep.file_path}::{dep.name} ({dep.symbol_type})")
    
# Get full context as combined code
full_code = resolution.full_context
```

### SurgicalPatcher

Modify code with validation and backup:

```python
from code_scalpel import SurgicalPatcher

# Create patcher for file
patcher = SurgicalPatcher("/path/to/module.py")

# Update a function (validates syntax before saving)
patcher.update_function("calculate_total", '''
def calculate_total(items: list[Item]) -> float:
    """Calculate total with tax."""
    subtotal = sum(item.price for item in items)
    return subtotal * 1.08
''')

# Update a method in a class
patcher.update_method("UserService", "authenticate", '''
def authenticate(self, username: str, password: str) -> bool:
    """Authenticate user with rate limiting."""
    if self.rate_limiter.is_blocked(username):
        raise RateLimitError()
    return self._verify_credentials(username, password)
''')

# Save changes (creates backup)
patcher.save()

# Or discard changes
patcher.discard_changes()
```

### MCP Tool: extract_code

```json
{
  "file_path": "/path/to/module.py",
  "name": "UserService",
  "extraction_type": "class",
  "include_cross_file_deps": true
}
```

Response includes:
- Extracted code
- Line numbers
- Dependencies
- Cross-file context (if enabled)

### MCP Tool: update_symbol

```json
{
  "file_path": "/path/to/module.py",
  "target_name": "process_data",
  "symbol_type": "function",
  "new_code": "def process_data(items):\n    return [transform(i) for i in items]"
}
```

Response includes:
- Success status
- Backup path
- Validation results

---

## AI Agent Integrations

### AutoGen

```python
from code_scalpel.integrations import AutogenScalpel

# Create Code Scalpel tool for AutoGen
scalpel = AutogenScalpel()

# Use in AutoGen conversation
assistant = autogen.AssistantAgent(
    name="code_analyst",
    tools=[scalpel.analyze_tool, scalpel.security_tool]
)

# Analyze code
result = scalpel.analyze("def foo(): pass")
```

### CrewAI

```python
from code_scalpel.integrations import CrewAIScalpel

scalpel = CrewAIScalpel()

# Create a CrewAI tool
analysis_tool = scalpel.create_tool("analyze")

# Use in CrewAI agent
from crewai import Agent, Task

analyst = Agent(
    role="Code Analyst",
    tools=[analysis_tool]
)

task = Task(
    description="Analyze this code for security issues",
    agent=analyst
)
```

### LangChain

```python
from langchain.tools import Tool
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()

# Create LangChain tool
code_analysis_tool = Tool(
    name="code_analyzer",
    description="Analyze Python code structure and complexity",
    func=lambda code: analyzer.analyze(code).model_dump()
)

# Use with LangChain agent
from langchain.agents import initialize_agent
agent = initialize_agent([code_analysis_tool], llm)
```

---

## CLI Reference

### analyze

Analyze code structure and metrics.

```bash
# Analyze a file
code-scalpel analyze path/to/file.py

# Analyze with JSON output
code-scalpel analyze file.py --format json

# Analyze with specific level
code-scalpel analyze file.py --level deep
```

### scan

Run security analysis.

```bash
# Scan for vulnerabilities
code-scalpel scan path/to/file.py

# Scan entire directory
code-scalpel scan src/ --recursive

# Output as JSON
code-scalpel scan file.py --format json
```

### server

Start the REST API server.

```bash
# Start on default port (8593)
code-scalpel server

# Custom port
code-scalpel server --port 9000

# With host binding
code-scalpel server --host 0.0.0.0 --port 8593
```

### mcp

Start the MCP server.

```bash
# stdio transport (default)
code-scalpel mcp

# HTTP transport
code-scalpel mcp --transport http --port 8593
```

### version

Show version information.

```bash
code-scalpel version
# Code Scalpel v1.1.0
```

---

## Real-World Examples

### FastAPI Security Scan

```python
# demos/real_world/fastapi_app.py
from fastapi import FastAPI, Query
import sqlite3

app = FastAPI()

@app.get("/api/users/search")
async def search_users(q: str = Query(...)):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # VULNERABLE: SQL Injection
    query = f"SELECT * FROM users WHERE name LIKE '%{q}%'"
    cursor.execute(query)
    return cursor.fetchall()
```

Scan result:
```
[HIGH] SQL Injection (CWE-89)
  Line 12: User input from query parameter concatenated into SQL
  Taint path: q -> query -> cursor.execute()
```

### Django Safe vs Unsafe Patterns

```python
# UNSAFE: Raw SQL with string formatting
def search_users_vulnerable(request):
    user_input = request.GET.get("q")
    cursor.execute(f"SELECT * FROM users WHERE name='{user_input}'")

# SAFE: ORM with parameterized queries
def search_users_safe(request):
    user_input = request.GET.get("q")
    return User.objects.filter(name=user_input)
```

### Test Generation from Symbolic Execution

```python
from code_scalpel.symbolic_execution_tools import (
    SymbolicAnalyzer, TestGenerator
)

code = """
def classify_age(age):
    if age < 0:
        return "invalid"
    elif age < 13:
        return "child"
    elif age < 20:
        return "teenager"
    else:
        return "adult"
"""

analyzer = SymbolicAnalyzer()
result = analyzer.analyze(code)

generator = TestGenerator()
tests = generator.generate(result, function_name="classify_age")

print(tests.pytest_code)
```

Output:
```python
import pytest
from module import classify_age

def test_classify_age_invalid():
    """Path: age < 0"""
    assert classify_age(-1) == "invalid"

def test_classify_age_child():
    """Path: 0 <= age < 13"""
    assert classify_age(5) == "child"

def test_classify_age_teenager():
    """Path: 13 <= age < 20"""
    assert classify_age(15) == "teenager"

def test_classify_age_adult():
    """Path: age >= 20"""
    assert classify_age(25) == "adult"
```

---

## Performance & Caching

### Content-Addressable Caching

Code Scalpel v1.0.0+ includes intelligent caching:

```python
# Cache key = SHA256(code_content + tool_version + config)
# Same code + same version = cache hit
```

Benchmark results:
- **First analysis**: ~50ms
- **Cached analysis**: ~0.25ms
- **Speedup**: ~200x

### Cache Configuration

```bash
# Disable caching
export SCALPEL_CACHE_ENABLED=0

# Set cache directory
export SCALPEL_CACHE_DIR=/path/to/cache
```

### Cache API

```python
from code_scalpel.utilities.cache import get_cache, clear_cache

cache = get_cache()

# Check cache stats
print(f"Cache hits: {cache.hits}")
print(f"Cache misses: {cache.misses}")
print(f"Hit rate: {cache.hit_rate:.1%}")

# Clear cache
clear_cache()
```

---

## Troubleshooting

### Common Issues

#### Z3 Not Found

```
ImportError: No module named 'z3'
```

**Solution:**
```bash
pip install z3-solver
```

#### Java Parsing Fails

```
ImportError: tree-sitter-java not found
```

**Solution:**
```bash
pip install tree-sitter tree-sitter-java
```

#### Preprocessing Errors

If you see syntax errors with valid code:

```python
# Try disabling preprocessing
tree = build_ast_from_file("file.py", preprocess=False)
```

#### MCP Server Connection Issues

1. Check port is not in use: `lsof -i :8593`
2. Verify MCP SDK is installed: `pip install mcp`
3. Check server logs: `code-scalpel mcp --verbose`

### Getting Help

- **GitHub Issues**: https://github.com/tescolopio/code-scalpel/issues
- **Documentation**: https://github.com/tescolopio/code-scalpel/docs

---

## API Reference

### code_scalpel

Main package exports:

```python
from code_scalpel import (
    # Core
    CodeAnalyzer,
    AnalysisResult,
    AnalysisLevel,
    analyze_code,
    
    # AST
    ASTAnalyzer,
    ASTBuilder,
    build_ast,
    build_ast_from_file,
    
    # PDG
    PDGBuilder,
    PDGAnalyzer,
    build_pdg,
    
    # Server
    create_app,
    run_server,
)
```

### code_scalpel.ast_tools

```python
from code_scalpel.ast_tools import (
    ASTBuilder,
    ASTAnalyzer,
    ASTTransformer,
    ASTVisualizer,
    ASTValidator,
    FunctionMetrics,
    ClassMetrics,
    build_ast,
    build_ast_from_file,
    visualize_ast,
)
```

### code_scalpel.pdg_tools

```python
from code_scalpel.pdg_tools import (
    PDGBuilder,
    PDGAnalyzer,
    ProgramSlicer,
    SlicingCriteria,
    SliceType,
    SliceInfo,
    NodeType,
    DependencyType,
    DataFlowAnomaly,
    SecurityVulnerability,
    build_pdg,
)
```

### code_scalpel.symbolic_execution_tools

```python
from code_scalpel.symbolic_execution_tools import (
    # Core
    SymbolicAnalyzer,
    SymbolicExecutionEngine,
    ConstraintSolver,
    
    # Security
    SecurityAnalyzer,
    TaintTracker,
    TaintLevel,
    TaintSource,
    SecuritySink,
    Vulnerability,
    
    # Utilities
    analyze_security,
    register_sanitizer,
)
```

### code_scalpel.mcp

```python
from code_scalpel.mcp import (
    mcp,        # FastMCP server instance
    run_server, # Start the MCP server
)
```

### code_scalpel.integrations

```python
from code_scalpel.integrations import (
    AutogenScalpel,
    CrewAIScalpel,
    create_app,
    run_rest_server,
)
```

---

## What's New in v1.2.3

**December 12, 2025 - Maintenance Release**

- Repository cleanup (removed 22 obsolete files)
- Development roadmap published ([DEVELOPMENT_ROADMAP.md](../DEVELOPMENT_ROADMAP.md))
- Test suite stability improvements
- Documentation updates

All critical bug fixes from v1.2.2 retained (100% vulnerability detection).

**Next Release:** v1.3.0 "Hardening" - January 2025

---

*Code Scalpel v1.2.3 - Built with surgical precision for AI-driven code analysis.*

*"Code Scalpel" is a trademark of 3D Tech Solutions LLC.*
