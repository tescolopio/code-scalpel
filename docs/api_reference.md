# Code Scalpel API Reference

This document provides detailed API documentation for Code Scalpel.

## Table of Contents

- [Core Classes](#core-classes)
  - [CodeAnalyzer](#codeanalyzer)
  - [AnalysisResult](#analysisresult)
  - [AnalysisMetrics](#analysismetrics)
- [AST Tools](#ast-tools)
  - [ASTAnalyzer](#astanalyzer)
  - [ASTBuilder](#astbuilder)
- [PDG Tools](#pdg-tools)
  - [PDGBuilder](#pdgbuilder)
  - [PDGAnalyzer](#pdganalyzer)
- [Security Tools](#security-tools)
  - [SecurityAnalyzer](#securityanalyzer)
  - [SecretScanner](#secretscanner)
- [Integrations](#integrations)
  - [AutogenScalpel](#autogenscalpel)
  - [CrewAIScalpel](#crewaigrscalpel)
  - [MCP Server](#mcp-server)
- [CLI](#cli)

---

## Core Classes

### CodeAnalyzer

The main entry point for code analysis.

```python
from code_scalpel import CodeAnalyzer, AnalysisLevel

analyzer = CodeAnalyzer(
    level=AnalysisLevel.STANDARD,  # BASIC, STANDARD, or FULL
    cache_enabled=True,            # Cache analysis results
    max_symbolic_depth=50,         # Max depth for symbolic execution
    max_loop_iterations=10         # Max loop iterations in symbolic execution
)
```

#### Methods

##### `analyze(code: str, level: Optional[AnalysisLevel] = None) -> AnalysisResult`

Perform comprehensive code analysis.

```python
result = analyzer.analyze("""
def hello(name):
    return f"Hello, {name}!"
""")

print(result.metrics.num_functions)  # 1
print(result.dead_code)              # []
print(result.errors)                 # []
```

**Parameters:**
- `code` (str): Python source code to analyze
- `level` (AnalysisLevel, optional): Override the default analysis level

**Returns:** `AnalysisResult` object containing all analysis data

##### `apply_refactor(code: str, refactor_type: str, **kwargs) -> str`

Apply a refactoring operation to code.

```python
# Remove dead code
new_code = analyzer.apply_refactor(code, 'remove_dead_code')

# Rename a variable
new_code = analyzer.apply_refactor(code, 'rename_variable', 
                                    old_name='x', new_name='count')

# Remove unused imports
new_code = analyzer.apply_refactor(code, 'remove_unused_imports')

# Inline a constant
new_code = analyzer.apply_refactor(code, 'inline_constant',
                                    constant_name='MAX_SIZE')
```

**Supported refactor types:**
- `remove_dead_code` - Remove all detected dead code
- `rename_variable` - Rename a variable (requires `old_name`, `new_name`)
- `remove_unused_imports` - Remove unused import statements
- `inline_constant` - Inline a constant value (requires `constant_name`)

##### `get_dead_code_summary(code: str) -> Dict[str, List]`

Get a categorized summary of dead code.

```python
summary = analyzer.get_dead_code_summary(code)
print(summary['functions'])   # List of unused functions
print(summary['variables'])   # List of unused variables
print(summary['imports'])     # List of unused imports
```

---

### AnalysisResult

Contains all results from code analysis.

```python
@dataclass
class AnalysisResult:
    code: str                                    # Original code
    ast_tree: Optional[ast.AST]                  # Parsed AST
    pdg: Optional[nx.DiGraph]                    # Program Dependence Graph
    call_graph: Optional[nx.DiGraph]             # Function call graph
    dead_code: List[DeadCodeItem]                # Detected dead code
    metrics: AnalysisMetrics                     # Code metrics
    security_issues: List[Dict[str, Any]]        # Security vulnerabilities
    refactor_suggestions: List[RefactorSuggestion]  # Improvement suggestions
    errors: List[str]                            # Any errors during analysis
    symbolic_paths: List[Dict[str, Any]]         # Symbolic execution paths
```

---

### AnalysisMetrics

Code metrics from analysis.

```python
@dataclass
class AnalysisMetrics:
    lines_of_code: int = 0
    num_functions: int = 0
    num_classes: int = 0
    num_variables: int = 0
    cyclomatic_complexity: int = 0
    analysis_time_seconds: float = 0.0
```

---

### DeadCodeItem

Represents detected dead code.

```python
@dataclass
class DeadCodeItem:
    name: str           # Name of the dead code element
    code_type: str      # 'function', 'variable', 'class', 'import', 'statement'
    line_start: int     # Starting line number
    line_end: int       # Ending line number
    reason: str         # Why it's considered dead
    confidence: float   # 0.0 to 1.0 confidence score
```

---

### AnalysisLevel

Enum for analysis depth.

```python
from code_scalpel import AnalysisLevel

class AnalysisLevel(Enum):
    BASIC = 'basic'        # AST only (fastest)
    STANDARD = 'standard'  # AST + PDG (default)
    FULL = 'full'          # AST + PDG + Symbolic Execution
```

---

## AST Tools

### ASTAnalyzer

Low-level AST analysis tools.

```python
from code_scalpel.ast_tools import ASTAnalyzer

analyzer = ASTAnalyzer(cache_enabled=True)
```

#### Methods

##### `parse_to_ast(code: str) -> ast.AST`

Parse code into an AST.

```python
tree = analyzer.parse_to_ast("def hello(): pass")
```

##### `ast_to_code(node: ast.AST) -> str`

Convert an AST back to source code.

```python
code = analyzer.ast_to_code(tree)
```

##### `analyze_function(node: ast.FunctionDef) -> FunctionMetrics`

Analyze a function node.

```python
metrics = analyzer.analyze_function(func_node)
print(metrics.name)
print(metrics.num_args)
print(metrics.complexity)
```

##### `analyze_class(node: ast.ClassDef) -> ClassMetrics`

Analyze a class node.

```python
metrics = analyzer.analyze_class(class_node)
print(metrics.name)
print(metrics.num_methods)
print(metrics.bases)
```

##### `find_security_issues(tree: ast.AST) -> List[Dict]`

Find potential security issues.

```python
issues = analyzer.find_security_issues(tree)
for issue in issues:
    print(f"{issue['type']}: {issue['description']}")
```

---

### ASTBuilder

Build and manipulate ASTs.

```python
from code_scalpel.ast_tools import ASTBuilder, build_ast

# Convenience function
tree = build_ast("def hello(): pass")

# Or use the class
builder = ASTBuilder()
tree = builder.build_ast(code, preprocess=True, validate=True)
```

---

## PDG Tools

### PDGBuilder

Build Program Dependence Graphs with full data and control dependency tracking.

```python
from code_scalpel.pdg_tools import PDGBuilder, build_pdg

# Convenience function
pdg, call_graph = build_pdg("x = 1; y = x + 1")

# Or use the class with options
builder = PDGBuilder(
    track_constants=True,    # Track constant values
    interprocedural=True     # Cross-function analysis
)
pdg, call_graph = builder.build(code)
```

The PDG is a NetworkX DiGraph where:
- **Nodes** represent statements (assign, if, for, while, call, function, class, try, except, return)
- **Edges** represent dependencies:
  - `data_dependency` - Variable def-use chains
  - `control_dependency` - Conditional execution
  - `loop_dependency` - Loop body membership
  - `exception_dependency` - Try/except flow
  - `call_dependency` - Function calls
  - `parameter_dependency` - Function parameters

#### Supported Constructs

| Construct | Handler | Dependencies |
|-----------|---------|-------------|
| `x = expr` | `visit_Assign` | Data deps from RHS variables |
| `x += expr` | `visit_AugAssign` | Data deps including target |
| `if cond:` | `visit_If` | Control deps to body/else |
| `for x in iter:` | `visit_For` | Loop + control deps |
| `while cond:` | `visit_While` | Loop + control deps |
| `try/except` | `visit_Try` | Exception deps |
| `def func():` | `visit_FunctionDef` | Parameter deps |
| `class Cls:` | `visit_ClassDef` | Control deps to body |
| `func()` | `visit_Call` | Call graph edges |

---

### PDGAnalyzer

Advanced analysis of Program Dependence Graphs including data flow, control flow, security analysis, and optimization opportunities.

```python
from code_scalpel.pdg_tools import PDGAnalyzer

analyzer = PDGAnalyzer(pdg)  # Pass PDG to constructor
```

#### Methods

##### `analyze_data_flow() -> Dict`

Comprehensive data flow analysis with caching.

```python
flow = analyzer.analyze_data_flow()
print(flow['anomalies'])           # Undefined/unused variables
print(flow['def_use_chains'])      # Definition-use chains
print(flow['live_variables'])      # Live variable analysis
print(flow['reaching_definitions']) # Reaching definitions
print(flow['value_ranges'])        # Variable value ranges
```

##### `analyze_control_flow() -> Dict`

Control flow analysis.

```python
flow = analyzer.analyze_control_flow()
print(flow['cyclomatic_complexity']) # McCabe complexity
print(flow['control_dependencies'])  # Control dep mapping
print(flow['unreachable_code'])      # Dead code detection
print(flow['loop_info'])             # Loop analysis
print(flow['dominators'])            # Dominator tree
```

##### `perform_security_analysis() -> List[SecurityVulnerability]`

Taint analysis and vulnerability detection.

```python
vulns = analyzer.perform_security_analysis()
for v in vulns:
    print(f"{v.type}: {v.source} -> {v.sink}")
    print(f"  Path: {v.path}")
    print(f"  Severity: {v.severity}")
```

**Detected Sources:** `input()`, `request.get`, `request.form`, `file.read`  
**Detected Sinks:** `execute()`, `eval()`, `subprocess.run`, `render_template`  
**Sanitizers:** `escape()`, `html.escape`, `bleach.clean`, `parameterize`

##### `find_optimization_opportunities() -> Dict`

Identify code optimization opportunities.

```python
opts = analyzer.find_optimization_opportunities()
print(opts['loop_invariant'])        # Code to hoist from loops
print(opts['common_subexpressions']) # CSE candidates
print(opts['dead_code'])             # Unreachable code
print(opts['redundant_computations']) # Duplicate computations
```

##### `compute_program_slice(criterion: str, direction: str) -> nx.DiGraph`

Compute a program slice.

```python
# Backward slice - what affects this node?
backward = analyzer.compute_program_slice('result_node', 'backward')

# Forward slice - what does this node affect?
forward = analyzer.compute_program_slice('input_node', 'forward')
```

---

### ProgramSlicer

Advanced program slicing with multiple strategies and caching.

```python
from code_scalpel.pdg_tools import ProgramSlicer, SlicingCriteria, SliceType

slicer = ProgramSlicer(pdg)
```

#### Slice Types

| Type | Description |
|------|-------------|
| `BACKWARD` | All nodes that affect the criterion |
| `FORWARD` | All nodes affected by the criterion |
| `THIN` | Direct dependencies only (no transitive) |
| `UNION` | Union of multiple slices |
| `INTERSECTION` | Intersection of multiple slices |
| `CONTROL` | Control dependencies only |
| `DATA` | Data dependencies only |

#### Methods

##### `compute_slice(criteria, slice_type) -> nx.DiGraph`

Compute a program slice.

```python
# Simple string criterion
slice_pdg = slicer.compute_slice("node_id", SliceType.BACKWARD)

# Detailed criteria
criteria = SlicingCriteria(
    nodes={"node1", "node2"},
    variables={"x", "y"},
    include_control=True,
    include_data=True
)
slice_pdg = slicer.compute_slice(criteria, SliceType.BACKWARD)
```

##### `compute_chop(source, target) -> nx.DiGraph`

Compute a program chop (intersection of forward and backward slices).

```python
source = SlicingCriteria(nodes={"input_node"}, variables=set())
target = SlicingCriteria(nodes={"output_node"}, variables=set())

# Get all nodes on paths from source to target
chop = slicer.compute_chop(source, target)
```

##### `decompose_slice(criteria) -> Dict[str, nx.DiGraph]`

Decompose a slice into meaningful components.

```python
components = slicer.decompose_slice(criteria)
print(components['data_only'])    # Data dependency subgraph
print(components['control_only']) # Control dependency subgraph
print(components['core'])         # Essential nodes
print(components['auxiliary'])    # Supporting nodes
```

##### `get_slice_info(sliced_pdg) -> SliceInfo`

Get metadata about a slice.

```python
info = slicer.get_slice_info(slice_pdg)
print(info.nodes)       # Set of node IDs
print(info.edges)       # Set of edge tuples
print(info.variables)   # Variables in slice
print(info.line_range)  # (start_line, end_line)
print(info.size)        # Number of nodes
print(info.complexity)  # Slice complexity score
```

##### `perform_security_analysis(pdg: nx.DiGraph, code: str) -> List[Dict]`

Perform security analysis using the PDG.

```python
vulnerabilities = analyzer.perform_security_analysis(pdg, code)
```

---

## Security Tools

### SecurityAnalyzer

Advanced security analysis using taint tracking and secret scanning.

```python
from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer

analyzer = SecurityAnalyzer()
```

#### Methods

##### `analyze(code: str) -> Dict`

Perform comprehensive security analysis.

```python
report = analyzer.analyze("""
import os
AWS_KEY = "AKIA..."
os.system(user_input)
""")

print(report['vulnerabilities'])
# [
#   {'type': 'HARDCODED_SECRET', 'severity': 'CRITICAL', ...},
#   {'type': 'COMMAND_INJECTION', 'severity': 'CRITICAL', ...}
# ]
```

### SecretScanner

Specialized scanner for hardcoded secrets.

```python
from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
import ast

scanner = SecretScanner()
tree = ast.parse("API_KEY = '12345'")
secrets = scanner.scan(tree)
```

---

## Integrations

### AutogenScalpel

Integration with Microsoft Autogen.

```python
from code_scalpel.integrations import AutogenScalpel

scalpel = AutogenScalpel(cache_enabled=True)
```

#### Methods

##### `async analyze_async(code: str) -> AnalysisResult`

Asynchronous code analysis.

```python
result = await scalpel.analyze_async(code)
print(result.success)
print(result.analysis)
print(result.issues)
print(result.suggestions)
```

##### `async refactor_async(code: str, task: str) -> Dict`

Asynchronous refactoring.

```python
result = await scalpel.refactor_async(code, "remove dead code")
print(result['refactored_code'])
```

##### `get_tool_description() -> Dict`

Get tool description for Autogen agents.

```python
tools = scalpel.get_tool_description()
```

---

### CrewAIScalpel

Integration with CrewAI.

```python
from code_scalpel.integrations import CrewAIScalpel

scalpel = CrewAIScalpel(cache_enabled=True)
```

#### Methods

##### `analyze(code: str) -> AnalysisResult`

Synchronous code analysis.

```python
result = scalpel.analyze(code)
```

##### `analyze_security(code: str) -> Dict`

Security-focused analysis.

```python
security = scalpel.analyze_security(code)
print(security['risk_level'])  # 'low', 'medium', 'high', 'critical'
print(security['issues'])
print(security['recommendations'])
```

##### `refactor(code: str, task: str) -> RefactorResult`

Refactor code based on task.

```python
result = scalpel.refactor(code, "improve performance")
print(result.refactored_code)
print(result.suggestions)
```

##### `get_crewai_tools() -> List[Dict]`

Get tools for CrewAI agents.

```python
tools = scalpel.get_crewai_tools()
```

---

### MCP Server

HTTP API server for AI agents.

```python
from code_scalpel.integrations import create_app, run_server, MCPServerConfig

# Create Flask app
config = MCPServerConfig(
    host="0.0.0.0",
    port=8080,
    debug=False,
    cache_enabled=True,
    max_code_size=100000
)
app = create_app(config)

# Or run directly
run_server(host="0.0.0.0", port=8080)
```

#### Endpoints

##### `GET /health`

Health check endpoint.

```json
{
  "status": "healthy",
  "service": "code-scalpel-mcp",
  "version": "0.1.0"
}
```

##### `POST /analyze`

Analyze code.

**Request:**
```json
{
  "code": "def hello(): return 42"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {...},
  "issues": [...],
  "suggestions": [...],
  "processing_time_ms": 15.2
}
```

##### `POST /refactor`

Refactor code.

**Request:**
```json
{
  "code": "def f(): x = 1; return 0",
  "task": "remove dead code"
}
```

**Response:**
```json
{
  "success": true,
  "original_code": "...",
  "refactored_code": "...",
  "analysis": {...},
  "suggestions": [...],
  "processing_time_ms": 25.5
}
```

##### `POST /security`

Security scan.

**Request:**
```json
{
  "code": "eval(user_input)"
}
```

**Response:**
```json
{
  "success": true,
  "issues": [
    {
      "type": "eval_usage",
      "description": "Use of eval() is dangerous",
      "line": 1
    }
  ],
  "risk_level": "high",
  "recommendations": [...]
}
```

---

## CLI

Command-line interface.

```bash
# Show help
code-scalpel --help

# Analyze a file
code-scalpel analyze myfile.py

# Analyze code string
code-scalpel analyze --code "def f(): pass"

# Output as JSON
code-scalpel analyze myfile.py --json

# Start MCP server
code-scalpel server --host 0.0.0.0 --port 8080

# Show version
code-scalpel version
```

---

## Convenience Functions

### `analyze_code(code: str) -> AnalysisResult`

Quick analysis without creating an analyzer.

```python
from code_scalpel import analyze_code

result = analyze_code("def hello(): pass")
```

### `build_ast(code: str) -> ast.AST`

Quick AST building.

```python
from code_scalpel import build_ast

tree = build_ast("x = 1 + 2")
```

### `build_pdg(code: str) -> Tuple[nx.DiGraph, nx.DiGraph]`

Quick PDG building.

```python
from code_scalpel import build_pdg

pdg, call_graph = build_pdg("x = 1; y = x + 1")
```

---

## Error Handling

All analysis methods handle errors gracefully:

```python
result = analyzer.analyze("def broken(")  # Syntax error

if result.errors:
    print(f"Analysis failed: {result.errors}")
else:
    print(f"Analysis succeeded: {result.metrics}")
```

For exceptions:

```python
try:
    result = analyzer.analyze(code)
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Type Hints

All public APIs include full type hints:

```python
from code_scalpel import CodeAnalyzer, AnalysisResult, AnalysisLevel
from typing import Optional, List, Dict, Any

def my_analysis(code: str) -> AnalysisResult:
    analyzer: CodeAnalyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    return analyzer.analyze(code)
```
