# Getting Started with Code Scalpel

Welcome to Code Scalpel! This guide will help you get up and running with AI-powered code analysis.

## What is Code Scalpel?

Code Scalpel is a Python toolkit for AI-driven code analysis and transformation. It provides:

- **AST Analysis** - Parse and analyze code structure (100% coverage)
- **PDG Analysis** - Build Program Dependence Graphs for data/control flow (100% coverage)
- **Program Slicing** - Extract relevant code subsets for analysis
- **Dead Code Detection** - Find unused functions, variables, and imports
- **Security Analysis** - Detect SQLi, XSS, command injection, and hardcoded secrets
- **AI Agent Integration** - Works with Autogen, CrewAI, and other frameworks
- **MCP Server** - HTTP API for AI agents to query code analysis

## Installation

### From PyPI (Recommended)

```bash
pip install code-scalpel
```

### From Source (Development)

```bash
git clone https://github.com/tescolopio/code-scalpel.git
cd code-scalpel
pip install -e .
```

### With Optional Dependencies

```bash
# For Autogen integration
pip install code-scalpel[autogen]

# For CrewAI integration
pip install code-scalpel[crewai]

# For all AI integrations
pip install code-scalpel[all]

# For development
pip install code-scalpel[dev]
```

## Quick Start

### 1. Basic Code Analysis

```python
from code_scalpel import CodeAnalyzer

# Create an analyzer
analyzer = CodeAnalyzer()

# Analyze some code
code = """
def unused_function():
    pass

def calculate_sum(a, b):
    result = a + b
    return result

total = calculate_sum(10, 20)
"""

result = analyzer.analyze(code)

# View metrics
print(f"Functions: {result.metrics.num_functions}")
print(f"Lines of code: {result.metrics.lines_of_code}")
print(f"Cyclomatic complexity: {result.metrics.cyclomatic_complexity}")

# Check for dead code
for item in result.dead_code:
    print(f"Dead code: {item.name} ({item.code_type}) - {item.reason}")

# View suggestions
for suggestion in result.refactor_suggestions:
    print(f"Suggestion: {suggestion.description}")
```

### 2. Using the CLI

```bash
# Analyze a file
code-scalpel analyze myfile.py

# Analyze code string
code-scalpel analyze --code "def hello(): return 42"

# Output as JSON
code-scalpel analyze myfile.py --json

# Start MCP server
code-scalpel server --port 8080

# Show version
code-scalpel version
```

### 3. MCP Server for AI Agents

Start the server:

```bash
code-scalpel server --port 8080
```

Query from your AI agent:

```python
import requests

# Analyze code
response = requests.post('http://localhost:8080/analyze', json={
    'code': 'def hello(): return 42'
})
print(response.json())

# Security scan
response = requests.post('http://localhost:8080/security', json={
    'code': 'eval(user_input)'  # Will detect security issue
})
print(response.json())

# Refactor code
response = requests.post('http://localhost:8080/refactor', json={
    'code': 'def f(): x=1; y=2; return x',
    'task': 'remove dead code'
})
print(response.json())
```

### 4. Autogen Integration

```python
from code_scalpel.integrations import AutogenScalpel

# Create the wrapper
scalpel = AutogenScalpel()

# Analyze code asynchronously
result = await scalpel.analyze_async(code)
print(result.analysis)
print(result.suggestions)

# Get tools for Autogen agent
tools = scalpel.get_tool_description()
```

### 5. CrewAI Integration

```python
from code_scalpel.integrations import CrewAIScalpel

# Create the wrapper
scalpel = CrewAIScalpel()

# Synchronous analysis
result = scalpel.analyze(code)

# Security analysis
security = scalpel.analyze_security(code)

# Get tools for CrewAI
tools = scalpel.get_crewai_tools()
```

### 6. PDG Analysis (Data/Control Flow)

```python
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer, ProgramSlicer

# Build a Program Dependence Graph
code = """
x = input("Enter value: ")
y = int(x)
if y > 10:
    result = y * 2
else:
    result = y + 5
print(result)
"""

pdg, call_graph = build_pdg(code)

# Analyze the PDG
analyzer = PDGAnalyzer(pdg)

# Find data flow anomalies
data_flow = analyzer.analyze_data_flow()
print(f"Anomalies: {data_flow['anomalies']}")

# Security analysis (taint tracking)
vulns = analyzer.perform_security_analysis()
for v in vulns:
    print(f"Vulnerability: {v.type}")
    print(f"  Path: {v.source} -> {v.sink}")

# Compute a program slice
from code_scalpel.pdg_tools import SlicingCriteria, SliceType

slicer = ProgramSlicer(pdg)
# Get all code that affects 'result'
result_nodes = [n for n, d in pdg.nodes(data=True) 
                if 'result' in d.get('targets', [])]
if result_nodes:
    criteria = SlicingCriteria(nodes={result_nodes[0]}, variables=set())
    slice_pdg = slicer.compute_slice(criteria, SliceType.BACKWARD)
    print(f"Slice size: {len(slice_pdg.nodes())} nodes")
```

## Analysis Levels

Code Scalpel supports three analysis levels:

```python
from code_scalpel import CodeAnalyzer, AnalysisLevel

# Basic: AST only (fastest)
analyzer = CodeAnalyzer(level=AnalysisLevel.BASIC)

# Standard: AST + PDG (default)
analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)

# Full: AST + PDG + Symbolic Execution (most thorough)
analyzer = CodeAnalyzer(level=AnalysisLevel.FULL)
```

## What Gets Detected

### Dead Code
- Unused functions
- Unused variables
- Unused imports
- Unreachable code after return/raise

### Security Issues
- `eval()` and `exec()` usage
- SQL injection patterns
- Command injection risks
- Hardcoded secrets

### Code Quality
- High cyclomatic complexity
- Long functions
- Deep nesting
- Missing docstrings

## Common Use Cases

### 1. Code Review Automation

```python
from code_scalpel import CodeAnalyzer

def review_code(filepath):
    with open(filepath) as f:
        code = f.read()
    
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(code)
    
    issues = []
    
    # Check for dead code
    if result.dead_code:
        issues.append(f"Found {len(result.dead_code)} dead code items")
    
    # Check for security issues
    if result.security_issues:
        issues.append(f"Found {len(result.security_issues)} security issues")
    
    # Check complexity
    if result.metrics.cyclomatic_complexity > 10:
        issues.append("High cyclomatic complexity")
    
    return issues
```

### 2. CI/CD Integration

```bash
#!/bin/bash
# Add to your CI pipeline

# Analyze and fail if issues found
code-scalpel analyze src/ --json > analysis.json

# Check for security issues
if grep -q '"security_issues": \[.\+\]' analysis.json; then
    echo "Security issues found!"
    exit 1
fi
```

### 3. IDE Integration

Use the MCP server with Claude Desktop or Cursor:

```json
{
  "mcpServers": {
    "code-scalpel": {
      "command": "code-scalpel",
      "args": ["server", "--port", "8080"]
    }
  }
}
```

## Next Steps

- [API Reference](api_reference.md) - Detailed API documentation
- [Examples](examples.md) - More code examples
- [Agent Integration](agent_integration.md) - AI framework guides

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/tescolopio/code-scalpel/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/tescolopio/code-scalpel/discussions)

## License

Code Scalpel is released under the MIT License.
