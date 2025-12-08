# Code Scalpel

[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://pypi.org/project/code-scalpel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1016%20passed-brightgreen.svg)](https://github.com/tescolopio/code-scalpel)

**Precision Code Analysis for the AI Era**

Code Scalpel gives AI agents the power to understand, analyze, and transform code with mathematical precision. Built for Claude, Copilot, and autonomous coding systems.

```bash
pip install code-scalpel
```

## Why Code Scalpel?

| Traditional Tools | Code Scalpel |
|------------------|--------------|
| Pattern matching (regex) | **Taint tracking** through variables |
| Single file analysis | **Cross-file** call graphs |
| Manual test writing | **Z3-powered** test generation |
| Generic output | **AI-optimized** structured responses |

## Quick Demo

### 1. Security: Find Hidden Vulnerabilities

```python
# The SQL injection is hidden through 3 variable assignments
# Regex linters miss this. Code Scalpel doesn't.

code-scalpel scan demos/vibe_check.py
# → SQL Injection (CWE-89) detected at line 38
#   Taint path: request.args → user_id → query_base → final_query
```

### 2. Analysis: Understand Complex Code

```python
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()
result = analyzer.analyze("""
def loan_approval(income, debt, credit_score):
    if credit_score < 600:
        return "REJECT"
    if income > 100000 and debt < 5000:
        return "INSTANT_APPROVE"
    return "STANDARD"
""")

print(f"Functions: {result.metrics.num_functions}")
print(f"Complexity: {result.metrics.cyclomatic_complexity}")
```

### 3. Test Generation: Cover Every Path

```bash
# Z3 solver derives exact inputs for all branches
code-scalpel analyze demos/test_gen_scenario.py

# Generates:
# - test_reject: credit_score=599
# - test_instant_approve: income=100001, debt=4999, credit_score=700
# - test_standard: income=50000, debt=20000, credit_score=700
```

## AI Agent Integration

### GitHub Copilot (VS Code)

Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "${workspaceFolder}"]
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "/path/to/project"]
    }
  }
}
```

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `analyze_code` | Parse structure, extract functions/classes/imports |
| `security_scan` | Detect SQLi, XSS, command injection via taint analysis |
| `symbolic_execute` | Explore all execution paths with Z3 |
| `generate_unit_tests` | Create pytest/unittest from symbolic paths |
| `simulate_refactor` | Verify changes are safe before applying |

## Features

### Polyglot Analysis
- **Python**: Full AST + PDG + Symbolic Execution
- **JavaScript**: Tree-sitter parsing + IR normalization
- **Java**: Enterprise-ready cross-file analysis

### Security Analysis
- SQL Injection (CWE-89)
- Cross-Site Scripting (CWE-79)
- Command Injection (CWE-78)
- Path Traversal (CWE-22)

### Performance
- **200x cache speedup** for unchanged files
- **5-second Z3 timeout** prevents hangs
- Content-addressable caching with version invalidation

## CLI Reference

```bash
# Analyze code structure
code-scalpel analyze app.py
code-scalpel analyze src/ --json

# Security scan
code-scalpel scan app.py
code-scalpel scan --code "cursor.execute(user_input)"

# Start MCP server
code-scalpel mcp                              # stdio (Claude Desktop)
code-scalpel mcp --http --port 8593           # HTTP (network)
code-scalpel mcp --root /project --allow-lan  # Team deployment
```

## Docker Deployment

```bash
# Build
docker build -t code-scalpel .

# Run MCP server
docker run -p 8593:8593 -v $(pwd):/app/code code-scalpel

# Connect at http://localhost:8593/mcp
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Agent Integration Guide](docs/agent_integration.md)
- [Demo Suite](demos/README.md)

## Contributing

```bash
git clone https://github.com/tescolopio/code-scalpel.git
cd code-scalpel
pip install -e ".[dev]"
pytest tests/
```

See [Contributing Guide](docs/guides/CONTRIBUTING.md) for details.

## Stats

- **1016** tests passing
- **100%** PDG coverage, **100%** AST coverage
- **3** languages supported
- **5** MCP tools
- **200x** cache speedup

## License

MIT License - see [LICENSE](LICENSE)

---

**Built for the AI Agent Era** | [PyPI](https://pypi.org/project/code-scalpel/) | [GitHub](https://github.com/tescolopio/code-scalpel)
