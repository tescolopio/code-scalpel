# Code Scalpel Documentation

**v1.0.1 - The AI Agent Toolkit for Precision Code Analysis**

---

## Quick Links

| Document | Description |
|----------|-------------|
| [**COMPREHENSIVE_GUIDE.md**](COMPREHENSIVE_GUIDE.md) | Full documentation with examples |
| [**README.md**](../README.md) | Project overview and quick start |

---

## Module Documentation

Detailed reference for each module:

| Module | File | Description |
|--------|------|-------------|
| **AST Tools** | [modules/AST_TOOLS.md](modules/AST_TOOLS.md) | Code parsing, analysis, transformation |
| **PDG Tools** | [modules/PDG_TOOLS.md](modules/PDG_TOOLS.md) | Data flow analysis, program slicing |
| **Symbolic Execution** | [modules/SYMBOLIC_EXECUTION.md](modules/SYMBOLIC_EXECUTION.md) | Path exploration, constraint solving |
| **MCP Server** | [modules/MCP_SERVER.md](modules/MCP_SERVER.md) | AI assistant integration |
| **Integrations** | [modules/INTEGRATIONS.md](modules/INTEGRATIONS.md) | AutoGen, CrewAI, LangChain |

---

## Architecture

```
code-scalpel/
├── src/code_scalpel/
│   ├── ast_tools/          # AST parsing and analysis
│   ├── pdg_tools/          # Program Dependence Graphs
│   ├── symbolic_execution_tools/  # Z3-powered symbolic execution
│   ├── mcp/                # MCP server
│   ├── integrations/       # AutoGen, CrewAI, LangChain
│   └── cli.py              # Command-line interface
├── docs/                   # Documentation
│   ├── modules/            # Module-specific docs
│   ├── guides/             # How-to guides
│   └── internal/           # Team documentation
├── demos/                  # Example code
│   └── real_world/         # Real-world vulnerability demos
└── tests/                  # Test suite (654 tests)
```

---

## Guides

| Guide | Description |
|-------|-------------|
| [guides/CONTRIBUTING.md](guides/CONTRIBUTING.md) | Developer contribution guide |

---

## Internal Documentation

Team-only documentation:

| Document | Description |
|----------|-------------|
| [internal/ROADMAP.md](internal/ROADMAP.md) | Development roadmap |
| [internal/PRODUCT_BACKLOG.md](internal/PRODUCT_BACKLOG.md) | Feature backlog |
| [internal/RELEASE_PROTOCOL.md](internal/RELEASE_PROTOCOL.md) | Release process |
| [internal/CHECKLIST.md](internal/CHECKLIST.md) | Milestone tracking |
| [internal/COVERAGE_ANALYSIS.md](internal/COVERAGE_ANALYSIS.md) | Test coverage analysis |

---

## API Reference

### Core Imports

```python
from code_scalpel import (
    # Analysis
    CodeAnalyzer,
    analyze_code,
    
    # AST
    ASTAnalyzer,
    ASTBuilder,
    build_ast,
    
    # PDG
    PDGBuilder,
    PDGAnalyzer,
    build_pdg,
    
    # Server
    run_server,
)
```

### Module-Specific Imports

```python
# Symbolic Execution
from code_scalpel.symbolic_execution_tools import (
    SymbolicAnalyzer,
    SecurityAnalyzer,
    analyze_security,
)

# MCP Server
from code_scalpel.mcp import mcp, run_server

# Integrations
from code_scalpel.integrations import (
    AutogenScalpel,
    CrewAIScalpel,
)
```

---

## Examples

### Quick Analysis

```python
from code_scalpel import analyze_code

result = analyze_code("""
def calculate(x, y):
    if x > y:
        return x - y
    return y - x
""")

print(f"Functions: {result.function_count}")
print(f"Complexity: {result.metrics.cyclomatic_complexity}")
```

### Security Scan

```python
from code_scalpel.symbolic_execution_tools import analyze_security

result = analyze_security("""
user_id = request.args.get("id")
query = f"SELECT * FROM users WHERE id={user_id}"
cursor.execute(query)
""")

if result.has_vulnerabilities:
    for vuln in result.vulnerabilities:
        print(f"[{vuln.severity}] {vuln.vulnerability_type}")
```

### MCP Server

```bash
# Start server
code-scalpel mcp

# Or with HTTP
code-scalpel mcp --transport http --port 8593
```

---

## Version History

| Version | Codename | Highlights |
|---------|----------|------------|
| v1.0.1 | - | Docker port alignment |
| v1.0.0 | "The Standard" | Caching, API freeze, Z3 hardening |
| v0.3.0 | "The Mathematician" | String support, security analysis |
| v0.2.0 | "Redemption" | Symbolic execution |

---

## Support

- **GitHub Issues**: [tescolopio/code-scalpel/issues](https://github.com/tescolopio/code-scalpel/issues)
- **PyPI**: [pypi.org/project/code-scalpel](https://pypi.org/project/code-scalpel/)

---

*Code Scalpel - Precision tools for AI-driven code analysis.*
