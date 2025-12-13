# Code Scalpel

[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://pypi.org/project/code-scalpel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1669%20passed-brightgreen.svg)](https://github.com/tescolopio/code-scalpel)

**MCP Server Toolkit for AI Agents**

Code Scalpel enables AI assistants (Claude, GitHub Copilot, Cursor) to perform surgical code operations without hallucination. Extract exactly what's needed, modify without collateral damage, verify before applying.

```bash
pip install code-scalpel
```

> **v1.3.0 STABLE RELEASE** (December 12, 2025)  
> Production-ready MCP server for AI agents with **100% vulnerability detection rate**.
>
> | Capability | Status | Notes |
> |------------|--------|-------|
> | Security Scanning | **100%** (15+ types) | SQL, XSS, NoSQL, LDAP injection, secrets |
> | Secret Detection | **30+ patterns** | AWS, GitHub, Stripe, private keys |
> | Test Generation | **Float support** | Int, str, bool, float properly inferred |
> | Surgical Extraction | **95%+ coverage** | Extract by symbol name, not line guessing |
>
> **What's New in v1.3.0:**
> - NoSQL injection detection (MongoDB find, aggregate, update, delete)
> - LDAP injection detection (python-ldap, ldap3)
> - Hardcoded secret scanning (30+ patterns including AWS, GitHub, Stripe)
> - Float type inference for symbolic execution
> - Path resolution improvements for extract_code
>
> See [RELEASE_NOTES_v1.3.0.md](docs/RELEASE_NOTES_v1.3.0.md) for technical details.

---

## The Revolution: Code as Graph, Not Text

Most AI coding tools treat your codebase like a book—they "read" as much as possible to understand context. This hits a hard ceiling: the **Context Window**.

**Code Scalpel changes the game.** It stops treating code as "text" and starts treating it as a **graph**—a deterministic pre-processor for probabilistic models.

### Breaking the Context Window Tyranny

| The Old Way (RAG/Chat) | The Code Scalpel Way |
|------------------------|----------------------|
| "Here are all 50 files. Good luck." | "Here's the variable definition, 3 callers, and 1 test. Nothing else." |
| Retrieve similar text chunks (fuzzy) | Trace variable dependencies (precise) |
| Context limit is a hard wall | Context limit is irrelevant—we slice to fit |
| "I think this fixes it" | "I have mathematically verified this path" |

### Why This Matters

**1. Operate on Million-Line Codebases with 4K Token Models**

Instead of stuffing files into context, Code Scalpel's **Program Dependence Graph (PDG)** surgically extracts *only* the code that matters:

```
User: "Refactor the calculate_tax function"
Old Way: Send 10 files (15,000 tokens) → Model confused
Scalpel: Send function + 3 dependencies (200 tokens) → Precise fix
```

**2. Turn "Dumb" Local LLMs into Geniuses**

Local models (Llama, Mistral) are fast and private but struggle with complex reasoning. Code Scalpel offloads the thinking:

- **Before:** "Does path A allow null?" → Model guesses
- **After:** Symbolic Engine proves it → Model receives fact: "Path A impossible. Path B crashes."

A 7B model + Code Scalpel outperforms a 70B model flying blind.

**3. From Chatbot to Operator (OODA Loop)**

Code Scalpel transforms LLMs from "suggestion machines" into autonomous operators:

1. **Observe:** `analyze_code` → Map the structure
2. **Orient:** `extract_code` → Isolate the bug's ecosystem  
3. **Decide:** `symbolic_execute` → Verify fix mathematically
4. **Act:** `update_symbol` → Apply without breaking syntax

---

## Quick Comparison

| Feature | Traditional Tools | Code Scalpel |
|---------|------------------|--------------|
| Pattern matching (regex) | ✓ | **Taint tracking** through variables |
| Single file analysis | ✓ | **Cross-file** dependency graphs |
| Manual test writing | ✓ | **Z3-powered** test generation |
| Generic output | ✓ | **AI-optimized** structured responses |
| Context strategy | Stuff everything | **Surgical slicing** |

---

## Quick Demo

### 1. Security: Find Hidden Vulnerabilities

```python
# The SQL injection is hidden through 3 variable assignments
# Regex linters miss this. Code Scalpel doesn't.

code-scalpel scan demos/vibe_check.py
# → SQL Injection (CWE-89) detected at line 38
#   Taint path: request.args → user_id → query_base → final_query
```

### 2. Secret Scanning: Detect Hardcoded Secrets

```python
# Detects AWS Keys, Stripe Secrets, Private Keys, and more
# Handles bytes, f-strings, and variable assignments

code-scalpel scan demos/config.py
# → Hardcoded Secret (AWS Access Key) detected at line 12
# → Hardcoded Secret (Stripe Secret Key) detected at line 45
```

### 3. Analysis: Understand Complex Code

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

### 4. Test Generation: Cover Every Path

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
| `extract_code` | **NEW** Surgically extract functions/classes with cross-file deps |
| `update_symbol` | **NEW** Safely replace functions/classes in files |
| `crawl_project` | **NEW** Discover project structure and file analysis |

## Features

### Polyglot Analysis
- **Python**: Full AST + PDG + Symbolic Execution
- **JavaScript**: Tree-sitter parsing + IR normalization
- **Java**: Enterprise-ready cross-file analysis

### Security Analysis
- SQL Injection (CWE-89)
- Cross-Site Scripting (CWE-79) - Flask/Django sinks
- Command Injection (CWE-78)
- Path Traversal (CWE-22)
- Code Injection (CWE-94) - eval/exec
- Insecure Deserialization (CWE-502) - pickle
- SSRF (CWE-918)
- Weak Cryptography (CWE-327) - MD5/SHA1
- Hardcoded Secrets (CWE-798) - 30+ patterns (AWS, GitHub, Stripe, private keys)
- NoSQL Injection (CWE-943) - MongoDB PyMongo/Motor
- LDAP Injection (CWE-90) - python-ldap/ldap3

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

## Roadmap

See [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) for the complete roadmap.

| Version | Status | Highlights |
|---------|--------|------------|
| **v1.3.0** | Current | NoSQL/LDAP injection, hardcoded secrets, 95% coverage |
| **v1.4.0** | Next | Context MCP tools (get_file_context, get_symbol_references) |
| **v1.5.0** | Planned | Project intelligence (get_project_map, get_call_graph) |
| **v2.0.0** | Planned | TypeScript/JavaScript full support |
| **v2.1.0** | Planned | AI verification tools (verify_behavior, suggest_fix) |

**Strategic Focus:** MCP server toolkit enabling AI agents to perform surgical code operations without hallucination.

## Stats

- **1,669** tests passing
- **100%** coverage: PDG, AST, Symbolic Execution, Security Analysis
- **95%+** coverage: Surgical Tools (SurgicalExtractor 95%, SurgicalPatcher 96%)
- **3** languages supported (Python full, JS/Java structural)
- **8** MCP tools for AI agents
- **15+** vulnerability types detected (SQL, XSS, NoSQL, LDAP, command injection, path traversal, secrets)
- **30+** secret detection patterns (AWS, GitHub, Stripe, private keys)
- **200x** cache speedup

## License

MIT License - see [LICENSE](LICENSE)

"Code Scalpel" is a trademark of 3D Tech Solutions LLC.

---

**Built for the AI Agent Era** | [PyPI](https://pypi.org/project/code-scalpel/) | [GitHub](https://github.com/tescolopio/code-scalpel)

<!-- mcp-name: io.github.tescolopio/code-scalpel -->
