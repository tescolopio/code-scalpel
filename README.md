# Code Scalpel

[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://pypi.org/project/code-scalpel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1669%20passed-brightgreen.svg)](https://github.com/tescolopio/code-scalpel)

**Precision Code Analysis for the AI Era**

Code Scalpel gives AI agents the power to understand, analyze, and transform code with mathematical precision. Built for Claude, Copilot, and autonomous coding systems.

```bash
pip install code-scalpel
```

> **✅ v1.2.3 STABLE RELEASE** (December 12, 2025)  
> Production-ready with **100% vulnerability detection rate** (validated by external team).
>
> | Capability | Status | Notes |
> |------------|--------|-------|
> | Security Scanning | **100%** (12/12 types) | SQL injection, XSS, command injection, path traversal, SSRF, weak crypto |
> | Test Generation | **Correct types** | Float, int, str, bool all properly inferred |
> | Symbolic Execution | **4/4 paths** | Full path exploration with deduplication |
> | Line Numbers | **Working** | Exact line numbers in all vulnerability reports |
>
> **What's New in v1.2.3:**
> - Flask XSS detection (`render_template_string`, `Response`)
> - Float type inference for test generation
> - Test path deduplication (no duplicates)
> - Meaningful assertions (`assert result is True/False`)
>
> See [RELEASE_NOTES_v1.2.2.md](RELEASE_NOTES_v1.2.2.md) for technical details.

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
- Hardcoded Secrets (CWE-798) - AWS keys, API tokens

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

- **1,669** tests passing
- **100%** coverage: PDG, AST, Symbolic Execution, Security Analysis
- **95%** coverage: Surgical Tools (SurgicalExtractor 94%, SurgicalPatcher 96%)
- **3** languages supported (Python full, JS/Java structural)
- **8** MCP tools
- **200x** cache speedup

## License

MIT License - see [LICENSE](LICENSE)

"Code Scalpel" is a trademark of 3D Tech Solutions LLC.

---

**Built for the AI Agent Era** | [PyPI](https://pypi.org/project/code-scalpel/) | [GitHub](https://github.com/tescolopio/code-scalpel)

<!-- mcp-name: io.github.tescolopio/code-scalpel -->
