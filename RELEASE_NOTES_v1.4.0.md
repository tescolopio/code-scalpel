# Code Scalpel v1.4.0 "Context" Release Notes

**Release Date:** December 12, 2024

## Overview

v1.4.0 "Context" focuses on **workspace-level intelligence** for AI agents. New MCP tools provide file overviews and symbol reference tracking without reading entire file contents, dramatically reducing token usage while improving navigation accuracy.

## New MCP Tools

### `get_file_context`

Returns a structured overview of a Python file without loading full content:

- **Functions**: Names, line numbers, parameters, docstrings
- **Classes**: Names, methods, base classes
- **Imports**: All import statements
- **Complexity Metrics**: Cyclomatic complexity estimates
- **Security Flags**: Quick scan for potential issues

**Token Efficiency**: ~200 tokens vs ~10,000 for full file read.

```python
# Example response structure
{
    "file_path": "/src/utils.py",
    "language": "python",
    "functions": [
        {"name": "calculate_tax", "line": 15, "parameters": ["amount", "rate"], "complexity": 3}
    ],
    "classes": [
        {"name": "TaxCalculator", "line": 45, "methods": ["__init__", "compute"], "bases": ["BaseCalculator"]}
    ],
    "imports": ["from decimal import Decimal", "import math"],
    "has_security_issues": false,
    "estimated_tokens": 1250
}
```

### `get_symbol_references`

Finds all usages of a function, class, or method across a project:

- **Definition Location**: Where the symbol is defined
- **Call Sites**: All locations where the symbol is used
- **Context Lines**: Surrounding code for each reference
- **Cross-File Search**: Scans entire project directory

**Use Cases**:
- Understanding function usage before refactoring
- Finding all call sites for API changes
- Tracing class inheritance and method overrides

```python
# Example: Find all usages of "calculate_tax"
{
    "symbol": "calculate_tax",
    "definition": {"file": "/src/tax.py", "line": 15},
    "references": [
        {"file": "/src/orders.py", "line": 42, "context": "total = calculate_tax(subtotal)"},
        {"file": "/src/reports.py", "line": 88, "context": "tax_amount = calculate_tax(line_total, rate=0.08)"}
    ],
    "total_references": 2
}
```

## New Security Detections

### XXE - XML External Entity Injection (CWE-611)

Detects unsafe XML parsing that could allow attackers to:
- Read arbitrary files from the server
- Perform Server-Side Request Forgery (SSRF)
- Execute denial-of-service attacks

**Vulnerable Sinks**:
- `xml.etree.ElementTree.parse`
- `xml.dom.minidom.parse`
- `xml.sax.parse`
- `lxml.etree.parse`, `lxml.etree.fromstring`

**Safe Alternatives** (Sanitizers):
- `defusedxml.parse`
- `defusedxml.ElementTree.parse`
- `defusedxml.minidom.parse`

### SSTI - Server-Side Template Injection (CWE-1336)

Detects template engines rendering user-controlled strings directly:

**Vulnerable Sinks**:
- `jinja2.Template(user_input)` - Direct string compilation
- `mako.template.Template(user_input)`
- `django.template.Template(user_input)`
- `tornado.template.Template(user_input)`

**Safe Alternatives** (Sanitizers):
- `render_template("file.html")` - File-based templates
- `flask.render_template()`

## Statistics

| Metric | Value |
|--------|-------|
| Tests Passing | 1,692 |
| MCP Tools | 10 |
| Security Sinks | 15 vulnerability types |
| Coverage | 95%+ |

## Vulnerability Coverage (v1.4.0)

| Vulnerability | CWE | Introduced |
|---------------|-----|------------|
| SQL Injection | CWE-89 | v1.0.0 |
| XSS | CWE-79 | v1.0.0 |
| Command Injection | CWE-78 | v1.0.0 |
| Path Traversal | CWE-22 | v1.0.0 |
| NoSQL Injection | CWE-943 | v1.3.0 |
| LDAP Injection | CWE-90 | v1.3.0 |
| Secret Detection | CWE-798 | v1.3.0 |
| Weak Cryptography | CWE-327 | v1.3.0 |
| SSRF | CWE-918 | v1.3.0 |
| **XXE** | CWE-611 | **v1.4.0** |
| **SSTI** | CWE-1336 | **v1.4.0** |

## MCP Tool Inventory (v1.4.0)

| Tool | Purpose |
|------|---------|
| `analyze_code` | Parse and extract code structure |
| `extract_code` | Surgical extraction by symbol name |
| `update_symbol` | Safe symbol replacement with backup |
| `security_scan` | Taint-based vulnerability detection |
| `generate_unit_tests` | Symbolic execution test generation |
| `simulate_refactor` | Verify refactor preserves behavior |
| `crawl_project` | Project-wide analysis |
| **`get_file_context`** | **File overview without full content** |
| **`get_symbol_references`** | **Find all usages of a symbol** |

## Upgrade Guide

```bash
pip install --upgrade code-scalpel
```

No breaking changes from v1.3.0. New tools are additive.

## What's Next (v1.5.0 Preview)

- JWT vulnerabilities (algorithm confusion, missing verification)
- Mass assignment detection (unfiltered request.json)
- Enhanced cross-file dependency resolution
- Java parser improvements

---

**Full Changelog**: https://github.com/code-scalpel/code-scalpel/compare/v1.3.0...v1.4.0
