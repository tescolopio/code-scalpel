# MCP Server - Complete Reference

The MCP (Model Context Protocol) server exposes Code Scalpel's analysis tools to AI assistants and agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Starting the Server](#starting-the-server)
4. [Available Tools](#available-tools)
5. [Tool Reference](#tool-reference)
6. [Configuration](#configuration)
7. [Client Integration](#client-integration)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The MCP server implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification, allowing AI assistants like Claude to use Code Scalpel's analysis capabilities.

### Architecture

```
┌─────────────────┐        MCP Protocol        ┌─────────────────┐
│                 │ ◄─────────────────────────► │                 │
│  Claude Desktop │         stdio/HTTP          │  Code Scalpel   │
│  Cursor         │                             │  MCP Server     │
│  Custom Client  │                             │                 │
└─────────────────┘                             └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Analysis Tools │
                                               │  - AST          │
                                               │  - PDG          │
                                               │  - Symbolic     │
                                               │  - Security     │
                                               └─────────────────┘
```

### Transport Options

| Transport | Use Case | Default |
|-----------|----------|---------|
| **stdio** | Local AI assistants (Claude Desktop, Cursor) | ✅ |
| **HTTP** | Network deployment, remote access | - |

---

## Installation

### Requirements

```bash
pip install code-scalpel mcp
```

### Verify Installation

```bash
python -c "from code_scalpel.mcp import run_server; print('MCP server ready')"
```

---

## Starting the Server

### stdio Transport (Default)

```bash
# Using module
python -m code_scalpel.mcp.server

# Using CLI
code-scalpel mcp
```

### HTTP Transport

```bash
# Using module
python -m code_scalpel.mcp.server --transport streamable-http --port 8593

# Using CLI
code-scalpel mcp --transport http --port 8593

# With custom host (for network access)
code-scalpel mcp --transport http --host 0.0.0.0 --port 8593
```

### Programmatic Start

```python
from code_scalpel.mcp import run_server

# Start with default settings (stdio)
run_server()

# Start with HTTP
run_server(transport="streamable-http", port=8593)
```

---

## Available Tools

The MCP server exposes 5 tools:

| Tool | Description | Best For |
|------|-------------|----------|
| `analyze_code` | Parse code structure | Understanding code organization |
| `security_scan` | Detect vulnerabilities | Security audits |
| `symbolic_execute` | Explore execution paths | Edge case discovery |
| `generate_unit_tests` | Create test cases | Automated testing |
| `simulate_refactor` | Verify safe refactoring | Code changes |

---

## Tool Reference

### analyze_code

Parse code and extract structural information.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | string | ✅ | Source code to analyze |
| `language` | string | - | Language: "python" (default) or "java" |

**Request:**

```json
{
  "name": "analyze_code",
  "arguments": {
    "code": "def hello(name):\n    return f'Hello, {name}!'",
    "language": "python"
  }
}
```

**Response:**

```json
{
  "success": true,
  "server_version": "1.0.1",
  "functions": ["hello"],
  "classes": [],
  "imports": [],
  "function_count": 1,
  "class_count": 0,
  "complexity": 1,
  "lines_of_code": 2,
  "issues": []
}
```

**Example Usage (Claude):**

```
User: Analyze this code and tell me about its structure.

[Provides code]

Claude: I'll analyze the code structure for you.

[Calls analyze_code tool]

The code contains:
- 3 functions: main, process_data, validate
- 1 class: DataProcessor
- Total complexity: 12
```

---

### security_scan

Detect security vulnerabilities using taint analysis.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | string | ✅ | Source code to scan |

**Request:**

```json
{
  "name": "security_scan",
  "arguments": {
    "code": "user_id = request.args.get('id')\nquery = 'SELECT * FROM users WHERE id=' + user_id\ncursor.execute(query)"
  }
}
```

**Response:**

```json
{
  "success": true,
  "server_version": "1.0.1",
  "has_vulnerabilities": true,
  "vulnerability_count": 1,
  "risk_level": "HIGH",
  "vulnerabilities": [
    {
      "type": "SQL Injection",
      "cwe": "CWE-89",
      "severity": "HIGH",
      "line": 3,
      "description": "User input concatenated into SQL query without sanitization"
    }
  ],
  "taint_sources": ["request.args.get"]
}
```

**Detected Vulnerabilities:**

| Type | CWE | Description |
|------|-----|-------------|
| SQL Injection | CWE-89 | User input in SQL queries |
| XSS | CWE-79 | User input in HTML output |
| Command Injection | CWE-78 | User input in shell commands |
| Path Traversal | CWE-22 | User input in file paths |

---

### symbolic_execute

Explore execution paths using symbolic execution.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | string | ✅ | Source code to analyze |
| `function_name` | string | - | Specific function to analyze |
| `max_paths` | integer | - | Maximum paths to explore (default: 50) |

**Request:**

```json
{
  "name": "symbolic_execute",
  "arguments": {
    "code": "def classify(x):\n    if x < 0:\n        return 'negative'\n    elif x == 0:\n        return 'zero'\n    else:\n        return 'positive'",
    "function_name": "classify",
    "max_paths": 10
  }
}
```

**Response:**

```json
{
  "success": true,
  "server_version": "1.0.1",
  "paths_explored": 3,
  "paths": [
    {
      "path_id": 1,
      "conditions": ["x < 0"],
      "final_state": {"return": "negative"},
      "reproduction_input": {"x": -1},
      "is_reachable": true
    },
    {
      "path_id": 2,
      "conditions": ["x >= 0", "x == 0"],
      "final_state": {"return": "zero"},
      "reproduction_input": {"x": 0},
      "is_reachable": true
    },
    {
      "path_id": 3,
      "conditions": ["x >= 0", "x != 0"],
      "final_state": {"return": "positive"},
      "reproduction_input": {"x": 1},
      "is_reachable": true
    }
  ],
  "symbolic_variables": ["x"],
  "constraints": ["x: Int"]
}
```

---

### generate_unit_tests

Generate test cases from symbolic execution paths.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `code` | string | ✅ | Source code containing function |
| `function_name` | string | ✅ | Function to generate tests for |
| `format` | string | - | Output format: "pytest" (default) or "unittest" |

**Request:**

```json
{
  "name": "generate_unit_tests",
  "arguments": {
    "code": "def is_adult(age):\n    return age >= 18",
    "function_name": "is_adult",
    "format": "pytest"
  }
}
```

**Response:**

```json
{
  "success": true,
  "server_version": "1.0.1",
  "function_name": "is_adult",
  "test_count": 2,
  "test_cases": [
    {
      "path_id": 1,
      "function_name": "is_adult",
      "inputs": {"age": 18},
      "description": "Path: age >= 18 -> True",
      "path_conditions": ["age >= 18"]
    },
    {
      "path_id": 2,
      "function_name": "is_adult",
      "inputs": {"age": 17},
      "description": "Path: age < 18 -> False",
      "path_conditions": ["age < 18"]
    }
  ],
  "pytest_code": "import pytest\n\ndef test_is_adult_adult():\n    \"\"\"Path: age >= 18\"\"\"\n    assert is_adult(18) == True\n\ndef test_is_adult_minor():\n    \"\"\"Path: age < 18\"\"\"\n    assert is_adult(17) == False\n",
  "unittest_code": "..."
}
```

---

### simulate_refactor

Verify a code change is safe before applying.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `original_code` | string | ✅ | Original source code |
| `refactored_code` | string | ✅ | Proposed refactored code |

**Request:**

```json
{
  "name": "simulate_refactor",
  "arguments": {
    "original_code": "def calc(x):\n    return x * 2",
    "refactored_code": "def calculate(x):\n    return x << 1"
  }
}
```

**Response:**

```json
{
  "success": true,
  "server_version": "1.0.1",
  "is_safe": true,
  "status": "safe",
  "reason": null,
  "security_issues": [],
  "structural_changes": {
    "functions_renamed": [["calc", "calculate"]],
    "functions_added": [],
    "functions_removed": []
  },
  "warnings": [
    "Bitwise shift used instead of multiplication - ensure this is intentional"
  ]
}
```

**Status Values:**

| Status | Meaning |
|--------|---------|
| `safe` | No issues found |
| `warning` | Minor concerns, review recommended |
| `unsafe` | Security or behavioral issues detected |
| `error` | Analysis failed |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCALPEL_CACHE_ENABLED` | Enable analysis caching | `1` |
| `SCALPEL_CACHE_DIR` | Cache directory | `~/.cache/code-scalpel` |
| `SCALPEL_LOG_LEVEL` | Logging level | `INFO` |
| `SCALPEL_MAX_CODE_SIZE` | Maximum code size (bytes) | `100000` |

### Server Options

```bash
python -m code_scalpel.mcp.server \
  --transport stdio \           # or streamable-http
  --port 8593 \                 # HTTP port
  --host 127.0.0.1 \            # Bind address
  --verbose                      # Debug logging
```

---

## Client Integration

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

Restart Claude Desktop to load the server.

### Cursor

Add to Cursor settings:

```json
{
  "mcp": {
    "servers": {
      "code-scalpel": {
        "command": "python",
        "args": ["-m", "code_scalpel.mcp.server"]
      }
    }
  }
}
```

### Custom Client (Python)

```python
from mcp import Client, StdioTransport
import subprocess

# Start server as subprocess
process = subprocess.Popen(
    ["python", "-m", "code_scalpel.mcp.server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

# Create client
transport = StdioTransport(process.stdin, process.stdout)
client = Client(transport)

# Call tool
result = await client.call_tool("analyze_code", {
    "code": "def hello(): pass"
})
print(result)
```

### HTTP Client

```python
import requests

# Start server with HTTP transport first
# python -m code_scalpel.mcp.server --transport streamable-http --port 8593

response = requests.post(
    "http://localhost:8593/tools/analyze_code",
    json={"code": "def hello(): pass"}
)
print(response.json())
```

---

## Security

### Code Safety

Code Scalpel **parses** code but never **executes** it:

```python
# Code is parsed with ast.parse(), not executed
import ast
tree = ast.parse(code)  # Safe
eval(code)              # NOT done
```

### Size Limits

Maximum code size is enforced:

```python
MAX_CODE_SIZE = 100_000  # 100KB

if len(code) > MAX_CODE_SIZE:
    return error("Code exceeds maximum size")
```

### Network Binding

HTTP transport binds to localhost by default:

```bash
# Safe: localhost only
python -m code_scalpel.mcp.server --transport http --port 8593

# Caution: network accessible
python -m code_scalpel.mcp.server --transport http --host 0.0.0.0 --port 8593
```

### Authentication

For production deployment, add authentication:

```python
# Example with API key
from code_scalpel.mcp import create_authenticated_server

server = create_authenticated_server(
    api_key=os.environ["SCALPEL_API_KEY"]
)
```

---

## Troubleshooting

### Server Won't Start

1. Check Python version:
   ```bash
   python --version  # Requires 3.9+
   ```

2. Verify installation:
   ```bash
   pip install code-scalpel mcp
   ```

3. Check for port conflicts:
   ```bash
   lsof -i :8593
   ```

### Connection Issues

1. For stdio, check client configuration points to correct Python

2. For HTTP, verify server is running:
   ```bash
   curl http://localhost:8593/health
   ```

3. Check logs:
   ```bash
   python -m code_scalpel.mcp.server --verbose
   ```

### Tool Errors

1. **Code too large**: Reduce code size or increase limit
2. **Timeout**: Increase timeout or simplify code
3. **Parse error**: Verify code syntax is valid

### Cache Issues

Clear cache if stale results:

```bash
rm -rf ~/.cache/code-scalpel
```

Or disable caching:

```bash
export SCALPEL_CACHE_ENABLED=0
```

---

## API Quick Reference

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": { ... }
  },
  "id": 1
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{ ... tool result ... }"
      }
    ]
  },
  "id": 1
}
```

### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Analysis failed: syntax error"
  },
  "id": 1
}
```

---

*MCP Server - Bringing Code Scalpel's power to AI assistants everywhere.*
