# AI Agent Integration Guide

This guide covers how to integrate Code Scalpel with AI agents and coding assistants.

## Overview

Code Scalpel is designed for AI-driven code analysis. It provides:

- **MCP Server**: Model Context Protocol for Claude, Copilot, Cursor
- **Surgical Tools**: Token-efficient extraction and modification
- **Structured Output**: Pydantic models optimized for LLM consumption

## MCP Server Integration

### Claude Desktop

Add to `claude_desktop_config.json` (usually at `~/.config/claude/`):

```json
{
  "mcpServers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "/path/to/your/project"]
    }
  }
}
```

Restart Claude Desktop. You'll see Code Scalpel tools in the tools menu.

### VS Code with Copilot / Cursor

Create `.vscode/mcp.json` in your workspace:

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

### Docker / Network Deployment

For team or remote deployment:

```bash
# Start HTTP server
docker run -p 8593:8593 -v /path/to/project:/app/code code-scalpel

# Or directly
code-scalpel mcp --http --port 8593 --allow-lan
```

## Available Tools

### 1. `extract_code` - Surgical Extraction

**Purpose**: Extract specific code elements without sending entire files.

**Token Savings**: Instead of sending a 5000-line file (~50k tokens), send only the function you need (~200 tokens).

```python
# Example: Agent wants to refactor calculate_tax
result = extract_code(
    file_path="/project/src/billing/calculator.py",
    target_type="function",
    target_name="calculate_tax",
    include_cross_file_deps=True  # Also gets TaxRate from models.py
)

# Agent receives:
# - target_code: The function (~50 lines)
# - context_code: External dependencies (TaxRate class from models.py)
# - token_estimate: ~150 tokens instead of 50,000
```

**Parameters**:
- `file_path`: Path to source file (server reads it - 0 tokens to agent)
- `target_type`: "function", "class", or "method"
- `target_name`: Name of the element (use "ClassName.method" for methods)
- `include_cross_file_deps`: Resolve imports from external files
- `include_context`: Include intra-file dependencies
- `context_depth`: How deep to traverse (1=direct, 2=transitive)

### 2. `update_symbol` - Surgical Modification

**Purpose**: Replace a function/class/method without rewriting entire file.

**Safety Features**:
- Automatic backup creation
- Syntax validation before write
- Preserves surrounding code exactly

```python
# Example: Agent refactored calculate_tax, now wants to update it
result = update_symbol(
    file_path="/project/src/billing/calculator.py",
    target_type="function",
    target_name="calculate_tax",
    new_code="""def calculate_tax(amount: float, rate: float = 0.1) -> float:
    \"\"\"Calculate tax with configurable rate.\"\"\"
    return round(amount * rate, 2)
""",
    create_backup=True
)

# Result:
# - success: true
# - backup_path: /project/src/billing/calculator.py.bak
# - lines_delta: -3 (new code is 3 lines shorter)
```

### 3. `crawl_project` - Project Discovery

**Purpose**: Understand project structure before making changes.

```python
result = crawl_project(
    root_path="/project",
    include_patterns=["*.py"],
    exclude_patterns=["**/test_*", "**/__pycache__/**"],
    include_analysis=True  # Adds function/class counts
)

# Returns tree structure with analysis metadata
```

### 4. `analyze_code` - Deep Analysis

**Purpose**: Get comprehensive code metrics and structure.

```python
result = analyze_code(
    code="def hello(): return 42"  # Or use file_path
)

# Returns metrics, dependencies, call graph
```

### 5. `security_scan` - Vulnerability Detection

**Purpose**: Find security issues via taint analysis.

```python
result = security_scan(
    code=suspicious_code
)

# Returns: SQL injection, XSS, command injection paths
```

### 6. `symbolic_execute` - Path Exploration

**Purpose**: Find all possible execution paths with Z3.

```python
result = symbolic_execute(
    code=complex_function,
    max_depth=50
)

# Returns: All reachable paths with concrete inputs
```

### 7. `generate_unit_tests` - Test Generation

**Purpose**: Create tests that cover all branches.

```python
result = generate_unit_tests(
    code=function_code,
    framework="pytest"
)

# Returns: Complete test file with branch coverage
```

### 8. `simulate_refactor` - Safe Refactoring

**Purpose**: Preview refactoring before applying.

```python
result = simulate_refactor(
    code=original_code,
    refactor_type="rename_variable",
    old_name="x",
    new_name="counter"
)

# Returns: Refactored code + validation results
```

## Token Optimization Strategy

### The Problem

Traditional LLM coding:
```
Agent: "Read file X" → 10,000 tokens
Agent: "Read file Y" → 8,000 tokens
Agent: "Find function Z in X" → reasoning
Agent: "Modify function Z" → send all 10,000 tokens back
TOTAL: ~28,000 tokens
```

### The Scalpel Solution

```
Agent: extract_code(file_path="X", target_name="Z", include_cross_file_deps=True)
Server: Reads X (0 tokens), finds Z, resolves deps from Y
Agent receives: ~200 tokens (just Z + deps)

Agent: "Modify function Z" → reasoning on 200 tokens

Agent: update_symbol(file_path="X", target_name="Z", new_code=modified)
Server: Locates Z, validates, writes
Agent receives: ~50 tokens (success confirmation)

TOTAL: ~500 tokens (56x reduction)
```

## Workflow Examples

### Refactoring a Function

```python
# 1. Agent asks for the function
extract_result = extract_code(
    file_path="src/utils.py",
    target_type="function",
    target_name="calculate_total",
    include_cross_file_deps=True
)

# 2. Agent reasons about the code (only ~200 tokens of context)
# "I see calculate_total uses TaxRate from models.py..."

# 3. Agent generates new code
new_code = """def calculate_total(items, tax_rate=None):
    ...
"""

# 4. Agent applies the change
update_result = update_symbol(
    file_path="src/utils.py",
    target_type="function",
    target_name="calculate_total",
    new_code=new_code
)
```

### Security Audit

```python
# 1. Crawl project
project = crawl_project(root_path="/app", include_patterns=["*.py"])

# 2. Scan each file
for file in project.files:
    scan = security_scan(file_path=file.path)
    if scan.vulnerabilities:
        # Extract the vulnerable function for closer inspection
        for vuln in scan.vulnerabilities:
            code = extract_code(
                file_path=file.path,
                target_type="function",
                target_name=vuln.function_name
            )
```

## Best Practices

1. **Use `file_path` over `code`**: Let the server read files (0 tokens to agent)

2. **Enable cross-file deps**: Get complete context without multiple reads

3. **Use `crawl_project` first**: Understand structure before diving in

4. **Create backups**: Always set `create_backup=True` on `update_symbol`

5. **Validate after update**: Run tests or `analyze_code` after modifications

## Troubleshooting

### "File not found"
- Check the file path is absolute or relative to `--root`
- Verify the MCP server has access to the workspace

### "Function not found"
- Check the exact function name (case-sensitive)
- For methods, use "ClassName.method_name" format
- Use `crawl_project` to discover available symbols

### "Cross-file deps not resolving"
- Ensure `file_path` is set (not `code`)
- Check that import paths are resolvable relative to the file
- Increase `context_depth` for transitive dependencies

## Framework-Specific Integration

### Autogen

```python
from autogen import AssistantAgent
from code_scalpel.integrations import AutogenScalpel

agent = AssistantAgent(
    name="CodeAnalyst",
    llm_config={"model": "gpt-4"},
)

# Add Code Scalpel tools
scalpel = AutogenScalpel()
agent.register_function(scalpel.analyze_code)
agent.register_function(scalpel.extract_code)
```

### CrewAI

```python
from crewai import Agent, Task, Crew
from code_scalpel.integrations import CrewAIScalpel

scalpel_tools = CrewAIScalpel()

analyst = Agent(
    role="Code Analyst",
    tools=[scalpel_tools.analyze, scalpel_tools.extract],
)
```

### LangChain

```python
from langchain.agents import initialize_agent
from code_scalpel.integrations import LangChainScalpel

tools = LangChainScalpel().get_tools()
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```
