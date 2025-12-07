# Dev.to Article: Code Scalpel v1.0

---
title: I Built a Z3-Powered Code Analyzer for AI Agents - Here's What I Learned
published: true
description: How symbolic execution and taint tracking can find vulnerabilities that regex misses
tags: python, security, ai, opensource
cover_image: https://dev-to-uploads.s3.amazonaws.com/uploads/articles/[YOUR_IMAGE].png
---

## The Problem That Started Everything

I was using Claude to review a Flask application for security issues. It flagged a few obvious things - `eval()` calls, missing CSRF tokens. Standard stuff.

But it completely missed this:

```python
def get_user(request):
    user_id = request.args.get('id')
    partial = "SELECT * FROM users WHERE id = '"
    query = partial + user_id + "'"
    cursor.execute(query)
    return cursor.fetchone()
```

Classic SQL injection. But because the user input wasn't directly next to `execute()`, the pattern matching didn't catch it.

I thought: **what if AI agents could actually follow data flow instead of matching patterns?**

## Enter Code Scalpel

Six months later, I shipped v1.0 of [Code Scalpel](https://github.com/tescolopio/code-scalpel) - an open-source toolkit that gives AI agents three superpowers:

### 1. Taint Tracking

Every piece of user input gets tagged as "tainted." Then we follow it:

```
request.args.get('id')  →  user_id  →  partial + user_id  →  query  →  cursor.execute()
          ↑                                                                    ↑
       SOURCE                                                                SINK
          (tainted)                           (still tainted)              (VULNERABILITY!)
```

Doesn't matter how many variables you pass through. If tainted data reaches a dangerous operation, we catch it.

### 2. Symbolic Execution

Here's where it gets interesting. Instead of running code with concrete values, we run it with *symbolic* values.

```python
def loan_approval(income, debt, credit_score):
    if credit_score < 600:
        return "REJECT"
    if income > 100000 and debt < 5000:
        return "INSTANT_APPROVE"
    return "STANDARD"
```

A normal test might try random inputs. Symbolic execution asks Z3 (Microsoft's theorem prover): **"What inputs make each branch true?"**

Z3 responds:
- `credit_score = 599` → REJECT
- `income = 100001, debt = 4999, credit_score = 700` → INSTANT_APPROVE
- `income = 50000, debt = 20000, credit_score = 700` → STANDARD

Mathematically derived test inputs. No guessing.

### 3. AI-Native Interface

Code Scalpel isn't a CLI tool you run manually. It's designed as an [MCP server](https://modelcontextprotocol.io/) that AI agents can use directly.

```json
// .vscode/mcp.json for GitHub Copilot
{
  "servers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "${workspaceFolder}"]
    }
  }
}
```

Now when you ask Copilot to "scan this file for security issues," it can use real dataflow analysis instead of regex.

## What I Learned Building This

### Lesson 1: Symbolic Execution Has Limits

Z3 is powerful, but it can hang on complex constraints. I learned this the hard way when a test file with nested loops froze my machine for 20 minutes.

**Solution:** Every solver call now has a 5-second timeout. If Z3 can't solve it in 5 seconds, we mark the path as "unknown" and move on.

### Lesson 2: Caching Is Non-Negotiable

Symbolic execution is slow. Analyzing a 500-line file takes 2-3 seconds. Users expect instant feedback.

**Solution:** Content-addressable caching. We hash the code + tool version + configuration. Same file? Instant cache hit. Changed your code? Fresh analysis.

Result: **200x speedup** on cache hits.

### Lesson 3: AI Agents Need Structure

Early versions returned human-readable text. AI agents struggled to parse it.

**Solution:** Every tool returns Pydantic models with strict schemas:

```python
class SecurityResult(BaseModel):
    success: bool
    has_vulnerabilities: bool
    vulnerability_count: int
    risk_level: str  # "low", "medium", "high", "critical"
    vulnerabilities: list[VulnerabilityInfo]
```

Structured output → better AI reasoning.

## Try It Yourself

```bash
pip install code-scalpel

# Scan for vulnerabilities
code-scalpel scan your_app.py

# Analyze code structure
code-scalpel analyze your_app.py --json

# Start MCP server for AI agents
code-scalpel mcp
```

The [demos folder](https://github.com/tescolopio/code-scalpel/tree/main/demos) has real-world examples:
- FastAPI app with 4 different vulnerability types
- Django views showing safe vs. unsafe patterns
- React components with XSS issues

## What's Next?

I'm working on:
- **TypeScript-specific analysis** (beyond generic JS)
- **Go support** (lots of demand)
- **VS Code extension** (visual taint flow diagrams)

What would be most useful for your workflow? Drop a comment!

---

*Code Scalpel is MIT licensed and free forever. [Star it on GitHub](https://github.com/tescolopio/code-scalpel) if you find it useful!*
