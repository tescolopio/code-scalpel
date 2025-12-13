# Code Scalpel Demo Outline

**Precision Code Analysis for the AI Era**

*"Code Scalpel" is a trademark of 3D Tech Solutions LLC.*

---

## Demo Overview

**Duration:** 30-45 minutes (adjustable)  
**Audience:** DevOps engineers, AI/ML developers, security teams, CTOs  
**Goal:** Demonstrate Code Scalpel's unique value proposition for AI-assisted development

---

## Part 1: The Problem (5 minutes)

### "Why Traditional Tools Fail in the AI Era"

**Live Demo: The Blind Regex**

```bash
# Show a "hidden" SQL injection that grep/regex misses
cat demos/vibe_check.py
grep -n "execute" demos/vibe_check.py  # Finds nothing obvious
```

**The Code:**
```python
def process_user(request):
    user_id = request.args.get('id')     # Line 10 - Taint source
    temp = user_id                         # Line 11 - Propagation
    query_base = f"SELECT * FROM users"    # Line 12 - Looks safe
    final_query = query_base + " WHERE id=" + temp  # Line 13 - Injection!
    cursor.execute(final_query)            # Line 14 - Sink
```

**Key Point:** "The vulnerability is spread across 5 lines. No pattern matcher catches this."

---

## Part 2: The Solution - Taint Analysis (10 minutes)

### "Code Scalpel Tracks Data, Not Patterns"

**Live Demo: Security Scan**

```bash
code-scalpel scan demos/vibe_check.py
```

**Expected Output:**
```
[CRITICAL] SQL Injection (CWE-89) at line 14
  Taint Path: request.args → user_id → temp → final_query → cursor.execute()
  Recommendation: Use parameterized queries
```

**Interactive Exploration:**
- Show the taint graph visualization
- Explain source → propagation → sink model
- Compare to Snyk/SonarQube (which miss this pattern)

### "Secret Detection That Actually Works"

```bash
code-scalpel scan demos/config.py
```

**Show detection of:**
- AWS keys in environment fallbacks
- Stripe secrets in f-strings
- Private keys as byte literals
- API keys assigned to variables (not just literals)

---

## Part 3: Symbolic Execution (10 minutes)

### "Find Bugs That Tests Miss"

**The Challenge:**
```python
def calculate_discount(price, quantity, customer_type):
    if customer_type == "VIP" and quantity > 100:
        if price > 1000:
            discount = 0.25
        else:
            discount = 0.15
    elif quantity > 50:
        discount = 0.10
    else:
        discount = 0.0
    
    return price * (1 - discount)  # Bug: negative price not handled
```

**Live Demo:**

```bash
code-scalpel analyze demos/discount.py --symbolic
```

**Expected Output:**
```
Execution Paths Explored: 6
Path 1: customer_type="VIP", quantity=101, price=1001 → discount=0.25
Path 2: customer_type="VIP", quantity=101, price=500 → discount=0.15
Path 3: customer_type="other", quantity=51, price=any → discount=0.10
Path 4: customer_type="other", quantity=10, price=any → discount=0.0
...

⚠️ Edge Case Detected:
  - No validation for negative price
  - Constraint: price < 0 produces negative return value
```

### "Auto-Generated Tests"

```bash
code-scalpel generate-tests demos/discount.py --framework pytest
```

**Show generated test file with:**
- Concrete inputs for each path
- Edge cases automatically discovered
- 100% path coverage

---

## Part 4: Surgical Tools - AI Agent Integration (10 minutes)

### "The Token-Saving Revolution"

**The Problem with Current AI Coding:**
```
User: "Fix the bug in calculate_discount"
AI: *dumps entire 500-line file into context* (1,500 tokens)
AI: *makes change*
AI: *dumps entire file again* (1,500 tokens)
Total: 3,000+ tokens for a 5-line change
```

**Code Scalpel Approach:**
```
User: "Fix the bug in calculate_discount"
AI: extract_code(file="discount.py", target="calculate_discount")
Server: Returns only the function (50 tokens)
AI: Makes change
AI: update_symbol(file="discount.py", target="calculate_discount", new_code="...")
Total: 100 tokens
```

**Live Demo: MCP Server**

```bash
# Terminal 1: Start MCP server
code-scalpel mcp --root ./demos

# Terminal 2: Show agent interaction
```

**Copilot/Claude Demo:**
1. Open VS Code with MCP configured
2. Ask: "Extract the UserController class from user_service.py"
3. Show: Only the class returned, not the entire file
4. Ask: "Add input validation to the create_user method"
5. Show: Surgical patch applied, backup created

### "Cross-File Dependency Resolution"

```python
# user_controller.py uses TaxCalculator from tax_utils.py
extract_code(
    file="user_controller.py",
    target="process_order",
    include_cross_file_deps=True
)
```

**Output includes:**
- The `process_order` function
- The `TaxCalculator` class from `tax_utils.py`
- All transitive dependencies

---

## Part 5: Enterprise Readiness (5 minutes)

### "Production-Grade Robustness"

**Memory Stability:**
```
✓ Analyzed 50 files sequentially
✓ Peak memory: 32MB (100MB threshold)
✓ No memory leaks detected
```

**Timeout Protection:**
```
✓ Symbolic engine bounded to 10 iterations
✓ Path explosion gracefully halted
✓ MCP server rejects malformed requests
```

**Coverage:**
| Module | Coverage |
|--------|----------|
| PDG Tools | 100% |
| AST Tools | 100% |
| Symbolic Execution | 100% |
| Security Analysis | 100% |
| Surgical Tools | 95% |

### "Docker Deployment"

```bash
docker run -p 8593:8593 -v $(pwd):/app/code code-scalpel
# Team-wide MCP server ready in 10 seconds
```

---

## Part 6: Roadmap Teaser (2 minutes)

### "What's Coming in v1.3.0"

**TypeScript Support (MVP):**
- Parser and analyzer ready (stub demonstrated)
- Same MCP interface - no client changes
- Target: Q1 2025

**Branding Evolution:**
- `extract_code` → `biopsy_code`
- `update_symbol` → `graft_code`
- Full medical metaphor across all tools

**Java Enterprise:**
- NullPointerException detection via symbolic execution
- Spring framework security patterns
- Target: Q2 2025

---

## Part 7: Call to Action (3 minutes)

### "Get Started in 60 Seconds"

```bash
pip install code-scalpel
code-scalpel scan your_project/
```

### "Integrate with Your AI Workflow"

**VS Code/Copilot:**
```json
// .vscode/mcp.json
{
  "servers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "${workspaceFolder}"]
    }
  }
}
```

**Claude Desktop:**
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "code-scalpel": {
      "command": "uvx",
      "args": ["code-scalpel", "mcp", "--root", "/your/project"]
    }
  }
}
```

### "Join the Community"

- GitHub: github.com/tescolopio/code-scalpel
- PyPI: pypi.org/project/code-scalpel
- Documentation: [link]

---

## Demo Script Assets Checklist

- [ ] `demos/vibe_check.py` - Hidden SQL injection
- [ ] `demos/config.py` - Hardcoded secrets
- [ ] `demos/discount.py` - Complex branching logic
- [ ] `demos/user_service.py` - Cross-file dependencies
- [ ] `demos/tax_utils.py` - Dependency target
- [ ] Docker image built and tested
- [ ] MCP server verified with Copilot/Claude
- [ ] Slides (optional) for key points

---

## Appendix: Competitive Positioning

| Feature | Code Scalpel | SonarQube | Snyk | Semgrep |
|---------|-------------|-----------|------|---------|
| Taint Tracking | ✓ Full path | Partial | Partial | Limited |
| Symbolic Execution | ✓ Z3-powered | ✗ | ✗ | ✗ |
| AI Agent Integration | ✓ MCP native | ✗ | ✗ | ✗ |
| Token Optimization | ✓ Surgical | ✗ | ✗ | ✗ |
| Self-hosted | ✓ Free | $$$ | $$$ | Partial |
| Python Full Support | ✓ | ✓ | ✓ | ✓ |
| TypeScript | Coming Q1 | ✓ | ✓ | ✓ |

**Key Differentiator:** "Code Scalpel is the only tool built *for* AI agents, not just *with* AI."

---

*Prepared by 3D Tech Solutions LLC*  
*Version: Demo v1.2.0*
