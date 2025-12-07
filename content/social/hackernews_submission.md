# Hacker News Submission

## Title (80 char limit)

```
Show HN: Code Scalpel – Z3-powered code analysis for AI agents (Python/JS/Java)
```

## URL

```
https://github.com/tescolopio/code-scalpel
```

## First Comment (Post immediately after submission)

---

Hi HN! I built Code Scalpel to solve a problem I kept hitting with AI coding assistants.

**The Problem:**

When I ask Claude or Copilot to review my code for security issues, they use pattern matching. "I see `eval()` - that's dangerous." But real vulnerabilities aren't that obvious. They're buried in data flows:

```python
user_id = request.args.get('id')  # tainted
part1 = "SELECT * FROM users WHERE id='"
query = part1 + user_id  # still tainted
cursor.execute(query)  # sink - but no obvious pattern
```

Regex linters miss this. I wanted something that could actually follow the data.

**The Solution:**

Code Scalpel uses three techniques:

1. **Taint tracking** - Tags user input as "tainted" and follows it through variable assignments until it reaches a dangerous operation

2. **Symbolic execution** - Uses Z3 to explore every possible execution path and find inputs that trigger specific branches

3. **MCP integration** - Designed as an MCP server so AI agents can use it directly, not just humans

**Technical details:**

- Built a unified IR (Intermediate Representation) so the same symbolic engine works across Python, JavaScript, and Java
- Content-addressable caching (SHA256 of code + version + config) for 200x speedup on unchanged files
- 5-second Z3 timeout to prevent hangs on complex constraints
- 654 tests, ~78% coverage

**What it's NOT:**

- Not a replacement for proper security audits
- Not production-ready for massive codebases (symbolic execution has limits)
- Not a commercial product - MIT licensed, free forever

Would love feedback on:
- What languages should I prioritize next? (Go? Rust? TypeScript separately from JS?)
- What other AI assistants should I support beyond Claude/Copilot?
- Are there specific vulnerability patterns you'd want detected?

Happy to answer any questions about the architecture or implementation!

---

## Anticipated Questions & Answers

**Q: How does this compare to Semgrep/CodeQL?**

A: Different goals. Semgrep/CodeQL are pattern-based rule engines - excellent for known vulnerability patterns at scale. Code Scalpel does dataflow analysis and symbolic execution - it can find things that don't match any pattern, but it's slower and more resource-intensive. Think of it as a complement, not a replacement.

**Q: Why Z3 instead of [other solver]?**

A: Z3 has the best Python bindings, handles the theory of strings well (important for web security), and is battle-tested at Microsoft scale. We also set a 5-second timeout per constraint to prevent hangs.

**Q: Symbolic execution doesn't scale. How do you handle real codebases?**

A: You're right - we don't try to analyze entire codebases symbolically. The tool is designed for targeted analysis: analyze a specific function, scan a specific file. The caching layer helps with repeated analysis. For whole-codebase scanning, combine with pattern-based tools.

**Q: Why MCP instead of just a CLI?**

A: AI agents work better with structured tool interfaces than parsing CLI output. MCP (Model Context Protocol) is becoming the standard for AI tool integration - Claude Desktop, Cursor, and Copilot all support it. The CLI exists for humans, but the MCP server is the primary interface.

**Q: Can I use this in CI/CD?**

A: Yes! `code-scalpel scan` returns exit code 2 if vulnerabilities are found. You can also use `--json` for machine-readable output. For CI integration, I'd recommend combining with Semgrep for speed, then using Code Scalpel for deep analysis on changed files.
