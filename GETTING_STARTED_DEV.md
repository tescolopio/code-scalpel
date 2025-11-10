# Getting Started with Code Scalpel Development

> Quick guide to start contributing to Code Scalpel's production readiness

---

## Overview

Code Scalpel is transforming from a development prototype into a production-ready AI code analysis toolkit with Model Context Protocol (MCP) support. This guide helps you get started with contributing to this effort.

---

## 📚 Documentation Files

1. **[PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md)** - Comprehensive backlog with all epics, features, and tasks
2. **[ROADMAP.md](ROADMAP.md)** - High-level roadmap and timeline
3. **[GITHUB_ISSUES_TEMPLATE.md](GITHUB_ISSUES_TEMPLATE.md)** - Templates for creating GitHub issues
4. **This file** - Quick start guide

---

## 🚀 Quick Start for Contributors

### 1. Clone and Setup (Current State - Won't Work Yet!)

```bash
# Clone the repository
git clone https://github.com/tescolopio/code-scalpel.git
cd code-scalpel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (THIS WILL FAIL - needs to be fixed first!)
pip install -e .
```

**Note:** The first priority task is fixing the package structure so installation works!

### 2. Understand What Needs to Be Done

Read in this order:
1. **[ROADMAP.md](ROADMAP.md)** - Understand the phases and timeline
2. **[PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md)** - Dive into specific epics and tasks
3. Pick a task that matches your skills and interests

### 3. Pick Your First Task

**For Python Experts:**
- Epic 1: Fix package structure and infrastructure
- Epic 2: Implement MCP server
- Epic 3: Write comprehensive tests

**For Documentation Writers:**
- Epic 4: Write getting started guides
- Epic 4: Complete API reference documentation
- Epic 4: Create MCP integration tutorials

**For DevOps Engineers:**
- Epic 6: Set up CI/CD pipelines
- Epic 9: Implement monitoring and logging
- Epic 11: Configure package distribution

**For Language Experts:**
- Epic 7: Implement JavaScript/TypeScript support
- Epic 7: Implement Java support
- Epic 7: Add other language parsers

---

## 🎯 Current Priorities (Week 1-2)

### Critical Path Tasks

#### Priority 1: Fix Package Structure (2 days)
**Why:** Nothing else works until this is done!

```bash
# What needs to happen:
1. Rename src/ to src/code_scalpel/
2. Fix pyproject.toml configuration
3. Make pip install -e . work
4. Update all imports
```

**Start Here:** See Epic 1, Feature 1.1 in PRODUCT_BACKLOG.md

#### Priority 2: MCP Server Core (3 days)
**Why:** This is the key differentiator for Code Scalpel

```bash
# What needs to happen:
1. Add fastmcp dependency
2. Create MCP server structure
3. Implement basic server
4. Add AST analysis tools
5. Add PDG analysis tools
```

**Start Here:** See Epic 2 in PRODUCT_BACKLOG.md

#### Priority 3: Basic Testing (2 days)
**Why:** Ensure quality as we build

```bash
# What needs to happen:
1. Set up pytest
2. Write tests for core functionality
3. Configure coverage reporting
```

**Start Here:** See Epic 3, Feature 3.1 in PRODUCT_BACKLOG.md

---

## 🛠️ Development Workflow

### Standard Contribution Flow

1. **Create an Issue**
   - Use templates from GITHUB_ISSUES_TEMPLATE.md
   - Add appropriate labels (P0, feature, etc.)
   - Link to epic/feature if applicable

2. **Create a Branch**
   ```bash
   git checkout -b feature/epic1-fix-package-structure
   ```

3. **Make Changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation

4. **Test Locally**
   ```bash
   # Run tests (once pytest is set up)
   pytest tests/
   
   # Run linters (once configured)
   black src/
   flake8 src/
   mypy src/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat(epic1): fix package structure for proper installation"
   git push origin feature/epic1-fix-package-structure
   ```

6. **Create Pull Request**
   - Reference related issues
   - Describe changes clearly
   - Wait for review

### Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(epic2): add MCP server core implementation
fix(epic1): correct import paths after restructure
docs(epic4): add getting started guide
test(epic3): add unit tests for AST analyzer
chore(epic6): set up GitHub Actions for CI
```

---

## 📋 What is Model Context Protocol (MCP)?

Since MCP is central to this project, here's a quick primer:

### What is MCP?

MCP (Model Context Protocol) is a standard protocol by Anthropic that allows AI agents (LLMs) to interact with external tools, data, and services. Think of it as "USB-C for AI" - a universal connector.

### Why MCP for Code Scalpel?

Instead of each AI framework (Autogen, CrewAI, Langchain) needing custom integrations, MCP provides:
- **Standard interface** - Write once, use everywhere
- **Tool discovery** - AI agents can discover available analysis tools
- **Structured calls** - Type-safe tool invocation
- **Resource access** - Read-only access to code patterns, templates, etc.

### MCP Architecture

```
┌─────────────────┐
│   AI Agent      │ (Claude, GPT-4, etc.)
│   (MCP Client)  │
└────────┬────────┘
         │
         │ MCP Protocol (JSON-RPC)
         │
┌────────▼────────┐
│  Code Scalpel   │
│  MCP Server     │
├─────────────────┤
│   Tools:        │
│   - parse_ast   │
│   - build_pdg   │
│   - analyze_deps│
│                 │
│   Resources:    │
│   - patterns    │
│   - templates   │
└─────────────────┘
```

### How AI Agents Will Use Code Scalpel

1. **Discovery**: Agent connects to MCP server, discovers available tools
2. **Invocation**: Agent calls tool (e.g., `parse_code_to_ast(code="...")`)
3. **Analysis**: Code Scalpel performs deep analysis
4. **Response**: Returns structured JSON response
5. **Action**: Agent uses results to provide insights to user

**Example Workflow:**
```
User: "Find all unused functions in this codebase"
  ↓
Agent: Calls Code Scalpel MCP tool `find_dead_code(code=...)`
  ↓
Code Scalpel: Builds PDG, analyzes reachability, returns unused functions
  ↓
Agent: Presents findings to user with explanations
```

---

## 📖 Key Technologies

### Core Technologies
- **Python 3.8+** - Primary implementation language
- **AST** (Abstract Syntax Tree) - Code structure analysis
- **NetworkX** - Graph algorithms for PDG
- **Z3 Solver** - Constraint solving for symbolic execution

### MCP Technologies
- **FastMCP** - Python framework for building MCP servers
- **Pydantic** - Data validation and schemas
- **JSON-RPC 2.0** - Protocol format

### Testing & Quality
- **Pytest** - Testing framework
- **Coverage.py** - Code coverage
- **Black** - Code formatting
- **Flake8** - Linting
- **MyPy** - Type checking

### CI/CD
- **GitHub Actions** - Automation pipeline
- **PyPI** - Package distribution
- **Codecov** - Coverage reporting

---

## 🤔 Common Questions

### Q: Where should I start?
**A:** Start with Epic 1 (Package Infrastructure) if you're comfortable with Python packaging, or Epic 4 (Documentation) if you prefer writing docs.

### Q: Do I need to understand MCP to contribute?
**A:** Not for all tasks! Testing, documentation, and parsers can be worked on independently.

### Q: How long will this take?
**A:** The roadmap estimates 12 weeks for v1.0, but with community help, we can accelerate!

### Q: Can I work on multiple languages?
**A:** Yes! Epic 7 covers multi-language support. We need experts in JavaScript, Java, C++, etc.

### Q: Is there a Discord/Slack?
**A:** Not yet! Use GitHub Discussions for now. (See Epic 12 for community building)

### Q: What if I find a bug not in the backlog?
**A:** Create an issue! The backlog is a living document.

---

## 🎓 Learning Resources

### Model Context Protocol (MCP)
- [Anthropic MCP Documentation](https://modelcontextprotocol.io/)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)

### Abstract Syntax Trees (AST)
- [Green Tree Snakes](https://greentreesnakes.readthedocs.io/) - AST guide
- [Python ast module docs](https://docs.python.org/3/library/ast.html)

### Program Dependence Graphs (PDG)
- [PDG Wikipedia](https://en.wikipedia.org/wiki/Program_dependence_graph)
- NetworkX documentation

### Symbolic Execution
- [Z3 Tutorial](https://ericpony.github.io/z3py-tutorial/)
- [Symbolic Execution Overview](https://en.wikipedia.org/wiki/Symbolic_execution)

---

## 💡 Tips for Success

1. **Start Small** - Pick one task, complete it well
2. **Ask Questions** - Use GitHub Discussions
3. **Read Code** - Understand existing implementation before changing
4. **Test Everything** - Quality over quantity
5. **Document** - Future you will thank you
6. **Communicate** - Update issues with progress
7. **Be Patient** - Building production software takes time

---

## 📞 Get Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and ideas
- **Pull Requests** - For code reviews

---

## 🙏 Thank You!

Code Scalpel is becoming a community project. Every contribution, no matter how small, helps make code analysis better for AI agents worldwide.

**Let's build something amazing together! 🚀**

---

**Last Updated:** 2025-11-10
**Next Steps:** Start with Epic 1, Feature 1.1 - Fix Package Structure
