# Code Scalpel 🔪

[![CI Pipeline](https://github.com/tescolopio/code-scalpel/actions/workflows/ci.yml/badge.svg)](https://github.com/tescolopio/code-scalpel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tescolopio/code-scalpel/branch/main/graph/badge.svg)](https://codecov.io/gh/tescolopio/code-scalpel)
[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://badge.fury.io/py/code-scalpel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-v0.2.0%20Redemption-brightgreen.svg)](https://pypi.org/project/code-scalpel/)

Code Scalpel is a precision tool set for AI-driven code analysis and transformation. Using advanced techniques like Abstract Syntax Trees (ASTs), Program Dependence Graphs (PDGs), and Symbolic Execution, Code Scalpel enables AI agents to perform deep analysis and surgical modifications of code with unprecedented accuracy.

> **⚠️ Project Scope:** Code Scalpel is a **Python toolkit and MCP server** designed for use by AI agents and automation systems. It is **NOT** a mobile application, web app, or GUI tool. Target users include AI coding assistants (Cursor, Cline, Claude Desktop), AI agent frameworks (Autogen, CrewAI, Langchain), and DevOps pipelines.

> **✅ Status:** Code Scalpel v0.2.0 "Redemption" is **live on PyPI**. Install with `pip install code-scalpel`. Symbolic Execution now works (Beta)!

## 🌟 Features

### 🔍 Deep Code Analysis
- **AST Analysis**: Parse and analyze code structure with surgical precision
- **Dependency Tracking**: Build and analyze Program Dependence Graphs
- **Dead Code Detection**: Identify and remove unused code segments
- **Security Scanning**: Detect dangerous patterns (eval, exec, SQL injection)

### 🤖 AI Agent Integration
- **Autogen Ready**: Seamless integration with Microsoft's Autogen framework
- **CrewAI Compatible**: Create specialized code analysis crews
- **MCP Server**: Model Context Protocol server for Claude Desktop, Cursor, Cline
- **Extensible**: Easy to integrate with other AI agent frameworks

### 🛠️ Code Surgery Tools
- **Code Review**: Automated, thorough code reviews
- **Optimization**: Identify and implement performance improvements
- **Security Analysis**: Detect potential vulnerabilities
- **Refactoring**: Suggest and apply code improvements

### 🔮 Symbolic Execution (Beta)
> **🆕 v0.2.0:** Symbolic Execution is now functional! Limited to Int/Bool types.

- **Path Analysis**: Explore all execution paths through your code
- **Constraint Solving**: Z3-powered satisfiability checking
- **Input Discovery**: Find inputs that trigger specific conditions

## **How We're Different** 🆚

Code Scalpel builds upon the foundations of static analysis while adding powerful AI agent capabilities. Here's how we compare to [SMAT-Lab/Scalpel](https://github.com/SMAT-Lab/Scalpel):

| Feature | **Code Scalpel** | **SMAT-Lab/Scalpel** |
|---------|-----------------|---------------------|
| **🤖 AI Agents** | ✅ **Yes** — Autogen, CrewAI, Langchain | ❌ No |
| **🔌 MCP Server** | ✅ **Yes** — Model Context Protocol | ❌ No |
| **📊 PDG Analysis** | ✅ **Yes** — Full data/control flow | ✅ Yes |
| **🌳 AST Analysis** | ✅ **Yes** | ✅ Yes |
| **🔍 Symbolic Execution** | ✅ **Beta** — Int/Bool support | ❌ No |
| **📞 Call Graph** | 🚧 Planned | ✅ Yes |
| **🔤 SSA Form** | 🚧 Planned | ✅ Yes |
| **🎯 Target Users** | **AI Agents & Automation** | Research & Academia |
| **🌍 Multi-Language** | 🚧 Planned (JS, Java, Go) | Python only |
| **⚡ Claude Optimized** | ✅ **Yes** — Structured for AI workflows | ❌ No |
| **📦 Package** | `code-scalpel` | `python-scalpel` |

> **💡 Key Insight:** While SMAT-Lab/Scalpel excels as a traditional static analysis framework for research, **Code Scalpel is purpose-built for the AI agent era** — enabling autonomous code analysis, review, and transformation by AI systems like Claude, Cursor, and Cline.

## 🚀 Quick Start

### Installation
```bash
pip install code-scalpel
```

> **Note:** Requires Python 3.9+. For optional AI integrations, install with extras: `pip install code-scalpel[all]`

### Basic Usage
```python
from code_scalpel import CodeAnalyzer

# Initialize the analyzer
analyzer = CodeAnalyzer()

# Analyze code
code = """
def example(x):
    y = x + 1
    if y > 10:
        return y * 2
    return y
"""

# Get analysis results
results = analyzer.analyze(code)
print(results.suggestions)
```

### AI Agent Integration

#### With Autogen
```python
from code_scalpel.integrations import AutogenScalpel

agent = AutogenScalpel(config)
analysis = await agent.analyze_code(code)
```

#### With CrewAI
```python
from code_scalpel.integrations import ScalpelCrew

crew = ScalpelCrew()
crew.add_analysis_task(code)
results = crew.execute()
```

## 📊 Example: Code Analysis Visualization

```python
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()
pdg = analyzer.build_pdg(code)
analyzer.visualize_pdg(pdg, "analysis.png")
```

![PDG Example](docs/images/pdg_example.png)

## 🎯 Use Cases

- **Code Review Automation**: Automate thorough code reviews with AI assistance
- **Technical Debt Analysis**: Identify and prioritize code improvements
- **Security Auditing**: Detect potential security vulnerabilities
- **Performance Optimization**: Find and fix performance bottlenecks
- **Refactoring Planning**: Get intelligent suggestions for code restructuring

## 🛡️ Features in Detail

### AST Analysis
- Function and class structure analysis
- Code complexity metrics
- Pattern matching and anti-pattern detection

### PDG Analysis
- Data flow analysis
- Control flow analysis
- Dependency chain visualization
- Dead code detection

### Symbolic Execution (Beta)
> **🆕 NEW in v0.2.0** - The "Redemption" release brings working symbolic execution!

```python
from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer

analyzer = SymbolicAnalyzer()
result = analyzer.analyze("""
x = 5
if x > 0:
    y = x * 2
else:
    y = -x
""")

print(f"Paths explored: {result.total_paths}")
print(f"Feasible paths: {result.feasible_count}")
```

**Current limitations (Phase 1):**
- Int and Bool types only
- No floating point or string support yet
- Loops bounded to 10 iterations

See [ROADMAP.md](ROADMAP.md) for Phase 2 plans.

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/code-scalpel.git
cd code-scalpel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## 📚 Documentation

Full documentation is available at [code-scalpel.readthedocs.io](https://code-scalpel.readthedocs.io/).

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Integration Guides](docs/integration_guides.md)
- [Examples](docs/examples.md)

## 🗺️ Project Planning & Roadmap

We're transforming Code Scalpel into a production-ready MCP-enabled toolkit:

- **[ROADMAP.md](ROADMAP.md)** - High-level roadmap and timeline (12-week plan to v1.0)
- **[PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md)** - Comprehensive backlog with all epics, features, and tasks
- **[GETTING_STARTED_DEV.md](GETTING_STARTED_DEV.md)** - Developer quick start guide
- **[GITHUB_ISSUES_TEMPLATE.md](GITHUB_ISSUES_TEMPLATE.md)** - Templates for creating issues

### What's Coming?

**Phase 1:** ✅ Complete - Package infrastructure + MCP server + PyPI release (v0.1.0)  
**Phase 2 (Current):** Testing + documentation + hotfixes  
**Phase 3:** Multi-language support + performance optimization  
**Phase 4:** Symbolic execution redemption (v0.2.0)  
**Phase 5:** Community building and ecosystem growth (v1.0.0)  

See [ROADMAP.md](ROADMAP.md) for full details.

### ✅ v0.2.0 "Redemption" Released!

Code Scalpel v0.2.0 is now available on PyPI:

```bash
pip install code-scalpel
```

**What's included:**
- AST Analysis (94% coverage)
- PDG Analysis (86% coverage)
- MCP HTTP Server
- CLI tool (`code-scalpel`)
- AI integrations (Autogen, CrewAI)
- **🆕 Symbolic Execution (Beta)** - 426 tests, 76% coverage

**What's new in v0.2.0:**
- Working `SymbolicAnalyzer` with Z3 constraint solver
- Path exploration with feasibility checking
- Type inference for Int/Bool expressions
- Bounded loop unrolling (max 10 iterations)

### 💬 Community Discussion

> **📌 Featured Discussion:** *Coming soon!* — We're creating a pinned discussion titled **"Code Scalpel: The AI Twist on Static Analysis"** in [GitHub Discussions](https://github.com/tescolopio/code-scalpel/discussions). Join the conversation about how AI agents are transforming code analysis!

## 📝 License

Code Scalpel is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/code-scalpel&type=Date)](https://star-history.com/#yourusername/code-scalpel&Date)

## 📧 Contact

- **GitHub Issues:** For bug reports and feature requests
- **GitHub Discussions:** For questions and community discussion
- **Documentation:** [code-scalpel.readthedocs.io](https://code-scalpel.readthedocs.io/) (coming soon)
- **Project Owner:** Tim Escolopio (3dtsus@gmail.com)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Read the [GETTING_STARTED_DEV.md](GETTING_STARTED_DEV.md) guide
2. Check the [ROADMAP.md](ROADMAP.md) for current priorities
3. Look at [PRODUCT_BACKLOG.md](PRODUCT_BACKLOG.md) for specific tasks
4. Create an issue or comment on existing ones
5. Submit a pull request

**Priority areas for contributors:**
- MCP server implementation
- Testing and test coverage
- Documentation and examples
- Multi-language parser support
- Performance optimization

---
Made with ❤️ by the Code Scalpel Team

**Join us in building the future of AI-driven code analysis!** 🚀