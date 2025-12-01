# Code Scalpel 🔪

[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://badge.fury.io/py/code-scalpel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)](https://github.com/tescolopio/code-scalpel)

Code Scalpel is a precision tool set for AI-driven code analysis and transformation. Using advanced techniques like Abstract Syntax Trees (ASTs), Program Dependence Graphs (PDGs), and Symbolic Execution, Code Scalpel enables AI agents to perform deep analysis and surgical modifications of code with unprecedented accuracy.

> **⚠️ Project Scope:** Code Scalpel is a **Python toolkit and MCP server** designed for use by AI agents and automation systems. It is **NOT** a mobile application, web app, or GUI tool. Target users include AI coding assistants (Cursor, Cline, Claude Desktop), AI agent frameworks (Autogen, CrewAI, Langchain), and DevOps pipelines.

> **🚧 Status:** Code Scalpel is currently being transformed from a development prototype into a production-ready toolkit with full Model Context Protocol (MCP) support. See our [ROADMAP.md](ROADMAP.md) for details.

## 🌟 Features

### 🔍 Deep Code Analysis
- **AST Analysis**: Parse and analyze code structure with surgical precision
- **Dependency Tracking**: Build and analyze Program Dependence Graphs
- **Symbolic Execution**: Understand possible execution paths and constraints
- **Dead Code Detection**: Identify and remove unused code segments

### 🤖 AI Agent Integration
- **Autogen Ready**: Seamless integration with Microsoft's Autogen framework
- **CrewAI Compatible**: Create specialized code analysis crews
- **Claude Optimized**: Structured for effective use with Anthropic's Claude
- **Extensible**: Easy to integrate with other AI agent frameworks

### 🛠️ Code Surgery Tools
- **Code Review**: Automated, thorough code reviews
- **Optimization**: Identify and implement performance improvements
- **Security Analysis**: Detect potential vulnerabilities
- **Refactoring**: Suggest and apply code improvements

## **How We're Different** 🆚

Code Scalpel builds upon the foundations of static analysis while adding powerful AI agent capabilities. Here's how we compare to [SMAT-Lab/Scalpel](https://github.com/SMAT-Lab/Scalpel):

| Feature | **Code Scalpel** | **SMAT-Lab/Scalpel** |
|---------|-----------------|---------------------|
| **🤖 AI Agents** | ✅ **Yes** — Autogen, CrewAI, Langchain | ❌ No |
| **🔌 MCP Server** | 🚧 Planned — Model Context Protocol | ❌ No |
| **🔍 Symbolic Execution** | ✅ **Yes** — Z3-powered path analysis | ❌ No |
| **📊 PDG Analysis** | ✅ **Yes** — Full data/control flow | ✅ Yes |
| **🌳 AST Analysis** | ✅ **Yes** | ✅ Yes |
| **📞 Call Graph** | 🚧 Planned | ✅ Yes |
| **🔤 SSA Form** | 🚧 Planned | ✅ Yes |
| **🎯 Target Users** | **AI Agents & Automation** | Research & Academia |
| **🌍 Multi-Language** | 🚧 Planned (JS, Java, Go) | Python only |
| **⚡ Claude Optimized** | ✅ **Yes** — Structured for AI workflows | ❌ No |
| **📦 Package** | `code-scalpel` | `python-scalpel` |

> **💡 Key Insight:** While SMAT-Lab/Scalpel excels as a traditional static analysis framework for research, **Code Scalpel is purpose-built for the AI agent era** — enabling autonomous code analysis, review, and transformation by AI systems like Claude, Cursor, and Cline.

## 🚀 Quick Start

> **⚠️ Note:** The package is currently being restructured. Installation via pip will not work until Epic 1 is completed. See [ROADMAP.md](ROADMAP.md) for progress.

### Installation (Coming Soon)
```bash
pip install code-scalpel
```

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

### Symbolic Execution
- Path condition analysis
- Constraint solving
- Bug detection
- Test case generation

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

**Phase 1 (Weeks 1-2):** Package infrastructure + MCP server implementation  
**Phase 2 (Weeks 3-4):** Comprehensive testing + documentation  
**Phase 3 (Weeks 5-6):** CI/CD + security + PyPI release  
**Phase 4 (Weeks 7-9):** Multi-language support + performance optimization  
**Phase 5 (Weeks 10+):** Community building and ecosystem growth  

See [ROADMAP.md](ROADMAP.md) for full details.

### 🚀 PyPI Alpha Release Checklist

To release `pip install code-scalpel==0.1.0a1`, the following must be completed:

1. **Package Structure** - Rename `src/` → `src/code_scalpel/` with proper `__init__.py`
2. **pyproject.toml** - Update version to `0.1.0a1`, add classifiers, URLs, and Python version requirements
3. **Dependencies** - Pin versions, add optional dev/test dependencies
4. **Basic Tests** - Core functionality tests passing
5. **Build & Verify** - Successfully build with `python -m build` and test install
6. **PyPI Account** - Configure TestPyPI first, then production PyPI
7. **Upload** - Use `twine upload` to publish

See the PyPI Alpha Release section in [ROADMAP.md](ROADMAP.md) for detailed steps.

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