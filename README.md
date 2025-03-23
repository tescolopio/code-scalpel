# Code Scalpel 🔪

[![PyPI version](https://badge.fury.io/py/code-scalpel.svg)](https://badge.fury.io/py/code-scalpel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code Scalpel is a precision tool set for AI-driven code analysis and transformation. Using advanced techniques like Abstract Syntax Trees (ASTs), Program Dependence Graphs (PDGs), and Symbolic Execution, Code Scalpel enables AI agents to perform deep analysis and surgical modifications of code with unprecedented accuracy.

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

## 🚀 Quick Start

### Installation
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

## 📝 License

Code Scalpel is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/code-scalpel&type=Date)](https://star-history.com/#yourusername/code-scalpel&Date)

## 📧 Contact

- GitHub Issues: For bug reports and feature requests
- Documentation: [code-scalpel.readthedocs.io](https://code-scalpel.readthedocs.io/)
- Email: your.email@example.com

---
Made with ❤️ by the Code Scalpel Team