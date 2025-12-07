# AST Tools - Complete Reference

The `ast_tools` module provides foundational code parsing and analysis capabilities for Code Scalpel.

---

## Table of Contents

1. [Overview](#overview)
2. [Polyglot Support](#polyglot-support)
3. [ASTBuilder](#astbuilder)
4. [ASTAnalyzer](#astanalyzer)
5. [ASTTransformer](#asttransformer)
6. [ASTVisualizer](#astvisualizer)
7. [ASTValidator](#astvalidator)
8. [Utility Functions](#utility-functions)
9. [Data Classes](#data-classes)
10. [Examples](#examples)

---

## Overview

The AST Tools module wraps Python's `ast` module with enhancements for:

- **Preprocessing**: Handle edge cases in source code
- **Validation**: Verify AST integrity
- **Metrics**: Extract function/class statistics
- **Transformation**: Modify AST nodes programmatically
- **Visualization**: Generate AST diagrams

### Quick Start

```python
from code_scalpel.ast_tools import build_ast, ASTAnalyzer

# Parse code
code = """
def greet(name):
    return f"Hello, {name}!"
"""
tree = build_ast(code)

# Analyze
analyzer = ASTAnalyzer(tree)
metrics = analyzer.analyze_function("greet")
print(f"Complexity: {metrics.cyclomatic_complexity}")
```

---

## Polyglot Support

Code Scalpel supports multiple programming languages through its **Unified IR (Intermediate Representation)** system. While Python has full native support, other languages use tree-sitter for parsing and normalize to the same IR.

### Supported Languages

| Language | Parser | Support Level | Since |
|----------|--------|---------------|-------|
| **Python** | `ast` (native) | Full | v0.1.0 |
| **Java** | tree-sitter | Structural | v0.3.0 |
| **JavaScript** | tree-sitter | Structural | v0.3.0 |

### Language Feature Matrix

| Feature | Python | Java | JavaScript |
|---------|--------|------|------------|
| Function extraction | ✅ | ✅ | ✅ |
| Class extraction | ✅ | ✅ | ✅ |
| Control flow analysis | ✅ | ✅ | ✅ |
| Type inference | ✅ | ⚠️ | ⚠️ |
| Symbolic execution | ✅ | ❌ | ❌ |
| Security scanning | ✅ | ❌ | ❌ |

**Legend:** ✅ Full support | ⚠️ Partial support | ❌ Not yet available

---

### Java Parsing

Java support uses tree-sitter for parsing. The `JavaNormalizer` converts Java CST (Concrete Syntax Tree) to Code Scalpel's Unified IR.

#### Installation

```bash
pip install tree-sitter tree-sitter-java
```

#### Usage

```python
from code_scalpel.ir.normalizers import JavaNormalizer

java_code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}
'''

normalizer = JavaNormalizer()
ir_module = normalizer.normalize(java_code)

# Extract functions
for node in ir_module.body:
    if hasattr(node, 'name'):
        print(f"Found: {node.name}")
# Output:
# Found: Calculator
# Found: add
# Found: factorial
```

#### Supported Java Constructs

| Construct | IR Node | Notes |
|-----------|---------|-------|
| Class declarations | `IRClassDef` | Includes methods |
| Method declarations | `IRFunctionDef` | Parameters extracted |
| If statements | `IRIf` | With else branches |
| While loops | `IRWhile` | Basic loop support |
| Return statements | `IRReturn` | Value extraction |
| Binary expressions | `IRBinOp` | Arithmetic, logical |
| Method invocations | `IRCall` | Function calls |
| Variable declarations | `IRAssign` | With initializers |
| Literals | `IRConstant` | int, string, boolean |

#### Limitations

- **No symbolic execution:** Java code cannot be symbolically executed (Python only)
- **No security scanning:** Taint analysis not available for Java
- **No type annotations:** Type information not extracted from Java generics

---

### JavaScript Parsing

JavaScript support uses tree-sitter for parsing. ES6+ syntax is supported.

#### Installation

```bash
pip install tree-sitter tree-sitter-javascript
```

#### Usage

```python
from code_scalpel.ir.normalizers import JavaScriptNormalizer

js_code = '''
function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
'''

normalizer = JavaScriptNormalizer()
ir_module = normalizer.normalize(js_code)

for node in ir_module.body:
    if hasattr(node, 'name'):
        print(f"Found: {node.name}")
```

---

### MCP Server Multi-Language Support

The MCP server's `analyze_code` tool accepts a `language` parameter:

```json
{
  "name": "analyze_code",
  "arguments": {
    "code": "public class Foo { }",
    "language": "java"
  }
}
```

**Supported values:** `"python"` (default), `"java"`, `"javascript"`

---

## ASTBuilder

The `ASTBuilder` class handles code parsing with preprocessing and validation.

### Constructor

```python
class ASTBuilder:
    def __init__(
        self,
        preprocess: bool = True,
        validate: bool = True,
        encoding: str = "utf-8"
    ):
        """
        Args:
            preprocess: Apply preprocessing to handle edge cases
            validate: Validate the resulting AST
            encoding: Character encoding for file reading
        """
```

### Methods

#### build_ast

```python
def build_ast(
    self,
    code: str,
    preprocess: bool = None,
    validate: bool = None
) -> ast.Module:
    """
    Build an AST from source code string.
    
    Args:
        code: Python source code
        preprocess: Override default preprocessing (optional)
        validate: Override default validation (optional)
        
    Returns:
        ast.Module: Parsed AST
        
    Raises:
        SyntaxError: If code cannot be parsed
    """
```

**Example:**

```python
builder = ASTBuilder()

# Normal parsing
tree = builder.build_ast("def foo(): pass")

# Raw parsing (no preprocessing)
tree = builder.build_ast(code, preprocess=False)

# Skip validation
tree = builder.build_ast(code, validate=False)
```

#### build_ast_from_file

```python
def build_ast_from_file(
    self,
    filepath: str,
    preprocess: bool = None,
    validate: bool = None
) -> ast.Module:
    """
    Build an AST from a source file.
    
    Args:
        filepath: Path to Python source file
        preprocess: Override default preprocessing
        validate: Override default validation
        
    Returns:
        ast.Module: Parsed AST
        
    Raises:
        FileNotFoundError: If file doesn't exist
        SyntaxError: If code cannot be parsed
    """
```

**Example:**

```python
builder = ASTBuilder()

# Parse file
tree = builder.build_ast_from_file("my_module.py")

# Parse with specific encoding
builder = ASTBuilder(encoding="latin-1")
tree = builder.build_ast_from_file("legacy_code.py")
```

### Preprocessing

The builder applies these preprocessing steps:

1. **Normalize line endings**: `\r\n` → `\n`
2. **Handle BOM**: Remove UTF-8 BOM if present
3. **Fix trailing whitespace**: Ensure newline at end
4. **Escape handling**: Normalize escape sequences

To disable preprocessing:

```python
tree = build_ast(code, preprocess=False)
```

---

## ASTAnalyzer

The `ASTAnalyzer` class extracts metrics and structural information from ASTs.

### Constructor

```python
class ASTAnalyzer:
    def __init__(self, tree: ast.Module):
        """
        Args:
            tree: Parsed AST module
        """
```

### Methods

#### analyze_function

```python
def analyze_function(self, function_name: str) -> FunctionMetrics:
    """
    Analyze a specific function.
    
    Args:
        function_name: Name of the function to analyze
        
    Returns:
        FunctionMetrics: Detailed function metrics
        
    Raises:
        ValueError: If function not found
    """
```

**Example:**

```python
code = """
def calculate(x, y, operation="add"):
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    else:
        raise ValueError(f"Unknown: {operation}")
"""

tree = build_ast(code)
analyzer = ASTAnalyzer(tree)
metrics = analyzer.analyze_function("calculate")

print(f"Parameters: {metrics.parameter_count}")  # 3
print(f"Lines: {metrics.lines_of_code}")         # 7
print(f"Complexity: {metrics.cyclomatic_complexity}")  # 3
print(f"Has docstring: {metrics.has_docstring}")  # False
```

#### analyze_class

```python
def analyze_class(self, class_name: str) -> ClassMetrics:
    """
    Analyze a specific class.
    
    Args:
        class_name: Name of the class to analyze
        
    Returns:
        ClassMetrics: Detailed class metrics
    """
```

**Example:**

```python
code = """
class Calculator:
    '''A simple calculator.'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result
    
    def clear(self):
        self.history = []
"""

analyzer = ASTAnalyzer(build_ast(code))
metrics = analyzer.analyze_class("Calculator")

print(f"Methods: {metrics.method_count}")       # 3
print(f"Attributes: {metrics.attribute_count}") # 1 (history)
print(f"Has docstring: {metrics.has_docstring}")  # True
print(f"Base classes: {metrics.base_classes}")  # []
```

#### get_all_functions

```python
def get_all_functions(self) -> list[str]:
    """Get names of all functions (including nested)."""
```

#### get_all_classes

```python
def get_all_classes(self) -> list[str]:
    """Get names of all classes (including nested)."""
```

#### get_imports

```python
def get_imports(self) -> list[str]:
    """Get all import statements as strings."""
```

#### get_global_variables

```python
def get_global_variables(self) -> list[str]:
    """Get names of all module-level variables."""
```

#### calculate_total_complexity

```python
def calculate_total_complexity(self) -> int:
    """Calculate cyclomatic complexity for entire module."""
```

**Example:**

```python
analyzer = ASTAnalyzer(tree)

print(f"Functions: {analyzer.get_all_functions()}")
print(f"Classes: {analyzer.get_all_classes()}")
print(f"Imports: {analyzer.get_imports()}")
print(f"Total complexity: {analyzer.calculate_total_complexity()}")
```

---

## ASTTransformer

Transform AST nodes programmatically.

### Methods

#### rename_function

```python
def rename_function(
    self,
    tree: ast.Module,
    old_name: str,
    new_name: str
) -> ast.Module:
    """Rename a function and all its call sites."""
```

**Example:**

```python
from code_scalpel.ast_tools import ASTTransformer

code = """
def old_name(x):
    return x * 2

result = old_name(5)
"""

transformer = ASTTransformer()
new_tree = transformer.rename_function(
    build_ast(code),
    "old_name",
    "new_name"
)

import ast
print(ast.unparse(new_tree))
# def new_name(x):
#     return x * 2
# result = new_name(5)
```

#### remove_function

```python
def remove_function(
    self,
    tree: ast.Module,
    function_name: str
) -> ast.Module:
    """Remove a function definition from the AST."""
```

#### add_import

```python
def add_import(
    self,
    tree: ast.Module,
    module: str,
    names: list[str] = None
) -> ast.Module:
    """
    Add an import statement.
    
    Args:
        tree: AST to modify
        module: Module name (e.g., "os.path")
        names: Specific names to import (e.g., ["join", "exists"])
    """
```

**Example:**

```python
# Add: import os
new_tree = transformer.add_import(tree, "os")

# Add: from os.path import join, exists
new_tree = transformer.add_import(
    tree,
    "os.path",
    names=["join", "exists"]
)
```

#### wrap_in_try_except

```python
def wrap_in_try_except(
    self,
    tree: ast.Module,
    function_name: str,
    exception_type: str = "Exception"
) -> ast.Module:
    """Wrap a function's body in try-except."""
```

---

## ASTVisualizer

Generate visual representations of ASTs.

### Methods

#### visualize

```python
def visualize(
    self,
    tree: ast.Module,
    output_file: str = "ast_visualization",
    format: str = "png",
    view: bool = True
) -> str:
    """
    Generate AST visualization.
    
    Args:
        tree: AST to visualize
        output_file: Base filename (without extension)
        format: Output format (png, svg, pdf, dot)
        view: Open visualization after generation
        
    Returns:
        str: Path to generated file
        
    Requires:
        graphviz package and system binary
    """
```

**Example:**

```python
from code_scalpel.ast_tools import visualize_ast, build_ast

code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

tree = build_ast(code)

# Generate PNG
visualize_ast(tree, "factorial_ast", format="png")

# Generate SVG (scalable)
visualize_ast(tree, "factorial_ast", format="svg")

# Just DOT file (no graphviz binary needed)
visualize_ast(tree, "factorial_ast", format="dot", view=False)
```

### Convenience Function

```python
from code_scalpel.ast_tools import visualize_ast

# Quick visualization
visualize_ast(tree, output_file="my_ast")
```

---

## ASTValidator

Validate AST structure and detect issues.

### Methods

#### validate

```python
def validate(self, tree: ast.Module) -> list[ValidationIssue]:
    """
    Validate an AST for structural issues.
    
    Returns:
        List of validation issues found
    """
```

#### check_syntax

```python
def check_syntax(self, code: str) -> tuple[bool, str | None]:
    """
    Check if code has valid Python syntax.
    
    Returns:
        (is_valid, error_message)
    """
```

**Example:**

```python
from code_scalpel.ast_tools import ASTValidator

validator = ASTValidator()

# Check syntax
is_valid, error = validator.check_syntax("def foo(")
if not is_valid:
    print(f"Syntax error: {error}")

# Validate AST
issues = validator.validate(tree)
for issue in issues:
    print(f"[{issue.severity}] Line {issue.line}: {issue.message}")
```

---

## Utility Functions

### build_ast

```python
def build_ast(
    code: str,
    preprocess: bool = True,
    validate: bool = True
) -> ast.Module:
    """
    Convenience function to build AST from code string.
    Uses a default ASTBuilder instance.
    """
```

### build_ast_from_file

```python
def build_ast_from_file(
    filepath: str,
    preprocess: bool = True,
    validate: bool = True
) -> ast.Module:
    """
    Convenience function to build AST from file.
    Uses a default ASTBuilder instance.
    """
```

### visualize_ast

```python
def visualize_ast(
    tree: ast.Module,
    output_file: str = "ast_visualization",
    format: str = "png",
    view: bool = True
) -> str:
    """
    Convenience function for AST visualization.
    """
```

### Helper Functions

```python
from code_scalpel.ast_tools import (
    get_all_names,    # Get all Name nodes in AST
    get_node_type,    # Get string type of AST node
    is_constant,      # Check if node is a constant
)

# Get all variable names
names = get_all_names(tree)

# Get node type
node_type = get_node_type(tree.body[0])  # "FunctionDef"

# Check if constant
is_const = is_constant(ast.Constant(value=42))  # True
```

---

## Data Classes

### FunctionMetrics

```python
@dataclass
class FunctionMetrics:
    name: str                    # Function name
    lines_of_code: int          # Line count
    parameter_count: int        # Number of parameters
    cyclomatic_complexity: int  # McCabe complexity
    cognitive_complexity: int   # Cognitive complexity
    has_docstring: bool         # Whether docstring exists
    docstring: str | None       # Docstring content
    is_async: bool              # async def?
    decorators: list[str]       # Decorator names
    return_type: str | None     # Return type annotation
    start_line: int             # Starting line number
    end_line: int               # Ending line number
```

### ClassMetrics

```python
@dataclass
class ClassMetrics:
    name: str                   # Class name
    lines_of_code: int         # Line count
    method_count: int          # Number of methods
    attribute_count: int       # Number of attributes
    has_docstring: bool        # Whether docstring exists
    docstring: str | None      # Docstring content
    base_classes: list[str]    # Parent class names
    decorators: list[str]      # Decorator names
    is_dataclass: bool         # @dataclass decorated?
    start_line: int            # Starting line number
    end_line: int              # Ending line number
```

---

## Examples

### Example 1: Code Quality Report

```python
from code_scalpel.ast_tools import build_ast_from_file, ASTAnalyzer

def generate_quality_report(filepath: str) -> dict:
    """Generate a code quality report for a Python file."""
    tree = build_ast_from_file(filepath)
    analyzer = ASTAnalyzer(tree)
    
    functions = analyzer.get_all_functions()
    classes = analyzer.get_all_classes()
    
    report = {
        "file": filepath,
        "total_functions": len(functions),
        "total_classes": len(classes),
        "total_complexity": analyzer.calculate_total_complexity(),
        "functions": [],
        "classes": [],
    }
    
    for func_name in functions:
        metrics = analyzer.analyze_function(func_name)
        report["functions"].append({
            "name": func_name,
            "complexity": metrics.cyclomatic_complexity,
            "lines": metrics.lines_of_code,
            "has_docstring": metrics.has_docstring,
        })
    
    for class_name in classes:
        metrics = analyzer.analyze_class(class_name)
        report["classes"].append({
            "name": class_name,
            "methods": metrics.method_count,
            "has_docstring": metrics.has_docstring,
        })
    
    return report

# Usage
report = generate_quality_report("my_module.py")
print(f"Total complexity: {report['total_complexity']}")

for func in report["functions"]:
    if func["complexity"] > 10:
        print(f"⚠️  {func['name']} has high complexity: {func['complexity']}")
```

### Example 2: Batch Rename

```python
from code_scalpel.ast_tools import (
    build_ast_from_file,
    ASTTransformer
)
import ast

def batch_rename_functions(filepath: str, renames: dict[str, str]) -> str:
    """
    Rename multiple functions in a file.
    
    Args:
        filepath: Path to source file
        renames: Mapping of old_name -> new_name
        
    Returns:
        Modified source code
    """
    tree = build_ast_from_file(filepath)
    transformer = ASTTransformer()
    
    for old_name, new_name in renames.items():
        tree = transformer.rename_function(tree, old_name, new_name)
    
    return ast.unparse(tree)

# Usage
new_code = batch_rename_functions("utils.py", {
    "calc": "calculate",
    "fmt": "format_output",
    "chk": "validate",
})

# Write to new file
with open("utils_renamed.py", "w") as f:
    f.write(new_code)
```

### Example 3: Find Complex Functions

```python
from code_scalpel.ast_tools import build_ast_from_file, ASTAnalyzer
from pathlib import Path

def find_complex_functions(
    directory: str,
    threshold: int = 10
) -> list[dict]:
    """Find all functions exceeding complexity threshold."""
    results = []
    
    for path in Path(directory).rglob("*.py"):
        try:
            tree = build_ast_from_file(str(path))
            analyzer = ASTAnalyzer(tree)
            
            for func_name in analyzer.get_all_functions():
                metrics = analyzer.analyze_function(func_name)
                if metrics.cyclomatic_complexity > threshold:
                    results.append({
                        "file": str(path),
                        "function": func_name,
                        "complexity": metrics.cyclomatic_complexity,
                        "line": metrics.start_line,
                    })
        except SyntaxError:
            continue  # Skip files with syntax errors
    
    return sorted(results, key=lambda x: x["complexity"], reverse=True)

# Usage
complex_funcs = find_complex_functions("src/", threshold=15)
for item in complex_funcs[:10]:
    print(f"{item['file']}:{item['line']} - {item['function']} "
          f"(complexity: {item['complexity']})")
```

---

## Error Handling

### Common Exceptions

```python
from code_scalpel.ast_tools import build_ast, build_ast_from_file

# SyntaxError for invalid code
try:
    tree = build_ast("def foo(")
except SyntaxError as e:
    print(f"Parse error: {e}")

# FileNotFoundError for missing files
try:
    tree = build_ast_from_file("nonexistent.py")
except FileNotFoundError:
    print("File not found")

# ValueError for missing functions/classes
try:
    metrics = analyzer.analyze_function("nonexistent")
except ValueError as e:
    print(f"Not found: {e}")
```

### Graceful Degradation

```python
from code_scalpel.ast_tools import ASTVisualizer

# Visualizer might not be available
if ASTVisualizer is not None:
    visualizer = ASTVisualizer()
    visualizer.visualize(tree, "output")
else:
    print("Visualization not available (install graphviz)")
```

---

*AST Tools - The foundation of Code Scalpel's analysis capabilities.*
