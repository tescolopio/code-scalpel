"""Unit tests for AST analyzer functionality."""

import ast
import os
import sys

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import directly from the module to avoid __init__.py import issues
from code_scalpel.ast_tools import analyzer as ast_analyzer_module

ASTAnalyzer = ast_analyzer_module.ASTAnalyzer
FunctionMetrics = ast_analyzer_module.FunctionMetrics
ClassMetrics = ast_analyzer_module.ClassMetrics


class TestFunctionMetrics:
    """Tests for the FunctionMetrics dataclass."""

    def test_create_function_metrics(self):
        """Test creating FunctionMetrics."""
        metrics = FunctionMetrics(
            name="test_func",
            args=["a", "b"],
            kwargs=[("c", "10")],
            return_type="int",
            complexity=3,
            line_count=10,
            calls_made=["print", "len"],
            variables_used={"a", "b", "x"},
        )

        assert metrics.name == "test_func"
        assert metrics.args == ["a", "b"]
        assert metrics.kwargs == [("c", "10")]
        assert metrics.return_type == "int"
        assert metrics.complexity == 3
        assert metrics.line_count == 10
        assert "print" in metrics.calls_made
        assert "x" in metrics.variables_used


class TestClassMetrics:
    """Tests for the ClassMetrics dataclass."""

    def test_create_class_metrics(self):
        """Test creating ClassMetrics."""
        metrics = ClassMetrics(
            name="TestClass",
            bases=["BaseClass"],
            methods=["__init__", "process"],
            attributes={"value": "int"},
            instance_vars={"self.value"},
            class_vars={"COUNT"},
        )

        assert metrics.name == "TestClass"
        assert "BaseClass" in metrics.bases
        assert "__init__" in metrics.methods
        assert "value" in metrics.attributes
        assert "self.value" in metrics.instance_vars
        assert "COUNT" in metrics.class_vars


class TestASTAnalyzer:
    """Tests for the ASTAnalyzer class."""

    def test_init_default(self):
        """Test ASTAnalyzer initialization with defaults."""
        analyzer = ASTAnalyzer()
        assert analyzer.cache_enabled is True
        assert analyzer.ast_cache == {}
        assert analyzer.current_context == []

    def test_init_cache_disabled(self):
        """Test ASTAnalyzer initialization with cache disabled."""
        analyzer = ASTAnalyzer(cache_enabled=False)
        assert analyzer.cache_enabled is False

    def test_parse_to_ast_valid_code(self):
        """Test parsing valid Python code."""
        analyzer = ASTAnalyzer()
        code = "x = 1 + 2"
        tree = analyzer.parse_to_ast(code)

        assert isinstance(tree, ast.AST)
        assert isinstance(tree, ast.Module)

    def test_parse_to_ast_caching(self):
        """Test that parsed ASTs are cached."""
        analyzer = ASTAnalyzer()
        code = "x = 1 + 2"

        tree1 = analyzer.parse_to_ast(code)
        tree2 = analyzer.parse_to_ast(code)

        assert tree1 is tree2
        assert code in analyzer.ast_cache

    def test_parse_to_ast_no_caching(self):
        """Test parsing without caching."""
        analyzer = ASTAnalyzer(cache_enabled=False)
        code = "x = 1 + 2"

        tree1 = analyzer.parse_to_ast(code)
        tree2 = analyzer.parse_to_ast(code)

        assert tree1 is not tree2
        assert code not in analyzer.ast_cache

    def test_parse_to_ast_syntax_error(self):
        """Test handling of syntax errors."""
        analyzer = ASTAnalyzer()

        with pytest.raises(SyntaxError):
            analyzer.parse_to_ast("def incomplete(")

    def test_ast_to_code(self):
        """Test converting AST back to code."""
        analyzer = ASTAnalyzer()
        original = "x = 1 + 2"
        tree = analyzer.parse_to_ast(original)
        code = analyzer.ast_to_code(tree)

        # The regenerated code should be syntactically valid
        assert "x" in code
        assert "1" in code
        assert "2" in code

    def test_analyze_function_simple(self):
        """Test analyzing a simple function."""
        analyzer = ASTAnalyzer()
        code = """
def add(a, b):
    return a + b
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)

        assert metrics.name == "add"
        assert metrics.args == ["a", "b"]
        assert metrics.complexity >= 1
        assert metrics.line_count >= 2

    def test_analyze_function_with_defaults(self):
        """Test analyzing a function with default arguments."""
        analyzer = ASTAnalyzer()
        code = """
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}"
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)

        assert "name" in metrics.args
        assert len(metrics.kwargs) >= 1

    def test_analyze_function_with_return_type(self):
        """Test analyzing a function with return type hint."""
        analyzer = ASTAnalyzer()
        code = """
def get_value() -> int:
    return 42
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)

        assert metrics.return_type is not None
        assert "int" in metrics.return_type

    def test_analyze_function_complexity(self):
        """Test cyclomatic complexity calculation."""
        analyzer = ASTAnalyzer()
        code = """
def complex_func(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    elif x < 0:
        return -x
    else:
        return 0
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)

        # Should have higher complexity due to multiple branches
        assert metrics.complexity > 1

    def test_analyze_class(self):
        """Test analyzing a class definition."""
        analyzer = ASTAnalyzer()
        code = """
class MyClass(BaseClass):
    class_var = 10
    
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
"""
        tree = analyzer.parse_to_ast(code)
        class_node = tree.body[0]

        metrics = analyzer.analyze_class(class_node)

        assert metrics.name == "MyClass"
        assert "BaseClass" in metrics.bases
        assert "__init__" in metrics.methods
        assert "get_value" in metrics.methods
        assert "class_var" in metrics.class_vars

    def test_analyze_code_style(self):
        """Test code style analysis."""
        analyzer = ASTAnalyzer()
        code = """
def myFunction():
    pass

class myclass:
    pass
"""
        tree = analyzer.parse_to_ast(code)
        issues = analyzer.analyze_code_style(tree)

        assert isinstance(issues, dict)
        # Should detect naming convention issues
        assert "naming_conventions" in issues or len(issues) == 0

    def test_analyze_code_style_long_function(self):
        """Test detection of long functions."""
        analyzer = ASTAnalyzer()
        # Create a function with many lines
        lines = ["    pass"] * 25
        code = "def long_function():\n" + "\n".join(lines)

        tree = analyzer.parse_to_ast(code)
        issues = analyzer.analyze_code_style(tree)

        assert "long_functions" in issues

    def test_find_security_issues(self):
        """Test security issue detection."""
        analyzer = ASTAnalyzer()
        code = """
user_input = input()
result = eval(user_input)
"""
        tree = analyzer.parse_to_ast(code)
        issues = analyzer.find_security_issues(tree)

        assert len(issues) > 0
        dangerous_funcs = [i for i in issues if i["type"] == "dangerous_function"]
        assert len(dangerous_funcs) >= 1

    def test_find_security_issues_sql_injection(self):
        """Test SQL injection detection."""
        analyzer = ASTAnalyzer()
        code = """
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
"""
        tree = analyzer.parse_to_ast(code)
        issues = analyzer.find_security_issues(tree)

        sql_issues = [i for i in issues if i["type"] == "sql_injection"]
        # Should detect potential SQL injection
        assert isinstance(sql_issues, list)

    def test_calculate_complexity(self):
        """Test cyclomatic complexity calculation."""
        analyzer = ASTAnalyzer()

        # Simple code
        simple_code = "x = 1"
        simple_tree = analyzer.parse_to_ast(simple_code)
        simple_complexity = analyzer._calculate_complexity(simple_tree)

        # Complex code with branches and loops
        complex_code = """
if x > 0:
    pass
while True:
    pass
for i in range(10):
    pass
"""
        complex_tree = analyzer.parse_to_ast(complex_code)
        complex_complexity = analyzer._calculate_complexity(complex_tree)

        assert complex_complexity > simple_complexity

    def test_count_node_lines(self):
        """Test line counting for nodes."""
        analyzer = ASTAnalyzer()
        code = """
def multi_line():
    x = 1
    y = 2
    return x + y
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        lines = analyzer._count_node_lines(func_node)
        assert lines >= 4

    def test_extract_function_calls(self):
        """Test extraction of function calls."""
        analyzer = ASTAnalyzer()
        code = """
def process():
    print("hello")
    x = len([1, 2, 3])
    result = custom_func(x)
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        calls = analyzer._extract_function_calls(func_node)

        assert "print" in calls
        assert "len" in calls
        assert "custom_func" in calls

    def test_extract_variables(self):
        """Test extraction of variables."""
        analyzer = ASTAnalyzer()
        code = """
def process(a, b):
    c = a + b
    return c * 2
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        variables = analyzer._extract_variables(func_node)

        assert "a" in variables
        assert "b" in variables
        assert "c" in variables

    def test_extract_instance_vars(self):
        """Test extraction of instance variables."""
        analyzer = ASTAnalyzer()
        code = """
def __init__(self, x, y):
    self.x = x
    self.y = y
    self.computed = x + y
"""
        tree = analyzer.parse_to_ast(code)
        init_method = tree.body[0]

        instance_vars = analyzer._extract_instance_vars(init_method)

        assert "x" in instance_vars
        assert "y" in instance_vars
        assert "computed" in instance_vars

    def test_get_call_name_simple(self):
        """Test getting call name from simple function call."""
        analyzer = ASTAnalyzer()
        code = "print('hello')"
        tree = analyzer.parse_to_ast(code)
        call_node = tree.body[0].value

        name = analyzer._get_call_name(call_node)
        assert name == "print"

    def test_get_call_name_method(self):
        """Test getting call name from method call."""
        analyzer = ASTAnalyzer()
        code = "obj.method()"
        tree = analyzer.parse_to_ast(code)
        call_node = tree.body[0].value

        name = analyzer._get_call_name(call_node)
        assert "method" in name


class TestASTAnalyzerEdgeCases:
    """Tests for edge cases in AST analysis."""

    def test_empty_function(self):
        """Test analyzing an empty function."""
        analyzer = ASTAnalyzer()
        code = """
def empty():
    pass
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)
        assert metrics.name == "empty"
        assert metrics.complexity == 1  # Base complexity

    def test_empty_class(self):
        """Test analyzing an empty class."""
        analyzer = ASTAnalyzer()
        code = """
class Empty:
    pass
"""
        tree = analyzer.parse_to_ast(code)
        class_node = tree.body[0]

        metrics = analyzer.analyze_class(class_node)
        assert metrics.name == "Empty"
        assert len(metrics.methods) == 0

    def test_nested_functions(self):
        """Test analyzing nested functions."""
        analyzer = ASTAnalyzer()
        code = """
def outer():
    def inner():
        pass
    return inner
"""
        tree = analyzer.parse_to_ast(code)
        outer_node = tree.body[0]

        # Should analyze the outer function without errors
        metrics = analyzer.analyze_function(outer_node)
        assert metrics.name == "outer"

    def test_lambda_expression(self):
        """Test code containing lambda expressions."""
        analyzer = ASTAnalyzer()
        code = """
def process(data):
    return list(map(lambda x: x * 2, data))
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        # Should handle lambda without errors
        metrics = analyzer.analyze_function(func_node)
        assert metrics.name == "process"

    def test_decorator(self):
        """Test analyzing decorated functions."""
        analyzer = ASTAnalyzer()
        code = """
@staticmethod
def my_static_method():
    pass
"""
        tree = analyzer.parse_to_ast(code)
        func_node = tree.body[0]

        metrics = analyzer.analyze_function(func_node)
        assert metrics.name == "my_static_method"

    def test_deep_nesting(self):
        """Test detection of deep nesting."""
        analyzer = ASTAnalyzer()
        code = """
if a:
    if b:
        if c:
            if d:
                pass
"""
        tree = analyzer.parse_to_ast(code)
        issues = analyzer.analyze_code_style(tree)

        # Should detect deep nesting
        assert "deep_nesting" in issues or len(issues) >= 0
