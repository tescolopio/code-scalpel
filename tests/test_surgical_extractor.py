"""Tests for the Surgical Extractor module.

Tests the precision extraction capabilities for token-efficient LLM interactions.
"""

import pytest
from code_scalpel.surgical_extractor import (
    SurgicalExtractor,
    ExtractionResult,
    ContextualExtraction,
    extract_function,
    extract_class,
    extract_method,
    extract_with_context,
)


class TestSurgicalExtractor:
    """Tests for the SurgicalExtractor class."""

    @pytest.fixture
    def sample_code(self):
        """Sample code with functions, classes, and dependencies."""
        return '''
import os
from typing import List

CONSTANT = 42

def helper():
    """A helper function."""
    return CONSTANT

def calculate(x, y):
    """Calculate using helper."""
    return helper() + x + y

def standalone():
    """No dependencies."""
    return 1 + 1

class Calculator:
    """A calculator class."""
    
    def __init__(self, value=0):
        self.value = value
    
    def add(self, x):
        """Add to value."""
        self.value += x
        return self.value
    
    def compute(self, x, y):
        """Use external function."""
        return calculate(x, y)

class AdvancedCalculator(Calculator):
    """Extended calculator."""
    
    def multiply(self, x):
        self.value *= x
        return self.value
'''

    def test_list_functions(self, sample_code):
        """Test listing all functions."""
        extractor = SurgicalExtractor(sample_code)
        functions = extractor.list_functions()
        
        assert "helper" in functions
        assert "calculate" in functions
        assert "standalone" in functions
        assert len(functions) == 3

    def test_list_classes(self, sample_code):
        """Test listing all classes."""
        extractor = SurgicalExtractor(sample_code)
        classes = extractor.list_classes()
        
        assert "Calculator" in classes
        assert "AdvancedCalculator" in classes
        assert len(classes) == 2

    def test_list_methods(self, sample_code):
        """Test listing methods of a class."""
        extractor = SurgicalExtractor(sample_code)
        methods = extractor.list_methods("Calculator")
        
        assert "__init__" in methods
        assert "add" in methods
        assert "compute" in methods
        assert len(methods) == 3

    def test_list_methods_nonexistent_class(self, sample_code):
        """Test listing methods of non-existent class."""
        extractor = SurgicalExtractor(sample_code)
        methods = extractor.list_methods("NonExistent")
        assert methods == []

    def test_get_function_success(self, sample_code):
        """Test extracting a function."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_function("helper")
        
        assert result.success is True
        assert result.name == "helper"
        assert "def helper" in result.code
        assert "return CONSTANT" in result.code
        assert result.node_type == "function"
        assert result.line_start > 0
        assert "CONSTANT" in result.dependencies

    def test_get_function_not_found(self, sample_code):
        """Test extracting non-existent function."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_function("nonexistent")
        
        assert result.success is False
        assert "not found" in result.error
        assert "Available:" in result.error

    def test_get_function_finds_dependencies(self, sample_code):
        """Test that dependencies are identified."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_function("calculate")
        
        assert result.success is True
        assert "helper" in result.dependencies

    def test_get_class_success(self, sample_code):
        """Test extracting a class."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_class("Calculator")
        
        assert result.success is True
        assert result.name == "Calculator"
        assert "class Calculator" in result.code
        assert "def add" in result.code
        assert result.node_type == "class"

    def test_get_class_not_found(self, sample_code):
        """Test extracting non-existent class."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_class("NonExistent")
        
        assert result.success is False
        assert "not found" in result.error

    def test_get_method_success(self, sample_code):
        """Test extracting a method."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_method("Calculator", "add")
        
        assert result.success is True
        assert result.name == "Calculator.add"
        assert "def add" in result.code
        assert result.node_type == "method"

    def test_get_method_class_not_found(self, sample_code):
        """Test extracting method from non-existent class."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_method("NonExistent", "add")
        
        assert result.success is False
        assert "Class" in result.error
        assert "not found" in result.error

    def test_get_method_method_not_found(self, sample_code):
        """Test extracting non-existent method."""
        extractor = SurgicalExtractor(sample_code)
        result = extractor.get_method("Calculator", "nonexistent")
        
        assert result.success is False
        assert "Method" in result.error
        assert "Available:" in result.error


class TestContextualExtraction:
    """Tests for extraction with context (dependencies)."""

    @pytest.fixture
    def dependency_code(self):
        """Code with a chain of dependencies."""
        return '''
import math

FACTOR = 10

def base():
    return FACTOR

def middle():
    return base() * 2

def top():
    return middle() + 1
'''

    def test_get_function_with_context(self, dependency_code):
        """Test extracting function with dependencies."""
        extractor = SurgicalExtractor(dependency_code)
        result = extractor.get_function_with_context("top")
        
        assert result.target.success is True
        assert "middle" in result.context_items
        # base is a dependency of middle
        assert "base" in result.context_items
        assert result.total_lines > 0
        
        # Full code should include all pieces
        full = result.full_code
        assert "def top" in full
        assert "def middle" in full
        assert "def base" in full

    def test_context_depth_limit(self, dependency_code):
        """Test that depth limit is respected."""
        extractor = SurgicalExtractor(dependency_code)
        
        # With depth 1, should get middle but maybe not base
        result = extractor.get_function_with_context("top", max_depth=1)
        assert result.target.success is True
        assert "middle" in result.context_items

    def test_context_includes_imports(self, dependency_code):
        """Test that required imports are included."""
        code_with_import = '''
import json

def parse_data(data):
    return json.loads(data)
'''
        extractor = SurgicalExtractor(code_with_import)
        result = extractor.get_function_with_context("parse_data")
        
        assert result.target.success is True
        assert "import json" in result.target.imports_needed

    def test_get_class_with_context(self):
        """Test extracting class with dependencies."""
        code = '''
def helper():
    return 42

class MyClass:
    def method(self):
        return helper()
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_class_with_context("MyClass")
        
        assert result.target.success is True
        assert "helper" in result.context_items

    def test_token_estimate(self):
        """Test token estimation."""
        code = "def foo(): return 42"
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("foo")
        
        # Token estimate is roughly chars / 4
        assert result.token_estimate > 0
        assert result.token_estimate < len(result.code)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_function(self):
        """Test extract_function convenience function."""
        code = "def hello(): return 'world'"
        result = extract_function(code, "hello")
        
        assert result.success is True
        assert "def hello" in result.code

    def test_extract_class(self):
        """Test extract_class convenience function."""
        code = "class Foo: pass"
        result = extract_class(code, "Foo")
        
        assert result.success is True
        assert "class Foo" in result.code

    def test_extract_method(self):
        """Test extract_method convenience function."""
        code = '''
class Bar:
    def baz(self):
        return 1
'''
        result = extract_method(code, "Bar", "baz")
        
        assert result.success is True
        assert "def baz" in result.code

    def test_extract_with_context_function(self):
        """Test extract_with_context for function."""
        code = '''
def a(): return 1
def b(): return a() + 1
'''
        result = extract_with_context(code, "b", "function")
        
        assert result.target.success is True
        assert "a" in result.context_items

    def test_extract_with_context_class(self):
        """Test extract_with_context for class."""
        code = '''
def helper(): return 42
class MyClass:
    def method(self): return helper()
'''
        result = extract_with_context(code, "MyClass", "class")
        
        assert result.target.success is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_python_code(self):
        """Test handling of invalid Python code."""
        extractor = SurgicalExtractor("def broken(")
        with pytest.raises(ValueError, match="Invalid Python code"):
            extractor.list_functions()  # Error on first operation

    def test_empty_code(self):
        """Test handling of empty code."""
        extractor = SurgicalExtractor("")
        assert extractor.list_functions() == []
        assert extractor.list_classes() == []

    def test_only_imports(self):
        """Test code with only imports."""
        code = "import os\nfrom sys import path"
        extractor = SurgicalExtractor(code)
        
        assert extractor.list_functions() == []
        assert extractor.list_classes() == []

    def test_async_function(self):
        """Test extraction of async functions."""
        code = '''
async def fetch():
    return await something()
'''
        extractor = SurgicalExtractor(code)
        functions = extractor.list_functions()
        
        assert "fetch" in functions
        
        result = extractor.get_function("fetch")
        assert result.success is True
        assert "async def fetch" in result.code

    def test_decorated_function(self):
        """Test extraction of decorated functions."""
        code = '''
@decorator
@another
def decorated():
    pass
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("decorated")
        
        assert result.success is True
        # Decorators should be included
        assert "decorator" in result.code or "def decorated" in result.code

    def test_nested_class(self):
        """Test that nested classes don't interfere."""
        code = '''
class Outer:
    class Inner:
        pass
    
    def method(self):
        pass
'''
        extractor = SurgicalExtractor(code)
        # Only top-level class should be listed
        classes = extractor.list_classes()
        assert "Outer" in classes
        # Inner is not top-level
        
        result = extractor.get_class("Outer")
        assert result.success is True
        assert "class Inner" in result.code

    def test_global_constants_as_dependencies(self):
        """Test that global constants are identified as dependencies."""
        code = '''
CONFIG = {"key": "value"}

def use_config():
    return CONFIG["key"]
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("use_config")
        
        assert result.success is True
        assert "CONFIG" in result.dependencies

    def test_builtin_functions_not_dependencies(self):
        """Test that builtins aren't listed as dependencies."""
        code = '''
def example():
    return len([1, 2, 3]) + sum([1, 2])
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("example")
        
        assert "len" not in result.dependencies
        assert "sum" not in result.dependencies

    def test_self_not_dependency(self):
        """Test that 'self' is not a dependency."""
        code = '''
class MyClass:
    def method(self):
        return self.value
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_method("MyClass", "method")
        
        assert "self" not in result.dependencies

    def test_loop_variable_not_dependency(self):
        """Test that loop variables aren't false dependencies."""
        code = '''
def iterate():
    for i in range(10):
        print(i)
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("iterate")
        
        # 'i' is defined in the loop, not a dependency
        assert "i" not in result.dependencies

    def test_comprehension_variable_not_dependency(self):
        """Test that comprehension variables handling."""
        code = '''
def comprehend(items):
    return [item * 2 for item in items]
'''
        extractor = SurgicalExtractor(code)
        result = extractor.get_function("comprehend")
        
        # 'items' is a parameter, not a dependency
        # 'item' is the comprehension variable
        # Neither should appear as external dependencies
        assert "items" not in result.dependencies
        assert "item" not in result.dependencies


class TestTokenSavingsScenario:
    """Tests demonstrating the token-saving value proposition."""

    def test_token_savings_calculation(self):
        """Demonstrate token savings with extraction."""
        # Simulate a large file
        large_code = '''
import os
import sys
from typing import List, Dict

CONFIG = {"debug": True}

def utility_1():
    return 1

def utility_2():
    return 2

def utility_3():
    return 3

def utility_4():
    return 4

def utility_5():
    return 5

def target_function():
    """This is what the LLM needs to fix."""
    return utility_1() + utility_2()

class UnrelatedClass:
    def __init__(self):
        self.value = 0
    
    def method_1(self):
        return self.value
    
    def method_2(self):
        return self.value * 2
    
    def method_3(self):
        return self.value * 3
'''
        total_chars = len(large_code)
        
        extractor = SurgicalExtractor(large_code)
        result = extractor.get_function_with_context("target_function")
        
        extracted_chars = len(result.full_code)
        
        # The extraction should be significantly smaller
        savings_percent = (1 - extracted_chars / total_chars) * 100
        
        # We expect at least 50% token savings
        assert savings_percent > 50, f"Expected >50% savings, got {savings_percent:.1f}%"
        
        # But the extraction should include the dependencies
        assert "utility_1" in result.context_items
        assert "utility_2" in result.context_items
        # And NOT include unrelated code
        assert "UnrelatedClass" not in result.full_code
        assert "utility_5" not in result.full_code


class TestSurgicalExtractorFromFile:
    """Tests for file-based extraction - the TOKEN-EFFICIENT path."""

    def test_from_file_basic(self, tmp_path):
        """Test creating extractor from a file."""
        file_content = '''
def hello():
    return "Hello, World!"
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(file_content)

        extractor = SurgicalExtractor.from_file(str(test_file))
        result = extractor.get_function("hello")

        assert result.success is True
        assert result.name == "hello"
        assert "Hello, World!" in result.code

    def test_from_file_stores_path(self, tmp_path):
        """Test that file_path is stored for potential cross-file resolution."""
        test_file = tmp_path / "module.py"
        test_file.write_text("def foo(): pass")

        extractor = SurgicalExtractor.from_file(str(test_file))
        
        assert extractor.file_path == str(test_file)

    def test_from_file_not_found(self):
        """Test FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            SurgicalExtractor.from_file("/nonexistent/path/to/file.py")

    def test_from_file_invalid_python(self, tmp_path):
        """Test ValueError for files with syntax errors."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def broken(")

        extractor = SurgicalExtractor.from_file(str(test_file))
        # The error happens at parse time, not at load time
        with pytest.raises(ValueError):
            extractor.get_function("broken")

    def test_from_file_utf8_encoding(self, tmp_path):
        """Test reading files with UTF-8 encoding (default)."""
        file_content = '''
def greet():
    return "Héllo Wörld! 日本語"
'''
        test_file = tmp_path / "unicode.py"
        test_file.write_text(file_content, encoding="utf-8")

        extractor = SurgicalExtractor.from_file(str(test_file))
        result = extractor.get_function("greet")

        assert result.success is True
        assert "Héllo" in result.code
        assert "日本語" in result.code

    def test_from_file_with_context(self, tmp_path):
        """Test extracting with context from a file."""
        file_content = '''
def helper():
    return 42

def main():
    return helper() + 1

def unrelated():
    return 0
'''
        test_file = tmp_path / "service.py"
        test_file.write_text(file_content)

        extractor = SurgicalExtractor.from_file(str(test_file))
        result = extractor.get_function_with_context("main")

        assert result.target.success is True
        assert "helper" in result.context_items
        assert "unrelated" not in result.context_items

    def test_from_file_token_efficiency(self, tmp_path):
        """Demonstrate that file-based extraction is token-efficient."""
        # Create a "large" file with many lines
        lines = []
        lines.append("# Large file header")
        for i in range(50):
            lines.append(f"def unused_{i}(): pass")
        lines.append("")
        lines.append("def target():")
        lines.append("    return 'needle in haystack'")
        lines.append("")
        for i in range(50, 100):
            lines.append(f"def unused_{i}(): pass")
        
        file_content = "\n".join(lines)
        test_file = tmp_path / "large.py"
        test_file.write_text(file_content)

        # File has ~100 functions, but we only want 1
        extractor = SurgicalExtractor.from_file(str(test_file))
        result = extractor.get_function("target")

        assert result.success is True
        assert "needle in haystack" in result.code
        # The extraction should be tiny
        assert result.token_estimate < 20  # ~2 lines
        # None of the unused functions should appear
        assert "unused_" not in result.code

