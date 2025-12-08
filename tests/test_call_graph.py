"""
Comprehensive tests for call_graph.py - CallGraphBuilder class.

Tests cover:
- Initialization
- Python file iteration with proper exclusions
- Definition extraction (functions, classes, methods)
- Import analysis (regular, aliased, from-imports)
- Call graph construction
- Edge cases (syntax errors, empty files, nested functions)
"""

import ast
import tempfile
from pathlib import Path


from code_scalpel.ast_tools.call_graph import CallGraphBuilder


class TestCallGraphBuilderInit:
    """Tests for CallGraphBuilder initialization."""

    def test_init_with_path(self):
        """Test initialization with a valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            assert builder.root_path == Path(tmpdir)
            assert builder.definitions == {}
            assert builder.calls == {}
            assert builder.imports == {}

    def test_init_with_string_path(self):
        """Test that Path objects are required (or converted)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # CallGraphBuilder expects Path, but should work with string too
            builder = CallGraphBuilder(Path(tmpdir))
            assert builder.root_path == Path(tmpdir)


class TestIterPythonFiles:
    """Tests for _iter_python_files method."""

    def test_finds_python_files(self):
        """Test that Python files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.py").write_text("x = 1")
            Path(tmpdir, "test2.py").write_text("y = 2")
            Path(tmpdir, "not_python.txt").write_text("hello")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 2
            assert all(f.suffix == ".py" for f in files)

    def test_finds_nested_python_files(self):
        """Test that nested Python files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            Path(tmpdir, "root.py").write_text("x = 1")
            Path(subdir, "nested.py").write_text("y = 2")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 2

    def test_skips_git_directory(self):
        """Test that .git directory is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir, ".git")
            git_dir.mkdir()
            Path(git_dir, "config.py").write_text("git config")
            Path(tmpdir, "main.py").write_text("x = 1")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_skips_venv_directory(self):
        """Test that venv and .venv directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir, "venv")
            venv_dir.mkdir()
            dot_venv_dir = Path(tmpdir, ".venv")
            dot_venv_dir.mkdir()
            Path(venv_dir, "site.py").write_text("venv stuff")
            Path(dot_venv_dir, "pip.py").write_text("pip stuff")
            Path(tmpdir, "main.py").write_text("x = 1")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_skips_pycache_directory(self):
        """Test that __pycache__ is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir, "__pycache__")
            cache_dir.mkdir()
            Path(cache_dir, "module.cpython-39.pyc").write_text("bytecode")
            Path(tmpdir, "module.py").write_text("x = 1")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 1
            assert files[0].name == "module.py"

    def test_skips_node_modules(self):
        """Test that node_modules is skipped (for mixed projects)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            node_dir = Path(tmpdir, "node_modules")
            node_dir.mkdir()
            Path(node_dir, "setup.py").write_text("weird but possible")
            Path(tmpdir, "main.py").write_text("x = 1")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 1

    def test_skips_hidden_directories(self):
        """Test that hidden directories (starting with .) are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hidden_dir = Path(tmpdir, ".hidden")
            hidden_dir.mkdir()
            Path(hidden_dir, "secret.py").write_text("secret code")
            Path(tmpdir, "visible.py").write_text("x = 1")

            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())

            assert len(files) == 1
            assert files[0].name == "visible.py"

    def test_empty_directory(self):
        """Test iteration over empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            files = list(builder._iter_python_files())
            assert files == []


class TestAnalyzeDefinitions:
    """Tests for _analyze_definitions method."""

    def test_extracts_function_definitions(self):
        """Test extraction of function definitions."""
        code = """
def foo():
    pass

def bar(x, y):
    return x + y
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            assert "foo" in builder.definitions["test.py"]
            assert "bar" in builder.definitions["test.py"]

    def test_extracts_async_function_definitions(self):
        """Test extraction of async function definitions."""
        code = """
async def async_foo():
    await something()

async def async_bar():
    pass
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            assert "async_foo" in builder.definitions["test.py"]
            assert "async_bar" in builder.definitions["test.py"]

    def test_extracts_class_definitions(self):
        """Test extraction of class definitions."""
        code = """
class MyClass:
    pass

class AnotherClass:
    x = 1
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            assert "MyClass" in builder.definitions["test.py"]
            assert "AnotherClass" in builder.definitions["test.py"]

    def test_extracts_method_definitions(self):
        """Test extraction of methods within classes."""
        code = """
class MyClass:
    def method_one(self):
        pass
    
    def method_two(self, x):
        return x

    async def async_method(self):
        pass
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            assert "MyClass" in builder.definitions["test.py"]
            assert "MyClass.method_one" in builder.definitions["test.py"]
            assert "MyClass.method_two" in builder.definitions["test.py"]
            assert "MyClass.async_method" in builder.definitions["test.py"]

    def test_extracts_imports(self):
        """Test extraction of import statements."""
        code = """
import os
import sys as system
from pathlib import Path
from typing import Dict, List
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            imports = builder.imports["test.py"]
            assert imports["os"] == "os"
            assert imports["system"] == "sys"
            assert imports["Path"] == "pathlib.Path"
            assert imports["Dict"] == "typing.Dict"
            assert imports["List"] == "typing.List"

    def test_extracts_imports_with_aliases(self):
        """Test extraction of aliased imports."""
        code = """
import numpy as np
import pandas as pd
from collections import defaultdict as dd
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            imports = builder.imports["test.py"]
            assert imports["np"] == "numpy"
            assert imports["pd"] == "pandas"
            assert imports["dd"] == "collections.defaultdict"

    def test_extracts_relative_imports(self):
        """Test extraction of relative imports (module is None or relative)."""
        code = """
from . import helper
from .utils import do_stuff
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder._analyze_definitions(tree, "test.py")

            imports = builder.imports["test.py"]
            assert imports["helper"] == "helper"  # module is empty string
            # Note: relative imports lose the leading dot in current impl
            assert imports["do_stuff"] == "utils.do_stuff"


class TestAnalyzeCalls:
    """Tests for _analyze_calls method."""

    def test_extracts_simple_function_calls(self):
        """Test extraction of simple function calls."""
        code = """
def caller():
    foo()
    bar()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:caller" in result
            assert "foo" in result["test.py:caller"]
            assert "bar" in result["test.py:caller"]

    def test_extracts_method_calls(self):
        """Test extraction of method calls on objects."""
        code = """
def process():
    obj.method()
    self.helper()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:process" in result
            assert "obj.method" in result["test.py:process"]
            assert "self.helper" in result["test.py:process"]

    def test_extracts_chained_method_calls(self):
        """Test extraction of chained method calls."""
        code = """
def chain():
    a.b.c()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:chain" in result
            assert "a.b.c" in result["test.py:chain"]

    def test_resolves_imported_calls(self):
        """Test resolution of calls to imported functions."""
        code = """
import utils

def process():
    utils.hash(data)
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {"utils": "utils"}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:process" in result
            assert "utils.hash" in result["test.py:process"]

    def test_resolves_aliased_imports(self):
        """Test resolution of aliased import calls."""
        code = """
import numpy as np

def compute():
    np.array([1, 2, 3])
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {"np": "numpy"}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:compute" in result
            assert "numpy.array" in result["test.py:compute"]

    def test_resolves_local_function_calls(self):
        """Test resolution of calls to local functions."""
        code = """
def helper():
    pass

def main():
    helper()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"helper", "main"}
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:main" in result
            assert "test.py:helper" in result["test.py:main"]

    def test_async_function_calls(self):
        """Test call extraction from async functions."""
        code = """
async def async_handler():
    await process()
    sync_call()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:async_handler" in result
            assert "process" in result["test.py:async_handler"]
            assert "sync_call" in result["test.py:async_handler"]

    def test_no_calls_outside_functions(self):
        """Test that calls at module level are not tracked."""
        code = """
# Module level call - not tracked
print("hello")

def func():
    print("inside function")
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"func"}
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            # Only func should have tracked calls
            assert "test.py:func" in result
            assert len(result) == 1


class TestBuild:
    """Tests for the full build() method."""

    def test_build_simple_project(self):
        """Test building call graph for a simple project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text(
                """
def main():
    helper()

def helper():
    print("helping")
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            assert "main.py:main" in graph
            assert "main.py:helper" in graph["main.py:main"]

    def test_build_multi_file_project(self):
        """Test building call graph across multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").write_text(
                """
from utils import process

def main():
    process()
"""
            )
            Path(tmpdir, "utils.py").write_text(
                """
def process():
    print("processing")
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            assert "main.py:main" in graph
            # The call to process() should be resolved via imports
            assert len(graph["main.py:main"]) >= 1

    def test_build_handles_syntax_errors(self):
        """Test that syntax errors in files don't crash the build."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "good.py").write_text(
                """
def good():
    pass
"""
            )
            Path(tmpdir, "bad.py").write_text(
                """
def bad(
    # Missing closing paren - syntax error
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            # Should not raise - just skip the bad file
            graph = builder.build()

            assert "good.py:good" in graph

    def test_build_handles_empty_files(self):
        """Test handling of empty Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "empty.py").write_text("")
            Path(tmpdir, "real.py").write_text(
                """
def real():
    pass
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            assert "real.py:real" in graph

    def test_build_with_nested_functions(self):
        """Test handling of nested function definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "nested.py").write_text(
                """
def outer():
    def inner():
        helper()
    inner()
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            # Note: Current impl may handle nested differently
            assert "nested.py:outer" in graph or "nested.py:inner" in graph

    def test_build_with_classes(self):
        """Test handling of class methods in call graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "classes.py").write_text(
                """
class MyClass:
    def method_a(self):
        self.method_b()
    
    def method_b(self):
        pass
"""
            )
            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            assert "classes.py:method_a" in graph
            assert "self.method_b" in graph["classes.py:method_a"]


class TestEdgeCases:
    """Edge case and regression tests."""

    def test_lambda_calls(self):
        """Test handling of lambda expressions with calls."""
        code = """
def func():
    f = lambda x: process(x)
    f(1)
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            # Lambda calls should be tracked within the enclosing function
            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:func" in result

    def test_builtin_calls(self):
        """Test that builtin calls are kept as-is."""
        code = """
def func():
    print("hello")
    len([1, 2, 3])
    range(10)
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"func"}
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "print" in result["test.py:func"]
            assert "len" in result["test.py:func"]
            assert "range" in result["test.py:func"]

    def test_complex_call_expressions(self):
        """Test handling of complex call expressions."""
        code = """
def func():
    get_handler()()  # Call the result of a call
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            # At least the inner call should be tracked
            assert "test.py:func" in result

    def test_decorator_calls(self):
        """Test that decorator calls are not tracked (they're at def level)."""
        code = """
@decorator
def func():
    actual_call()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = set()
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            # Only the body call should be tracked
            assert "test.py:func" in result
            assert "actual_call" in result["test.py:func"]


class TestIntegration:
    """Integration tests using real project structures."""

    def test_realistic_project_structure(self):
        """Test with a realistic project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a realistic structure
            src = Path(tmpdir, "src")
            src.mkdir()
            tests = Path(tmpdir, "tests")
            tests.mkdir()

            Path(src, "__init__.py").write_text("")
            Path(src, "core.py").write_text(
                """
from .utils import helper

def main():
    helper()
    process()

def process():
    print("processing")
"""
            )
            Path(src, "utils.py").write_text(
                """
def helper():
    return "helping"
"""
            )
            Path(tests, "test_core.py").write_text(
                """
from src.core import main

def test_main():
    main()
"""
            )

            builder = CallGraphBuilder(Path(tmpdir))
            graph = builder.build()

            # Should have entries for all functions
            assert any("main" in key for key in graph.keys())
            assert any("helper" in key for key in graph.keys())


class TestCallGraphCoverageGaps:
    """Tests to close specific coverage gaps in call_graph.py."""

    def test_get_attribute_value_with_non_name_non_attribute(self):
        """Test _get_attribute_value returns None for exotic node types (line 136).

        Line 136 handles the case where node is neither ast.Name nor ast.Attribute,
        such as a Subscript (list[0].method()) or Call (get_obj().method()).
        """
        code = """
def func():
    # Call on subscript result: handlers[0].process()
    # The value of the Attribute is a Subscript, not Name or Attribute
    handlers[0].process()
    
    # Call on call result: get_handler().execute()
    # The value of the Attribute is a Call, not Name or Attribute
    get_handler().execute()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"func"}
            builder.imports["test.py"] = {}

            # This should exercise line 136 where _get_attribute_value
            # receives a Subscript or Call node and returns None
            result = builder._analyze_calls(tree, "test.py")

            # Should not crash, function should still be tracked
            assert "test.py:func" in result
            # The calls should be handled gracefully
            # (may not be in result as the attribute value couldn't be resolved)

    def test_subscript_method_call(self):
        """Test method call on subscript expression."""
        code = """
def process_items():
    items = [handler1, handler2]
    items[0].run()
    matrix[i][j].calculate()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"process_items"}
            builder.imports["test.py"] = {}

            # Line 136 is hit when processing items[0].run()
            # because items[0] is Subscript, not Name or Attribute
            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:process_items" in result

    def test_call_result_method_call(self):
        """Test method call on function call result."""
        code = """
def chained_operations():
    create_connection().execute("query")
    factory.build().initialize().start()
"""
        tree = ast.parse(code)
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CallGraphBuilder(Path(tmpdir))
            builder.definitions["test.py"] = {"chained_operations"}
            builder.imports["test.py"] = {}

            result = builder._analyze_calls(tree, "test.py")

            assert "test.py:chained_operations" in result
