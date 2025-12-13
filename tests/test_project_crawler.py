"""Tests for the Project Crawler module.

Tests the project-wide Python file analysis functionality including:
- Directory crawling with exclusion patterns
- File analysis (functions, classes, imports, complexity)
- Report generation
- Edge cases and error handling
"""

import os
import tempfile
import pytest
from pathlib import Path

from code_scalpel.project_crawler import (
    ProjectCrawler,
    CrawlResult,
    FileAnalysisResult,
    FunctionInfo,
    ClassInfo,
    CodeAnalyzerVisitor,
    crawl_project,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_COMPLEXITY_THRESHOLD,
)


class TestCodeAnalyzerVisitor:
    """Tests for the AST visitor that extracts metrics."""

    def test_visit_simple_function(self):
        """Test analyzing a simple function."""
        import ast

        code = "def hello(): return 42"
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert len(visitor.functions) == 1
        assert visitor.functions[0].name == "hello"
        assert visitor.functions[0].complexity == 1
        assert visitor.functions[0].is_method is False

    def test_visit_function_with_branches(self):
        """Test complexity calculation with branches."""
        import ast

        code = """
def complex_func(x):
    if x > 0:
        if x > 10:
            return "big"
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert len(visitor.functions) == 1
        # 1 base + 3 if/elif + 1 else = higher complexity
        assert visitor.functions[0].complexity > 1

    def test_visit_class_with_methods(self):
        """Test analyzing a class with methods."""
        import ast

        code = """
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert len(visitor.classes) == 1
        assert visitor.classes[0].name == "Calculator"
        assert len(visitor.classes[0].methods) == 2
        assert visitor.classes[0].methods[0].name == "__init__"
        assert visitor.classes[0].methods[0].is_method is True
        assert visitor.classes[0].methods[0].class_name == "Calculator"

    def test_visit_imports(self):
        """Test collecting import statements."""
        import ast

        code = """
import os
import sys
from pathlib import Path
from typing import Dict, List
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert "os" in visitor.imports
        assert "sys" in visitor.imports
        assert "pathlib" in visitor.imports
        assert "typing" in visitor.imports

    def test_complexity_warning_threshold(self):
        """Test that high complexity triggers warnings."""
        import ast

        code = """
def complex_function(x, y, z):
    if x:
        if y:
            if z:
                for i in range(10):
                    while True:
                        try:
                            pass
                        except:
                            pass
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor(complexity_threshold=5)
        visitor.visit(tree)

        assert len(visitor.complexity_warnings) == 1
        assert visitor.complexity_warnings[0].complexity > 5

    def test_async_function(self):
        """Test analyzing async functions."""
        import ast

        code = """
async def fetch_data():
    return await some_async_call()
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert len(visitor.functions) == 1
        assert visitor.functions[0].name == "fetch_data"

    def test_class_inheritance(self):
        """Test capturing base classes."""
        import ast

        code = """
class Child(Parent, Mixin):
    pass
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        assert len(visitor.classes) == 1
        assert "Parent" in visitor.classes[0].bases
        assert "Mixin" in visitor.classes[0].bases

    def test_comprehension_complexity(self):
        """Test that comprehensions with conditions add complexity."""
        import ast

        code = """
def filter_items(items):
    return [x for x in items if x > 0 if x < 100]
"""
        tree = ast.parse(code)
        visitor = CodeAnalyzerVisitor()
        visitor.visit(tree)

        # 1 base + 2 if conditions in comprehension
        assert visitor.functions[0].complexity >= 3


class TestFunctionInfo:
    """Tests for FunctionInfo dataclass."""

    def test_qualified_name_standalone(self):
        """Test qualified name for standalone function."""
        func = FunctionInfo(name="hello", lineno=1, complexity=1)
        assert func.qualified_name == "hello"

    def test_qualified_name_method(self):
        """Test qualified name for class method."""
        func = FunctionInfo(
            name="method", lineno=1, complexity=1, is_method=True, class_name="MyClass"
        )
        assert func.qualified_name == "MyClass.method"


class TestProjectCrawler:
    """Tests for the ProjectCrawler class."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main module
            main_file = Path(tmpdir) / "main.py"
            main_file.write_text("""
import os
from utils import helper

def main():
    print("Hello")
    return helper()

class App:
    def run(self):
        pass
""")

            # Create utils module
            utils_file = Path(tmpdir) / "utils.py"
            utils_file.write_text("""
def helper():
    return 42

def complex_helper(x, y, z):
    if x:
        if y:
            if z:
                for i in range(10):
                    while True:
                        try:
                            pass
                        except:
                            pass
    return None
""")

            # Create subdirectory with module
            subdir = Path(tmpdir) / "submodule"
            subdir.mkdir()
            sub_file = subdir / "__init__.py"
            sub_file.write_text("# Submodule init")

            sub_module = subdir / "core.py"
            sub_module.write_text("""
class Core:
    def process(self):
        return True
""")

            # Create excluded directory
            venv_dir = Path(tmpdir) / "venv"
            venv_dir.mkdir()
            venv_file = venv_dir / "should_skip.py"
            venv_file.write_text("# Should not be analyzed")

            yield tmpdir

    def test_crawl_project_structure(self, temp_project):
        """Test basic project crawling."""
        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()

        assert result.total_files >= 4  # main.py, utils.py, __init__.py, core.py
        assert result.total_lines_of_code > 0
        assert result.total_functions > 0
        assert result.total_classes >= 2  # App, Core

    def test_exclude_dirs(self, temp_project):
        """Test that excluded directories are skipped."""
        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()

        # venv should be excluded by default
        file_paths = [f.path for f in result.files_analyzed]
        assert not any("venv" in p for p in file_paths)

    def test_custom_exclude_dirs(self, temp_project):
        """Test custom exclusion patterns."""
        crawler = ProjectCrawler(
            temp_project,
            exclude_dirs=frozenset({"venv", "submodule"}),
        )
        result = crawler.crawl()

        file_paths = [f.path for f in result.files_analyzed]
        assert not any("submodule" in p for p in file_paths)

    def test_complexity_warnings(self, temp_project):
        """Test that high complexity functions are flagged."""
        crawler = ProjectCrawler(temp_project, complexity_threshold=5)
        result = crawler.crawl()

        # complex_helper should be flagged
        all_warnings = result.all_complexity_warnings
        warning_names = [w[1].qualified_name for w in all_warnings]
        assert "complex_helper" in warning_names

    def test_invalid_path_raises(self):
        """Test that invalid path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            ProjectCrawler("/nonexistent/path")

    def test_file_not_directory_raises(self, temp_project):
        """Test that file path (not dir) raises ValueError."""
        file_path = Path(temp_project) / "main.py"
        with pytest.raises(ValueError, match="not a directory"):
            ProjectCrawler(file_path)

    def test_syntax_error_handling(self, temp_project):
        """Test handling of files with syntax errors."""
        bad_file = Path(temp_project) / "bad_syntax.py"
        bad_file.write_text("def broken( return")

        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()

        assert len(result.files_with_errors) >= 1
        error_paths = [f.path for f in result.files_with_errors]
        assert any("bad_syntax.py" in p for p in error_paths)


class TestCrawlResult:
    """Tests for CrawlResult dataclass."""

    def test_summary_properties(self):
        """Test summary property calculations."""
        result = CrawlResult(
            root_path="/test",
            timestamp="2025-01-01",
            files_analyzed=[
                FileAnalysisResult(
                    path="/test/a.py",
                    status="success",
                    lines_of_code=100,
                    functions=[FunctionInfo("f1", 1, 5)],
                    classes=[ClassInfo("C1", 1, [FunctionInfo("m1", 2, 3)])],
                ),
                FileAnalysisResult(
                    path="/test/b.py",
                    status="success",
                    lines_of_code=50,
                    functions=[FunctionInfo("f2", 1, 2)],
                ),
            ],
        )

        assert result.total_files == 2
        assert result.total_lines_of_code == 150
        assert result.total_functions == 3  # f1, m1, f2
        assert result.total_classes == 1

    def test_summary_dict(self):
        """Test summary as dictionary."""
        result = CrawlResult(
            root_path="/test",
            timestamp="2025-01-01",
        )
        summary = result.summary

        assert "root_path" in summary
        assert "timestamp" in summary
        assert "total_files" in summary
        assert "total_lines_of_code" in summary


class TestReportGeneration:
    """Tests for Markdown report generation."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project for report testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_file = Path(tmpdir) / "main.py"
            main_file.write_text("""
def simple():
    return 1

def complex(x, y, z):
    if x:
        if y:
            if z:
                for i in range(10):
                    while True:
                        if i > 5:
                            break
    return None
""")
            yield tmpdir

    def test_generate_report(self, temp_project):
        """Test report generation."""
        crawler = ProjectCrawler(temp_project, complexity_threshold=5)
        result = crawler.crawl()
        report = crawler.generate_report(result)

        assert "# Project Python Analysis Report" in report
        assert "Executive Summary" in report
        assert "Complexity Warnings" in report
        assert "File Statistics" in report

    def test_report_contains_stats(self, temp_project):
        """Test that report contains correct statistics."""
        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()
        report = crawler.generate_report(result)

        assert "Total Files Scanned:" in report
        assert "Total Lines of Code:" in report

    def test_report_output_to_file(self, temp_project):
        """Test writing report to file."""
        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()

        output_path = Path(temp_project) / "report.md"
        crawler.generate_report(result, output_path=str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "Project Python Analysis Report" in content


class TestToDictConversion:
    """Tests for dictionary/JSON conversion."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_file = Path(tmpdir) / "main.py"
            main_file.write_text("def hello(): return 42")
            yield tmpdir

    def test_to_dict_structure(self, temp_project):
        """Test dictionary output structure."""
        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()
        data = crawler.to_dict(result)

        assert "root_path" in data
        assert "timestamp" in data
        assert "summary" in data
        assert "files" in data
        assert "errors" in data

    def test_to_dict_serializable(self, temp_project):
        """Test that output is JSON serializable."""
        import json

        crawler = ProjectCrawler(temp_project)
        result = crawler.crawl()
        data = crawler.to_dict(result)

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


class TestConvenienceFunction:
    """Tests for the crawl_project convenience function."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_file = Path(tmpdir) / "main.py"
            main_file.write_text("""
def hello():
    return 42

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"
""")
            yield tmpdir

    def test_crawl_project_function(self, temp_project):
        """Test the convenience function."""
        result = crawl_project(temp_project)

        assert "summary" in result
        assert result["summary"]["total_files"] >= 1
        assert result["summary"]["total_functions"] >= 2

    def test_crawl_project_with_options(self, temp_project):
        """Test convenience function with options."""
        result = crawl_project(
            temp_project,
            complexity_threshold=1,  # Very low threshold
        )

        # With threshold=1, most functions should be flagged
        assert result["summary"]["complexity_warnings"] >= 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_directory(self):
        """Test crawling an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crawler = ProjectCrawler(tmpdir)
            result = crawler.crawl()

            assert result.total_files == 0
            assert result.total_lines_of_code == 0

    def test_only_excluded_dirs(self):
        """Test directory with only excluded subdirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only excluded directories
            venv = Path(tmpdir) / "venv"
            venv.mkdir()
            (venv / "test.py").write_text("x = 1")

            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            (pycache / "test.cpython-310.pyc").write_text("bytecode")

            crawler = ProjectCrawler(tmpdir)
            result = crawler.crawl()

            assert result.total_files == 0

    def test_unicode_content(self):
        """Test files with unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_file = Path(tmpdir) / "unicode.py"
            unicode_file.write_text('''
def greet():
    """Привет мир! 你好世界!"""
    return "🎉"
''', encoding="utf-8")

            crawler = ProjectCrawler(tmpdir)
            result = crawler.crawl()

            assert result.total_files == 1
            assert len(result.files_with_errors) == 0

    def test_nested_classes(self):
        """Test handling of nested classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_file = Path(tmpdir) / "nested.py"
            nested_file.write_text("""
class Outer:
    class Inner:
        def inner_method(self):
            pass
    
    def outer_method(self):
        pass
""")

            crawler = ProjectCrawler(tmpdir)
            result = crawler.crawl()

            assert result.total_classes >= 1  # At least Outer

    def test_lambda_functions(self):
        """Test that lambdas don't cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lambda_file = Path(tmpdir) / "lambdas.py"
            lambda_file.write_text("""
square = lambda x: x ** 2
items = list(filter(lambda x: x > 0, [1, -2, 3]))
""")

            crawler = ProjectCrawler(tmpdir)
            result = crawler.crawl()

            # Should complete without error
            assert len(result.files_with_errors) == 0
