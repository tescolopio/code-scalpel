"""
Project Crawler - Whole-project Python analysis tool.

Crawls an entire project directory, analyzes all Python files, and generates
comprehensive metrics including:
- Structure analysis (classes, functions, imports)
- Cyclomatic complexity estimation
- Lines of code counts
- Complexity hotspot detection

This module integrates with Code Scalpel's existing AST analysis tools
and can be used standalone or via the MCP server.

Usage:
    from code_scalpel.project_crawler import ProjectCrawler

    crawler = ProjectCrawler("/path/to/project")
    result = crawler.crawl()
    print(result.summary)

    # Generate markdown report
    report = crawler.generate_report()
"""

from __future__ import annotations

import ast
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default directories to exclude from crawling
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "egg-info",
    ".egg-info",
    ".tox",
    ".nox",
    "htmlcov",
    "site-packages",
})

# Default complexity threshold for warnings
DEFAULT_COMPLEXITY_THRESHOLD: int = 10


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    lineno: int
    complexity: int
    is_method: bool = False
    class_name: str | None = None

    @property
    def qualified_name(self) -> str:
        """Return the fully qualified name."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    lineno: int
    methods: list[FunctionInfo] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""

    path: str
    status: str  # "success" or "error"
    lines_of_code: int = 0
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    complexity_warnings: list[FunctionInfo] = field(default_factory=list)
    error: str | None = None

    @property
    def total_functions(self) -> int:
        """Total number of functions including class methods."""
        return len(self.functions) + sum(len(c.methods) for c in self.classes)


@dataclass
class CrawlResult:
    """Result of crawling an entire project."""

    root_path: str
    timestamp: str
    files_analyzed: list[FileAnalysisResult] = field(default_factory=list)
    files_with_errors: list[FileAnalysisResult] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        """Total number of files scanned."""
        return len(self.files_analyzed) + len(self.files_with_errors)

    @property
    def total_lines_of_code(self) -> int:
        """Total lines of code across all files."""
        return sum(f.lines_of_code for f in self.files_analyzed)

    @property
    def total_functions(self) -> int:
        """Total number of functions across all files."""
        return sum(f.total_functions for f in self.files_analyzed)

    @property
    def total_classes(self) -> int:
        """Total number of classes across all files."""
        return sum(len(f.classes) for f in self.files_analyzed)

    @property
    def all_complexity_warnings(self) -> list[tuple[str, FunctionInfo]]:
        """All complexity warnings with file paths."""
        warnings = []
        for file_result in self.files_analyzed:
            for func in file_result.complexity_warnings:
                warnings.append((file_result.path, func))
        return warnings

    @property
    def summary(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "root_path": self.root_path,
            "timestamp": self.timestamp,
            "total_files": self.total_files,
            "successful_files": len(self.files_analyzed),
            "failed_files": len(self.files_with_errors),
            "total_lines_of_code": self.total_lines_of_code,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "complexity_warnings": len(self.all_complexity_warnings),
        }


class CodeAnalyzerVisitor(ast.NodeVisitor):
    """AST visitor that extracts code metrics."""

    def __init__(self, complexity_threshold: int = DEFAULT_COMPLEXITY_THRESHOLD):
        self.complexity_threshold = complexity_threshold
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[str] = []
        self.complexity_warnings: list[FunctionInfo] = []
        self._current_class: ClassInfo | None = None

    def visit_Import(self, node: ast.Import) -> None:
        """Handle import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle from ... import statements."""
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions."""
        bases = [
            self._get_base_name(base)
            for base in node.bases
            if self._get_base_name(base)
        ]
        class_info = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            bases=bases,
        )
        self.classes.append(class_info)

        # Visit class body with context
        old_class = self._current_class
        self._current_class = class_info
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definitions."""
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self._process_function(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function or async function definition."""
        complexity = self._calculate_complexity(node)

        func_info = FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            complexity=complexity,
            is_method=self._current_class is not None,
            class_name=self._current_class.name if self._current_class else None,
        )

        if self._current_class:
            self._current_class.methods.append(func_info)
        else:
            self.functions.append(func_info)

        if complexity > self.complexity_threshold:
            self.complexity_warnings.append(func_info)

        self.generic_visit(node)

    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity.

        Complexity = 1 + number of decision points (if, for, while, try, etc.)
        """
        score = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                score += 1
            elif isinstance(child, ast.Try):
                score += 1
            elif isinstance(child, ast.ExceptHandler):
                score += 1
            elif isinstance(child, ast.BoolOp):
                # and/or add decision points
                score += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                # List/dict/set comprehensions with if clauses
                score += len(child.ifs)
        return score

    @staticmethod
    def _get_base_name(node: ast.expr) -> str | None:
        """Extract base class name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class ProjectCrawler:
    """
    Crawls a project directory and analyzes all Python files.

    Example:
        crawler = ProjectCrawler("/path/to/project")
        result = crawler.crawl()
        print(f"Analyzed {result.total_files} files")
        print(f"Total LOC: {result.total_lines_of_code}")
    """

    def __init__(
        self,
        root_path: str | Path,
        exclude_dirs: frozenset[str] | None = None,
        complexity_threshold: int = DEFAULT_COMPLEXITY_THRESHOLD,
    ):
        """
        Initialize the project crawler.

        Args:
            root_path: Root directory to crawl
            exclude_dirs: Directory names to exclude (uses defaults if None)
            complexity_threshold: Complexity score that triggers a warning
        """
        self.root_path = Path(root_path).resolve()
        self.exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS
        self.complexity_threshold = complexity_threshold

        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.root_path}")

    def crawl(self) -> CrawlResult:
        """
        Crawl the project and analyze all Python files.

        Returns:
            CrawlResult with analysis data for all files
        """
        result = CrawlResult(
            root_path=str(self.root_path),
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        for root, dirs, files in os.walk(self.root_path):
            # Filter excluded directories in-place
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for filename in files:
                if filename.endswith(".py"):
                    file_path = os.path.join(root, filename)
                    file_result = self._analyze_file(file_path)

                    if file_result.status == "success":
                        result.files_analyzed.append(file_result)
                    else:
                        result.files_with_errors.append(file_result)

        return result

    def _analyze_file(self, file_path: str) -> FileAnalysisResult:
        """
        Analyze a single Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            FileAnalysisResult with metrics
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            tree = ast.parse(code, filename=file_path)
            visitor = CodeAnalyzerVisitor(self.complexity_threshold)
            visitor.visit(tree)

            return FileAnalysisResult(
                path=file_path,
                status="success",
                lines_of_code=len(code.splitlines()),
                functions=visitor.functions,
                classes=visitor.classes,
                imports=visitor.imports,
                complexity_warnings=visitor.complexity_warnings,
            )

        except SyntaxError as e:
            return FileAnalysisResult(
                path=file_path,
                status="error",
                error=f"Syntax error at line {e.lineno}: {e.msg}",
            )
        except Exception as e:
            return FileAnalysisResult(
                path=file_path,
                status="error",
                error=str(e),
            )

    def generate_report(
        self, result: CrawlResult | None = None, output_path: str | None = None
    ) -> str:
        """
        Generate a Markdown report of the crawl results.

        Args:
            result: CrawlResult to report on (crawls if None)
            output_path: Optional path to write the report

        Returns:
            Markdown report string
        """
        if result is None:
            result = self.crawl()

        md_lines = [
            "# Project Python Analysis Report",
            "",
            f"**Target:** `{result.root_path}`",
            f"**Date:** {result.timestamp}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Files Scanned:** {result.total_files}",
            f"- **Successful Analyses:** {len(result.files_analyzed)}",
            f"- **Failed Analyses:** {len(result.files_with_errors)}",
            f"- **Total Lines of Code:** {result.total_lines_of_code:,}",
            f"- **Total Functions:** {result.total_functions}",
            f"- **Total Classes:** {result.total_classes}",
            f"- **Complexity Hotspots:** {len(result.all_complexity_warnings)}",
            "",
            "---",
            "",
        ]

        # Complexity warnings section
        md_lines.append(f"## Complexity Warnings (Score > {self.complexity_threshold})")
        md_lines.append("")

        warnings = result.all_complexity_warnings
        if not warnings:
            md_lines.append("No overly complex functions detected.")
        else:
            md_lines.append("| File | Function | Complexity | Line |")
            md_lines.append("|------|----------|------------|------|")
            for file_path, func in sorted(warnings, key=lambda x: x[1].complexity, reverse=True):
                rel_path = os.path.relpath(file_path, result.root_path)
                md_lines.append(
                    f"| `{rel_path}` | `{func.qualified_name}` | **{func.complexity}** | {func.lineno} |"
                )
        md_lines.append("")

        # File statistics section
        md_lines.append("## File Statistics")
        md_lines.append("")
        md_lines.append("| File | LOC | Classes | Functions | Imports |")
        md_lines.append("|------|-----|---------|-----------|---------|")

        for file_result in sorted(
            result.files_analyzed, key=lambda x: x.lines_of_code, reverse=True
        ):
            rel_path = os.path.relpath(file_result.path, result.root_path)
            md_lines.append(
                f"| `{rel_path}` | {file_result.lines_of_code} | "
                f"{len(file_result.classes)} | {file_result.total_functions} | "
                f"{len(file_result.imports)} |"
            )
        md_lines.append("")

        # Error section
        if result.files_with_errors:
            md_lines.append("## Analysis Errors")
            md_lines.append("")
            md_lines.append("| File | Error |")
            md_lines.append("|------|-------|")
            for file_result in result.files_with_errors:
                rel_path = os.path.relpath(file_result.path, result.root_path)
                error_msg = file_result.error or "Unknown error"
                md_lines.append(f"| `{rel_path}` | {error_msg} |")
            md_lines.append("")

        report = "\n".join(md_lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def to_dict(self, result: CrawlResult | None = None) -> dict[str, Any]:
        """
        Convert crawl results to a dictionary for JSON serialization.

        Args:
            result: CrawlResult to convert (crawls if None)

        Returns:
            Dictionary representation of the results
        """
        if result is None:
            result = self.crawl()

        def file_result_to_dict(fr: FileAnalysisResult) -> dict[str, Any]:
            return {
                "path": os.path.relpath(fr.path, result.root_path),
                "status": fr.status,
                "lines_of_code": fr.lines_of_code,
                "functions": [
                    {
                        "name": f.qualified_name,
                        "lineno": f.lineno,
                        "complexity": f.complexity,
                    }
                    for f in fr.functions
                ],
                "classes": [
                    {
                        "name": c.name,
                        "lineno": c.lineno,
                        "methods": [
                            {
                                "name": m.name,
                                "lineno": m.lineno,
                                "complexity": m.complexity,
                            }
                            for m in c.methods
                        ],
                        "bases": c.bases,
                    }
                    for c in fr.classes
                ],
                "imports": fr.imports,
                "complexity_warnings": [
                    {
                        "name": f.qualified_name,
                        "lineno": f.lineno,
                        "complexity": f.complexity,
                    }
                    for f in fr.complexity_warnings
                ],
                "error": fr.error,
            }

        return {
            "root_path": result.root_path,
            "timestamp": result.timestamp,
            "summary": result.summary,
            "files": [file_result_to_dict(f) for f in result.files_analyzed],
            "errors": [file_result_to_dict(f) for f in result.files_with_errors],
        }


def crawl_project(
    root_path: str,
    exclude_dirs: list[str] | None = None,
    complexity_threshold: int = DEFAULT_COMPLEXITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Convenience function to crawl a project and return results as a dictionary.

    Args:
        root_path: Path to the project root
        exclude_dirs: Optional list of directory names to exclude
        complexity_threshold: Complexity score that triggers a warning

    Returns:
        Dictionary with crawl results
    """
    exclude_set = frozenset(exclude_dirs) if exclude_dirs else None
    crawler = ProjectCrawler(
        root_path,
        exclude_dirs=exclude_set,
        complexity_threshold=complexity_threshold,
    )
    result = crawler.crawl()
    return crawler.to_dict(result)
