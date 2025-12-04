#!/usr/bin/env python3

import ast
import subprocess
import tokenize
from collections import defaultdict
from io import StringIO
from typing import Any, Optional

from ..base_parser import BaseParser, Language, ParseResult, PreprocessorConfig


class PythonParser(BaseParser):
    def parse_code(
        self,
        code: str,
        preprocess: bool = True,
        config: Optional[PreprocessorConfig] = None,
    ) -> ParseResult:
        if preprocess:
            code = self._preprocess_code(code, config or PreprocessorConfig())
        return self._parse_python(code)

    def _preprocess_code(self, code: str, config: PreprocessorConfig) -> str:
        # Python-specific preprocessing logic
        if config.remove_comments:
            code = self._remove_comments(code, Language.PYTHON)
        if config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        return code

    def _parse_python(self, code: str) -> ParseResult:
        """Parse Python code with detailed analysis."""
        errors = []
        warnings = []
        metrics = defaultdict(int)

        try:
            # Parse into AST
            tree = ast.parse(code)

            # Get token stream
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))

            # Analyze code structure
            metrics.update(self._analyze_python_code(tree))

            # Check for potential issues using mypy
            warnings.extend(self._check_python_code_with_mypy(code))

            return ParseResult(
                ast=tree,
                errors=errors,
                warnings=warnings,
                tokens=tokens,
                metrics=dict(metrics),
                language=Language.PYTHON,
            )

        except SyntaxError as e:
            errors.append(self._format_syntax_error(e))
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=warnings,
                tokens=[],
                metrics=dict(metrics),
                language=Language.PYTHON,
            )

    def _analyze_python_code(self, tree: ast.AST) -> dict[str, int]:
        """Analyze Python code structure."""
        metrics = defaultdict(int)

        for node in ast.walk(tree):
            # Count different node types
            metrics[f"count_{type(node).__name__}"] += 1

            # Analyze complexity
            if isinstance(node, ast.FunctionDef):
                metrics["function_count"] += 1
                metrics["max_function_complexity"] = max(
                    metrics["max_function_complexity"], self._calculate_complexity(node)
                )
            elif isinstance(node, ast.ClassDef):
                metrics["class_count"] += 1

        return dict(metrics)

    def _check_python_code_with_mypy(self, code: str) -> list[str]:
        """Check for potential code issues using mypy."""
        warnings = []

        # Run mypy as a subprocess
        result = subprocess.run(
            [
                "mypy",
                "--show-error-codes",
                "--hide-error-context",
                "--hide-error-traceback",
                "--no-color-output",
                "--no-error-summary",
                "--show-column-numbers",
                "--ignore-missing-imports",
                "--command",
                code,
            ],
            text=True,
            capture_output=True,
        )

        if result.returncode != 0 and result.stdout:
            for line in result.stdout.splitlines():
                warnings.append(line)

        return warnings

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the complexity of a function node."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        return complexity

    def _format_syntax_error(self, e: SyntaxError) -> dict[str, Any]:
        """Format a syntax error for inclusion in the error list."""
        return {
            "type": "SyntaxError",
            "message": e.msg,
            "line": e.lineno,
            "column": e.offset,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Python parser using mypy for analysis"
    )
    parser.add_argument("filename", help="Path to the Python file")
    args = parser.parse_args()

    try:
        with open(args.filename) as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.filename} not found.")
        exit(1)

    python_parser = PythonParser()
    result = python_parser.parse_code(code)

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(error)

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(warning)

    print("Metrics:")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")
