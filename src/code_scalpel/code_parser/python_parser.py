import ast
from typing import Any, List
from .interface import IParser, ParseResult, Language


class PythonParser(IParser):
    """Python implementation of the parser interface."""

    def parse(self, code: str) -> ParseResult:
        errors = []
        metrics = {}
        try:
            tree = ast.parse(code)
            metrics["complexity"] = self._calculate_complexity(tree)
            return ParseResult(
                ast=tree,
                errors=[],
                warnings=[],
                metrics=metrics,
                language=Language.PYTHON,
            )
        except SyntaxError as e:
            errors.append(
                {
                    "type": "SyntaxError",
                    "message": e.msg,
                    "line": e.lineno,
                    "column": e.offset,
                    "text": e.text.strip() if e.text else None,
                }
            )
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=[],
                metrics={},
                language=Language.PYTHON,
            )

    def get_functions(self, ast_tree: Any) -> List[str]:
        if not isinstance(ast_tree, ast.AST):
            return []
        return [
            node.name
            for node in ast.walk(ast_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def get_classes(self, ast_tree: Any) -> List[str]:
        if not isinstance(ast_tree, ast.AST):
            return []
        return [
            node.name for node in ast.walk(ast_tree) if isinstance(node, ast.ClassDef)
        ]

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
