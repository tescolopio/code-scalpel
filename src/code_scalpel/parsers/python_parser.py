import ast
from typing import Any, List
from .base_parser import BaseParser


class PythonParser(BaseParser):
    """
    Python implementation of the BaseParser using the 'ast' module.
    """

    def parse(self, code: str) -> ast.AST:
        return ast.parse(code)

    def get_functions(self, ast_tree: Any) -> List[str]:
        if not isinstance(ast_tree, ast.AST):
            raise TypeError("Expected ast.AST object")

        functions = []
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        return functions

    def get_classes(self, ast_tree: Any) -> List[str]:
        if not isinstance(ast_tree, ast.AST):
            raise TypeError("Expected ast.AST object")

        classes = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes

    def get_imports(self, ast_tree: Any) -> List[str]:
        if not isinstance(ast_tree, ast.AST):
            raise TypeError("Expected ast.AST object")

        imports = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
