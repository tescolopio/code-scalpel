import ast
import logging
from typing import Optional, Dict, Callable, List
import tokenize
from io import StringIO
from functools import lru_cache

logger = logging.getLogger(__name__)


class ASTBuilder:
    """
    Advanced AST builder with preprocessing and validation capabilities.
    """

    def __init__(self):
        self.preprocessing_hooks: List[Callable[[str], str]] = []
        self.validation_hooks: List[Callable[[ast.AST], None]] = []
        self.ast_cache: Dict[str, ast.AST] = {}

    def build_ast(
        self, code: str, preprocess: bool = True, validate: bool = True
    ) -> Optional[ast.AST]:
        """
        Build an AST from Python code with optional preprocessing and validation.

        Args:
            code (str): The Python code to parse.
            preprocess (bool): Whether to apply preprocessing hooks.
            validate (bool): Whether to apply validation hooks.

        Returns:
            Optional[ast.AST]: The parsed AST, or None if an error occurred.
        """
        if code in self.ast_cache:
            return self.ast_cache[code]

        try:
            if preprocess:
                code = self._preprocess_code(code)

            tree = ast.parse(code)

            if validate:
                self._validate_ast(tree)

            self.ast_cache[code] = tree
            return tree
        except SyntaxError as e:
            self._handle_syntax_error(e)
            return None
        except Exception as e:
            logger.error(f"Unexpected error building AST: {str(e)}")
            return None

    @lru_cache(maxsize=100)
    def build_ast_from_file(
        self, filepath: str, preprocess: bool = True, validate: bool = True
    ) -> Optional[ast.AST]:
        """
        Build an AST from a Python source file with caching.

        Args:
            filepath (str): The path to the Python source file.
            preprocess (bool): Whether to apply preprocessing hooks.
            validate (bool): Whether to apply validation hooks.

        Returns:
            Optional[ast.AST]: The parsed AST, or None if an error occurred.
        """
        try:
            with tokenize.open(filepath) as file:
                code = file.read()
                return self.build_ast(code, preprocess, validate)
        except FileNotFoundError:
            logger.error(f"Error: File not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            return None

    def add_preprocessing_hook(self, hook: Callable[[str], str]) -> None:
        """Add a preprocessing hook to modify code before parsing."""
        self.preprocessing_hooks.append(hook)

    def remove_preprocessing_hook(self, hook: Callable[[str], str]) -> None:
        """Remove a preprocessing hook."""
        self.preprocessing_hooks.remove(hook)

    def add_validation_hook(self, hook: Callable[[ast.AST], None]) -> None:
        """Add a validation hook to check the AST after parsing."""
        self.validation_hooks.append(hook)

    def remove_validation_hook(self, hook: Callable[[ast.AST], None]) -> None:
        """Remove a validation hook."""
        self.validation_hooks.remove(hook)

    def _preprocess_code(self, code: str) -> str:
        """Apply all preprocessing hooks to the code."""
        processed_code = code

        # Remove comments
        processed_code = self._remove_comments(processed_code)

        # Apply custom preprocessing hooks
        for hook in self.preprocessing_hooks:
            processed_code = hook(processed_code)

        return processed_code

    def _validate_ast(self, tree: ast.AST) -> None:
        """Apply all validation hooks to the AST."""
        for hook in self.validation_hooks:
            hook(tree)

    @staticmethod
    def _remove_comments(code: str) -> str:
        """Remove comments while preserving line numbers."""
        result = []
        prev_toktype = tokenize.INDENT
        first_line = True

        tokens = tokenize.generate_tokens(StringIO(code).readline)

        for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokens:
            if toktype == tokenize.COMMENT:
                continue
            elif toktype == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    result.append(" ")
                result.append(ttext)
            elif toktype == tokenize.NEWLINE:
                result.append(ttext)
            elif toktype == tokenize.INDENT:
                result.append(ttext)
            elif toktype == tokenize.DEDENT:
                pass
            else:
                if not first_line and prev_toktype != tokenize.INDENT:
                    result.append(" ")
                result.append(ttext)
            prev_toktype = toktype
            first_line = False

        return "".join(result)

    def _handle_syntax_error(self, error: SyntaxError) -> None:
        """Handle syntax errors with detailed information."""
        logger.error(f"Syntax Error at line {error.lineno}, column {error.offset}:")
        logger.error(f"  {error.text.strip()}")
        logger.error("  " + " " * (error.offset - 1) + "^")
        logger.error(f"Error message: {str(error)}")

    def clear_cache(self) -> None:
        """Clear the AST cache."""
        self.ast_cache.clear()
