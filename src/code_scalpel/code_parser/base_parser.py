# src/code_parser/base_parser.py

import ast  # Import the ast module
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class Language(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"


@dataclass
class ParseResult:
    """Result of code parsing."""

    ast: Optional[Any]  # AST structure
    errors: list[Any]  # Parsing errors
    warnings: list[str]  # Parse warnings
    tokens: list[Any]  # Token stream
    metrics: dict[str, Any]  # Parse metrics
    language: Language


@dataclass
class PreprocessorConfig:
    """Configuration for code preprocessing."""

    remove_comments: bool = False
    normalize_whitespace: bool = False


class BaseParser(ABC):
    @abstractmethod
    def parse_code(
        self,
        code: str,
        preprocess: bool = True,
        config: Optional[PreprocessorConfig] = None,
    ) -> ParseResult:
        pass

    def _preprocess_code(
        self, code: str, language: Language, config: PreprocessorConfig
    ) -> str:
        """Preprocess code according to configuration."""
        if config.remove_comments:
            code = self._remove_comments(code, language)
        if config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        return code

    @staticmethod
    def _remove_comments(code: str, language: Language) -> str:
        """Remove comments while preserving line numbers."""
        import re

        if language == Language.PYTHON:
            # Remove Python comments (# ...)
            lines = code.split("\n")
            result = []
            for line in lines:
                # Remove inline comments but keep the line
                if "#" in line:
                    # Don't remove # inside strings (simple heuristic)
                    in_string = False
                    quote_char = None
                    new_line = []
                    i = 0
                    while i < len(line):
                        c = line[i]
                        if c in "\"'":
                            if not in_string:
                                in_string = True
                                quote_char = c
                            elif c == quote_char:
                                in_string = False
                        elif c == "#" and not in_string:
                            break
                        new_line.append(c)
                        i += 1
                    result.append("".join(new_line).rstrip())
                else:
                    result.append(line)
            return "\n".join(result)
        elif language in (Language.JAVASCRIPT, Language.JAVA, Language.CPP):
            # Remove C-style comments (// and /* */)
            # Remove single-line comments
            code = re.sub(r"//[^\n]*", "", code)
            # Remove multi-line comments
            code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
            return code
        return code

    @staticmethod
    def _normalize_whitespace(code: str) -> str:
        """Normalize whitespace in the code."""
        import re

        # Replace multiple spaces with single space (but preserve indentation)
        lines = code.split("\n")
        result = []
        for line in lines:
            # Preserve leading whitespace
            stripped = line.lstrip()
            leading = line[: len(line) - len(stripped)]
            # Normalize internal whitespace
            normalized = re.sub(r" +", " ", stripped)
            result.append(leading + normalized)
        return "\n".join(result)

    @staticmethod
    def _calculate_complexity(node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    @staticmethod
    def _format_syntax_error(error: SyntaxError) -> dict[str, Any]:
        """Format syntax error with detailed information."""
        return {
            "type": "SyntaxError",
            "message": error.msg,
            "line": error.lineno,
            "column": error.offset,
            "text": error.text.strip() if error.text else None,
            "filename": error.filename,
        }


class CodeParser(BaseParser):
    """Advanced code parser with multi-language support."""

    def __init__(self):
        self.parsers = {}
        self.preprocessors = {}
        self._setup_logging()
        self._init_parsers()

    def parse_code(
        self,
        code: str,
        language: Language,
        preprocess: bool = True,
        config: Optional[PreprocessorConfig] = None,
    ) -> ParseResult:
        """
        Parse code with comprehensive analysis.

        Args:
            code: Source code
            language: Programming language
            preprocess: Whether to preprocess code
            config: Preprocessing configuration

        Returns:
            Parse result with AST and analysis
        """
        try:
            # Preprocess if requested
            if preprocess:
                code = self._preprocess_code(
                    code, language, config or PreprocessorConfig()
                )

            # Parse code
            if language == Language.PYTHON:
                return self._parse_python(code)
            elif language == Language.JAVASCRIPT:
                return self._parse_javascript(code)
            elif language == Language.JAVA:
                return self._parse_java(code)
            elif language == Language.CPP:
                return self._parse_cpp(code)
            else:
                raise ValueError(f"Unsupported language: {language}")

        except Exception as e:
            self.logger.error(f"Parsing error: {str(e)}")
            raise

    def _setup_logging(self):
        """Setup logging configuration."""
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("CodeParser")

    def _init_parsers(self):
        """Initialize language-specific parsers."""
        # Initialize parsers for different languages
        pass

    def _parse_python(self, code: str) -> ParseResult:
        """Parse Python code."""
        errors = []
        warnings = []
        tokens = []
        metrics = {}

        try:
            tree = ast.parse(code)
            metrics["complexity"] = self._calculate_complexity(tree)
            return ParseResult(
                ast=tree,
                errors=errors,
                warnings=warnings,
                tokens=tokens,
                metrics=metrics,
                language=Language.PYTHON,
            )
        except SyntaxError as e:
            errors.append(self._format_syntax_error(e))
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=warnings,
                tokens=tokens,
                metrics=metrics,
                language=Language.PYTHON,
            )

    def _parse_javascript(self, code: str) -> ParseResult:
        """Parse JavaScript code."""
        # Basic stub implementation - returns minimal ParseResult
        return ParseResult(
            ast={"type": "Program", "body": [], "sourceType": "script"},
            errors=[],
            warnings=[],
            tokens=[],
            metrics={},
            language=Language.JAVASCRIPT,
        )

    def _parse_java(self, code: str) -> ParseResult:
        """Parse Java code."""
        # Basic stub implementation - returns minimal ParseResult
        return ParseResult(
            ast={"type": "CompilationUnit", "body": []},
            errors=[],
            warnings=[],
            tokens=[],
            metrics={},
            language=Language.JAVA,
        )

    def _parse_cpp(self, code: str) -> ParseResult:
        """Parse C++ code."""
        # Basic stub implementation - returns minimal ParseResult
        return ParseResult(
            ast={"type": "TranslationUnit", "body": []},
            errors=[],
            warnings=[],
            tokens=[],
            metrics={},
            language=Language.CPP,
        )


def parse_code(code: str, language: Language) -> ParseResult:
    """Convenience function to parse code."""
    parser = CodeParser()
    return parser.parse_code(code, language)
