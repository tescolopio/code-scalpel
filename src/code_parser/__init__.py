# __init__.py

from .base_parser import BaseParser, Language, ParseResult, PreprocessorConfig
from typing import Optional
from .python_parsers.python_parsers_flake8 import PythonParser as Flake8PythonParser
from .python_parsers.python_parsers_mypy import PythonParser as MypyPythonParser
from .python_parsers.python_parsers_prospector import PythonParser as ProspectorPythonParser
from .python_parsers.python_parsers_pylint import PythonParser as PylintPythonParser
from .python_parsers.python_parsers_bandit import PythonParser as BanditPythonParser
from .python_parsers.python_parsers_pydocstyle import PythonParser as PydocstylePythonParser
from .python_parsers.python_parsers_pycodestyle import PythonParser as PycodestylePythonParser

class CodeParser:
    """
    CodeParser is a multi-language code parser that supports various programming languages.
    It dynamically initializes language-specific parsers and provides a unified interface for parsing code.
    """

    def __init__(self):
        self.parsers = {
            Language.PYTHON: {
                'flake8': Flake8PythonParser(),
                'mypy': MypyPythonParser(),
                'prospector': ProspectorPythonParser(),
                'pylint': PylintPythonParser(),
                'bandit': BanditPythonParser(),
                'pydocstyle': PydocstylePythonParser(),
                'pycodestyle': PycodestylePythonParser(),
            }
        }

    def parse_code(self, code: str, language: Language, tool: str, preprocess: bool = True, config: Optional[PreprocessorConfig] = None) -> ParseResult:
        """
        Parse code with comprehensive analysis.

        Args:
            code (str): Source code to parse.
            language (Language): Programming language of the source code.
            tool (str): The specific tool to use for parsing (e.g., 'flake8', 'mypy').
            preprocess (bool): Whether to preprocess the code.
            config (Optional[PreprocessorConfig]): Configuration for preprocessing.

        Returns:
            ParseResult: Result of the parsing process, including AST, errors, warnings, tokens, and metrics.

        Raises:
            ValueError: If the specified language or tool is not supported.
        """
        parser = self.parsers.get(language, {}).get(tool)
        if parser:
            return parser.parse_code(code, preprocess, config)
        else:
            raise ValueError(f"Unsupported language or tool: {language}, {tool}")

__all__ = [
    "BaseParser", "Language", "ParseResult", "PreprocessorConfig",
    "Flake8PythonParser", "MypyPythonParser", "ProspectorPythonParser", "PylintPythonParser",
    "BanditPythonParser", "PydocstylePythonParser", "PycodestylePythonParser",
    "CodeParser"
]