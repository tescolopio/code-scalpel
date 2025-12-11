from abc import ABC, abstractmethod
from typing import Any, List


class BaseParser(ABC):
    """
    Abstract base class for language-specific parsers.
    """

    @abstractmethod
    def parse(self, code: str) -> Any:
        """
        Parse source code into an AST.

        Args:
            code: Source code string

        Returns:
            AST object (type depends on language)
        """
        pass

    @abstractmethod
    def get_functions(self, ast_tree: Any) -> List[str]:
        """Get list of function names from AST."""
        pass

    @abstractmethod
    def get_classes(self, ast_tree: Any) -> List[str]:
        """Get list of class names from AST."""
        pass

    @abstractmethod
    def get_imports(self, ast_tree: Any) -> List[str]:
        """Get list of imported modules."""
        pass
