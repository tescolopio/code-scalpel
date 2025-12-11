"""
Base Normalizer - Abstract interface for language-specific normalizers.

A normalizer converts a language's native AST/CST into Unified IR.
All normalizers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union

from ..nodes import IRModule, IRNode


class BaseNormalizer(ABC):
    """
    Abstract base class for language-specific normalizers.

    A normalizer:
    1. Parses source code using the language's native parser
    2. Converts the native AST/CST to Unified IR
    3. Sets source_language on all nodes for semantic dispatch

    Example:
        >>> normalizer = PythonNormalizer()
        >>> ir = normalizer.normalize("x = 1 + 2")
        >>> ir.source_language
        'python'
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """
        Return the language name.

        This value is set on all IR nodes' source_language field.

        Returns:
            Language identifier (e.g., "python", "javascript")
        """
        pass

    @abstractmethod
    def normalize(self, source: str, filename: str = "<string>") -> IRModule:
        """
        Parse source code and normalize to Unified IR.

        Args:
            source: Source code string
            filename: Optional filename for error messages

        Returns:
            IRModule representing the normalized program

        Raises:
            SyntaxError: If source cannot be parsed
        """
        pass

    @abstractmethod
    def normalize_node(self, node: Any) -> Union[IRNode, List[IRNode], None]:
        """
        Normalize a single native AST/CST node to IR.

        This method dispatches based on node type to specific handlers.

        Args:
            node: Native AST/CST node from the language parser

        Returns:
            IRNode: Single normalized node
            List[IRNode]: Multiple nodes (e.g., multiple statements)
            None: Node should be skipped (comments, whitespace)

        Raises:
            NotImplementedError: If node type is not supported
        """
        pass

    def _set_language(self, node: IRNode) -> IRNode:
        """
        Set source_language on a node.

        Helper method for subclasses.
        """
        node.source_language = self.language
        return node
