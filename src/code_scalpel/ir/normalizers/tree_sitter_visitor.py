"""
TreeSitter Visitor - Base class for walking tree-sitter CST nodes.

Tree-sitter produces Concrete Syntax Trees (CSTs) that include ALL tokens:
- Keywords (function, if, return)
- Operators (+, -, *, /)
- Delimiters (parentheses, braces, commas, semicolons)
- Whitespace (in some grammars)
- Comments

Unlike Python's ast module which gives you a clean AST, tree-sitter gives you
the raw parse tree. This visitor provides infrastructure for walking CSTs
and filtering out syntactic noise to produce IR nodes.

Design Philosophy:
    1. Type-safe: Map CST node types to visitor methods
    2. Flexible: Allow subclasses to override any node handler
    3. Debuggable: Track parent chain for error messages
    4. Noise-aware: Default handlers for common noise nodes

Usage:
    class JavaScriptVisitor(TreeSitterVisitor):
        language = "javascript"
        
        def visit_function_declaration(self, node):
            # node.children contains: 'function', name, params, body
            name = self.get_child_by_field(node, 'name')
            params = self.get_child_by_field(node, 'parameters')
            body = self.get_child_by_field(node, 'body')
            ...

Node Field Access:
    Tree-sitter nodes have TWO access patterns:
    1. Positional: node.children[i]
    2. Named fields: node.child_by_field_name('name')
    
    Always prefer named fields - they're stable across grammar versions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from ..nodes import IRNode, SourceLocation


# Type variable for the tree-sitter Node type
TSNode = TypeVar("TSNode")


@dataclass
class VisitorContext:
    """
    Context passed through the visitor tree.
    
    Attributes:
        filename: Source filename for error messages
        source: Original source code (for extracting text)
        parent_chain: Stack of parent nodes for debugging
        scope_stack: Current scope hierarchy (for name resolution)
    """
    filename: str = "<string>"
    source: str = ""
    parent_chain: List[Any] = None
    scope_stack: List[str] = None
    
    def __post_init__(self):
        if self.parent_chain is None:
            self.parent_chain = []
        if self.scope_stack is None:
            self.scope_stack = []


class TreeSitterVisitor(ABC, Generic[TSNode]):
    """
    Abstract base class for tree-sitter CST visitors.
    
    Subclasses must implement:
        - language: The language identifier
        - _get_node_type(node): Extract node type string
        - _get_children(node): Get child nodes
        - _get_text(node): Get node's source text
        - _get_location(node): Get source location
    
    Subclasses should implement visit_* methods for each node type:
        - visit_program(node) -> IRModule
        - visit_function_declaration(node) -> IRFunctionDef
        - visit_binary_expression(node) -> IRBinaryOp
        - etc.
    
    Node types with hyphens are converted to underscores:
        - "arrow_function" -> visit_arrow_function
        - "if_statement" -> visit_if_statement
    """
    
    def __init__(self):
        self.ctx: VisitorContext = VisitorContext()
        self._handlers: Dict[str, Callable] = {}
        self._register_handlers()
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language identifier (e.g., 'javascript')."""
        pass
    
    @abstractmethod
    def _get_node_type(self, node: TSNode) -> str:
        """Get the node type string (e.g., 'function_declaration')."""
        pass
    
    @abstractmethod
    def _get_children(self, node: TSNode) -> List[TSNode]:
        """Get list of child nodes."""
        pass
    
    @abstractmethod
    def _get_named_children(self, node: TSNode) -> List[TSNode]:
        """Get list of named (non-anonymous) child nodes."""
        pass
    
    @abstractmethod
    def _get_text(self, node: TSNode) -> str:
        """Get the source text of a node."""
        pass
    
    @abstractmethod
    def _get_location(self, node: TSNode) -> SourceLocation:
        """Get source location for a node."""
        pass
    
    @abstractmethod
    def _get_child_by_field(self, node: TSNode, field_name: str) -> Optional[TSNode]:
        """Get a child node by its field name."""
        pass
    
    @abstractmethod
    def _get_children_by_field(self, node: TSNode, field_name: str) -> List[TSNode]:
        """Get all children with a given field name."""
        pass
    
    # =========================================================================
    # Handler Registration
    # =========================================================================
    
    def _register_handlers(self) -> None:
        """
        Auto-register visit_* methods as handlers.
        
        Scans for methods named visit_<node_type> and registers them.
        """
        for name in dir(self):
            if name.startswith("visit_"):
                node_type = name[6:]  # Remove 'visit_' prefix
                method = getattr(self, name)
                if callable(method):
                    self._handlers[node_type] = method
    
    def register_handler(self, node_type: str, handler: Callable) -> None:
        """
        Manually register a handler for a node type.
        
        Args:
            node_type: The CST node type (e.g., 'function_declaration')
            handler: Callable that takes a node and returns IR
        """
        self._handlers[node_type] = handler
    
    # =========================================================================
    # Core Visitor Logic
    # =========================================================================
    
    def visit(self, node: TSNode) -> Union[IRNode, List[IRNode], None]:
        """
        Visit a node and return its IR representation.
        
        Dispatch order:
        1. Look up registered handler by node type
        2. Fall back to generic_visit if no handler
        
        Args:
            node: Tree-sitter node to visit
            
        Returns:
            IRNode, list of IRNodes, or None (for noise nodes)
        """
        node_type = self._get_node_type(node)
        
        # Push to parent chain for debugging
        self.ctx.parent_chain.append(node)
        
        try:
            # Look up handler
            handler = self._handlers.get(node_type)
            
            if handler is not None:
                return handler(node)
            else:
                return self.generic_visit(node)
                
        finally:
            # Pop from parent chain
            self.ctx.parent_chain.pop()
    
    def generic_visit(self, node: TSNode) -> Union[IRNode, List[IRNode], None]:
        """
        Default visitor for unhandled node types.
        
        Default behavior: visit all children and collect results.
        Override this for different default behavior.
        
        Returns:
            List of IR nodes from children, or None if all children return None
        """
        results = []
        for child in self._get_named_children(node):
            result = self.visit(child)
            if result is not None:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        
        return results if results else None
    
    def visit_children(self, node: TSNode) -> List[IRNode]:
        """
        Visit all children and collect non-None results.
        
        Convenience method for compound nodes.
        """
        results = []
        for child in self._get_named_children(node):
            result = self.visit(child)
            if result is not None:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        return results
    
    # =========================================================================
    # Noise Filtering
    # =========================================================================
    
    # Common syntactic noise that should return None
    NOISE_TYPES = frozenset({
        # Punctuation
        ",", ";", ":", ".", "?",
        "(", ")", "[", "]", "{", "}",
        # Operators as separate tokens (we read them from parent)
        "+", "-", "*", "/", "%", "**",
        "=", "==", "!=", "<", ">", "<=", ">=",
        "+=", "-=", "*=", "/=",
        "&&", "||", "!", "&", "|", "^", "~",
        "=>",
        # Keywords (extracted by parent nodes)
        "function", "class", "if", "else", "for", "while",
        "return", "break", "continue", "const", "let", "var",
        "async", "await", "import", "export", "from",
        # Whitespace and comments
        "comment", "line_comment", "block_comment",
    })
    
    def is_noise(self, node: TSNode) -> bool:
        """
        Check if a node is syntactic noise.
        
        Override in subclass to customize noise filtering.
        """
        node_type = self._get_node_type(node)
        return node_type in self.NOISE_TYPES
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def make_location(self, node: TSNode) -> SourceLocation:
        """Create SourceLocation from node."""
        return self._get_location(node)
    
    def get_text(self, node: TSNode) -> str:
        """Get source text for a node."""
        return self._get_text(node)
    
    def get_child_by_field(self, node: TSNode, field: str) -> Optional[TSNode]:
        """Get child by field name."""
        return self._get_child_by_field(node, field)
    
    def get_children_by_field(self, node: TSNode, field: str) -> List[TSNode]:
        """Get all children with field name."""
        return self._get_children_by_field(node, field)
    
    def find_child_by_type(self, node: TSNode, node_type: str) -> Optional[TSNode]:
        """Find first child of a specific type."""
        for child in self._get_children(node):
            if self._get_node_type(child) == node_type:
                return child
        return None
    
    def find_children_by_type(self, node: TSNode, node_type: str) -> List[TSNode]:
        """Find all children of a specific type."""
        return [
            child for child in self._get_children(node)
            if self._get_node_type(child) == node_type
        ]
    
    def error(self, message: str, node: TSNode) -> None:
        """
        Raise an error with source location context.
        
        Args:
            message: Error message
            node: Node where error occurred
        """
        loc = self._get_location(node)
        raise ValueError(f"{loc}: {message}")
    
    def warn(self, message: str, node: TSNode) -> None:
        """
        Log a warning with source location context.
        
        Args:
            message: Warning message
            node: Node where warning occurred
        """
        import warnings
        loc = self._get_location(node)
        warnings.warn(f"{loc}: {message}")
    
    def debug_node(self, node: TSNode, indent: int = 0) -> str:
        """
        Generate debug string showing node structure.
        
        Useful for understanding tree-sitter output.
        """
        prefix = "  " * indent
        node_type = self._get_node_type(node)
        text = self._get_text(node)
        
        # Truncate long text
        if len(text) > 50:
            text = text[:47] + "..."
        text = text.replace("\n", "\\n")
        
        lines = [f"{prefix}{node_type}: {text!r}"]
        
        for child in self._get_children(node):
            lines.append(self.debug_node(child, indent + 1))
        
        return "\n".join(lines)
