# Core imports
from .analyzer import ASTAnalyzer, ClassMetrics, FunctionMetrics
from .builder import ASTBuilder

# These imports might fail due to incomplete implementations, handle gracefully
try:
    from .transformer import ASTTransformer
except ImportError:
    ASTTransformer = None

try:
    from .visualizer import ASTVisualizer
except ImportError:
    ASTVisualizer = None

try:
    from .validator import ASTValidator
except ImportError:
    ASTValidator = None

try:
    from .utils import (
        get_all_names,
        get_node_type,
        is_constant,
    )
except ImportError:
    is_constant = None
    get_node_type = None
    get_all_names = None


# Create convenience functions using the default ASTBuilder instance
_default_builder = ASTBuilder()


def build_ast(code: str, preprocess: bool = True, validate: bool = True):
    """Build an AST from Python code."""
    return _default_builder.build_ast(code, preprocess, validate)


def build_ast_from_file(filepath: str, preprocess: bool = True, validate: bool = True):
    """Build an AST from a Python source file."""
    return _default_builder.build_ast_from_file(filepath, preprocess, validate)


# Create convenience function for visualization
def visualize_ast(tree, output_file="ast_visualization", format="png", view=True):
    """Visualize an AST using graphviz."""
    if ASTVisualizer is not None:
        visualizer = ASTVisualizer()
        return visualizer.visualize(tree, output_file, format, view)
    raise ImportError("ASTVisualizer not available")


__all__ = [
    "ASTAnalyzer",
    "FunctionMetrics",
    "ClassMetrics",
    "ASTBuilder",
    "ASTTransformer",
    "ASTVisualizer",
    "ASTValidator",
    "is_constant",
    "get_node_type",
    "get_all_names",
    "build_ast",
    "build_ast_from_file",
    "visualize_ast",
]
