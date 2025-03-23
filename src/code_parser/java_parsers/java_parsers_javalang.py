from collections import defaultdict
from typing import Optional, Dict, List, Any, Callable
import javalang

from ..base_parser import BaseParser, ParseResult, PreprocessorConfig, Language

class JavaParser(BaseParser):
    """
    JavaParser is responsible for parsing and analyzing Java code.

    This class uses the javalang library to parse Java code into an Abstract Syntax Tree (AST),
    performs preprocessing steps, analyzes the code structure, and checks for potential issues.

    Attributes:
        None

    Methods:
        _preprocess_code(code: str, config: Optional[PreprocessorConfig]) -> str:
            Preprocess the Java code based on the provided configuration.
        
        _parse_java(code: str) -> ParseResult:
            Parses Java code with detailed analysis and returns the result.
        
        _analyze_java_code(ast: javalang.tree.CompilationUnit) -> Dict[str, int]:
            Analyzes the Java code structure and returns metrics.
        
        _visit_node(node: javalang.ast.Node, metrics: Dict[str, int]) -> None:
            Visits nodes in the AST and updates metrics.
        
        _check_java_code(ast: javalang.tree.CompilationUnit) -> List[str]:
            Checks for potential code issues and returns warnings.
        
        _visit_for_warnings(node: javalang.ast.Node, warnings: List[str], find_identifiers: Callable) -> None:
            Visits nodes in the AST and collects warnings.
        
        get_children(node: javalang.ast.Node) -> List[javalang.ast.Node]:
            Returns the child nodes of a given node.
    """
    def _preprocess_code(self, code: str, config: Optional[PreprocessorConfig]) -> str:
        """
        Preprocess the Java code.

        :param code: The Java code to preprocess.
        :param config: Configuration for preprocessing.
        :return: The preprocessed code.
        """
        if config is None:
            config = PreprocessorConfig(remove_comments=False, normalize_whitespace=False)
        if config.remove_comments:
            code = self._remove_comments(code, Language.JAVA)
        if config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        return code

    def _parse_java(self, code: str) -> ParseResult:
        """
        Parses Java code with detailed analysis.

        :param code: The Java code to parse.
        :return: The result of parsing the code.
        """
        errors = []
        warnings = []
        metrics = defaultdict(int)

        try:
            # Parse Java code into AST using javalang
            ast = javalang.parse.parse(code)

            # Analyze code structure
            metrics.update(self._analyze_java_code(ast))

            # Check for potential issues
            warnings.extend(self._check_java_code(ast))

            return ParseResult(
                ast=ast,
                errors=errors,
                warnings=warnings,
                tokens=[],  # Tokens not implemented for Java
                metrics=dict(metrics),
                language=Language.JAVA
            )

        except javalang.parser.JavaSyntaxError as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'line': e.position[0] if e.position else None,
                'column': e.position[1] if e.position else None
            }
            errors.append(error_info)
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=warnings,
                tokens=[],  # Tokens not implemented for Java
                metrics=dict(metrics),
                language=Language.JAVA
            )

    def _analyze_java_code(self, ast: javalang.tree.CompilationUnit) -> Dict[str, int]:
        """
        Analyzes the Java code structure and returns metrics.

        :param ast: The Abstract Syntax Tree (AST) of the Java code.
        :return: A dictionary of metrics.
        """
        metrics = defaultdict(int)
        self._visit_node(ast, metrics)
        return dict(metrics)

    def _visit_node(self, node: javalang.ast.Node, metrics: Dict[str, int]) -> None:
        """
        Visit nodes in the AST and update metrics.

        :param node: The current AST node.
        :param metrics: The metrics dictionary to update.
        """
        # Count different node types
        metrics[f'count_{type(node).__name__}'] += 1

        # Analyze complexity
        if isinstance(node, javalang.tree.MethodDeclaration):
            metrics['method_count'] += 1
            metrics['max_method_complexity'] = max(
                metrics['max_method_complexity'],
                self._calculate_complexity(node)
            )
        elif isinstance(node, javalang.tree.ClassDeclaration):
            metrics['class_count'] += 1

        # Recursively visit child nodes
        for child in self.get_children(node):
            self._visit_node(child, metrics)

    def _calculate_complexity(self, node: javalang.tree.MethodDeclaration) -> int:
        """
        Calculate the complexity of a method node.

        :param node: The method node.
        :return: The complexity of the method.
        """
        complexity = 1  # Base complexity
        for child in self.get_children(node):
            if isinstance(child, (javalang.tree.IfStatement, javalang.tree.ForStatement, javalang.tree.WhileStatement)):
                complexity += 1
            complexity += self._calculate_complexity(child)
        return complexity

    def _check_java_code(self, ast: javalang.tree.CompilationUnit) -> List[str]:
        """
        Check for potential code issues.

        :param ast: The Abstract Syntax Tree (AST) of the Java code.
        :return: A list of warnings.
        """
        warnings = []

        def find_identifiers(node: javalang.ast.Node, name: str) -> List[javalang.tree.Identifier]:
            """Find identifiers with the given name in the AST."""
            identifiers = []

            def visit(node: javalang.ast.Node) -> None:
                if isinstance(node, javalang.tree.Identifier) and node.value == name:
                    identifiers.append(node)
                for child in self.get_children(node):
                    visit(child)

            visit(node)
            return identifiers

        self._visit_for_warnings(ast, warnings, find_identifiers)
        return warnings

    def _visit_for_warnings(self, node: javalang.ast.Node, warnings: List[str], find_identifiers: Callable) -> None:
        """
        Visit nodes in the AST and collect warnings.

        :param node: The current AST node.
        :param warnings: The list of warnings to update.
        :param find_identifiers: Function to find identifiers in the AST.
        """
        # Check for unused variables
        if isinstance(node, javalang.tree.VariableDeclarator):
            if not find_identifiers(node, node.name):
                warnings.append(f"Unused variable '{node.name}' at line {node.position.line}")

        # Check for unreachable code
        if isinstance(node, javalang.tree.ReturnStatement) and hasattr(node, 'parent'):
            parent_children = self.get_children(node.parent)
            if any(isinstance(n, javalang.tree.Statement) for n in parent_children[parent_children.index(node) + 1:]):
                warnings.append(f"Unreachable code after return statement at line {node.position.line}")

        # Recursively visit child nodes
        for child in self.get_children(node):
            self._visit_for_warnings(child, warnings, find_identifiers)

    def get_children(self, node: javalang.ast.Node) -> List[javalang.ast.Node]:
        """
        Returns the child nodes of a given node.

        :param node: The node to get children from.
        :return: A list of child nodes.
        """
        children = []
        if isinstance(node, list):
            children.extend([item for item in node if isinstance(item, javalang.ast.Node)])
        elif isinstance(node, dict):
            children.extend([v for k, v in node.items() if isinstance(v, (dict, list))])
        elif isinstance(node, javalang.ast.Node):
            for field in node.__dict__.values():
                if isinstance(field, (list, dict, javalang.ast.Node)):
                    children.append(field)
        return children