from collections import defaultdict
from typing import Callable, Optional

import esprima
import esprima.error_handler

from ..base_parser import BaseParser, Language, ParseResult, PreprocessorConfig


class JavaScriptParser(BaseParser):
    """
    JavaScriptParser is responsible for parsing and analyzing JavaScript code.

    This class uses the Esprima library to parse JavaScript code into an Abstract Syntax Tree (AST),
    performs preprocessing steps, analyzes the code structure, and checks for potential issues.

    Attributes:
        None

    Methods:
        _preprocess_code(code: str, config: Optional[PreprocessorConfig]) -> str:
            Preprocess the JavaScript code based on the provided configuration.

        _parse_javascript(code: str) -> ParseResult:
            Parses JavaScript code with detailed analysis and returns the result.

        _analyze_javascript_code(ast: esprima.nodes.Node) -> Dict[str, int]:
            Analyzes the JavaScript code structure and returns metrics.

        _visit_node(node: esprima.nodes.Node, metrics: Dict[str, int]) -> None:
            Visits nodes in the AST and updates metrics.

        _check_javascript_code(ast: esprima.nodes.Node) -> List[str]:
            Checks for potential code issues and returns warnings.

        _visit_for_warnings(node: esprima.nodes.Node, warnings: List[str], find_identifiers: Callable) -> None:
            Visits nodes in the AST and collects warnings.

        get_children(node: esprima.nodes.Node) -> List[esprima.nodes.Node]:
            Returns the child nodes of a given node.
    """

    def _preprocess_code(self, code: str, config: Optional[PreprocessorConfig]) -> str:
        """
        Preprocess the JavaScript code.

        :param code: The JavaScript code to preprocess.
        :param config: Configuration for preprocessing.
        :return: The preprocessed code.
        """
        if config is None:
            config = PreprocessorConfig(
                remove_comments=False, normalize_whitespace=False
            )
        if config.remove_comments:
            code = self._remove_comments(code, Language.JAVASCRIPT)
        if config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        return code

    def _parse_javascript(self, code: str) -> ParseResult:
        """
        Parses JavaScript code with detailed analysis.

        :param code: The JavaScript code to parse.
        :return: The result of parsing the code.
        """
        errors = []
        warnings = []
        metrics = defaultdict(int)

        try:
            # Parse JavaScript code into AST using Esprima
            ast = esprima.parseScript(code, loc=True, tolerant=True)

            # Set parent nodes
            self._set_parent_nodes(ast)

            # Analyze code structure
            metrics.update(self._analyze_javascript_code(ast))

            # Check for potential issues
            warnings.extend(self._check_javascript_code(ast))

            return ParseResult(
                ast=ast,
                errors=errors,
                warnings=warnings,
                tokens=[],  # Tokens not implemented for JavaScript
                metrics=dict(metrics),
                language=Language.JAVASCRIPT,
            )

        except esprima.error_handler.Error as e:
            error_info = {
                "type": type(e).__name__,
                "message": e.message,
                "line": e.lineNumber,
                "column": e.column,
            }
            errors.append(error_info)
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=warnings,
                tokens=[],  # Tokens not implemented for JavaScript
                metrics=dict(metrics),
                language=Language.JAVASCRIPT,
            )

    def _analyze_javascript_code(self, ast: esprima.nodes.Node) -> dict[str, int]:
        """
        Analyzes the JavaScript code structure and returns metrics.

        :param ast: The Abstract Syntax Tree (AST) of the JavaScript code.
        :return: A dictionary of metrics.
        """
        metrics = defaultdict(int)
        self._visit_node(ast, metrics)
        return dict(metrics)

    def _visit_node(self, node: esprima.nodes.Node, metrics: dict[str, int]) -> None:
        """
        Visit nodes in the AST and update metrics.

        :param node: The current AST node.
        :param metrics: The metrics dictionary to update.
        """
        # Count different node types
        metrics[f"count_{type(node).__name__}"] += 1

        # Analyze complexity
        if isinstance(node, esprima.nodes.FunctionDeclaration):
            metrics["function_count"] += 1
            metrics["max_function_complexity"] = max(
                metrics["max_function_complexity"], self._calculate_complexity(node)
            )
        elif isinstance(node, esprima.nodes.ClassDeclaration):
            metrics["class_count"] += 1

        # Recursively visit child nodes
        for child in self.get_children(node):
            self._visit_node(child, metrics)

    def _calculate_complexity(self, node: esprima.nodes.FunctionDeclaration) -> int:
        """
        Calculate the complexity of a function node.

        :param node: The function node.
        :return: The complexity of the function.
        """
        complexity = 1  # Base complexity
        for child in self.get_children(node):
            if isinstance(
                child,
                (
                    esprima.nodes.IfStatement,
                    esprima.nodes.ForStatement,
                    esprima.nodes.WhileStatement,
                ),
            ):
                complexity += 1
            complexity += self._calculate_complexity(child)
        return complexity

    def _check_javascript_code(self, ast: esprima.nodes.Node) -> list[str]:
        """
        Check for potential code issues.

        :param ast: The Abstract Syntax Tree (AST) of the JavaScript code.
        :return: A list of warnings.
        """
        warnings = []

        def find_identifiers(
            node: esprima.nodes.Node, name: str
        ) -> list[esprima.nodes.Identifier]:
            """Find identifiers with the given name in the AST."""
            identifiers = []

            def visit(node: esprima.nodes.Node) -> None:
                if isinstance(node, esprima.nodes.Identifier) and node.name == name:
                    identifiers.append(node)
                for child in self.get_children(node):
                    visit(child)

            visit(node)
            return identifiers

        self._visit_for_warnings(ast, warnings, find_identifiers)
        return warnings

    def _visit_for_warnings(
        self, node: esprima.nodes.Node, warnings: list[str], find_identifiers: Callable
    ) -> None:
        """
        Visit nodes in the AST and collect warnings.

        :param node: The current AST node.
        :param warnings: The list of warnings to update.
        :param find_identifiers: Function to find identifiers in the AST.
        """
        # Check for unused variables
        if isinstance(node, esprima.nodes.VariableDeclarator):
            if not find_identifiers(node, node.id.name):
                warnings.append(
                    f"Unused variable '{node.id.name}' at line {node.loc.start.line}"
                )

        # Check for unreachable code
        if isinstance(node, esprima.nodes.ReturnStatement) and hasattr(node, "parent"):
            parent_children = self.get_children(node.parent)
            if any(
                isinstance(n, esprima.nodes.Statement)
                for n in parent_children[parent_children.index(node) + 1 :]
            ):
                warnings.append(
                    f"Unreachable code after return statement at line {node.loc.start.line}"
                )

        # Recursively visit child nodes
        for child in self.get_children(node):
            self._visit_for_warnings(child, warnings, find_identifiers)

    def get_children(self, node: esprima.nodes.Node) -> list[esprima.nodes.Node]:
        """
        Returns the child nodes of a given node.

        :param node: The node to get children from.
        :return: A list of child nodes.
        """
        children = []
        if isinstance(node, list):
            children.extend(
                [item for item in node if isinstance(item, esprima.nodes.Node)]
            )
        elif isinstance(node, dict):
            children.extend([v for k, v in node.items() if isinstance(v, (dict, list))])
        elif isinstance(node, esprima.nodes.Node):
            for field in node.__dict__.values():
                if isinstance(field, (list, dict, esprima.nodes.Node)):
                    children.append(field)
        return children

    def _set_parent_nodes(
        self, node: esprima.nodes.Node, parent: Optional[esprima.nodes.Node] = None
    ) -> None:
        """
        Set parent nodes for the AST.

        :param node: The current AST node.
        :param parent: The parent node.
        """
        node.parent = parent
        for child in self.get_children(node):
            self._set_parent_nodes(child, node)
