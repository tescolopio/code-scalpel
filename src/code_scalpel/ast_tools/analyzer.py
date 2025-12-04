import ast
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import astor
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FunctionMetrics:
    """Metrics for a function."""

    name: str
    args: List[str]
    kwargs: List[Tuple[str, Optional[str]]]  # (arg_name, default_value)
    return_type: Optional[str]
    complexity: int
    line_count: int
    calls_made: List[str]
    variables_used: Set[str]


@dataclass
class ClassMetrics:
    """Metrics for a class."""

    name: str
    bases: List[str]
    methods: List[str]
    attributes: Dict[str, Optional[str]]  # attribute_name -> type_hint
    instance_vars: Set[str]
    class_vars: Set[str]


class ASTAnalyzer:
    """
    Advanced Python code analyzer using Abstract Syntax Trees (ASTs).
    """

    def __init__(self, cache_enabled: bool = True):
        self.ast_cache: Dict[str, ast.AST] = {}
        self.cache_enabled = cache_enabled
        self.current_context: List[str] = []  # Track current function/class context

    def parse_to_ast(self, code: str) -> ast.AST:
        """Parse Python code into an AST with caching."""
        if self.cache_enabled and code in self.ast_cache:
            return self.ast_cache[code]

        try:
            tree = ast.parse(code)
            if self.cache_enabled:
                self.ast_cache[code] = tree
            return tree
        except SyntaxError as e:
            logger.error(f"Syntax error while parsing code: {e}")
            raise

    def ast_to_code(self, node: ast.AST) -> str:
        """Convert AST back to source code with formatting."""
        return astor.to_source(node)

    def analyze_function(self, node: ast.FunctionDef) -> FunctionMetrics:
        """Analyze a function definition comprehensively."""
        # Extract arguments
        args = [arg.arg for arg in node.args.args]
        kwargs = [
            (
                node.args.args[len(node.args.args) - len(node.args.defaults) + i].arg,
                astor.to_source(default).strip(),
            )
            for i, default in enumerate(node.args.defaults)
        ]

        # Extract return type
        return_type = astor.to_source(node.returns).strip() if node.returns else None

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        # Count lines
        line_count = self._count_node_lines(node)

        # Analyze function calls
        calls_made = self._extract_function_calls(node)

        # Analyze variables
        variables_used = self._extract_variables(node)

        return FunctionMetrics(
            name=node.name,
            args=args,
            kwargs=kwargs,
            return_type=return_type,
            complexity=complexity,
            line_count=line_count,
            calls_made=calls_made,
            variables_used=variables_used,
        )

    def analyze_class(self, node: ast.ClassDef) -> ClassMetrics:
        """Analyze a class definition comprehensively."""
        # Extract base classes
        bases = [astor.to_source(base).strip() for base in node.bases]

        # Extract methods and attributes
        methods = []
        attributes = {}
        instance_vars = set()
        class_vars = set()

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes[item.target.id] = (
                    astor.to_source(item.annotation).strip()
                    if item.annotation
                    else None
                )
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_vars.add(target.id)

        # Find instance variables in __init__
        init_method = next(
            (
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name == "__init__"
            ),
            None,
        )
        if init_method:
            instance_vars = self._extract_instance_vars(init_method)

        return ClassMetrics(
            name=node.name,
            bases=bases,
            methods=methods,
            attributes=attributes,
            instance_vars=instance_vars,
            class_vars=class_vars,
        )

    def analyze_code_style(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code style and potential issues."""
        issues = defaultdict(list)

        # Check function lengths
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._count_node_lines(node) > 20:
                    issues["long_functions"].append(
                        f"Function '{node.name}' is too long ({self._count_node_lines(node)} lines)"
                    )

        # Check nesting depth
        self._check_nesting_depth(tree, issues)

        # Check naming conventions
        self._check_naming_conventions(tree, issues)

        return dict(issues)

    def find_security_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify potential security issues in the code."""
        issues = []

        # Check for dangerous function calls
        dangerous_functions = {
            "eval",
            "exec",
            "os.system",
            "subprocess.call",
            "subprocess.Popen",
            "pickle.loads",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in dangerous_functions:
                    issues.append(
                        {
                            "type": "dangerous_function",
                            "function": func_name,
                            "line": getattr(node, "lineno", None),
                        }
                    )

        # Check for SQL injection vulnerabilities
        self._check_sql_injection(tree, issues)

        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a code block."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity

    def _count_node_lines(self, node: ast.AST) -> int:
        """Count the number of lines in a node."""
        return node.end_lineno - node.lineno + 1 if hasattr(node, "end_lineno") else 1

    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract all function calls within a node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(
                        f"{astor.to_source(child.func.value).strip()}.{child.func.attr}"
                    )
        return calls

    def _extract_variables(self, node: ast.AST) -> Set[str]:
        """Extract all variables used within a node."""
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables.add(child.id)
        return variables

    def _extract_instance_vars(self, init_method: ast.FunctionDef) -> Set[str]:
        """Extract instance variables from __init__ method."""
        instance_vars = set()
        for node in ast.walk(init_method):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                instance_vars.add(node.attr)
        return instance_vars

    def _check_nesting_depth(self, tree: ast.AST, issues: defaultdict) -> None:
        """Check for excessive nesting depth."""

        def get_nesting_depth(node, current_depth=0):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                current_depth += 1
                if current_depth > 3:
                    issues["deep_nesting"].append(
                        f"Deep nesting detected at line {node.lineno}"
                    )
            for child in ast.iter_child_nodes(node):
                get_nesting_depth(child, current_depth)

        get_nesting_depth(tree)

    def _check_naming_conventions(self, tree: ast.AST, issues: defaultdict) -> None:
        """Check Python naming conventions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name[0].isupper():
                issues["naming_conventions"].append(
                    f"Class '{node.name}' should use CapWords convention"
                )
            elif isinstance(node, ast.FunctionDef) and not node.name.islower():
                issues["naming_conventions"].append(
                    f"Function '{node.name}' should use lowercase_with_underscores"
                )

    def _check_sql_injection(self, tree: ast.AST, issues: List[Dict[str, Any]]) -> None:
        """Check for potential SQL injection vulnerabilities."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"execute", "executemany"}:
                    # Check if string formatting or concatenation is used
                    if any(
                        isinstance(arg, (ast.BinOp, ast.JoinedStr)) for arg in node.args
                    ):
                        issues.append(
                            {
                                "type": "sql_injection",
                                "line": node.lineno,
                                "message": "Possible SQL injection vulnerability",
                            }
                        )

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the full name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{astor.to_source(node.func.value).strip()}.{node.func.attr}"
        return ""
