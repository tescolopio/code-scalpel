import ast
from dataclasses import dataclass
from typing import Callable, Optional, Union

import astor


@dataclass
class TransformationRule:
    """Defines a transformation rule."""

    pattern: Union[str, ast.AST]  # Code pattern to match
    replacement: Union[str, ast.AST]  # Replacement pattern
    condition: Optional[Callable] = None  # Optional condition for applying the rule


class ASTTransformer(ast.NodeTransformer):
    """Advanced AST transformer with pattern matching and complex transformations."""

    def __init__(self):
        super().__init__()
        self.var_mapping: dict[str, str] = {}
        self.func_mapping: dict[str, str] = {}
        self.transformation_rules: list[TransformationRule] = []
        self.context: list[ast.AST] = []
        self.modified = False

    def add_transformation_rule(self, rule: TransformationRule) -> None:
        """Add a new transformation rule."""
        self.transformation_rules.append(rule)

    def rename_variable(
        self, old_name: str, new_name: str, scope: Optional[str] = None
    ) -> None:
        """Register a variable rename transformation."""
        self.var_mapping[(old_name, scope if scope else "")] = new_name

    def rename_function(self, old_name: str, new_name: str) -> None:
        """Register a function rename transformation."""
        self.func_mapping[old_name] = new_name

    def visit(self, node: ast.AST) -> ast.AST:
        """Enhanced visit method with context tracking."""
        self.context.append(node)
        result = super().visit(node)
        self.context.pop()
        return result

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Transform variable names with scope awareness."""
        current_scope = self._get_current_scope()

        # Check scoped mapping first, then global mapping
        new_name = self.var_mapping.get(
            (node.id, current_scope)
        ) or self.var_mapping.get((node.id, ""))

        if new_name:
            self.modified = True
            return ast.Name(id=new_name, ctx=node.ctx)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Transform function definitions."""
        # Handle function renaming
        new_name = self.func_mapping.get(node.name)
        if new_name:
            self.modified = True
            node.name = new_name

        # Transform function body
        node.body = [self.visit(stmt) for stmt in node.body]

        # Transform decorators
        if node.decorator_list:
            node.decorator_list = [self.visit(d) for d in node.decorator_list]

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Transform function calls with pattern matching."""
        for rule in self.transformation_rules:
            if self._matches_pattern(node, rule.pattern):
                if not rule.condition or rule.condition(node):
                    self.modified = True
                    return self._apply_replacement(node, rule.replacement)

        # Transform function name and arguments
        node.func = self.visit(node.func)
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [self.visit(kw) for kw in node.keywords]

        return node

    def extract_method(
        self, node: ast.AST, new_func_name: str, args: list[str] = None
    ) -> tuple[ast.FunctionDef, ast.Call]:
        """Extract a code block into a new method."""
        # Analyze used variables
        used_vars = set()
        defined_vars = set()

        class VarCollector(ast.NodeVisitor):
            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Load):
                    used_vars.add(n.id)
                elif isinstance(n.ctx, ast.Store):
                    defined_vars.add(n.id)

        VarCollector().visit(node)

        # Determine parameters
        params = args if args else list(used_vars - defined_vars)

        # Create new function
        new_func = ast.FunctionDef(
            name=new_func_name,
            args=ast.arguments(
                args=[ast.arg(arg=p) for p in params],
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[node] if isinstance(node, ast.stmt) else [ast.Expr(node)],
            decorator_list=[],
        )

        # Create function call
        call = ast.Call(
            func=ast.Name(id=new_func_name, ctx=ast.Load()),
            args=[ast.Name(id=p, ctx=ast.Load()) for p in params],
            keywords=[],
        )

        return new_func, call

    def inline_variable(self, node: ast.Name) -> Optional[ast.AST]:
        """Inline a variable's value at its use sites."""
        # Find variable definition
        assignment = self._find_variable_definition(node.id)
        if assignment and isinstance(assignment, ast.Assign):
            return self.visit(assignment.value)
        return None

    def transform_code(self, code: str) -> str:
        """Transform code with all registered transformations."""
        tree = ast.parse(code)
        self.modified = False
        transformed = self.visit(tree)

        # Fix any AST inconsistencies
        ast.fix_missing_locations(transformed)

        return astor.to_source(transformed)

    def _matches_pattern(self, node: ast.AST, pattern: Union[str, ast.AST]) -> bool:
        """Check if a node matches a pattern."""
        if isinstance(pattern, str):
            pattern = ast.parse(pattern).body[0]
            if isinstance(pattern, ast.Expr):
                pattern = pattern.value

        # Compare AST structures
        return self._compare_nodes(node, pattern)

    def _compare_nodes(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Compare two AST nodes for structural equality."""
        if type(node1) is not type(node2):
            return False

        for field in node1._fields:
            val1 = getattr(node1, field)
            val2 = getattr(node2, field)

            if isinstance(val1, list):
                if not isinstance(val2, list) or len(val1) != len(val2):
                    return False
                for v1, v2 in zip(val1, val2):
                    if not self._compare_nodes(v1, v2):
                        return False
            elif isinstance(val1, ast.AST):
                if not self._compare_nodes(val1, val2):
                    return False
            elif val1 != val2:
                return False

        return True

    def _apply_replacement(
        self, node: ast.AST, replacement: Union[str, ast.AST]
    ) -> ast.AST:
        """Apply a replacement pattern."""
        if isinstance(replacement, str):
            replacement = ast.parse(replacement).body[0]
            if isinstance(replacement, ast.Expr):
                replacement = replacement.value

        # Copy relevant attributes from original node
        for attr in ["lineno", "col_offset"]:
            if hasattr(node, attr):
                setattr(replacement, attr, getattr(node, attr))

        return replacement

    def _get_current_scope(self) -> str:
        """Get the name of the current function/class scope."""
        for node in reversed(self.context):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                return node.name
        return ""
