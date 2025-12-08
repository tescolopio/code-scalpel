import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import networkx as nx


class NodeType(Enum):
    """Types of nodes in the PDG."""

    ASSIGN = "assign"
    IF = "if"
    WHILE = "while"
    FOR = "for"
    CALL = "call"
    RETURN = "return"
    FUNCTION = "function"
    CLASS = "class"
    TRY = "try"
    EXCEPT = "except"
    IMPORT = "import"


@dataclass
class Scope:
    """Represents a scope in the code."""

    type: str
    name: str
    node_id: str
    parent: Optional["Scope"] = None
    variables: dict[str, str] = None  # var_name -> defining_node_id

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}


class PDGBuilder(ast.NodeVisitor):
    """Enhanced Program Dependence Graph Builder."""

    def __init__(self, track_constants: bool = True, interprocedural: bool = True):
        self.graph = nx.DiGraph()
        self.scopes: list[Scope] = []
        self.control_deps: list[str] = []
        self.loop_deps: list[str] = []
        self.exception_deps: list[str] = []
        self.call_graph: nx.DiGraph = nx.DiGraph()
        self.track_constants = track_constants
        self.interprocedural = interprocedural
        self.current_function: Optional[str] = None
        self.node_counter = defaultdict(int)

    def build(self, code: str) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build PDG and call graph from code."""
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph, self.call_graph

    def visit_Module(self, node: ast.Module):
        """Handle module-level code by creating a module scope."""
        # Create module-level scope for variable tracking
        self.enter_scope("module", "__module__", "module_0")
        
        # Visit all top-level statements
        for stmt in node.body:
            self.visit(stmt)
        
        # Exit module scope
        self.exit_scope()

    def enter_scope(self, type_: str, name: str, node_id: str):
        """Enter a new scope."""
        parent = self.scopes[-1] if self.scopes else None
        scope = Scope(type_, name, node_id, parent)
        self.scopes.append(scope)
        return scope

    def exit_scope(self):
        """Exit the current scope."""
        return self.scopes.pop() if self.scopes else None

    def get_current_scope(self) -> Optional[Scope]:
        """Get the current scope."""
        return self.scopes[-1] if self.scopes else None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions."""
        node_id = self._get_node_id("function")

        # Add function node
        self.graph.add_node(
            node_id,
            type=NodeType.FUNCTION.value,
            name=node.name,
            args=[arg.arg for arg in node.args.args],
            returns=ast.unparse(node.returns) if node.returns else None,
            lineno=node.lineno,
        )

        # Add to call graph
        self.call_graph.add_node(node.name, node_id=node_id)

        # Handle decorators
        for decorator in node.decorator_list:
            self._process_decorator(decorator, node_id)

        # Enter function scope
        prev_function = self.current_function
        self.current_function = node.name
        scope = self.enter_scope("function", node.name, node_id)

        # Add parameter nodes
        for arg in node.args.args:
            arg_id = self._get_node_id("param")
            self.graph.add_node(
                arg_id, type="parameter", name=arg.arg, lineno=arg.lineno
            )
            self.graph.add_edge(node_id, arg_id, type="parameter_dependency")
            scope.variables[arg.arg] = arg_id

        # Process function body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")

        # Exit function scope
        self.exit_scope()
        self.current_function = prev_function

    def visit_If(self, node: ast.If):
        """Handle if statements."""
        node_id = self._get_node_id("if")

        # Add if node
        self.graph.add_node(
            node_id,
            type=NodeType.IF.value,
            condition=ast.unparse(node.test),
            lineno=node.lineno,
        )

        # Add data dependencies for condition
        for var in self._extract_variables(node.test):
            if def_node := self._find_definition(var):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Enter control context
        self.control_deps.append(node_id)

        # Process if body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")

        # Process else/elif body
        for stmt in node.orelse:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")

        # Exit control context
        self.control_deps.pop()

        return node_id

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        node_id = self._get_node_id("class")

        # Add class node
        self.graph.add_node(
            node_id,
            type=NodeType.CLASS.value,
            name=node.name,
            bases=[ast.unparse(base) for base in node.bases],
            lineno=node.lineno,
        )

        # Enter class scope
        self.enter_scope("class", node.name, node_id)

        # Process class body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")

        # Exit class scope
        self.exit_scope()

    def visit_For(self, node: ast.For):
        """Handle for loops."""
        node_id = self._get_node_id("for")

        # Add for loop node
        self.graph.add_node(
            node_id,
            type=NodeType.FOR.value,
            target=ast.unparse(node.target),
            iter=ast.unparse(node.iter),
            lineno=node.lineno,
        )

        # Add data dependencies for iterator
        for var in self._extract_variables(node.iter):
            if def_node := self._find_definition(var):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Handle loop variable
        self._process_loop_variable(node.target, node_id)

        # Enter loop context
        self.control_deps.append(node_id)
        self.loop_deps.append(node_id)

        # Process loop body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")
            self.graph.add_edge(node_id, stmt_id, type="loop_dependency")

        # Process else block if present
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
                stmt_id = list(self.graph.nodes)[-1]
                self.graph.add_edge(node_id, stmt_id, type="control_dependency")

        # Exit loop context
        self.control_deps.pop()
        self.loop_deps.pop()

    def visit_While(self, node: ast.While):
        """Handle while loops."""
        node_id = self._get_node_id("while")

        # Add while loop node
        self.graph.add_node(
            node_id,
            type=NodeType.WHILE.value,
            condition=ast.unparse(node.test),
            lineno=node.lineno,
        )

        # Add data dependencies for condition
        for var in self._extract_variables(node.test):
            if def_node := self._find_definition(var):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Enter loop context
        self.control_deps.append(node_id)
        self.loop_deps.append(node_id)

        # Process loop body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(node_id, stmt_id, type="control_dependency")
            self.graph.add_edge(node_id, stmt_id, type="loop_dependency")

        # Exit loop context
        self.control_deps.pop()
        self.loop_deps.pop()

    def visit_Try(self, node: ast.Try):
        """Handle try-except blocks."""
        try_id = self._get_node_id("try")

        # Add try node
        self.graph.add_node(try_id, type=NodeType.TRY.value, lineno=node.lineno)

        # Enter try context
        self.control_deps.append(try_id)
        self.exception_deps.append(try_id)

        # Process try body
        for stmt in node.body:
            self.visit(stmt)
            stmt_id = list(self.graph.nodes)[-1]
            self.graph.add_edge(try_id, stmt_id, type="control_dependency")

        # Process except handlers
        for handler in node.handlers:
            handler_id = self._get_node_id("except")
            self.graph.add_node(
                handler_id,
                type=NodeType.EXCEPT.value,
                exception_type=ast.unparse(handler.type) if handler.type else None,
                lineno=handler.lineno,
            )
            self.graph.add_edge(try_id, handler_id, type="exception_dependency")

            # Process except body
            for stmt in handler.body:
                self.visit(stmt)
                stmt_id = list(self.graph.nodes)[-1]
                self.graph.add_edge(handler_id, stmt_id, type="control_dependency")

        # Exit try context
        self.control_deps.pop()
        self.exception_deps.pop()

    def visit_Assign(self, node: ast.Assign):
        """Handle assignment statements."""
        node_id = self._get_node_id("assign")

        # Extract target variable names
        target_names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        target_names.append(elt.id)

        # Add assignment node
        self.graph.add_node(
            node_id,
            type=NodeType.ASSIGN.value,
            targets=target_names,
            value=ast.unparse(node.value),
            lineno=node.lineno,
            defines=target_names,
        )

        # Add control dependency if inside a control structure
        if self.control_deps:
            for ctrl_id in self.control_deps:
                self.graph.add_edge(ctrl_id, node_id, type="control_dependency")

        # Add data dependencies for variables used in RHS
        for var in self._extract_variables(node.value):
            if def_node := self._find_definition(var):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Visit any calls in the value expression
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call):
                call_id = self.visit_Call(child)
                if call_id:  # pragma: no branch - visit_Call always returns node_id
                    self.graph.add_edge(call_id, node_id, type="data_dependency")

        # Register variable definitions
        for var_name in target_names:
            self._add_variable_definition(var_name, node_id)

        return node_id

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment statements (+=, -=, etc.)."""
        node_id = self._get_node_id("assign")

        # Get target name
        target_name = node.target.id if isinstance(node.target, ast.Name) else None

        # Add assignment node
        self.graph.add_node(
            node_id,
            type=NodeType.ASSIGN.value,
            targets=[target_name] if target_name else [],
            value=ast.unparse(node.value),
            op=type(node.op).__name__,
            lineno=node.lineno,
            defines=[target_name] if target_name else [],
        )

        # Add control dependency if inside a control structure
        if self.control_deps:
            for ctrl_id in self.control_deps:
                self.graph.add_edge(ctrl_id, node_id, type="control_dependency")

        # Add data dependencies - includes the target itself (x += 1 uses x)
        if target_name:
            if def_node := self._find_definition(target_name):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Add data dependencies for variables used in RHS
        for var in self._extract_variables(node.value):
            if def_node := self._find_definition(var):
                self.graph.add_edge(def_node, node_id, type="data_dependency")

        # Register variable definition
        if target_name:
            self._add_variable_definition(target_name, node_id)

        return node_id

    def visit_Call(self, node: ast.Call):
        """Handle function calls."""
        node_id = self._get_node_id("call")

        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
        else:
            func_name = ast.unparse(node.func)

        # Add call node
        self.graph.add_node(
            node_id, type=NodeType.CALL.value, function=func_name, lineno=node.lineno
        )

        # Add to call graph if we're in a function
        if self.current_function:
            self.call_graph.add_edge(self.current_function, func_name)

        # Process arguments
        for idx, arg in enumerate(node.args):
            self._process_call_argument(arg, node_id, idx)

        # Process keyword arguments
        for keyword in node.keywords:
            self._process_call_keyword(keyword, node_id)

        return node_id

    def _process_call_argument(self, arg: ast.AST, call_id: str, index: int):
        """Process a function call argument."""
        # Add data dependencies for variables in argument
        for var in self._extract_variables(arg):
            if def_node := self._find_definition(var):
                self.graph.add_edge(
                    def_node, call_id, type="data_dependency", arg_index=index
                )

    def _process_call_keyword(self, keyword: ast.keyword, call_id: str):
        """Process a function call keyword argument."""
        for var in self._extract_variables(keyword.value):
            if def_node := self._find_definition(var):
                self.graph.add_edge(
                    def_node, call_id, type="data_dependency", keyword=keyword.arg
                )

    def _process_decorator(self, decorator: ast.AST, function_id: str):
        """Process a function decorator."""
        decorator_id = self.visit(decorator)
        # Only add edge if decorator was processed (some decorators return None)
        if decorator_id is not None:
            self.graph.add_edge(decorator_id, function_id, type="decorator_dependency")

    def _process_loop_variable(self, target: ast.AST, loop_id: str):
        """Process loop variable assignment."""
        if isinstance(target, ast.Name):
            # Single loop variable
            self._add_variable_definition(target.id, loop_id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Multiple loop variables
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._add_variable_definition(elt.id, loop_id)

    def _find_definition(self, var_name: str) -> Optional[str]:
        """Find the most recent definition of a variable in the current scope chain."""
        for scope in reversed(self.scopes):
            if var_name in scope.variables:
                return scope.variables[var_name]
        return None

    def _add_variable_definition(self, var_name: str, node_id: str):
        """Add a variable definition to the current scope."""
        if scope := self.get_current_scope():
            scope.variables[var_name] = node_id

    def _get_node_id(self, prefix: str) -> str:
        """Generate a unique node ID."""
        self.node_counter[prefix] += 1
        return f"{prefix}_{self.node_counter[prefix]}"

    def _extract_variables(self, node: ast.AST) -> set[str]:
        """Extract all variables used in an AST node."""
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables.add(child.id)
        return variables


def build_pdg(
    code: str, track_constants: bool = True, interprocedural: bool = True
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """
    Build a Program Dependence Graph from Python code.

    Args:
        code: The Python source code
        track_constants: Whether to track constant values
        interprocedural: Whether to perform interprocedural analysis

    Returns:
        Tuple containing the PDG and call graph
    """
    builder = PDGBuilder(track_constants, interprocedural)
    return builder.build(code)
