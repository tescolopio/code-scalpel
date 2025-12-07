import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

class CallGraphBuilder:
    """
    Builds a static call graph for a Python project.
    """

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.definitions: Dict[str, Set[str]] = {}  # file_path -> set of defined functions/classes
        self.calls: Dict[str, List[str]] = {}       # "file:function" -> list of called function names
        self.imports: Dict[str, Dict[str, str]] = {} # file_path -> { alias -> full_name }

    def build(self) -> Dict[str, List[str]]:
        """
        Build the call graph.
        Returns an adjacency list: {"module:caller": ["module:callee", ...]}
        """
        # 1. First pass: Collect definitions and imports
        for file_path in self._iter_python_files():
            rel_path = str(file_path.relative_to(self.root_path))
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                tree = ast.parse(code)
                self._analyze_definitions(tree, rel_path)
            except Exception:
                continue

        # 2. Second pass: Analyze calls and resolve them
        graph = {}
        for file_path in self._iter_python_files():
            rel_path = str(file_path.relative_to(self.root_path))
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                tree = ast.parse(code)
                file_calls = self._analyze_calls(tree, rel_path)
                graph.update(file_calls)
            except Exception:
                continue
                
        return graph

    def _iter_python_files(self):
        """Iterate over all Python files in the project, skipping hidden/ignored dirs."""
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for file in files:
                if file.endswith(".py"):
                    yield Path(root) / file

    def _analyze_definitions(self, tree: ast.AST, rel_path: str):
        """Extract function/class definitions and imports."""
        self.definitions[rel_path] = set()
        self.imports[rel_path] = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.definitions[rel_path].add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.definitions[rel_path].add(node.name)
                # Also add methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.definitions[rel_path].add(f"{node.name}.{item.name}")
            
            # Collect imports for resolution
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    self.imports[rel_path][asname] = name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    full_name = f"{module}.{name}" if module else name
                    self.imports[rel_path][asname] = full_name

    def _analyze_calls(self, tree: ast.AST, rel_path: str) -> Dict[str, List[str]]:
        """Extract calls from functions and resolve them."""
        file_graph = {}
        
        class CallVisitor(ast.NodeVisitor):
            def __init__(self, builder, current_file):
                self.builder = builder
                self.current_file = current_file
                self.current_scope = None
                self.calls = []

            def visit_FunctionDef(self, node):
                old_scope = self.current_scope
                self.current_scope = node.name
                self.calls = []
                self.generic_visit(node)
                
                # Store calls for this function
                key = f"{self.current_file}:{node.name}"
                file_graph[key] = self.calls
                
                self.current_scope = old_scope

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_Call(self, node):
                if self.current_scope:
                    callee = self._get_callee_name(node)
                    if callee:
                        resolved = self._resolve_callee(callee)
                        self.calls.append(resolved)
                self.generic_visit(node)

            def _get_callee_name(self, node):
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle obj.method() - simplified
                    value = self._get_attribute_value(node.func.value)
                    if value:
                        return f"{value}.{node.func.attr}"
                return None

            def _get_attribute_value(self, node):
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    val = self._get_attribute_value(node.value)
                    return f"{val}.{node.attr}" if val else None
                return None

            def _resolve_callee(self, callee):
                # 1. Check local imports
                imports = self.builder.imports.get(self.current_file, {})
                
                # Case: alias.method() where alias is imported
                parts = callee.split(".")
                if parts[0] in imports:
                    # e.g. "utils.hash" where "import my_utils as utils" -> "my_utils.hash"
                    # or "hash" where "from utils import hash" -> "utils.hash"
                    resolved_base = imports[parts[0]]
                    if len(parts) > 1:
                        return f"{resolved_base}.{'.'.join(parts[1:])}"
                    return resolved_base
                
                # 2. Check if it's a local definition in the same file
                if callee in self.builder.definitions.get(self.current_file, set()):
                    return f"{self.current_file}:{callee}"
                
                # 3. Fallback: return as is (likely external lib or built-in)
                return callee

        visitor = CallVisitor(self, rel_path)
        visitor.visit(tree)
        return file_graph
