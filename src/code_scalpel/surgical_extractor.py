"""
Surgical Extractor - Precision code extraction for token-efficient LLM interactions.

This module provides surgical extraction of code elements (functions, classes, methods)
from source files. Instead of feeding entire files to LLMs, extract only the relevant
pieces plus their dependencies.

Key Principle: "Feed the LLM 50 lines, not 5,000 lines."

Usage:
    from code_scalpel.surgical_extractor import SurgicalExtractor

    extractor = SurgicalExtractor(code)
    
    # Extract just one function (saves tokens)
    func_code = extractor.get_function("calculate_tax")
    
    # Extract with context (dependencies)
    func_with_deps = extractor.get_function_with_context("calculate_tax")
    
    # Extract a class method
    method_code = extractor.get_method("Calculator", "add")
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CrossFileSymbol:
    """A symbol resolved from an external file."""

    name: str
    source_file: str
    code: str
    node_type: str  # "function", "class", "variable"
    import_statement: str  # e.g., "from models import TaxRate"


@dataclass
class CrossFileResolution:
    """Result of cross-file dependency resolution."""

    success: bool
    target: "ExtractionResult"
    external_symbols: list[CrossFileSymbol] = field(default_factory=list)
    unresolved_imports: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def full_code(self) -> str:
        """Get combined code: external dependencies + target."""
        parts = []
        # Group by source file for cleaner output
        by_file: dict[str, list[CrossFileSymbol]] = {}
        for sym in self.external_symbols:
            by_file.setdefault(sym.source_file, []).append(sym)

        for source_file, symbols in by_file.items():
            parts.append(f"# From {source_file}")
            for sym in symbols:
                parts.append(sym.code)

        if parts:
            parts.append("")  # blank line separator
        parts.append(self.target.code)
        return "\n\n".join(parts)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate."""
        return len(self.full_code) // 4


@dataclass
class ExtractionResult:
    """Result of a surgical extraction."""

    success: bool
    name: str
    code: str
    node_type: str  # "function", "class", "method"
    line_start: int = 0
    line_end: int = 0
    dependencies: list[str] = field(default_factory=list)
    imports_needed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens (chars / 4)."""
        return len(self.code) // 4


@dataclass
class ContextualExtraction:
    """Extraction with all required context for LLM understanding."""

    target: ExtractionResult
    context_code: str  # Combined code of all dependencies
    total_lines: int
    context_items: list[str]  # Names of included dependencies

    @property
    def full_code(self) -> str:
        """Get the complete code block for LLM consumption."""
        if self.context_code:
            return f"{self.context_code}\n\n{self.target.code}"
        return self.target.code

    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens."""
        return len(self.full_code) // 4


class SurgicalExtractor:
    """
    Precision code extractor using AST analysis.

    Extracts specific code elements while preserving structure and
    identifying dependencies for context-aware extraction.

    Example (from string):
        >>> code = '''
        ... def helper():
        ...     return 42
        ...
        ... def main():
        ...     return helper() + 1
        ... '''
        >>> extractor = SurgicalExtractor(code)
        >>> result = extractor.get_function("main")
        >>> print(result.code)
        def main():
            return helper() + 1
        >>> print(result.dependencies)
        ['helper']

    Example (from file - TOKEN SAVER):
        >>> # Agent asks: "Get me calculate_tax from utils.py"
        >>> # Agent pays ~50 tokens, Server does the heavy lifting
        >>> extractor = SurgicalExtractor.from_file("/path/to/utils.py")
        >>> result = extractor.get_function("calculate_tax")
        >>> # Agent receives only the function (~200 tokens)
    """

    def __init__(self, code: str, file_path: str | None = None):
        """
        Initialize the extractor with source code.

        Args:
            code: Python source code to analyze
            file_path: Optional path to the source file (for cross-file resolution)
        """
        self.code = code
        self.file_path = file_path
        self.source_lines = code.splitlines()
        self._tree: ast.Module | None = None
        self._functions: dict[str, ast.FunctionDef] = {}
        self._classes: dict[str, ast.ClassDef] = {}
        self._imports: list[ast.Import | ast.ImportFrom] = []
        self._global_assigns: dict[str, ast.Assign] = {}
        self._parsed = False

    @classmethod
    def from_file(cls, file_path: str, encoding: str = "utf-8") -> "SurgicalExtractor":
        """
        Create an extractor by reading directly from a file.

        This is the TOKEN-EFFICIENT path. The Agent specifies a file path,
        the Server reads it (0 token cost to Agent), and returns only the
        requested symbol.

        Args:
            file_path: Path to the Python source file
            encoding: File encoding (default: utf-8)

        Returns:
            SurgicalExtractor instance ready for extraction

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file can't be read or parsed

        Example:
            >>> extractor = SurgicalExtractor.from_file("src/utils.py")
            >>> func = extractor.get_function("calculate_tax")
            >>> # Agent receives ~50 lines, not 5000
        """

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding=encoding) as f:
                code = f.read()
        except IOError as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")

        return cls(code, file_path=file_path)

    def _ensure_parsed(self) -> None:
        """Parse the code if not already done."""
        if self._parsed:
            return

        try:
            self._tree = ast.parse(self.code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

        # Index top-level definitions
        for node in self._tree.body:
            if isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node
            elif isinstance(node, ast.AsyncFunctionDef):
                self._functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self._classes[node.name] = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._imports.append(node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._global_assigns[target.id] = node

        self._parsed = True

    def list_functions(self) -> list[str]:
        """List all top-level function names."""
        self._ensure_parsed()
        return list(self._functions.keys())

    def list_classes(self) -> list[str]:
        """List all class names."""
        self._ensure_parsed()
        return list(self._classes.keys())

    def list_methods(self, class_name: str) -> list[str]:
        """List all methods of a class."""
        self._ensure_parsed()
        if class_name not in self._classes:
            return []

        methods = []
        for node in self._classes[class_name].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(node.name)
        return methods

    def get_function(self, name: str) -> ExtractionResult:
        """
        Extract a function by name.

        Args:
            name: Function name to extract

        Returns:
            ExtractionResult with the function code
        """
        self._ensure_parsed()

        if name not in self._functions:
            return ExtractionResult(
                success=False,
                name=name,
                code="",
                node_type="function",
                error=f"Function '{name}' not found. Available: {list(self._functions.keys())}",
            )

        node = self._functions[name]
        code = self._node_to_code(node)
        deps = self._find_dependencies(node)
        imports = self._find_required_imports(node)

        return ExtractionResult(
            success=True,
            name=name,
            code=code,
            node_type="function",
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            dependencies=deps,
            imports_needed=imports,
        )

    def get_class(self, name: str) -> ExtractionResult:
        """
        Extract a class by name.

        Args:
            name: Class name to extract

        Returns:
            ExtractionResult with the class code
        """
        self._ensure_parsed()

        if name not in self._classes:
            return ExtractionResult(
                success=False,
                name=name,
                code="",
                node_type="class",
                error=f"Class '{name}' not found. Available: {list(self._classes.keys())}",
            )

        node = self._classes[name]
        code = self._node_to_code(node)
        deps = self._find_dependencies(node)
        imports = self._find_required_imports(node)

        return ExtractionResult(
            success=True,
            name=name,
            code=code,
            node_type="class",
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            dependencies=deps,
            imports_needed=imports,
        )

    def get_method(self, class_name: str, method_name: str) -> ExtractionResult:
        """
        Extract a specific method from a class.

        Args:
            class_name: Name of the class
            method_name: Name of the method

        Returns:
            ExtractionResult with the method code
        """
        self._ensure_parsed()

        if class_name not in self._classes:
            return ExtractionResult(
                success=False,
                name=f"{class_name}.{method_name}",
                code="",
                node_type="method",
                error=f"Class '{class_name}' not found.",
            )

        class_node = self._classes[class_name]
        method_node = None

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    method_node = node
                    break

        if method_node is None:
            available = self.list_methods(class_name)
            return ExtractionResult(
                success=False,
                name=f"{class_name}.{method_name}",
                code="",
                node_type="method",
                error=f"Method '{method_name}' not found in class '{class_name}'. Available: {available}",
            )

        code = self._node_to_code(method_node)
        deps = self._find_dependencies(method_node)
        imports = self._find_required_imports(method_node)

        return ExtractionResult(
            success=True,
            name=f"{class_name}.{method_name}",
            code=code,
            node_type="method",
            line_start=method_node.lineno,
            line_end=getattr(method_node, "end_lineno", method_node.lineno),
            dependencies=deps,
            imports_needed=imports,
        )

    def get_function_with_context(
        self, name: str, max_depth: int = 2
    ) -> ContextualExtraction:
        """
        Extract a function with all its dependencies.

        This is the key token-saving operation: instead of giving the LLM
        the entire file, give it just the function plus the things it calls.

        Args:
            name: Function name to extract
            max_depth: How deep to follow dependencies (default: 2)

        Returns:
            ContextualExtraction with target and context
        """
        target = self.get_function(name)
        if not target.success:
            return ContextualExtraction(
                target=target,
                context_code="",
                total_lines=0,
                context_items=[],
            )

        # Gather context recursively
        context_items = []
        context_code_parts = []
        visited = {name}

        def gather_deps(deps: list[str], depth: int) -> None:
            if depth > max_depth:
                return

            for dep in deps:
                if dep in visited:
                    continue
                visited.add(dep)

                # Try function first
                if dep in self._functions:
                    dep_result = self.get_function(dep)
                    if dep_result.success:
                        context_items.append(dep)
                        context_code_parts.append(dep_result.code)
                        gather_deps(dep_result.dependencies, depth + 1)

                # Then try class
                elif dep in self._classes:
                    dep_result = self.get_class(dep)
                    if dep_result.success:
                        context_items.append(dep)
                        context_code_parts.append(dep_result.code)

                # Then try global assignment
                elif dep in self._global_assigns:
                    node = self._global_assigns[dep]
                    context_items.append(dep)
                    context_code_parts.append(self._node_to_code(node))

        gather_deps(target.dependencies, 1)

        # Add required imports
        imports_code = self._get_imports_code(target.imports_needed)
        if imports_code:
            context_code_parts.insert(0, imports_code)

        context_code = "\n\n".join(context_code_parts)
        total_lines = len(context_code.splitlines()) + len(target.code.splitlines())

        return ContextualExtraction(
            target=target,
            context_code=context_code,
            total_lines=total_lines,
            context_items=context_items,
        )

    def get_class_with_context(
        self, name: str, max_depth: int = 2
    ) -> ContextualExtraction:
        """
        Extract a class with all its dependencies.

        Args:
            name: Class name to extract
            max_depth: How deep to follow dependencies

        Returns:
            ContextualExtraction with target and context
        """
        target = self.get_class(name)
        if not target.success:
            return ContextualExtraction(
                target=target,
                context_code="",
                total_lines=0,
                context_items=[],
            )

        # Similar logic to function context
        context_items = []
        context_code_parts = []
        visited = {name}

        def gather_deps(deps: list[str], depth: int) -> None:
            if depth > max_depth:
                return

            for dep in deps:
                if dep in visited:
                    continue
                visited.add(dep)

                if dep in self._functions:
                    dep_result = self.get_function(dep)
                    if dep_result.success:
                        context_items.append(dep)
                        context_code_parts.append(dep_result.code)
                        gather_deps(dep_result.dependencies, depth + 1)
                elif dep in self._classes:
                    dep_result = self.get_class(dep)
                    if dep_result.success:
                        context_items.append(dep)
                        context_code_parts.append(dep_result.code)
                elif dep in self._global_assigns:
                    node = self._global_assigns[dep]
                    context_items.append(dep)
                    context_code_parts.append(self._node_to_code(node))

        gather_deps(target.dependencies, 1)

        imports_code = self._get_imports_code(target.imports_needed)
        if imports_code:
            context_code_parts.insert(0, imports_code)

        context_code = "\n\n".join(context_code_parts)
        total_lines = len(context_code.splitlines()) + len(target.code.splitlines())

        return ContextualExtraction(
            target=target,
            context_code=context_code,
            total_lines=total_lines,
            context_items=context_items,
        )

    def resolve_cross_file_dependencies(
        self,
        target_name: str,
        target_type: str = "function",
        max_depth: int = 1,
    ) -> CrossFileResolution:
        """
        Extract a symbol and resolve its dependencies from external files.

        This is the CROSS-FILE token saver. When calculate_tax uses TaxRate
        from models.py, this method will:
        1. Extract calculate_tax from the current file
        2. Find the import: "from models import TaxRate"
        3. Resolve models.py path relative to current file
        4. Extract TaxRate from models.py
        5. Return both as a combined context

        Args:
            target_name: Name of function/class to extract
            target_type: "function" or "class"
            max_depth: How many levels of imports to follow (default: 1)

        Returns:
            CrossFileResolution with target and external symbols

        Example:
            >>> # utils.py imports TaxRate from models.py
            >>> extractor = SurgicalExtractor.from_file("utils.py")
            >>> result = extractor.resolve_cross_file_dependencies("calculate_tax")
            >>> print(result.full_code)
            # From models.py
            class TaxRate:
                ...

            def calculate_tax(amount):
                rate = TaxRate()
                ...

        Requires:
            - file_path must be set (use from_file() or pass to __init__)
        """
        # Extract the target
        if target_type == "class":
            target = self.get_class(target_name)
        else:
            target = self.get_function(target_name)

        if not target.success:
            return CrossFileResolution(
                success=False,
                target=target,
                error=target.error,
            )

        if not self.file_path:
            return CrossFileResolution(
                success=True,
                target=target,
                error="file_path not set - cannot resolve cross-file imports",
            )

        # Build import map: symbol_name -> (module_path, import_statement)
        import_map = self._build_import_map()

        # Find which imports are actually used by the target
        used_imports = set(target.imports_needed)

        # Also check dependencies that might be imported
        for dep in target.dependencies:
            if dep in import_map:
                used_imports.add(dep)

        # Resolve external symbols
        external_symbols: list[CrossFileSymbol] = []
        unresolved: list[str] = []
        # Track (file_path, symbol_name) to avoid duplicate resolution
        visited_symbols: set[tuple[str, str]] = set()
        # Cache extractors to avoid re-parsing same files
        extractor_cache: dict[str, SurgicalExtractor] = {}

        def get_extractor(path: str) -> SurgicalExtractor:
            """Get or create an extractor for a file (cached)."""
            if path not in extractor_cache:
                extractor_cache[path] = SurgicalExtractor.from_file(path)
            return extractor_cache[path]

        def resolve_symbol(
            symbol_name: str,
            module_info: tuple[str | None, str, str],
            depth: int,
        ) -> None:
            """Recursively resolve a symbol from an external file."""
            if depth > max_depth:
                return

            module_path, import_stmt, actual_name = module_info

            if module_path is None:
                unresolved.append(f"{symbol_name} (module not found)")
                return

            # Check if we've already resolved this specific symbol
            visit_key = (module_path, actual_name)
            if visit_key in visited_symbols:
                return  # Already resolved this symbol

            visited_symbols.add(visit_key)

            try:
                ext_extractor = get_extractor(module_path)

                # Try to extract as function first, then class
                result = ext_extractor.get_function(actual_name)
                if not result.success:
                    result = ext_extractor.get_class(actual_name)

                if result.success:
                    external_symbols.append(
                        CrossFileSymbol(
                            name=symbol_name,
                            source_file=module_path,
                            code=result.code,
                            node_type=result.node_type,
                            import_statement=import_stmt,
                        )
                    )

                    # Recursively resolve dependencies of this symbol
                    if depth < max_depth:
                        ext_import_map = ext_extractor._build_import_map()
                        for dep in result.dependencies:
                            if dep in ext_import_map:
                                resolve_symbol(
                                    dep,
                                    ext_import_map[dep],
                                    depth + 1,
                                )
                else:
                    # Try as a global variable
                    ext_extractor._ensure_parsed()
                    if actual_name in ext_extractor._global_assigns:
                        node = ext_extractor._global_assigns[actual_name]
                        external_symbols.append(
                            CrossFileSymbol(
                                name=symbol_name,
                                source_file=module_path,
                                code=ext_extractor._node_to_code(node),
                                node_type="variable",
                                import_statement=import_stmt,
                            )
                        )
                    else:
                        unresolved.append(f"{symbol_name} (not found in {module_path})")

            except (FileNotFoundError, ValueError) as e:
                unresolved.append(f"{symbol_name} ({e})")

        # Resolve each imported symbol
        for dep in target.dependencies:
            if dep in import_map:
                resolve_symbol(dep, import_map[dep], 1)

        return CrossFileResolution(
            success=True,
            target=target,
            external_symbols=external_symbols,
            unresolved_imports=unresolved,
        )

    def _build_import_map(self) -> dict[str, tuple[str | None, str, str]]:
        """
        Build a mapping from imported symbol names to their source modules.

        Returns:
            Dict mapping symbol_name -> (resolved_path, import_statement, actual_name)
            resolved_path is None if module cannot be found
        """
        self._ensure_parsed()
        import_map: dict[str, tuple[str | None, str, str]] = {}

        base_dir = Path(self.file_path).parent if self.file_path else Path(".")

        for imp in self._imports:
            if isinstance(imp, ast.ImportFrom):
                # from module import name1, name2
                module_name = imp.module or ""
                module_path = self._resolve_module_path(
                    module_name, base_dir, imp.level
                )
                import_stmt = ast.unparse(imp)

                for alias in imp.names:
                    # Handle "from module import *" - skip
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    actual_name = alias.name
                    import_map[local_name] = (module_path, import_stmt, actual_name)

            elif isinstance(imp, ast.Import):
                # import module, module2
                for alias in imp.names:
                    module_name = alias.name
                    local_name = alias.asname or module_name.split(".")[0]
                    module_path = self._resolve_module_path(module_name, base_dir, 0)
                    import_stmt = ast.unparse(imp)
                    import_map[local_name] = (module_path, import_stmt, module_name)

        return import_map

    def _resolve_module_path(
        self, module_name: str, base_dir: Path, level: int = 0
    ) -> str | None:
        """
        Resolve a module name to a file path.

        Args:
            module_name: Module name (e.g., "models", "utils.helpers")
            base_dir: Directory to resolve relative imports from
            level: Number of dots for relative imports (0 = absolute)

        Returns:
            Resolved file path or None if not found
        """
        # Handle relative imports (level > 0 means relative)
        if level > 0:
            # Go up 'level' directories from base_dir
            search_dir = base_dir
            for _ in range(level - 1):  # level=1 means current dir
                search_dir = search_dir.parent

            if module_name:
                parts = module_name.split(".")
                search_path = search_dir / "/".join(parts)
            else:
                search_path = search_dir
        else:
            # Absolute import - search relative to base_dir first
            # (common in packages), then could extend to sys.path
            parts = module_name.split(".")
            search_path = base_dir / "/".join(parts)

        # Try module.py
        py_file = search_path.with_suffix(".py")
        if py_file.exists():
            return str(py_file)

        # Try module/__init__.py
        init_file = search_path / "__init__.py"
        if init_file.exists():
            return str(init_file)

        # Try searching from parent directories (common package structure)
        # e.g., src/package/module.py when importing from src/package/subdir/file.py
        current = base_dir
        for _ in range(5):  # Don't go too far up
            current = current.parent
            if not current.exists():
                break

            test_path = current / "/".join(parts)
            py_file = test_path.with_suffix(".py")
            if py_file.exists():
                return str(py_file)

            init_file = test_path / "__init__.py"
            if init_file.exists():
                return str(init_file)

        return None

    def _node_to_code(self, node: ast.AST) -> str:
        """Convert an AST node back to source code."""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback to source lines if available
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                return "\n".join(self.source_lines[node.lineno - 1 : node.end_lineno])
            raise

    def _find_dependencies(self, node: ast.AST) -> list[str]:
        """
        Find names that this node depends on.

        Returns names of functions, classes, and variables used.
        """
        deps = set()
        defined_in_scope = set()

        # First pass: collect all names that are defined in scope
        class DefinitionCollector(ast.NodeVisitor):
            def visit_FunctionDef(inner_self, n: ast.FunctionDef) -> None:
                # Function args are defined in scope
                for arg in n.args.args:
                    defined_in_scope.add(arg.arg)
                if n.args.vararg:
                    defined_in_scope.add(n.args.vararg.arg)
                if n.args.kwarg:
                    defined_in_scope.add(n.args.kwarg.arg)
                for arg in n.args.kwonlyargs:
                    defined_in_scope.add(arg.arg)
                inner_self.generic_visit(n)

            def visit_AsyncFunctionDef(inner_self, n: ast.AsyncFunctionDef) -> None:
                for arg in n.args.args:
                    defined_in_scope.add(arg.arg)
                if n.args.vararg:
                    defined_in_scope.add(n.args.vararg.arg)
                if n.args.kwarg:
                    defined_in_scope.add(n.args.kwarg.arg)
                for arg in n.args.kwonlyargs:
                    defined_in_scope.add(arg.arg)
                inner_self.generic_visit(n)

            def visit_For(inner_self, n: ast.For) -> None:
                # Loop variable is defined
                if isinstance(n.target, ast.Name):
                    defined_in_scope.add(n.target.id)
                elif isinstance(n.target, ast.Tuple):
                    for elt in n.target.elts:
                        if isinstance(elt, ast.Name):
                            defined_in_scope.add(elt.id)
                inner_self.generic_visit(n)

            def visit_comprehension(inner_self, n: ast.comprehension) -> None:
                # Comprehension target variable
                if isinstance(n.target, ast.Name):
                    defined_in_scope.add(n.target.id)
                elif isinstance(n.target, ast.Tuple):
                    for elt in n.target.elts:
                        if isinstance(elt, ast.Name):
                            defined_in_scope.add(elt.id)
                inner_self.generic_visit(n)

            def visit_Name(inner_self, n: ast.Name) -> None:
                if isinstance(n.ctx, ast.Store):
                    defined_in_scope.add(n.id)
                inner_self.generic_visit(n)

            def visit_ExceptHandler(inner_self, n: ast.ExceptHandler) -> None:
                if n.name:
                    defined_in_scope.add(n.name)
                inner_self.generic_visit(n)

            def visit_With(inner_self, n: ast.With) -> None:
                for item in n.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined_in_scope.add(item.optional_vars.id)
                inner_self.generic_visit(n)

        # Collect definitions first
        DefinitionCollector().visit(node)

        # Second pass: collect all names that are loaded (used)
        class UsageCollector(ast.NodeVisitor):
            def visit_Name(inner_self, n: ast.Name) -> None:
                if isinstance(n.ctx, ast.Load):
                    if n.id not in defined_in_scope:
                        deps.add(n.id)
                inner_self.generic_visit(n)

        UsageCollector().visit(node)

        # Filter out builtins and known stdlib
        builtins = {
            "print",
            "len",
            "range",
            "int",
            "str",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "sum",
            "min",
            "max",
            "abs",
            "all",
            "any",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "None",
            "True",
            "False",
            "self",
            "cls",
            "super",
        }

        return [d for d in deps if d not in builtins]

    def _find_required_imports(self, node: ast.AST) -> list[str]:
        """Find which imports are needed for this node."""
        # Get all names used in the node
        names_used = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(inner_self, n: ast.Name) -> None:
                if isinstance(n.ctx, ast.Load):
                    names_used.add(n.id)
                inner_self.generic_visit(n)

            def visit_Attribute(inner_self, n: ast.Attribute) -> None:
                # Get the base name (e.g., 'os' from 'os.path')
                if isinstance(n.value, ast.Name):
                    names_used.add(n.value.id)
                inner_self.generic_visit(n)

        NameCollector().visit(node)

        # Check which imports provide these names
        required = []
        for imp in self._imports:
            if isinstance(imp, ast.Import):
                for alias in imp.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name in names_used:
                        required.append(ast.unparse(imp))
                        break
            elif isinstance(imp, ast.ImportFrom):
                for alias in imp.names:
                    name = alias.asname or alias.name
                    if name in names_used:
                        required.append(ast.unparse(imp))
                        break

        return required

    def _get_imports_code(self, import_statements: list[str]) -> str:
        """Combine import statements into code block."""
        if not import_statements:
            return ""
        return "\n".join(sorted(set(import_statements)))


def extract_function(code: str, name: str) -> ExtractionResult:
    """
    Convenience function to extract a function from code.

    Args:
        code: Python source code
        name: Function name to extract

    Returns:
        ExtractionResult with the function code
    """
    return SurgicalExtractor(code).get_function(name)


def extract_class(code: str, name: str) -> ExtractionResult:
    """
    Convenience function to extract a class from code.

    Args:
        code: Python source code
        name: Class name to extract

    Returns:
        ExtractionResult with the class code
    """
    return SurgicalExtractor(code).get_class(name)


def extract_method(code: str, class_name: str, method_name: str) -> ExtractionResult:
    """
    Convenience function to extract a method from code.

    Args:
        code: Python source code
        class_name: Class name
        method_name: Method name

    Returns:
        ExtractionResult with the method code
    """
    return SurgicalExtractor(code).get_method(class_name, method_name)


def extract_with_context(
    code: str, name: str, target_type: str = "function", max_depth: int = 2
) -> ContextualExtraction:
    """
    Convenience function to extract code with dependencies.

    Args:
        code: Python source code
        name: Name of function or class to extract
        target_type: "function" or "class"
        max_depth: Dependency depth

    Returns:
        ContextualExtraction with target and context
    """
    extractor = SurgicalExtractor(code)
    if target_type == "class":
        return extractor.get_class_with_context(name, max_depth)
    return extractor.get_function_with_context(name, max_depth)
