"""
Surgical Patcher - Precision code modification for safe LLM-driven refactoring.

This module provides surgical replacement of code elements (functions, classes, methods)
in source files. Instead of having LLMs output entire files (risking accidental deletions),
the LLM provides only the new code for a specific symbol, and we handle the replacement.

Key Principle: "Replace the function, not the file."

Usage:
    from code_scalpel.surgical_patcher import SurgicalPatcher

    patcher = SurgicalPatcher.from_file("src/utils.py")

    # Replace a function
    result = patcher.update_function("calculate_tax", new_code)

    # Replace a class
    result = patcher.update_class("TaxCalculator", new_class_code)

    # Replace a method within a class
    result = patcher.update_method("TaxCalculator", "compute", new_method_code)

    # Write changes back to file
    patcher.save()

Safety Features:
    - Creates backup before modification
    - Validates new code parses correctly
    - Preserves surrounding code exactly
    - Atomic write (temp file + rename)
"""

from __future__ import annotations

import ast
import os
import shutil
import tempfile
from dataclasses import dataclass


@dataclass
class PatchResult:
    """Result of a surgical patch operation."""

    success: bool
    file_path: str
    target_name: str
    target_type: str  # "function", "class", "method"
    lines_before: int = 0  # Lines in original symbol
    lines_after: int = 0  # Lines in replacement
    backup_path: str | None = None
    error: str | None = None

    @property
    def lines_delta(self) -> int:
        """Change in line count."""
        return self.lines_after - self.lines_before


@dataclass
class _SymbolLocation:
    """Internal: Location of a symbol in source code."""

    name: str
    node_type: str
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    col_offset: int
    node: ast.AST
    parent_class: str | None = None  # For methods


class SurgicalPatcher:
    """
    Precision code patcher using AST-guided line replacement.

    Unlike naive string replacement, this:
    1. Parses the file to find exact symbol boundaries
    2. Validates the replacement code is syntactically correct
    3. Preserves everything outside the target symbol
    4. Creates backups before modification

    Example:
        >>> patcher = SurgicalPatcher.from_file("calculator.py")
        >>> new_code = '''
        ... def add(a, b):
        ...     # Now with logging!
        ...     print(f"Adding {a} + {b}")
        ...     return a + b
        ... '''
        >>> result = patcher.update_function("add", new_code)
        >>> if result.success:
        ...     patcher.save()
    """

    def __init__(self, code: str, file_path: str | None = None):
        """
        Initialize the patcher with source code.

        Args:
            code: Python source code to modify
            file_path: Path to the source file (required for save())
        """
        self.original_code = code
        self.current_code = code
        self.file_path = file_path
        self._lines = code.splitlines(keepends=True)
        self._tree: ast.Module | None = None
        self._symbols: dict[str, _SymbolLocation] = {}
        self._parsed = False
        self._modified = False
        self._backup_path: str | None = None

    @classmethod
    def from_file(cls, file_path: str, encoding: str = "utf-8") -> "SurgicalPatcher":
        """
        Create a patcher by reading from a file.

        Args:
            file_path: Path to the Python source file
            encoding: File encoding (default: utf-8)

        Returns:
            SurgicalPatcher instance ready for modifications

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file can't be read
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding=encoding) as f:
                code = f.read()
        except IOError as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")

        return cls(code, file_path=os.path.abspath(file_path))

    def _ensure_parsed(self) -> None:
        """Parse the code and index all symbols."""
        if self._parsed:
            return

        try:
            self._tree = ast.parse(self.current_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

        self._index_symbols()
        self._parsed = True

    def _index_symbols(self) -> None:
        """Build an index of all functions, classes, and methods."""
        self._symbols.clear()

        for node in ast.walk(self._tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this is a method (inside a class)
                parent_class = self._find_parent_class(node)
                if parent_class:
                    key = f"{parent_class}.{node.name}"
                    symbol = _SymbolLocation(
                        name=node.name,
                        node_type="method",
                        line_start=node.lineno,
                        line_end=self._get_end_line(node),
                        col_offset=node.col_offset,
                        node=node,
                        parent_class=parent_class,
                    )
                else:
                    key = node.name
                    symbol = _SymbolLocation(
                        name=node.name,
                        node_type="function",
                        line_start=node.lineno,
                        line_end=self._get_end_line(node),
                        col_offset=node.col_offset,
                        node=node,
                    )
                self._symbols[key] = symbol

            elif isinstance(node, ast.ClassDef):
                key = node.name
                symbol = _SymbolLocation(
                    name=node.name,
                    node_type="class",
                    line_start=node.lineno,
                    line_end=self._get_end_line(node),
                    col_offset=node.col_offset,
                    node=node,
                )
                self._symbols[key] = symbol

    def _find_parent_class(self, node: ast.AST) -> str | None:
        """Find the parent class of a method node."""
        for potential_parent in ast.walk(self._tree):
            if isinstance(potential_parent, ast.ClassDef):
                for child in ast.walk(potential_parent):
                    if child is node and child is not potential_parent:
                        return potential_parent.name
        return None

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line of an AST node, handling decorators."""
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno

        # Fallback for older Python: estimate from body
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno") and child.lineno:
                max_line = max(max_line, child.lineno)
            if hasattr(child, "end_lineno") and child.end_lineno:
                max_line = max(max_line, child.end_lineno)
        return max_line

    def _get_decorator_start(self, node: ast.AST) -> int:
        """Get the starting line including decorators."""
        if hasattr(node, "decorator_list") and node.decorator_list:
            return min(d.lineno for d in node.decorator_list)
        return node.lineno

    def _validate_replacement(self, new_code: str, target_type: str) -> None:
        """Validate that replacement code is syntactically correct."""
        try:
            tree = ast.parse(new_code)
        except SyntaxError as e:
            raise ValueError(f"Replacement code has syntax error: {e}")

        # Verify it contains the expected type
        body = tree.body
        if not body:
            raise ValueError("Replacement code is empty")

        if target_type == "function":
            if not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError(
                    "Replacement for function must be a function definition"
                )
        elif target_type == "class":
            if not isinstance(body[0], ast.ClassDef):
                raise ValueError("Replacement for class must be a class definition")
        elif target_type == "method":
            if not isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError("Replacement for method must be a function definition")

    def _apply_patch(
        self, symbol: _SymbolLocation, new_code: str
    ) -> tuple[str, int, int]:
        """
        Apply a patch to the current code.

        Returns:
            Tuple of (new_code, lines_removed, lines_added)
        """
        lines = self.current_code.splitlines(keepends=True)

        # Include decorators in the replacement range
        start_line = self._get_decorator_start(symbol.node)
        end_line = symbol.line_end

        # Determine indentation from the original
        original_indent = ""
        if start_line <= len(lines):
            original_line = lines[start_line - 1]
            original_indent = original_line[
                : len(original_line) - len(original_line.lstrip())
            ]

        # Prepare replacement lines with proper indentation
        new_lines = new_code.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Apply indentation to replacement (detect its base indent and adjust)
        if new_lines:
            # Find the base indentation of the new code
            first_non_empty = next(
                (line for line in new_lines if line.strip()), new_lines[0]
            )
            new_base_indent = first_non_empty[
                : len(first_non_empty) - len(first_non_empty.lstrip())
            ]

            # Reindent if needed
            if new_base_indent != original_indent:
                adjusted_lines = []
                for line in new_lines:
                    if line.strip():  # Non-empty line
                        if line.startswith(new_base_indent):
                            line = original_indent + line[len(new_base_indent) :]
                    adjusted_lines.append(line)
                new_lines = adjusted_lines

        # Replace lines
        lines_removed = end_line - start_line + 1
        lines_added = len(new_lines)

        result_lines = lines[: start_line - 1] + new_lines + lines[end_line:]
        return "".join(result_lines), lines_removed, lines_added

    def update_function(self, name: str, new_code: str) -> PatchResult:
        """
        Replace a function definition with new code.

        Args:
            name: Name of the function to replace
            new_code: New function definition (including def line and body)

        Returns:
            PatchResult indicating success or failure
        """
        self._ensure_parsed()

        if name not in self._symbols:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="function",
                error=f"Function '{name}' not found",
            )

        symbol = self._symbols[name]
        if symbol.node_type != "function":
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="function",
                error=f"'{name}' is a {symbol.node_type}, not a function",
            )

        try:
            self._validate_replacement(new_code, "function")
        except ValueError as e:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="function",
                error=str(e),
            )

        lines_before = symbol.line_end - self._get_decorator_start(symbol.node) + 1
        new_code_str, _, lines_added = self._apply_patch(symbol, new_code)

        self.current_code = new_code_str
        self._parsed = False  # Need to re-parse after modification
        self._modified = True

        return PatchResult(
            success=True,
            file_path=self.file_path or "",
            target_name=name,
            target_type="function",
            lines_before=lines_before,
            lines_after=lines_added,
        )

    def update_class(self, name: str, new_code: str) -> PatchResult:
        """
        Replace a class definition with new code.

        Args:
            name: Name of the class to replace
            new_code: New class definition (including class line and body)

        Returns:
            PatchResult indicating success or failure
        """
        self._ensure_parsed()

        if name not in self._symbols:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="class",
                error=f"Class '{name}' not found",
            )

        symbol = self._symbols[name]
        if symbol.node_type != "class":
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="class",
                error=f"'{name}' is a {symbol.node_type}, not a class",
            )

        try:
            self._validate_replacement(new_code, "class")
        except ValueError as e:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=name,
                target_type="class",
                error=str(e),
            )

        lines_before = symbol.line_end - self._get_decorator_start(symbol.node) + 1
        new_code_str, _, lines_added = self._apply_patch(symbol, new_code)

        self.current_code = new_code_str
        self._parsed = False
        self._modified = True

        return PatchResult(
            success=True,
            file_path=self.file_path or "",
            target_name=name,
            target_type="class",
            lines_before=lines_before,
            lines_after=lines_added,
        )

    def update_method(
        self, class_name: str, method_name: str, new_code: str
    ) -> PatchResult:
        """
        Replace a method within a class.

        Args:
            class_name: Name of the containing class
            method_name: Name of the method to replace
            new_code: New method definition (including def line and body)

        Returns:
            PatchResult indicating success or failure
        """
        self._ensure_parsed()

        key = f"{class_name}.{method_name}"
        if key not in self._symbols:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=key,
                target_type="method",
                error=f"Method '{method_name}' not found in class '{class_name}'",
            )

        symbol = self._symbols[key]
        if symbol.node_type != "method":
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=key,
                target_type="method",
                error=f"'{key}' is a {symbol.node_type}, not a method",
            )

        try:
            self._validate_replacement(new_code, "method")
        except ValueError as e:
            return PatchResult(
                success=False,
                file_path=self.file_path or "",
                target_name=key,
                target_type="method",
                error=str(e),
            )

        lines_before = symbol.line_end - self._get_decorator_start(symbol.node) + 1
        new_code_str, _, lines_added = self._apply_patch(symbol, new_code)

        self.current_code = new_code_str
        self._parsed = False
        self._modified = True

        return PatchResult(
            success=True,
            file_path=self.file_path or "",
            target_name=key,
            target_type="method",
            lines_before=lines_before,
            lines_after=lines_added,
        )

    def get_modified_code(self) -> str:
        """Get the current (possibly modified) code."""
        return self.current_code

    def save(self, backup: bool = True) -> str | None:
        """
        Write modified code back to the file.

        Args:
            backup: If True, create a backup file before saving

        Returns:
            Path to backup file if created, None otherwise

        Raises:
            ValueError: If no file_path was provided
            IOError: If file cannot be written
        """
        if not self.file_path:
            raise ValueError("Cannot save: no file_path specified")

        if not self._modified:
            return None  # Nothing to save

        backup_path = None
        if backup:
            backup_path = f"{self.file_path}.bak"
            shutil.copy2(self.file_path, backup_path)
            self._backup_path = backup_path

        # Atomic write: write to temp file, then rename
        dir_path = os.path.dirname(self.file_path)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, delete=False, suffix=".tmp"
        ) as f:
            f.write(self.current_code)
            temp_path = f.name

        os.replace(temp_path, self.file_path)
        self._modified = False
        self.original_code = self.current_code

        return backup_path

    def discard_changes(self) -> None:
        """Discard all modifications and revert to original code."""
        self.current_code = self.original_code
        self._parsed = False
        self._modified = False


# Convenience functions for one-shot operations
def update_function_in_file(
    file_path: str, function_name: str, new_code: str, backup: bool = True
) -> PatchResult:
    """
    Update a function in a file (convenience function).

    Args:
        file_path: Path to the Python file
        function_name: Name of the function to replace
        new_code: New function definition
        backup: Whether to create a backup

    Returns:
        PatchResult indicating success or failure
    """
    patcher = SurgicalPatcher.from_file(file_path)
    result = patcher.update_function(function_name, new_code)
    if result.success:
        result.backup_path = patcher.save(backup=backup)
    return result


def update_class_in_file(
    file_path: str, class_name: str, new_code: str, backup: bool = True
) -> PatchResult:
    """
    Update a class in a file (convenience function).

    Args:
        file_path: Path to the Python file
        class_name: Name of the class to replace
        new_code: New class definition
        backup: Whether to create a backup

    Returns:
        PatchResult indicating success or failure
    """
    patcher = SurgicalPatcher.from_file(file_path)
    result = patcher.update_class(class_name, new_code)
    if result.success:
        result.backup_path = patcher.save(backup=backup)
    return result


def update_method_in_file(
    file_path: str,
    class_name: str,
    method_name: str,
    new_code: str,
    backup: bool = True,
) -> PatchResult:
    """
    Update a method in a file (convenience function).

    Args:
        file_path: Path to the Python file
        class_name: Name of the containing class
        method_name: Name of the method to replace
        new_code: New method definition
        backup: Whether to create a backup

    Returns:
        PatchResult indicating success or failure
    """
    patcher = SurgicalPatcher.from_file(file_path)
    result = patcher.update_method(class_name, method_name, new_code)
    if result.success:
        result.backup_path = patcher.save(backup=backup)
    return result
