"""
File path resolution utilities for Code Scalpel.

Handles conversion of relative/absolute paths, workspace detection,
and common project structure patterns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class PathResolutionError(Exception):
    """Raised when a file path cannot be resolved."""

    pass


def resolve_file_path(
    file_path: str,
    workspace_root: Optional[str] = None,
    check_exists: bool = True,
) -> str:
    """
    Resolve a file path to an absolute path.

    This function handles:
    - Absolute paths (returned as-is)
    - Relative paths from workspace root
    - Relative paths from current working directory
    - Common project structures (src/, lib/, app/, etc.)

    Args:
        file_path: Path to resolve (relative or absolute)
        workspace_root: Optional workspace root directory
        check_exists: If True, raise FileNotFoundError if file doesn't exist

    Returns:
        Absolute path to the file

    Raises:
        FileNotFoundError: If check_exists=True and file doesn't exist
        PathResolutionError: If path cannot be resolved

    Examples:
        >>> resolve_file_path("/abs/path/file.py")
        "/abs/path/file.py"

        >>> resolve_file_path("utils.py", workspace_root="/project")
        "/project/utils.py"

        >>> resolve_file_path("src/utils.py")
        "/current/dir/src/utils.py"
    """
    path = Path(file_path)

    # Case 1: Already absolute
    if path.is_absolute():
        if check_exists and not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return str(path)

    # Case 2: Try relative to workspace root
    if workspace_root:
        workspace_path = Path(workspace_root) / path
        if workspace_path.exists():
            return str(workspace_path.resolve())

    # Case 3: Try relative to current working directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path.resolve())

    # Case 4: Try common project structures
    common_prefixes = ["src", "lib", "app", "pkg", "python", "code"]
    for prefix in common_prefixes:
        candidate = Path(prefix) / path
        if candidate.exists():
            return str(candidate.resolve())

    # Case 5: Try stripping common prefixes (maybe user provided "src/utils.py" from workspace root)
    for prefix in common_prefixes:
        if path.parts and path.parts[0] == prefix:
            # Try without the prefix
            stripped_path = Path(*path.parts[1:])
            if workspace_root:
                candidate = Path(workspace_root) / prefix / stripped_path
                if candidate.exists():
                    return str(candidate.resolve())

    # If check_exists=False, return the best guess
    if not check_exists:
        if workspace_root:
            return str((Path(workspace_root) / path).resolve())
        return str((Path.cwd() / path).resolve())

    # Failed to resolve
    raise FileNotFoundError(
        f"Cannot resolve path: {file_path}\n"
        f"Tried:\n"
        f"  - Absolute: {path}\n"
        f"  - Workspace: {Path(workspace_root) / path if workspace_root else 'N/A'}\n"
        f"  - CWD: {Path.cwd() / path}\n"
        f"  - Common prefixes: {', '.join(f'{p}/{path}' for p in common_prefixes)}"
    )


def get_workspace_root(start_path: Optional[str] = None) -> Optional[str]:
    """
    Detect workspace root by looking for common markers.

    Searches up the directory tree for:
    - .git directory
    - pyproject.toml
    - setup.py
    - package.json
    - go.mod

    Args:
        start_path: Directory to start search from (defaults to CWD)

    Returns:
        Absolute path to workspace root, or None if not found

    Examples:
        >>> get_workspace_root("/project/src/utils")
        "/project"  # if /project/.git exists
    """
    if start_path is None:
        start_path = os.getcwd()

    markers = [".git", "pyproject.toml", "setup.py", "package.json", "go.mod"]

    current = Path(start_path).resolve()
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return str(current)
        current = current.parent

    return None


def normalize_path(path: str) -> str:
    """
    Normalize a path to use forward slashes on all platforms.

    Args:
        path: Path to normalize

    Returns:
        Normalized path with forward slashes

    Examples:
        >>> normalize_path("C:\\\\Users\\\\file.py")
        "C:/Users/file.py"

        >>> normalize_path("/home/user/file.py")
        "/home/user/file.py"
    """
    return str(Path(path)).replace("\\", "/")


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get relative path from base_path to file_path.

    Args:
        file_path: Target file path
        base_path: Base directory path

    Returns:
        Relative path from base to target

    Examples:
        >>> get_relative_path("/project/src/utils.py", "/project")
        "src/utils.py"
    """
    try:
        return str(Path(file_path).relative_to(base_path))
    except ValueError:
        # Not relative, return absolute
        return str(Path(file_path).resolve())
