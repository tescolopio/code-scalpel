# src/util_tools/file_manager.py

import os
import shutil
from typing import Any


def read_file(filepath: str) -> str:
    """
    Reads the contents of a file.

    Args:
      filepath (str): The path to the file.

    Returns:
      str: The contents of the file.
    """
    try:
        with open(filepath) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def write_file(filepath: str, content: str):
    """
    Writes content to a file.

    Args:
      filepath (str): The path to the file.
      content (str): The content to write.
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
    except OSError as e:
        raise OSError(f"Error writing to file {filepath}: {e}")


def create_directory(directory: str):
    """
    Creates a directory if it doesn't exist.

    Args:
      directory (str): The path to the directory.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise OSError(f"Error creating directory {directory}: {e}")


def delete_file(filepath: str):
    """
    Deletes a file.

    Args:
      filepath (str): The path to the file.
    """
    try:
        os.remove(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:
        raise OSError(f"Error deleting file {filepath}: {e}")


def delete_directory(directory: str):
    """
    Deletes a directory and its contents.

    Args:
      directory (str): The path to the directory.
    """
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {directory}")
    except OSError as e:
        raise OSError(f"Error deleting directory {directory}: {e}")


def copy_file(src: str, dst: str):
    """
    Copies a file from src to dst.

    Args:
      src (str): The source file path.
      dst (str): The destination file path.
    """
    try:
        shutil.copy2(src, dst)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {src}")
    except OSError as e:
        raise OSError(f"Error copying file from {src} to {dst}: {e}")


def move_file(src: str, dst: str):
    """
    Moves a file from src to dst.

    Args:
      src (str): The source file path.
      dst (str): The destination file path.
    """
    try:
        shutil.move(src, dst)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {src}")
    except OSError as e:
        raise OSError(f"Error moving file from {src} to {dst}: {e}")


def list_files(directory: str) -> list[str]:
    """
    Lists all files in a directory.

    Args:
      directory (str): The path to the directory.

    Returns:
      List[str]: A list of file paths.
    """
    try:
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {directory}")
    except OSError as e:
        raise OSError(f"Error listing files in directory {directory}: {e}")


def list_directories(directory: str) -> list[str]:
    """
    Lists all directories in a directory.

    Args:
      directory (str): The path to the directory.

    Returns:
      List[str]: A list of directory paths.
    """
    try:
        return [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {directory}")
    except OSError as e:
        raise OSError(f"Error listing directories in directory {directory}: {e}")


def get_file_metadata(filepath: str) -> dict[str, Any]:
    """
    Gets metadata for a file.

    Args:
      filepath (str): The path to the file.

    Returns:
      Dict[str, Any]: A dictionary of file metadata.
    """
    try:
        stat = os.stat(filepath)
        return {
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:
        raise OSError(f"Error getting metadata for file {filepath}: {e}")


def join_paths(*paths: str) -> str:
    """
    Joins multiple paths into one.

    Args:
      *paths (str): The paths to join.

    Returns:
      str: The joined path.
    """
    return os.path.join(*paths)


def split_path(path: str) -> list[str]:
    """
    Splits a path into its components.

    Args:
      path (str): The path to split.

    Returns:
      List[str]: A list of path components.
    """
    return path.split(os.sep)


def normalize_path(path: str) -> str:
    """
    Normalizes a path.

    Args:
      path (str): The path to normalize.

    Returns:
      str: The normalized path.
    """
    return os.path.normpath(path)


def is_valid_path(path: str) -> bool:
    """
    Checks if a path is valid.

    Args:
      path (str): The path to check.

    Returns:
      bool: True if the path is valid, False otherwise.
    """
    return os.path.exists(path)


# ------------------------------------------------------------------------------
# Further Expansion Ideas:
# ------------------------------------------------------------------------------

# 1. File System Monitoring:
#   - Add functions to monitor file system changes (e.g., file creation,
#     modification, deletion).

# ------------------------------------------------------------------------------
