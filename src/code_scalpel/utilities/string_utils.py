# src/util_tools/string_utils.py

import re
from typing import Optional


def clean_code(code: str) -> str:
    """
    Cleans code by removing comments and extra whitespace.

    Args:
      code (str): The code to clean.

    Returns:
      str: The cleaned code.
    """
    # Remove single-line comments
    code = re.sub(r"#.*", "", code)
    # Remove multi-line comments (this is a simple approach, might not handle all cases)
    code = re.sub(r"\"\"\".*?\"\"\"", "", code, flags=re.DOTALL)
    # Remove extra whitespace
    code = " ".join(code.split())
    return code


def extract_function_signature(code: str) -> str:
    """
    Extracts the function signature from a Python function definition.

    Args:
      code (str): The code containing the function definition.

    Returns:
      str: The function signature (e.g., "def my_function(a, b):").
    """
    match = re.search(r"def\s+\w+\s*\(.*?\)\s*:", code)
    if match:
        return match.group(0)
    else:
        return ""


def split_string(s: str, delimiter: str) -> list[str]:
    """
    Splits a string based on a delimiter.

    Args:
      s (str): The string to split.
      delimiter (str): The delimiter to split by.

    Returns:
      List[str]: The list of split strings.
    """
    return s.split(delimiter)


def join_strings(strings: list[str], delimiter: str) -> str:
    """
    Joins a list of strings with a delimiter.

    Args:
      strings (List[str]): The list of strings to join.
      delimiter (str): The delimiter to join with.

    Returns:
      str: The joined string.
    """
    return delimiter.join(strings)


def convert_case(s: str, to_upper: bool = True) -> str:
    """
    Converts the case of a string.

    Args:
      s (str): The string to convert.
      to_upper (bool): Whether to convert to upper case (default: True).

    Returns:
      str: The converted string.
    """
    return s.upper() if to_upper else s.lower()


def replace_substring(s: str, old: str, new: str) -> str:
    """
    Replaces a substring with another substring.

    Args:
      s (str): The original string.
      old (str): The substring to replace.
      new (str): The substring to replace with.

    Returns:
      str: The modified string.
    """
    return s.replace(old, new)


def trim_whitespace(s: str) -> str:
    """
    Trims leading and trailing whitespace from a string.

    Args:
      s (str): The string to trim.

    Returns:
      str: The trimmed string.
    """
    return s.strip()


def search_pattern(s: str, pattern: str) -> Optional[re.Match]:
    """
    Searches for a pattern in a string.

    Args:
      s (str): The string to search.
      pattern (str): The pattern to search for.

    Returns:
      Optional[re.Match]: The match object if found, otherwise None.
    """
    return re.search(pattern, s)


def extract_matching_groups(s: str, pattern: str) -> list[tuple[str]]:
    """
    Extracts matching groups from a string based on a pattern.

    Args:
      s (str): The string to search.
      pattern (str): The pattern to search for.

    Returns:
      List[Tuple[str]]: The list of matching groups.
    """
    return re.findall(pattern, s)


def replace_pattern(s: str, pattern: str, replacement: str) -> str:
    """
    Replaces a pattern in a string with a replacement.

    Args:
      s (str): The original string.
      pattern (str): The pattern to replace.
      replacement (str): The replacement string.

    Returns:
      str: The modified string.
    """
    return re.sub(pattern, replacement, s)


def format_code_snippet(code: str, indent: int = 4) -> str:
    """
    Formats a code snippet with consistent indentation.

    Args:
      code (str): The code snippet to format.
      indent (int): The number of spaces to use for indentation (default: 4).

    Returns:
      str: The formatted code snippet.
    """
    lines = code.split("\n")
    formatted_lines = [line.strip() for line in lines]
    return "\n".join(" " * indent + line for line in formatted_lines)


def wrap_lines(s: str, width: int) -> str:
    """
    Wraps lines to a specified width.

    Args:
      s (str): The string to wrap.
      width (int): The maximum line width.

    Returns:
      str: The wrapped string.
    """
    return "\n".join([s[i : i + width] for i in range(0, len(s), width)])


def validate_identifier(identifier: str) -> bool:
    """
    Validates if a string is a valid Python identifier.

    Args:
      identifier (str): The string to validate.

    Returns:
      bool: True if the string is a valid identifier, False otherwise.
    """
    return identifier.isidentifier()


def extract_keywords(code: str) -> list[str]:
    """
    Extracts Python keywords from code.

    Args:
      code (str): The code to extract keywords from.

    Returns:
      List[str]: The list of extracted keywords.
    """
    keywords = {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
    return [word for word in code.split() if word in keywords]
