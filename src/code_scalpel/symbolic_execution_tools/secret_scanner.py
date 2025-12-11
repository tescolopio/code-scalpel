"""
Secret Scanner - The High-Entropy Hunter.

This module detects hardcoded secrets in source code using regex patterns.
It scans AST Constant nodes for high-entropy strings like API keys,
tokens, and private keys.
"""

import ast
import re
from typing import List, Tuple, Pattern, Set

from .taint_tracker import Vulnerability, SecuritySink, TaintSource


class SecretScanner(ast.NodeVisitor):
    """
    Scans AST for hardcoded secrets.
    """

    def __init__(self) -> None:
        self.vulnerabilities: List[Vulnerability] = []

        # High-confidence patterns for string literals
        self.string_patterns: List[Tuple[str, Pattern]] = [
            ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
            ("Stripe Secret", re.compile(r"sk_live_[0-9a-zA-Z]{24}")),
            ("Private Key", re.compile(r"-----BEGIN RSA PRIVATE KEY-----")),
        ]

        # Generic API Key heuristic
        # Matches assignments like: api_key = "..."
        self.generic_value_pattern = re.compile(r"^[a-zA-Z0-9_\-]{20,}$")
        self.generic_keys: Set[str] = {"api_key", "access_token"}

    def scan(self, tree: ast.AST) -> List[Vulnerability]:
        """
        Scan an AST for hardcoded secrets.

        Args:
            tree: The AST to scan

        Returns:
            List of detected vulnerabilities
        """
        self.vulnerabilities = []
        self.visit(tree)
        return self.vulnerabilities

    def visit_Constant(self, node: ast.Constant) -> None:
        """Check string literals against high-confidence patterns."""
        value = node.value
        if isinstance(value, bytes):
            # errors='ignore' ensures this never raises - invalid bytes are skipped
            value = value.decode("utf-8", errors="ignore")

        if isinstance(value, str):
            for name, pattern in self.string_patterns:
                if pattern.search(value):
                    self._add_vuln(name, node)

        # Continue traversal (though Constant usually has no children)
        self.generic_visit(node)

    def visit_Str(self, node: ast.Str) -> None:
        """Support for Python < 3.8 where strings are ast.Str."""
        # In Python 3.9+, ast.Str is deprecated but might still appear in some ASTs?
        # Actually 3.9+ parses to Constant. But let's be safe.
        for name, pattern in self.string_patterns:
            if pattern.search(node.s):
                self._add_vuln(name, node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for generic API key assignments."""
        # Check if value is a string literal
        value_node = node.value
        string_val = None

        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            string_val = value_node.value
        elif isinstance(value_node, ast.Str):  # pragma: no cover
            string_val = value_node.s

        if string_val:
            # Check targets
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in self.generic_keys:
                        if self.generic_value_pattern.match(string_val):
                            self._add_vuln("Generic API Key", value_node)

        self.generic_visit(node)

    def _add_vuln(self, name: str, node: ast.AST) -> None:
        """Add a vulnerability to the list."""
        # Avoid duplicates if multiple patterns match or visited multiple times
        # (Simple check based on location)
        loc = (node.lineno, node.col_offset) if hasattr(node, "lineno") else (0, 0)

        # Check if we already have this vuln
        for v in self.vulnerabilities:
            if v.sink_location == loc and v.sink_type == SecuritySink.HARDCODED_SECRET:
                return

        vuln = Vulnerability(
            sink_type=SecuritySink.HARDCODED_SECRET,
            taint_source=TaintSource.HARDCODED,
            taint_path=[],
            sink_location=loc,
            source_location=loc,
            sanitizers_applied=set(),
        )
        # We store the secret type (e.g., "AWS Access Key") in the taint_path
        # so it can be reported in the summary.
        vuln.taint_path = [name]

        self.vulnerabilities.append(vuln)
