"""
Security Analyzer - The Vulnerability Hunter.

This module provides high-level security analysis by combining:
- Symbolic execution (engine.py)
- Taint tracking (taint_tracker.py)
- Sink detection (SINK_PATTERNS)

It can detect:
- SQL Injection (CWE-89)
- Cross-Site Scripting (CWE-79)
- Path Traversal (CWE-22)
- Command Injection (CWE-78)

Usage:
    analyzer = SecurityAnalyzer()
    vulns = analyzer.analyze('''
        user_id = request.args.get("id")
        query = "SELECT * FROM users WHERE id=" + user_id
        cursor.execute(query)
    ''')
    
    for v in vulns:
        print(f"{v.vulnerability_type} at line {v.sink_location[0]}")
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .taint_tracker import (
    TaintTracker,
    TaintInfo,
    TaintLevel,
    SecuritySink,
    Vulnerability,
    TAINT_SOURCE_PATTERNS,
    SINK_PATTERNS,
    SANITIZER_PATTERNS,
    SANITIZER_REGISTRY,
    load_sanitizers_from_config,
)

# Auto-load custom sanitizers from pyproject.toml on module import
_config_loaded = False


def _ensure_config_loaded() -> None:
    """Load config once per process."""
    global _config_loaded
    if not _config_loaded:
        load_sanitizers_from_config()
        _config_loaded = True


@dataclass
class SecurityAnalysisResult:
    """
    Result from security analysis.

    Attributes:
        vulnerabilities: List of detected vulnerabilities
        taint_flows: Map of variable names to their taint info
        analyzed_lines: Number of lines analyzed
        functions_analyzed: Names of functions that were analyzed
    """

    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    taint_flows: Dict[str, TaintInfo] = field(default_factory=dict)
    analyzed_lines: int = 0
    functions_analyzed: List[str] = field(default_factory=list)

    @property
    def has_vulnerabilities(self) -> bool:
        """Check if any vulnerabilities were found."""
        return len(self.vulnerabilities) > 0

    @property
    def vulnerability_count(self) -> int:
        """Get total number of vulnerabilities."""
        return len(self.vulnerabilities)

    def get_by_type(self, vuln_type: str) -> List[Vulnerability]:
        """Get vulnerabilities of a specific type."""
        return [v for v in self.vulnerabilities if v.vulnerability_type == vuln_type]

    def get_sql_injections(self) -> List[Vulnerability]:
        """Get SQL injection vulnerabilities."""
        return [
            v for v in self.vulnerabilities if v.sink_type == SecuritySink.SQL_QUERY
        ]

    def get_xss(self) -> List[Vulnerability]:
        """Get XSS vulnerabilities."""
        return [
            v for v in self.vulnerabilities if v.sink_type == SecuritySink.HTML_OUTPUT
        ]

    def get_path_traversals(self) -> List[Vulnerability]:
        """Get path traversal vulnerabilities."""
        return [
            v for v in self.vulnerabilities if v.sink_type == SecuritySink.FILE_PATH
        ]

    def get_command_injections(self) -> List[Vulnerability]:
        """Get command injection vulnerabilities."""
        return [
            v for v in self.vulnerabilities if v.sink_type == SecuritySink.SHELL_COMMAND
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "vulnerability_count": self.vulnerability_count,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "taint_flows": {
                name: {
                    "source": info.source.name,
                    "level": info.level.name,
                    "path": info.propagation_path,
                }
                for name, info in self.taint_flows.items()
            },
            "analyzed_lines": self.analyzed_lines,
            "functions_analyzed": self.functions_analyzed,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        if not self.has_vulnerabilities:
            return "No vulnerabilities detected."

        lines = [f"Found {self.vulnerability_count} vulnerability(ies):"]

        for v in self.vulnerabilities:
            loc = f"line {v.sink_location[0]}" if v.sink_location else "unknown"
            lines.append(f"  - {v.vulnerability_type} ({v.cwe_id}) at {loc}")

        return "\n".join(lines)


class SecurityAnalyzer:
    """
    High-level security analyzer for Python code.

    Combines AST analysis with taint tracking to detect
    security vulnerabilities in Python source code.

    Example:
        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        if result.has_vulnerabilities:
            for vuln in result.vulnerabilities:
                print(f"SECURITY: {vuln}")
    """

    def __init__(self):
        """Initialize the security analyzer."""
        self._taint_tracker: Optional[TaintTracker] = None
        self._current_taint_map: Dict[str, TaintInfo] = {}

    def analyze(self, code: str) -> SecurityAnalysisResult:
        """
        Analyze Python code for security vulnerabilities.

        Args:
            code: Python source code

        Returns:
            SecurityAnalysisResult with detected vulnerabilities
        """
        if not code or not code.strip():
            return SecurityAnalysisResult()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return SecurityAnalysisResult()

        # Initialize fresh tracker
        self._taint_tracker = TaintTracker()
        self._current_taint_map = {}

        # Analyze the AST
        result = SecurityAnalysisResult(
            analyzed_lines=code.count("\n") + 1,
        )

        # Visit all nodes
        self._analyze_node(tree, result)

        # Collect results
        result.vulnerabilities = self._taint_tracker.get_vulnerabilities()
        result.taint_flows = {
            name: self._taint_tracker.get_taint(name)
            for name in self._current_taint_map.keys()
            if self._taint_tracker.get_taint(name) is not None
        }

        return result

    def _analyze_node(self, node: ast.AST, result: SecurityAnalysisResult) -> None:
        """Recursively analyze an AST node."""

        if isinstance(node, ast.Module):
            for child in node.body:
                self._analyze_node(child, result)

        elif isinstance(node, ast.FunctionDef):
            result.functions_analyzed.append(node.name)
            for child in node.body:
                self._analyze_node(child, result)

        elif isinstance(node, ast.Assign):
            self._analyze_assignment(node)

        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                self._analyze_call(node.value, (node.lineno, node.col_offset))

        elif isinstance(node, ast.If):
            for child in node.body:
                self._analyze_node(child, result)
            for child in node.orelse:
                self._analyze_node(child, result)

        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            for child in node.body:
                self._analyze_node(child, result)

        elif isinstance(node, ast.With):
            for child in node.body:
                self._analyze_node(child, result)

        elif isinstance(node, ast.Try):
            for child in node.body:
                self._analyze_node(child, result)
            for handler in node.handlers:
                for child in handler.body:
                    self._analyze_node(child, result)

    def _analyze_assignment(self, node: ast.Assign) -> None:
        """Analyze an assignment for taint propagation."""
        # Get target name(s)
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)

        if not targets:
            return

        # Check if RHS is a call that might be a sink (even if also an assignment)
        # e.g., html = render_template_string(user) - user reaches the sink
        if isinstance(node.value, ast.Call):
            self._analyze_call(node.value, (node.lineno, node.col_offset))

        # Check if RHS introduces taint
        source_info = self._check_taint_source(
            node.value, (node.lineno, node.col_offset)
        )

        if source_info is not None:
            # RHS is a taint source
            for target in targets:
                self._taint_tracker.mark_tainted(target, source_info)
                self._current_taint_map[target] = source_info
        else:
            # Check if RHS is a sanitizer call wrapping tainted data
            sanitizer_result = self._check_sanitizer_call(node.value)

            if sanitizer_result is not None:
                sanitizer_name, sanitized_taint = sanitizer_result
                for target in targets:
                    # Apply sanitizer to propagated taint
                    final_taint = sanitized_taint.apply_sanitizer(sanitizer_name)
                    self._taint_tracker.mark_tainted(target, final_taint)
                    self._current_taint_map[target] = final_taint
            else:
                # Check if RHS propagates taint (no sanitizer)
                source_vars = self._extract_variable_names(node.value)
                for target in targets:
                    propagated = self._taint_tracker.propagate_assignment(
                        target, source_vars
                    )
                    if propagated is not None:
                        self._current_taint_map[target] = propagated

    def _check_sanitizer_call(self, node: ast.expr) -> Optional[Tuple[str, TaintInfo]]:
        """
        Check if an expression is a sanitizer call wrapping tainted data.

        Returns:
            Tuple of (sanitizer_name, source_taint) if sanitizer found, None otherwise
        """
        if not isinstance(node, ast.Call):
            return None

        func_name = self._get_call_name(node)
        if func_name is None:
            return None

        # Check if this function is a registered sanitizer
        if func_name not in SANITIZER_REGISTRY and func_name not in SANITIZER_PATTERNS:
            return None

        # Get the sanitizer name (prefer registry, fallback to patterns)
        sanitizer_name = func_name

        # Find tainted arguments
        for arg in node.args:  # pragma: no branch - loop continuation
            if isinstance(arg, ast.Name):
                taint = self._taint_tracker.get_taint(arg.id)
                if taint is not None:
                    return (sanitizer_name, taint)
            elif isinstance(arg, ast.BinOp):  # pragma: no branch
                # Tainted expression in argument
                arg_vars = self._extract_variable_names(arg)
                for var in arg_vars:  # pragma: no branch - loop continuation
                    taint = self._taint_tracker.get_taint(var)
                    if taint is not None:  # pragma: no branch
                        return (sanitizer_name, taint)

        return None

    def _analyze_call(self, node: ast.Call, location: Tuple[int, int]) -> None:
        """Analyze a function call for sink detection."""
        # Recursively analyze chained calls like hashlib.md5(...).hexdigest()
        # The inner call (hashlib.md5) is in node.func.value when node.func is Attribute
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Call
        ):
            self._analyze_call(node.func.value, location)

        # Also check args that are calls
        # Also check keyword args that are calls (e.g., annotate(val=RawSQL(...)))
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Call):
                self._analyze_call(keyword.value, location)
        for arg in node.args:
            if isinstance(arg, ast.Call):
                self._analyze_call(arg, location)

        # Get the function name
        func_name = self._get_call_name(node)

        if func_name is None:
            return

        # Check if this is a security sink
        sink = SINK_PATTERNS.get(func_name)

        if sink is not None:
            # Check all arguments for taint
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self._taint_tracker.check_sink(arg.id, sink, location)
                elif isinstance(arg, ast.BinOp):
                    # String concatenation in argument
                    arg_vars = self._extract_variable_names(arg)
                    for var in arg_vars:
                        self._taint_tracker.check_sink(var, sink, location)
                elif isinstance(arg, ast.JoinedStr):
                    # f-string
                    arg_vars = self._extract_variable_names(arg)
                    for var in arg_vars:
                        self._taint_tracker.check_sink(var, sink, location)
                elif isinstance(arg, ast.Call):
                    # Method call on tainted variable: var.method()
                    # e.g., user_data.encode() where user_data is tainted
                    arg_vars = self._extract_variable_names(arg)
                    for var in arg_vars:
                        self._taint_tracker.check_sink(var, sink, location)

        # Check if this is a sanitizer
        sanitizer = SANITIZER_PATTERNS.get(func_name)

        if sanitizer is not None:
            # Get the variable being sanitized
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self._taint_tracker.apply_sanitizer(arg.id, sanitizer)

    def _check_taint_source(
        self, node: ast.expr, location: Tuple[int, int]
    ) -> Optional[TaintInfo]:
        """Check if an expression is a taint source."""

        if isinstance(node, ast.Call):
            func_name = self._get_call_name(node)

            if func_name is not None:
                source = TAINT_SOURCE_PATTERNS.get(func_name)

                if source is not None:
                    return TaintInfo(
                        source=source,
                        level=TaintLevel.HIGH,
                        source_location=location,
                        propagation_path=[],
                    )

        elif isinstance(node, ast.Subscript):
            # e.g., request.args["id"]
            call_name = self._get_subscript_base(node)

            if call_name is not None:
                source = TAINT_SOURCE_PATTERNS.get(call_name)

                if source is not None:
                    return TaintInfo(
                        source=source,
                        level=TaintLevel.HIGH,
                        source_location=location,
                        propagation_path=[],
                    )

        return None

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the full dotted name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func

            while isinstance(current, ast.Attribute):  # pragma: no branch - loop exit
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))

        return None

    def _get_subscript_base(self, node: ast.Subscript) -> Optional[str]:
        """Get the base name for a subscript like request.args["id"]."""
        if isinstance(node.value, ast.Attribute):
            parts = []
            current = node.value

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))

        return None

    def _extract_variable_names(self, node: ast.expr) -> List[str]:
        """Extract all variable names referenced in an expression."""
        names = []

        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.BinOp):
            names.extend(self._extract_variable_names(node.left))
            names.extend(self._extract_variable_names(node.right))
        elif isinstance(node, ast.Call):
            # Extract from arguments
            for arg in node.args:
                names.extend(self._extract_variable_names(arg))
            # Extract from method receiver: user_data.encode() -> user_data
            if isinstance(node.func, ast.Attribute):
                names.extend(self._extract_variable_names(node.func.value))
        elif isinstance(node, ast.JoinedStr):
            # f-string
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    names.extend(self._extract_variable_names(value.value))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                names.append(node.value.id)

        return names


def analyze_security(code: str) -> SecurityAnalysisResult:
    """
    Convenience function to analyze code for security vulnerabilities.

    Automatically loads custom sanitizers from pyproject.toml if present.

    Args:
        code: Python source code

    Returns:
        SecurityAnalysisResult with detected vulnerabilities

    Example:
        result = analyze_security('''
            user_input = input("Enter ID: ")
            os.system("rm " + user_input)
        ''')

        if result.has_vulnerabilities:
            print(result.summary())
    """
    _ensure_config_loaded()
    analyzer = SecurityAnalyzer()
    return analyzer.analyze(code)


def find_sql_injections(code: str) -> List[Vulnerability]:
    """
    Find SQL injection vulnerabilities in code.

    Args:
        code: Python source code

    Returns:
        List of SQL injection vulnerabilities
    """
    result = analyze_security(code)
    return result.get_sql_injections()


def find_xss(code: str) -> List[Vulnerability]:
    """
    Find XSS vulnerabilities in code.

    Args:
        code: Python source code

    Returns:
        List of XSS vulnerabilities
    """
    result = analyze_security(code)
    return result.get_xss()


def find_command_injections(code: str) -> List[Vulnerability]:
    """
    Find command injection vulnerabilities in code.

    Args:
        code: Python source code

    Returns:
        List of command injection vulnerabilities
    """
    result = analyze_security(code)
    return result.get_command_injections()


def find_path_traversals(code: str) -> List[Vulnerability]:
    """
    Find path traversal vulnerabilities in code.

    Args:
        code: Python source code

    Returns:
        List of path traversal vulnerabilities
    """
    result = analyze_security(code)
    return result.get_path_traversals()
