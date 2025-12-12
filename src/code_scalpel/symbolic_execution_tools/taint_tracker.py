"""
Taint Tracking - The Bloodhound of Security Analysis.

This module provides taint propagation for detecting security vulnerabilities:
- SQL Injection
- XSS (Cross-Site Scripting)
- Path Traversal
- Command Injection

CRITICAL CONCEPT: Taint Sources and Sinks
==========================================

TAINT SOURCE: Where untrusted data enters the system
    - User input (request.args, request.form)
    - File reads
    - Network data
    - Database queries (sometimes)
    - Environment variables

TAINT SINK: Where data reaches dangerous operations
    - SQL queries (cursor.execute)
    - HTML output (render_template, innerHTML)
    - File paths (open(), os.path.join)
    - Shell commands (os.system, subprocess)

A VULNERABILITY exists when:
    TAINTED DATA flows from SOURCE → SINK without SANITIZATION

This module tracks taint through:
1. Variable assignments (x = tainted_input)
2. String operations (query = "SELECT " + tainted_input)
3. Function returns (may or may not propagate taint)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from z3 import ExprRef, String


class TaintSource(Enum):
    """
    Categories of taint sources.

    Each source has different security implications:
    - USER_INPUT: Most dangerous, directly controlled by attacker
    - FILE_CONTENT: Dangerous if file path is also tainted
    - NETWORK_DATA: Remote attacker controlled
    - DATABASE: May be pre-tainted from prior injection
    - ENVIRONMENT: Less common attack vector
    """

    USER_INPUT = auto()  # request.args, request.form, sys.argv
    FILE_CONTENT = auto()  # open().read()
    NETWORK_DATA = auto()  # socket.recv(), requests.get()
    DATABASE = auto()  # cursor.fetchone()
    ENVIRONMENT = auto()  # os.environ
    HARDCODED = auto()  # Hardcoded secrets
    UNKNOWN = auto()  # Source couldn't be determined


class SecuritySink(Enum):
    """
    Categories of security sinks where tainted data is dangerous.

    Each sink type corresponds to a different vulnerability class:
    - SQL_QUERY: SQL Injection (CWE-89)
    - HTML_OUTPUT: XSS (CWE-79)
    - FILE_PATH: Path Traversal (CWE-22)
    - SHELL_COMMAND: Command Injection (CWE-78)
    - EVAL: Code Injection (CWE-94)
    - DESERIALIZATION: Insecure Deserialization (CWE-502)
    - WEAK_CRYPTO: Use of Weak Cryptographic Algorithm (CWE-327)
    - SSRF: Server-Side Request Forgery (CWE-918)
    """

    SQL_QUERY = auto()  # cursor.execute(), Session.execute()
    HTML_OUTPUT = auto()  # render_template(), innerHTML
    FILE_PATH = auto()  # open(), os.path.join()
    SHELL_COMMAND = auto()  # os.system(), subprocess.run()
    EVAL = auto()  # eval(), exec()
    DESERIALIZATION = auto()  # pickle.loads(), yaml.load()
    LOG_OUTPUT = auto()  # logging.info() - can leak sensitive data
    HEADER = auto()  # HTTP header injection
    WEAK_CRYPTO = auto()  # hashlib.md5(), hashlib.sha1(), DES
    SSRF = auto()  # requests.get(), urllib.request.urlopen()
    HARDCODED_SECRET = auto()  # Hardcoded secrets (AWS keys, etc.)


class TaintLevel(Enum):
    """
    Confidence level of taint.

    HIGH: Definitely tainted (direct assignment from source)
    MEDIUM: Probably tainted (flows through operations)
    LOW: Possibly tainted (partial sanitization applied)
    NONE: Not tainted (concrete value or sanitized)
    """

    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    NONE = auto()


@dataclass
class TaintInfo:
    """
    Taint metadata attached to a symbolic value.

    Attributes:
        source: Where the taint originated
        level: Confidence level of taintedness
        source_location: (line, column) in source code
        propagation_path: List of variable names taint flowed through
        sanitizers_applied: Set of sanitization functions applied
        cleared_sinks: Sinks that are safe due to sanitization
    """

    source: TaintSource
    level: TaintLevel = TaintLevel.HIGH
    source_location: Optional[Tuple[int, int]] = None
    propagation_path: List[str] = field(default_factory=list)
    sanitizers_applied: Set[str] = field(default_factory=set)
    cleared_sinks: Set[SecuritySink] = field(default_factory=set)

    def propagate(self, through_var: str) -> TaintInfo:
        """
        Create new TaintInfo when taint propagates through a variable.

        Args:
            through_var: Name of variable taint is flowing through

        Returns:
            New TaintInfo with updated propagation path
        """
        return TaintInfo(
            source=self.source,
            level=self.level,
            source_location=self.source_location,
            propagation_path=self.propagation_path + [through_var],
            sanitizers_applied=self.sanitizers_applied.copy(),
            cleared_sinks=self.cleared_sinks.copy(),
        )

    def apply_sanitizer(self, sanitizer: str) -> TaintInfo:
        """
        Record that a sanitizer was applied and clear relevant sinks.

        Args:
            sanitizer: Name of sanitization function

        Returns:
            New TaintInfo with sanitizer recorded, level lowered, and sinks cleared
        """
        new_sanitizers = self.sanitizers_applied | {sanitizer}

        # Get which sinks this sanitizer clears
        sanitizer_info = SANITIZER_REGISTRY.get(sanitizer)
        new_cleared = self.cleared_sinks.copy()

        if sanitizer_info is not None:
            if sanitizer_info.full_clear:
                # Type coercion (int, float, bool) clears ALL sinks
                new_cleared = set(SecuritySink)
            else:
                # Partial clear - only specific sinks
                new_cleared |= sanitizer_info.clears_sinks

        # Lower taint level based on sanitizer
        new_level = TaintLevel.LOW if len(new_sanitizers) > 0 else self.level

        # If all dangerous sinks are cleared, mark as NONE
        if new_cleared >= {
            SecuritySink.SQL_QUERY,
            SecuritySink.HTML_OUTPUT,
            SecuritySink.FILE_PATH,
            SecuritySink.SHELL_COMMAND,
        }:
            new_level = TaintLevel.NONE

        return TaintInfo(
            source=self.source,
            level=new_level,
            source_location=self.source_location,
            propagation_path=self.propagation_path.copy(),
            sanitizers_applied=new_sanitizers,
            cleared_sinks=new_cleared,
        )

    def is_dangerous_for(self, sink: SecuritySink) -> bool:
        """
        Check if this taint is dangerous for a specific sink.

        Some sanitizers are sink-specific:
        - html.escape() → safe for HTML_OUTPUT, NOT for SQL_QUERY
        - int() → safe for ALL sinks (type coercion)

        Args:
            sink: The security sink to check

        Returns:
            True if tainted data reaching this sink is dangerous
        """
        if self.level == TaintLevel.NONE:
            return False

        # Check if this specific sink was cleared by a sanitizer
        if sink in self.cleared_sinks:
            return False

        # Backward compatibility: check SINK_SANITIZERS
        safe_sanitizers = SINK_SANITIZERS.get(sink, set())
        if self.sanitizers_applied & safe_sanitizers:
            return False

        return True


# Mapping of sinks to sanitizers that make them safe
SINK_SANITIZERS: Dict[SecuritySink, Set[str]] = {
    SecuritySink.SQL_QUERY: {
        "parameterized_query",
        "sqlalchemy_text_bindparams",
        "escape_string",
    },
    SecuritySink.HTML_OUTPUT: {
        "html.escape",
        "markupsafe.escape",
        "bleach.clean",
        "cgi.escape",
    },
    SecuritySink.FILE_PATH: {
        "os.path.basename",
        "pathlib.Path.name",
        "secure_filename",
    },
    SecuritySink.SHELL_COMMAND: {
        "shlex.quote",
        "pipes.quote",
    },
    SecuritySink.EVAL: set(),  # Almost never safe
    SecuritySink.DESERIALIZATION: set(),  # Almost never safe
}


# =============================================================================
# Sanitizer Registry (RFC-002: The Silencer)
# =============================================================================


@dataclass
class SanitizerInfo:
    """
    Information about a sanitizer function.

    Attributes:
        name: Full function name (e.g., "html.escape")
        clears_sinks: Which sink types this sanitizer protects against
        full_clear: If True, clears ALL taint (e.g., int(), float())
    """

    name: str
    clears_sinks: Set[SecuritySink] = field(default_factory=set)
    full_clear: bool = False


# Built-in sanitizer registry
# Users can extend via pyproject.toml [tool.code-scalpel.sanitizers]
SANITIZER_REGISTRY: Dict[str, SanitizerInfo] = {
    # XSS sanitizers
    "html.escape": SanitizerInfo("html.escape", {SecuritySink.HTML_OUTPUT}),
    "markupsafe.escape": SanitizerInfo("markupsafe.escape", {SecuritySink.HTML_OUTPUT}),
    "markupsafe.Markup": SanitizerInfo("markupsafe.Markup", {SecuritySink.HTML_OUTPUT}),
    "bleach.clean": SanitizerInfo("bleach.clean", {SecuritySink.HTML_OUTPUT}),
    "cgi.escape": SanitizerInfo("cgi.escape", {SecuritySink.HTML_OUTPUT}),
    # SQL sanitizers
    "escape_string": SanitizerInfo("escape_string", {SecuritySink.SQL_QUERY}),
    "mysql.connector.escape_string": SanitizerInfo(
        "mysql.connector.escape_string", {SecuritySink.SQL_QUERY}
    ),
    # Path sanitizers
    "os.path.basename": SanitizerInfo("os.path.basename", {SecuritySink.FILE_PATH}),
    "werkzeug.utils.secure_filename": SanitizerInfo(
        "werkzeug.utils.secure_filename", {SecuritySink.FILE_PATH}
    ),
    "secure_filename": SanitizerInfo("secure_filename", {SecuritySink.FILE_PATH}),
    # Shell sanitizers
    "shlex.quote": SanitizerInfo("shlex.quote", {SecuritySink.SHELL_COMMAND}),
    "pipes.quote": SanitizerInfo("pipes.quote", {SecuritySink.SHELL_COMMAND}),
    # Type coercion - FULL CLEAR (converts to safe type)
    "int": SanitizerInfo("int", set(), full_clear=True),
    "float": SanitizerInfo("float", set(), full_clear=True),
    "bool": SanitizerInfo("bool", set(), full_clear=True),
    "str": SanitizerInfo("str", set(), full_clear=False),  # str() doesn't sanitize!
    "abs": SanitizerInfo("abs", set(), full_clear=True),
    "len": SanitizerInfo("len", set(), full_clear=True),
    "ord": SanitizerInfo("ord", set(), full_clear=True),
    "hex": SanitizerInfo("hex", set(), full_clear=True),
}


def register_sanitizer(
    name: str,
    clears_sinks: Optional[Set[SecuritySink]] = None,
    full_clear: bool = False,
) -> None:
    """
    Register a custom sanitizer function.

    Args:
        name: Full function name (e.g., "my_lib.clean_sql")
        clears_sinks: Which sink types this sanitizer protects against
        full_clear: If True, clears ALL taint

    Example:
        register_sanitizer("my_lib.clean_sql", {SecuritySink.SQL_QUERY})
    """
    SANITIZER_REGISTRY[name] = SanitizerInfo(
        name=name,
        clears_sinks=clears_sinks or set(),
        full_clear=full_clear,
    )


def load_sanitizers_from_config(config_path: Optional[str] = None) -> int:
    """
    Load custom sanitizers from pyproject.toml.

    Expected format:
        [tool.code-scalpel.sanitizers]
        "my_lib.clean_sql" = ["SQL_QUERY"]
        "utils.strip_tags" = ["HTML_OUTPUT"]
        "utils.super_clean" = ["ALL"]  # Full clear

    Args:
        config_path: Path to config file. If None, searches for pyproject.toml
                     in current directory and parent directories.

    Returns:
        Number of sanitizers loaded

    Example pyproject.toml:
        [tool.code-scalpel.sanitizers]
        "my_utils.clean_sql" = ["SQL_QUERY"]
        "my_utils.safe_print" = ["HTML_OUTPUT"]
        "my_utils.super_clean" = ["ALL"]
    """
    import os

    # Find config file
    if config_path is None:
        config_path = _find_config_file()

    if config_path is None or not os.path.exists(config_path):
        return 0

    try:
        config = _load_toml(config_path)
        if config is None:
            return 0

        sanitizers = (
            config.get("tool", {}).get("code-scalpel", {}).get("sanitizers", {})
        )

        count = 0
        for func_name, sinks in sanitizers.items():
            if not isinstance(sinks, list):
                continue  # Invalid format, skip

            # Check for full clear
            if "ALL" in sinks or "*" in sinks:
                register_sanitizer(func_name, full_clear=True)
            else:
                sink_set = set()
                for sink_name in sinks:
                    try:
                        sink_set.add(SecuritySink[sink_name])
                    except KeyError:
                        pass  # Unknown sink name, skip
                if (
                    sink_set
                ):  # Only register if we matched at least one sink  # pragma: no branch
                    register_sanitizer(func_name, sink_set)
            count += 1

        return count

    except Exception:
        # Don't crash on config errors, just skip loading
        return 0


def _find_config_file() -> Optional[str]:
    """Search for pyproject.toml in current and parent directories."""
    import os

    current = os.getcwd()

    # Search up to 10 levels
    for _ in range(10):  # pragma: no branch
        candidate = os.path.join(current, "pyproject.toml")
        if os.path.exists(candidate):
            return candidate

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return None


def _load_toml(path: str) -> Optional[Dict[str, Any]]:
    """Load a TOML file using available parser."""
    # Python 3.11+ has tomllib built-in
    try:
        import tomllib  # pragma: no cover

        with open(path, "rb") as f:  # pragma: no cover
            return tomllib.load(f)  # pragma: no cover
    except ImportError:
        pass

    # Fallback to tomli (pip install tomli)
    try:
        import tomli

        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass

    # No TOML parser available
    return None


@dataclass
class TaintedValue:
    """
    A symbolic value with taint information attached.

    This wraps a Z3 expression with taint metadata for tracking
    data flow through the program.

    Attributes:
        expr: The Z3 symbolic expression
        taint: Taint metadata (None if not tainted)
    """

    expr: ExprRef
    taint: Optional[TaintInfo] = None

    @property
    def is_tainted(self) -> bool:
        """Check if this value is tainted."""
        return self.taint is not None and self.taint.level != TaintLevel.NONE

    def __repr__(self) -> str:
        if self.is_tainted:
            return f"TaintedValue({self.expr}, taint={self.taint.source.name})"
        return f"TaintedValue({self.expr}, clean)"


class TaintTracker:
    """
    Tracks taint propagation through symbolic execution.

    This class maintains a shadow state alongside the symbolic state,
    tracking which variables are tainted and how taint flows through
    operations.

    Example:
        tracker = TaintTracker()

        # Mark user input as tainted
        user_input = tracker.taint_source("user_input", TaintSource.USER_INPUT)

        # Track operations
        query = tracker.concat(StringVal("SELECT * FROM users WHERE id="), user_input)

        # Check for vulnerabilities
        if tracker.reaches_sink(query, SecuritySink.SQL_QUERY):
            print("SQL Injection vulnerability!")
    """

    def __init__(self):
        """Initialize the taint tracker."""
        self._taint_map: Dict[str, TaintInfo] = {}
        self._vulnerabilities: List[Vulnerability] = []

    # =========================================================================
    # Taint Sources
    # =========================================================================

    def taint_source(
        self, name: str, source: TaintSource, location: Optional[Tuple[int, int]] = None
    ) -> TaintedValue:
        """
        Create a tainted symbolic string from a source.

        Args:
            name: Variable name
            source: Type of taint source
            location: Source code location (line, col)

        Returns:
            TaintedValue with symbolic string and taint info
        """
        expr = String(name)
        taint = TaintInfo(
            source=source,
            level=TaintLevel.HIGH,
            source_location=location,
            propagation_path=[name],
        )

        self._taint_map[name] = taint

        return TaintedValue(expr=expr, taint=taint)

    def mark_tainted(self, name: str, taint_info: TaintInfo) -> None:
        """
        Mark an existing variable as tainted.

        Args:
            name: Variable name
            taint_info: Taint metadata
        """
        self._taint_map[name] = taint_info

    def get_taint(self, name: str) -> Optional[TaintInfo]:
        """
        Get taint info for a variable.

        Args:
            name: Variable name

        Returns:
            TaintInfo if tainted, None otherwise
        """
        return self._taint_map.get(name)

    def is_tainted(self, name: str) -> bool:
        """
        Check if a variable is tainted.

        Args:
            name: Variable name

        Returns:
            True if variable is tainted
        """
        taint = self._taint_map.get(name)
        return taint is not None and taint.level != TaintLevel.NONE

    # =========================================================================
    # Taint Propagation
    # =========================================================================

    def propagate_assignment(
        self, target: str, source_names: List[str]
    ) -> Optional[TaintInfo]:
        """
        Propagate taint through an assignment.

        If any source is tainted, the target becomes tainted.

        Args:
            target: Target variable name
            source_names: Names of variables used in RHS

        Returns:
            TaintInfo if target is now tainted
        """
        # Merge taint from all sources
        merged_taint = None

        for source_name in source_names:
            source_taint = self._taint_map.get(source_name)
            if source_taint is not None:
                if merged_taint is None:
                    merged_taint = source_taint.propagate(target)
                else:
                    # Merge: take highest taint level
                    if source_taint.level.value < merged_taint.level.value:
                        merged_taint = TaintInfo(
                            source=source_taint.source,
                            level=source_taint.level,
                            source_location=merged_taint.source_location,
                            propagation_path=merged_taint.propagation_path + [target],
                            sanitizers_applied=merged_taint.sanitizers_applied
                            & source_taint.sanitizers_applied,
                        )

        if merged_taint is not None:
            self._taint_map[target] = merged_taint
        else:
            # Target is clean - remove any existing taint
            self._taint_map.pop(target, None)

        return merged_taint

    def propagate_concat(
        self, result_name: str, operand_names: List[str]
    ) -> Optional[TaintInfo]:
        """
        Propagate taint through string concatenation.

        If ANY operand is tainted, the result is tainted.
        This is the key propagation rule for injection vulnerabilities.

        Args:
            result_name: Name of result variable
            operand_names: Names of concatenated strings

        Returns:
            TaintInfo if result is tainted
        """
        return self.propagate_assignment(result_name, operand_names)

    def apply_sanitizer(self, var_name: str, sanitizer: str) -> Optional[TaintInfo]:
        """
        Record that a sanitizer was applied to a variable.

        Args:
            var_name: Variable name
            sanitizer: Name of sanitization function

        Returns:
            Updated TaintInfo
        """
        current_taint = self._taint_map.get(var_name)
        if current_taint is None:
            return None

        new_taint = current_taint.apply_sanitizer(sanitizer)
        self._taint_map[var_name] = new_taint
        return new_taint

    # =========================================================================
    # Sink Detection
    # =========================================================================

    def check_sink(
        self,
        var_name: str,
        sink: SecuritySink,
        location: Optional[Tuple[int, int]] = None,
    ) -> Optional["Vulnerability"]:
        """
        Check if tainted data reaches a security sink.

        Args:
            var_name: Name of variable being used at sink
            sink: Type of security sink
            location: Source code location

        Returns:
            Vulnerability if detected, None if safe
        """
        taint = self._taint_map.get(var_name)

        if taint is None:
            return None

        if not taint.is_dangerous_for(sink):
            return None

        # Found a vulnerability!
        vuln = Vulnerability(
            sink_type=sink,
            taint_source=taint.source,
            taint_path=taint.propagation_path,
            sink_location=location,
            source_location=taint.source_location,
            sanitizers_applied=taint.sanitizers_applied,
        )

        self._vulnerabilities.append(vuln)
        return vuln

    def get_vulnerabilities(self) -> List["Vulnerability"]:
        """Get all detected vulnerabilities."""
        return self._vulnerabilities.copy()

    # =========================================================================
    # State Management
    # =========================================================================

    def fork(self) -> "TaintTracker":
        """
        Create an isolated copy for branching.

        Returns:
            New TaintTracker with copied state
        """
        forked = TaintTracker()
        forked._taint_map = {k: v for k, v in self._taint_map.items()}
        forked._vulnerabilities = self._vulnerabilities.copy()
        return forked

    def clear(self) -> None:
        """Reset all taint tracking state."""
        self._taint_map.clear()
        self._vulnerabilities.clear()


@dataclass
class Vulnerability:
    """
    A detected security vulnerability.

    Attributes:
        sink_type: Type of dangerous operation
        taint_source: Where the tainted data originated
        taint_path: Variables the taint flowed through
        sink_location: Where the vulnerability is (line, col)
        source_location: Where tainted data entered (line, col)
        sanitizers_applied: Sanitizers that were applied (but insufficient)
    """

    sink_type: SecuritySink
    taint_source: TaintSource
    taint_path: List[str]
    sink_location: Optional[Tuple[int, int]] = None
    source_location: Optional[Tuple[int, int]] = None
    sanitizers_applied: Set[str] = field(default_factory=set)

    @property
    def vulnerability_type(self) -> str:
        """Get the common name for this vulnerability type."""
        mapping = {
            SecuritySink.SQL_QUERY: "SQL Injection",
            SecuritySink.HTML_OUTPUT: "Cross-Site Scripting (XSS)",
            SecuritySink.FILE_PATH: "Path Traversal",
            SecuritySink.SHELL_COMMAND: "Command Injection",
            SecuritySink.EVAL: "Code Injection",
            SecuritySink.DESERIALIZATION: "Insecure Deserialization",
            SecuritySink.LOG_OUTPUT: "Log Injection",
            SecuritySink.HEADER: "HTTP Header Injection",
            SecuritySink.WEAK_CRYPTO: "Use of Weak Cryptographic Hash",
            SecuritySink.SSRF: "Server-Side Request Forgery (SSRF)",
            SecuritySink.HARDCODED_SECRET: "Hardcoded Secret",
        }
        return mapping.get(self.sink_type, "Unknown Vulnerability")

    @property
    def cwe_id(self) -> str:
        """Get the CWE identifier for this vulnerability."""
        mapping = {
            SecuritySink.SQL_QUERY: "CWE-89",
            SecuritySink.HTML_OUTPUT: "CWE-79",
            SecuritySink.FILE_PATH: "CWE-22",
            SecuritySink.SHELL_COMMAND: "CWE-78",
            SecuritySink.EVAL: "CWE-94",
            SecuritySink.DESERIALIZATION: "CWE-502",
            SecuritySink.LOG_OUTPUT: "CWE-117",
            SecuritySink.HEADER: "CWE-113",
            SecuritySink.WEAK_CRYPTO: "CWE-327",
            SecuritySink.SSRF: "CWE-918",
            SecuritySink.HARDCODED_SECRET: "CWE-798",
        }
        return mapping.get(self.sink_type, "CWE-Unknown")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.vulnerability_type,
            "cwe": self.cwe_id,
            "sink": self.sink_type.name,
            "source": self.taint_source.name,
            "taint_path": self.taint_path,
            "sink_location": self.sink_location,
            "source_location": self.source_location,
            "sanitizers": list(self.sanitizers_applied),
        }

    def __repr__(self) -> str:  # pragma: no cover
        path_str = " → ".join(self.taint_path)
        return (
            f"Vulnerability({self.vulnerability_type}, "
            f"flow: {path_str}, "
            f"{self.cwe_id})"
        )


# =============================================================================
# Known Taint Sources - Pattern Matching
# =============================================================================

# Function calls that introduce taint
TAINT_SOURCE_PATTERNS: Dict[str, TaintSource] = {
    # Flask/Django request handling
    "request.args.get": TaintSource.USER_INPUT,
    "request.form.get": TaintSource.USER_INPUT,
    "request.form": TaintSource.USER_INPUT,
    "request.args": TaintSource.USER_INPUT,
    "request.data": TaintSource.USER_INPUT,
    "request.json": TaintSource.USER_INPUT,
    "request.cookies.get": TaintSource.USER_INPUT,
    "request.headers.get": TaintSource.USER_INPUT,
    "request.GET.get": TaintSource.USER_INPUT,
    "request.POST.get": TaintSource.USER_INPUT,
    "request.GET": TaintSource.USER_INPUT,
    "request.POST": TaintSource.USER_INPUT,
    # Standard input
    "input": TaintSource.USER_INPUT,
    "sys.argv": TaintSource.USER_INPUT,
    # File operations
    "open.read": TaintSource.FILE_CONTENT,
    "file.read": TaintSource.FILE_CONTENT,
    "Path.read_text": TaintSource.FILE_CONTENT,
    # Network
    "socket.recv": TaintSource.NETWORK_DATA,
    "requests.get": TaintSource.NETWORK_DATA,
    "urllib.request.urlopen": TaintSource.NETWORK_DATA,
    # Database
    "cursor.fetchone": TaintSource.DATABASE,
    "cursor.fetchall": TaintSource.DATABASE,
    "cursor.fetchmany": TaintSource.DATABASE,
    # Environment
    "os.environ.get": TaintSource.ENVIRONMENT,
    "os.getenv": TaintSource.ENVIRONMENT,
}

# Function calls that are security sinks
SINK_PATTERNS: Dict[str, SecuritySink] = {
    # SQL
    "cursor.execute": SecuritySink.SQL_QUERY,
    "connection.execute": SecuritySink.SQL_QUERY,
    "session.execute": SecuritySink.SQL_QUERY,
    "engine.execute": SecuritySink.SQL_QUERY,
    "RawSQL": SecuritySink.SQL_QUERY,
    "django.db.models.expressions.RawSQL": SecuritySink.SQL_QUERY,
    "django.db.models.RawSQL": SecuritySink.SQL_QUERY,
    "extra": SecuritySink.SQL_QUERY,
    "QuerySet.extra": SecuritySink.SQL_QUERY,
    "text": SecuritySink.SQL_QUERY,
    "sqlalchemy.text": SecuritySink.SQL_QUERY,
    "sqlalchemy.sql.expression.text": SecuritySink.SQL_QUERY,
    # HTML/XSS
    "render_template_string": SecuritySink.HTML_OUTPUT,
    "flask.render_template_string": SecuritySink.HTML_OUTPUT,
    "Response": SecuritySink.HTML_OUTPUT,
    "flask.Response": SecuritySink.HTML_OUTPUT,
    "make_response": SecuritySink.HTML_OUTPUT,
    "flask.make_response": SecuritySink.HTML_OUTPUT,
    "Markup": SecuritySink.HTML_OUTPUT,
    "flask.Markup": SecuritySink.HTML_OUTPUT,
    "markupsafe.Markup": SecuritySink.HTML_OUTPUT,
    "innerHTML": SecuritySink.HTML_OUTPUT,
    # File paths
    "open": SecuritySink.FILE_PATH,
    "os.path.join": SecuritySink.FILE_PATH,
    "pathlib.Path": SecuritySink.FILE_PATH,
    "shutil.copy": SecuritySink.FILE_PATH,
    # Shell commands
    "os.system": SecuritySink.SHELL_COMMAND,
    "os.popen": SecuritySink.SHELL_COMMAND,
    "subprocess.run": SecuritySink.SHELL_COMMAND,
    "subprocess.call": SecuritySink.SHELL_COMMAND,
    "subprocess.Popen": SecuritySink.SHELL_COMMAND,
    # Eval
    "eval": SecuritySink.EVAL,
    "exec": SecuritySink.EVAL,
    "compile": SecuritySink.EVAL,
    # Deserialization
    "pickle.load": SecuritySink.DESERIALIZATION,
    "pickle.loads": SecuritySink.DESERIALIZATION,
    "_pickle.load": SecuritySink.DESERIALIZATION,
    "_pickle.loads": SecuritySink.DESERIALIZATION,
    "yaml.load": SecuritySink.DESERIALIZATION,
    "yaml.unsafe_load": SecuritySink.DESERIALIZATION,
    "marshal.loads": SecuritySink.DESERIALIZATION,
    # Weak Cryptography (CWE-327)
    "hashlib.md5": SecuritySink.WEAK_CRYPTO,
    "hashlib.sha1": SecuritySink.WEAK_CRYPTO,
    "cryptography.hazmat.primitives.ciphers.algorithms.DES": SecuritySink.WEAK_CRYPTO,
    "Crypto.Cipher.DES": SecuritySink.WEAK_CRYPTO,  # PyCryptodome
    "Crypto.Hash.MD5": SecuritySink.WEAK_CRYPTO,
    "Crypto.Hash.SHA": SecuritySink.WEAK_CRYPTO,
    "DES": SecuritySink.WEAK_CRYPTO,
    "MD5.new": SecuritySink.WEAK_CRYPTO,
    "SHA.new": SecuritySink.WEAK_CRYPTO,
    # SSRF - Server-Side Request Forgery (CWE-918)
    "requests.get": SecuritySink.SSRF,
    "requests.post": SecuritySink.SSRF,
    "requests.put": SecuritySink.SSRF,
    "requests.delete": SecuritySink.SSRF,
    "requests.head": SecuritySink.SSRF,
    "requests.patch": SecuritySink.SSRF,
    "urllib.request.urlopen": SecuritySink.SSRF,
    "urllib.request.Request": SecuritySink.SSRF,
    "urlopen": SecuritySink.SSRF,
    "Request": SecuritySink.SSRF,
    "httpx.get": SecuritySink.SSRF,
    "httpx.post": SecuritySink.SSRF,
    "httpx.AsyncClient.get": SecuritySink.SSRF,
    "aiohttp.ClientSession.get": SecuritySink.SSRF,
}

# Sanitizer function patterns
SANITIZER_PATTERNS: Dict[str, str] = {
    "html.escape": "html.escape",
    "markupsafe.escape": "markupsafe.escape",
    "bleach.clean": "bleach.clean",
    "cgi.escape": "cgi.escape",
    "shlex.quote": "shlex.quote",
    "os.path.basename": "os.path.basename",
    "werkzeug.utils.secure_filename": "secure_filename",
}
