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

import z3
from z3 import ExprRef, BoolRef, StringSort, String, StringVal


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
    USER_INPUT = auto()      # request.args, request.form, sys.argv
    FILE_CONTENT = auto()    # open().read()
    NETWORK_DATA = auto()    # socket.recv(), requests.get()
    DATABASE = auto()        # cursor.fetchone()
    ENVIRONMENT = auto()     # os.environ
    UNKNOWN = auto()         # Source couldn't be determined


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
    """
    SQL_QUERY = auto()       # cursor.execute(), Session.execute()
    HTML_OUTPUT = auto()     # render_template(), innerHTML
    FILE_PATH = auto()       # open(), os.path.join()
    SHELL_COMMAND = auto()   # os.system(), subprocess.run()
    EVAL = auto()            # eval(), exec()
    DESERIALIZATION = auto() # pickle.loads(), yaml.load()
    LOG_OUTPUT = auto()      # logging.info() - can leak sensitive data
    HEADER = auto()          # HTTP header injection


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
    """
    source: TaintSource
    level: TaintLevel = TaintLevel.HIGH
    source_location: Optional[Tuple[int, int]] = None
    propagation_path: List[str] = field(default_factory=list)
    sanitizers_applied: Set[str] = field(default_factory=set)
    
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
        )
    
    def apply_sanitizer(self, sanitizer: str) -> TaintInfo:
        """
        Record that a sanitizer was applied.
        
        Args:
            sanitizer: Name of sanitization function
            
        Returns:
            New TaintInfo with sanitizer recorded and level lowered
        """
        new_sanitizers = self.sanitizers_applied | {sanitizer}
        
        # Lower taint level based on sanitizer
        new_level = TaintLevel.LOW if len(new_sanitizers) > 0 else self.level
        
        return TaintInfo(
            source=self.source,
            level=new_level,
            source_location=self.source_location,
            propagation_path=self.propagation_path.copy(),
            sanitizers_applied=new_sanitizers,
        )
    
    def is_dangerous_for(self, sink: SecuritySink) -> bool:
        """
        Check if this taint is dangerous for a specific sink.
        
        Some sanitizers are sink-specific:
        - html.escape() → safe for HTML_OUTPUT, NOT for SQL_QUERY
        - parameterized queries → safe for SQL_QUERY
        
        Args:
            sink: The security sink to check
            
        Returns:
            True if tainted data reaching this sink is dangerous
        """
        if self.level == TaintLevel.NONE:
            return False
        
        # Check if appropriate sanitizer was applied
        safe_sanitizers = SINK_SANITIZERS.get(sink, set())
        if self.sanitizers_applied & safe_sanitizers:
            return False
        
        return True


# Mapping of sinks to sanitizers that make them safe
SINK_SANITIZERS: Dict[SecuritySink, Set[str]] = {
    SecuritySink.SQL_QUERY: {
        'parameterized_query',
        'sqlalchemy_text_bindparams',
        'escape_string',
    },
    SecuritySink.HTML_OUTPUT: {
        'html.escape',
        'markupsafe.escape',
        'bleach.clean',
        'cgi.escape',
    },
    SecuritySink.FILE_PATH: {
        'os.path.basename',
        'pathlib.Path.name',
        'secure_filename',
    },
    SecuritySink.SHELL_COMMAND: {
        'shlex.quote',
        'pipes.quote',
    },
    SecuritySink.EVAL: set(),  # Almost never safe
    SecuritySink.DESERIALIZATION: set(),  # Almost never safe
}


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
        self,
        name: str,
        source: TaintSource,
        location: Optional[Tuple[int, int]] = None
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
    
    def mark_tainted(
        self,
        name: str,
        taint_info: TaintInfo
    ) -> None:
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
        self,
        target: str,
        source_names: List[str]
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
                            sanitizers_applied=merged_taint.sanitizers_applied & source_taint.sanitizers_applied,
                        )
        
        if merged_taint is not None:
            self._taint_map[target] = merged_taint
        else:
            # Target is clean - remove any existing taint
            self._taint_map.pop(target, None)
        
        return merged_taint
    
    def propagate_concat(
        self,
        result_name: str,
        operand_names: List[str]
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
    
    def apply_sanitizer(
        self,
        var_name: str,
        sanitizer: str
    ) -> Optional[TaintInfo]:
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
        location: Optional[Tuple[int, int]] = None
    ) -> Optional['Vulnerability']:
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
    
    def get_vulnerabilities(self) -> List['Vulnerability']:
        """Get all detected vulnerabilities."""
        return self._vulnerabilities.copy()
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def fork(self) -> 'TaintTracker':
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
    
    def __repr__(self) -> str:
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
    
    # HTML/XSS
    "render_template_string": SecuritySink.HTML_OUTPUT,
    "Markup": SecuritySink.HTML_OUTPUT,
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
    "pickle.loads": SecuritySink.DESERIALIZATION,
    "yaml.load": SecuritySink.DESERIALIZATION,
    "yaml.unsafe_load": SecuritySink.DESERIALIZATION,
    "marshal.loads": SecuritySink.DESERIALIZATION,
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
