"""
Code Scalpel MCP Server - Real MCP Protocol Implementation.

This server implements the Model Context Protocol (MCP) specification using
the official Python SDK. It exposes Code Scalpel's analysis tools to any
MCP-compliant client (Claude Desktop, Cursor, etc.).

Transports:
- stdio: Default. Client spawns server as subprocess. Best for local use.
- streamable-http: Network deployment. Requires explicit --transport flag.

Usage:
    # stdio (default)
    python -m code_scalpel.mcp.server

    # HTTP transport for network access
    python -m code_scalpel.mcp.server --transport streamable-http --port 8080

Security:
    - Code is PARSED, never executed (ast.parse only)
    - Maximum code size enforced
    - HTTP transport binds to 127.0.0.1 by default
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from code_scalpel import SurgicalExtractor

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

__version__ = "1.0.2"

# Setup logging
logger = logging.getLogger(__name__)

# Maximum code size to prevent resource exhaustion
MAX_CODE_SIZE = 100_000

# Project root for resources (default to current directory)
PROJECT_ROOT = Path.cwd()

# Caching enabled by default
CACHE_ENABLED = os.environ.get("SCALPEL_CACHE_ENABLED", "1") != "0"


# ============================================================================
# CACHING
# ============================================================================


def _get_cache():
    """Get the analysis cache (lazy initialization)."""
    if not CACHE_ENABLED:
        return None
    try:
        from code_scalpel.utilities.cache import get_cache

        return get_cache()
    except ImportError:
        logger.warning("Cache module not available")
        return None


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================


class FunctionInfo(BaseModel):
    """Information about a function."""

    name: str = Field(description="Function name")
    lineno: int = Field(description="Line number where function starts")
    end_lineno: int | None = Field(default=None, description="Line number where function ends")
    is_async: bool = Field(default=False, description="Whether function is async")


class ClassInfo(BaseModel):
    """Information about a class."""

    name: str = Field(description="Class name")
    lineno: int = Field(description="Line number where class starts")
    end_lineno: int | None = Field(default=None, description="Line number where class ends")
    methods: list[str] = Field(default_factory=list, description="Method names in class")


class AnalysisResult(BaseModel):
    """Result of code analysis."""

    success: bool = Field(description="Whether analysis succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    functions: list[str] = Field(description="List of function names found")
    classes: list[str] = Field(description="List of class names found")
    imports: list[str] = Field(description="List of import statements")
    function_count: int = Field(description="Total number of functions found")
    class_count: int = Field(description="Total number of classes found")
    complexity: int = Field(description="Cyclomatic complexity estimate")
    lines_of_code: int = Field(description="Total lines of code")
    issues: list[str] = Field(default_factory=list, description="Issues found")
    error: str | None = Field(default=None, description="Error message if failed")
    # v1.3.0: Detailed info with line numbers
    function_details: list[FunctionInfo] = Field(default_factory=list, description="Detailed function info with line numbers")
    class_details: list[ClassInfo] = Field(default_factory=list, description="Detailed class info with line numbers")


class VulnerabilityInfo(BaseModel):
    """Information about a detected vulnerability."""

    type: str = Field(description="Vulnerability type (e.g., SQL Injection)")
    cwe: str = Field(description="CWE identifier")
    severity: str = Field(description="Severity level")
    line: int | None = Field(default=None, description="Line number if known")
    description: str = Field(description="Description of the vulnerability")


class SecurityResult(BaseModel):
    """Result of security analysis."""

    success: bool = Field(description="Whether analysis succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    has_vulnerabilities: bool = Field(description="Whether vulnerabilities were found")
    vulnerability_count: int = Field(description="Number of vulnerabilities")
    risk_level: str = Field(description="Overall risk level")
    vulnerabilities: list[VulnerabilityInfo] = Field(
        default_factory=list, description="List of vulnerabilities"
    )
    taint_sources: list[str] = Field(
        default_factory=list, description="Identified taint sources"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class PathCondition(BaseModel):
    """A condition along an execution path."""

    condition: str = Field(description="The condition expression")
    is_satisfiable: bool = Field(description="Whether condition is satisfiable")


class ExecutionPath(BaseModel):
    """An execution path discovered by symbolic execution."""

    path_id: int = Field(description="Unique path identifier")
    conditions: list[str] = Field(description="Conditions along the path")
    final_state: dict[str, Any] = Field(description="Variable values at path end")
    reproduction_input: dict[str, Any] | None = Field(
        default=None, description="Input values that trigger this path"
    )
    is_reachable: bool = Field(description="Whether path is reachable")


class SymbolicResult(BaseModel):
    """Result of symbolic execution."""

    success: bool = Field(description="Whether analysis succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    paths_explored: int = Field(description="Number of execution paths explored")
    paths: list[ExecutionPath] = Field(
        default_factory=list, description="Discovered execution paths"
    )
    symbolic_variables: list[str] = Field(
        default_factory=list, description="Variables treated symbolically"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Discovered constraints"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class GeneratedTestCase(BaseModel):
    """A generated test case."""

    path_id: int = Field(description="Path ID this test covers")
    function_name: str = Field(description="Function being tested")
    inputs: dict[str, Any] = Field(description="Input values for this test")
    description: str = Field(description="Human-readable description")
    path_conditions: list[str] = Field(
        default_factory=list, description="Conditions that define this path"
    )


class TestGenerationResult(BaseModel):
    """Result of test generation."""

    success: bool = Field(description="Whether generation succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    function_name: str = Field(description="Function tests were generated for")
    test_count: int = Field(description="Number of test cases generated")
    test_cases: list[GeneratedTestCase] = Field(
        default_factory=list, description="Generated test cases"
    )
    pytest_code: str = Field(default="", description="Generated pytest code")
    unittest_code: str = Field(default="", description="Generated unittest code")
    error: str | None = Field(default=None, description="Error message if failed")


class RefactorSecurityIssue(BaseModel):
    """A security issue found in refactored code."""

    type: str = Field(description="Vulnerability type")
    severity: str = Field(description="Severity level")
    line: int | None = Field(default=None, description="Line number")
    description: str = Field(description="Issue description")
    cwe: str | None = Field(default=None, description="CWE identifier")


class RefactorSimulationResult(BaseModel):
    """Result of refactor simulation."""

    success: bool = Field(description="Whether simulation succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    is_safe: bool = Field(description="Whether the refactor is safe to apply")
    status: str = Field(description="Status: safe, unsafe, warning, or error")
    reason: str | None = Field(default=None, description="Reason if not safe")
    security_issues: list[RefactorSecurityIssue] = Field(
        default_factory=list, description="Security issues found"
    )
    structural_changes: dict[str, Any] = Field(
        default_factory=dict, description="Functions/classes added/removed"
    )
    warnings: list[str] = Field(default_factory=list, description="Warnings")
    error: str | None = Field(default=None, description="Error message if failed")


class CrawlFunctionInfo(BaseModel):
    """Information about a function from project crawl."""

    name: str = Field(description="Function name (qualified if method)")
    lineno: int = Field(description="Line number")
    complexity: int = Field(description="Cyclomatic complexity")


class CrawlClassInfo(BaseModel):
    """Information about a class from project crawl."""

    name: str = Field(description="Class name")
    lineno: int = Field(description="Line number")
    methods: list[CrawlFunctionInfo] = Field(
        default_factory=list, description="Methods in the class"
    )
    bases: list[str] = Field(default_factory=list, description="Base classes")


class CrawlFileResult(BaseModel):
    """Result of analyzing a single file during crawl."""

    path: str = Field(description="Relative path to the file")
    status: str = Field(description="success or error")
    lines_of_code: int = Field(default=0, description="Lines of code")
    functions: list[CrawlFunctionInfo] = Field(
        default_factory=list, description="Top-level functions"
    )
    classes: list[CrawlClassInfo] = Field(
        default_factory=list, description="Classes found"
    )
    imports: list[str] = Field(default_factory=list, description="Import statements")
    complexity_warnings: list[CrawlFunctionInfo] = Field(
        default_factory=list, description="High-complexity functions"
    )
    error: str | None = Field(default=None, description="Error if failed")


class CrawlSummary(BaseModel):
    """Summary statistics from project crawl."""

    total_files: int = Field(description="Total files scanned")
    successful_files: int = Field(description="Files analyzed successfully")
    failed_files: int = Field(description="Files that failed analysis")
    total_lines_of_code: int = Field(description="Total lines of code")
    total_functions: int = Field(description="Total functions found")
    total_classes: int = Field(description="Total classes found")
    complexity_warnings: int = Field(description="Number of high-complexity functions")


class ProjectCrawlResult(BaseModel):
    """Result of crawling an entire project."""

    success: bool = Field(description="Whether crawl succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    root_path: str = Field(description="Project root path")
    timestamp: str = Field(description="When the crawl was performed")
    summary: CrawlSummary = Field(description="Summary statistics")
    files: list[CrawlFileResult] = Field(
        default_factory=list, description="Analyzed files"
    )
    errors: list[CrawlFileResult] = Field(
        default_factory=list, description="Files with errors"
    )
    markdown_report: str = Field(default="", description="Markdown report")
    error: str | None = Field(default=None, description="Error if failed")


class SurgicalExtractionResult(BaseModel):
    """Result of surgical code extraction."""

    success: bool = Field(description="Whether extraction succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    name: str = Field(description="Name of extracted element")
    code: str = Field(description="Extracted source code")
    node_type: str = Field(description="Type: function, class, or method")
    line_start: int = Field(default=0, description="Starting line number")
    line_end: int = Field(default=0, description="Ending line number")
    dependencies: list[str] = Field(
        default_factory=list, description="Names of dependencies"
    )
    imports_needed: list[str] = Field(
        default_factory=list, description="Required import statements"
    )
    token_estimate: int = Field(default=0, description="Estimated token count")
    error: str | None = Field(default=None, description="Error if failed")


class ContextualExtractionResult(BaseModel):
    """Result of extraction with dependencies included."""

    success: bool = Field(description="Whether extraction succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    target_name: str = Field(description="Name of target element")
    target_code: str = Field(description="Target element source code")
    context_code: str = Field(description="Combined dependency source code")
    full_code: str = Field(description="Complete code block for LLM consumption")
    context_items: list[str] = Field(
        default_factory=list, description="Names of included dependencies"
    )
    total_lines: int = Field(default=0, description="Total lines in extraction")
    # v1.3.0: Line number information
    line_start: int = Field(default=0, description="Starting line number of target")
    line_end: int = Field(default=0, description="Ending line number of target")
    token_estimate: int = Field(default=0, description="Estimated token count")
    error: str | None = Field(default=None, description="Error if failed")


class PatchResultModel(BaseModel):
    """Result of a surgical code modification."""

    success: bool = Field(description="Whether the patch was applied successfully")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    file_path: str = Field(description="Path to the modified file")
    target_name: str = Field(description="Name of the modified symbol")
    target_type: str = Field(description="Type: function, class, or method")
    lines_before: int = Field(default=0, description="Lines in original code")
    lines_after: int = Field(default=0, description="Lines in replacement code")
    lines_delta: int = Field(default=0, description="Change in line count")
    backup_path: str | None = Field(default=None, description="Path to backup file")
    error: str | None = Field(default=None, description="Error message if failed")


# [20251212_FEATURE] v1.4.0 - New MCP tool models for enhanced AI context

class FileContextResult(BaseModel):
    """Result of get_file_context - file overview without full content."""

    success: bool = Field(description="Whether analysis succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    file_path: str = Field(description="Path to the analyzed file")
    language: str = Field(default="python", description="Detected language")
    line_count: int = Field(description="Total lines in file")
    functions: list[str] = Field(default_factory=list, description="Function names")
    classes: list[str] = Field(default_factory=list, description="Class names")
    imports: list[str] = Field(default_factory=list, description="Import statements")
    exports: list[str] = Field(default_factory=list, description="Exported symbols (__all__)")
    complexity_score: int = Field(default=0, description="Overall cyclomatic complexity")
    has_security_issues: bool = Field(default=False, description="Whether file has security issues")
    summary: str = Field(default="", description="Brief description of file purpose")
    error: str | None = Field(default=None, description="Error message if failed")


class SymbolReference(BaseModel):
    """A single reference to a symbol."""

    file: str = Field(description="File path containing the reference")
    line: int = Field(description="Line number of the reference")
    column: int = Field(default=0, description="Column number")
    context: str = Field(description="Code snippet showing usage context")
    is_definition: bool = Field(default=False, description="Whether this is the definition")


class SymbolReferencesResult(BaseModel):
    """Result of get_symbol_references - all usages of a symbol."""

    success: bool = Field(description="Whether search succeeded")
    server_version: str = Field(default=__version__, description="Code Scalpel version")
    symbol_name: str = Field(description="Name of the searched symbol")
    definition_file: str | None = Field(default=None, description="File where symbol is defined")
    definition_line: int | None = Field(default=None, description="Line where symbol is defined")
    references: list[SymbolReference] = Field(default_factory=list, description="All references found")
    total_references: int = Field(default=0, description="Total reference count")
    error: str | None = Field(default=None, description="Error message if failed")


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP(
    name="Code Scalpel",
    instructions=f"""Code Scalpel v{__version__} - AI-powered code analysis tools:

**TOKEN-EFFICIENT EXTRACTION (READ):**
- extract_code: Surgically extract functions/classes/methods by FILE PATH.
  The SERVER reads the file - YOU pay ~50 tokens instead of ~10,000.
  Example: extract_code(file_path="/src/utils.py", target_type="function", target_name="calculate_tax")

**SURGICAL MODIFICATION (WRITE):**
- update_symbol: Replace a function/class/method in a file with new code.
  YOU provide only the new symbol - the SERVER handles safe replacement.
  Example: update_symbol(file_path="/src/utils.py", target_type="function", 
           target_name="calculate_tax", new_code="def calculate_tax(amount): ...")
  Creates backup, validates syntax, preserves surrounding code.

**ANALYSIS TOOLS:**
- analyze_code: Parse Python/Java code, extract structure (functions, classes, imports)
- security_scan: Detect vulnerabilities using taint analysis (SQL injection, XSS, etc.)
- symbolic_execute: Explore execution paths using symbolic execution
- generate_unit_tests: Generate pytest/unittest tests from symbolic execution paths
- simulate_refactor: Verify a code change is safe before applying it
- crawl_project: Crawl entire project directory, analyze all Python files

**WORKFLOW OPTIMIZATION:**
1. Use extract_code(file_path=...) to get ONLY the symbol you need
2. Modify the extracted code
3. Use update_symbol(file_path=..., new_code=...) to apply the change safely

Code is PARSED only, never executed.""",
)


def _validate_code(code: str) -> tuple[bool, str | None]:
    """Validate code before analysis."""
    if not code:
        return False, "Code cannot be empty"
    if not isinstance(code, str):
        return False, "Code must be a string"
    if len(code) > MAX_CODE_SIZE:
        return False, f"Code exceeds maximum size of {MAX_CODE_SIZE} characters"
    return True, None


def _count_complexity(tree: ast.AST) -> int:
    """Estimate cyclomatic complexity."""
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            complexity += len(node.values) - 1
    return complexity


def _analyze_java_code(code: str) -> AnalysisResult:
    """Analyze Java code using tree-sitter."""
    try:
        from code_scalpel.code_parser.java_parsers.java_parser_treesitter import (
            JavaParser,
        )

        parser = JavaParser()
        result = parser.parse(code)
        return AnalysisResult(
            success=True,
            functions=result["functions"],
            classes=result["classes"],
            imports=result["imports"],
            function_count=len(result["functions"]),
            class_count=len(result["classes"]),
            complexity=result["complexity"],
            lines_of_code=result["lines_of_code"],
            issues=result["issues"],
        )
    except ImportError:
        return AnalysisResult(
            success=False,
            functions=[],
            classes=[],
            imports=[],
            function_count=0,
            class_count=0,
            complexity=0,
            lines_of_code=0,
            error="Java support not available. Please install tree-sitter and tree-sitter-java.",
        )
    except Exception as e:
        return AnalysisResult(
            success=False,
            functions=[],
            classes=[],
            imports=[],
            function_count=0,
            class_count=0,
            complexity=0,
            lines_of_code=0,
            error=f"Java analysis failed: {str(e)}",
        )


def _analyze_code_sync(code: str, language: str = "python") -> AnalysisResult:
    """Synchronous implementation of analyze_code."""
    valid, error = _validate_code(code)
    if not valid:
        return AnalysisResult(
            success=False,
            functions=[],
            classes=[],
            imports=[],
            function_count=0,
            class_count=0,
            complexity=0,
            lines_of_code=0,
            error=error,
        )

    # Check cache first
    cache = _get_cache()
    cache_config = {"language": language}
    if cache:
        cached = cache.get(code, "analysis", cache_config)
        if cached is not None:
            logger.debug("Cache hit for analyze_code")
            # Convert dict back to AnalysisResult if needed
            if isinstance(cached, dict):
                return AnalysisResult(**cached)
            return cached

    if language.lower() == "java":
        result = _analyze_java_code(code)
        if cache and result.success:
            cache.set(code, "analysis", result.model_dump(), cache_config)
        return result

    try:
        tree = ast.parse(code)

        functions = []
        function_details = []
        classes = []
        class_details = []
        imports = []
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                function_details.append(FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=getattr(node, 'end_lineno', None),
                    is_async=False,
                ))
                # Flag potential issues
                if len(node.name) < 2:
                    issues.append(f"Function '{node.name}' has very short name")
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(f"async {node.name}")
                function_details.append(FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=getattr(node, 'end_lineno', None),
                    is_async=True,
                ))
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Extract method names
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                class_details.append(ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=getattr(node, 'end_lineno', None),
                    methods=methods,
                ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        result = AnalysisResult(
            success=True,
            functions=functions,
            classes=classes,
            imports=imports,
            function_count=len(functions),
            class_count=len(classes),
            complexity=_count_complexity(tree),
            lines_of_code=len(code.splitlines()),
            issues=issues,
            function_details=function_details,
            class_details=class_details,
        )

        # Cache successful result
        if cache:
            cache.set(code, "analysis", result.model_dump(), cache_config)

        return result

    except SyntaxError as e:
        return AnalysisResult(
            success=False,
            functions=[],
            classes=[],
            imports=[],
            function_count=0,
            class_count=0,
            complexity=0,
            lines_of_code=0,
            error=f"Syntax error at line {e.lineno}: {e.msg}. Please check your code syntax.",
        )
    except Exception as e:
        return AnalysisResult(
            success=False,
            functions=[],
            classes=[],
            imports=[],
            function_count=0,
            class_count=0,
            complexity=0,
            lines_of_code=0,
            error=f"Analysis failed: {str(e)}",
        )


@mcp.tool()
async def analyze_code(code: str, language: str = "python") -> AnalysisResult:
    """
    Analyze source code structure.

    Use this tool to understand the high-level architecture (classes, functions, imports)
    of a file before attempting to edit it. This helps prevent hallucinating non-existent
    methods or classes.

    Args:
        code: Source code to analyze
        language: Language of the code ("python", "java")

    Returns:
        Structured analysis result with code metrics and structure
    """
    return await asyncio.to_thread(_analyze_code_sync, code, language)


def _security_scan_sync(code: str) -> SecurityResult:
    """Synchronous implementation of security_scan."""
    valid, error = _validate_code(code)
    if not valid:
        return SecurityResult(
            success=False,
            has_vulnerabilities=False,
            vulnerability_count=0,
            risk_level="unknown",
            error=error,
        )

    # Check cache first
    cache = _get_cache()
    if cache:
        cached = cache.get(code, "security")
        if cached is not None:
            logger.debug("Cache hit for security_scan")
            if isinstance(cached, dict):
                # Reconstruct VulnerabilityInfo objects
                if "vulnerabilities" in cached:
                    cached["vulnerabilities"] = [
                        VulnerabilityInfo(**v) for v in cached["vulnerabilities"]
                    ]
                return SecurityResult(**cached)
            return cached

    try:
        # Import here to avoid circular imports
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code).to_dict()

        vulnerabilities = []
        taint_sources = []

        for vuln in result.get("vulnerabilities", []):
            # Extract line number from sink_location tuple (line, col)
            sink_loc = vuln.get("sink_location")
            line_number = sink_loc[0] if sink_loc and isinstance(sink_loc, (list, tuple)) else None
            
            vulnerabilities.append(
                VulnerabilityInfo(
                    type=vuln.get("type", "Unknown"),
                    cwe=vuln.get("cwe", "Unknown"),
                    severity=vuln.get("severity", "medium"),
                    line=line_number,
                    description=vuln.get("description", ""),
                )
            )

        for source in result.get("taint_sources", []):
            taint_sources.append(str(source))

        vuln_count = len(vulnerabilities)
        if vuln_count == 0:
            risk_level = "low"
        elif vuln_count <= 2:
            risk_level = "medium"
        elif vuln_count <= 5:
            risk_level = "high"
        else:
            risk_level = "critical"

        security_result = SecurityResult(
            success=True,
            has_vulnerabilities=vuln_count > 0,
            vulnerability_count=vuln_count,
            risk_level=risk_level,
            vulnerabilities=vulnerabilities,
            taint_sources=taint_sources,
        )

        # Cache successful result
        if cache:
            cache.set(code, "security", security_result.model_dump())

        return security_result

    except ImportError:
        # Fallback to basic pattern matching if SecurityAnalyzer not available
        return _basic_security_scan(code)
    except Exception as e:
        return SecurityResult(
            success=False,
            has_vulnerabilities=False,
            vulnerability_count=0,
            risk_level="unknown",
            error=f"Security scan failed: {str(e)}",
        )


@mcp.tool()
async def security_scan(code: str) -> SecurityResult:
    """
    Scan Python code for security vulnerabilities using taint analysis.

    Use this tool to audit code for security vulnerabilities before deploying
    or committing changes. It tracks data flow from sources to sinks.

    Detects:
    - SQL Injection (CWE-89)
    - NoSQL Injection (CWE-943) - MongoDB
    - LDAP Injection (CWE-90)
    - Cross-Site Scripting (CWE-79)
    - Command Injection (CWE-78)
    - Path Traversal (CWE-22)
    - XXE - XML External Entity (CWE-611) [v1.4.0]
    - SSTI - Server-Side Template Injection (CWE-1336) [v1.4.0]
    - Hardcoded Secrets (CWE-798) - 30+ patterns

    Args:
        code: Python source code to scan

    Returns:
        Security analysis result with vulnerabilities and risk assessment
    """
    return await asyncio.to_thread(_security_scan_sync, code)


def _basic_security_scan(code: str) -> SecurityResult:
    """Fallback security scan using pattern matching."""
    vulnerabilities = []
    taint_sources = []

    # Detect common dangerous patterns
    patterns = [
        (
            "execute(",
            "SQL Injection",
            "CWE-89",
            "Possible SQL injection via execute()",
        ),
        ("cursor.execute", "SQL Injection", "CWE-89", "SQL query execution detected"),
        ("os.system(", "Command Injection", "CWE-78", "os.system() call detected"),
        (
            "subprocess.call(",
            "Command Injection",
            "CWE-78",
            "subprocess.call() detected",
        ),
        ("eval(", "Code Injection", "CWE-94", "eval() call detected"),
        ("exec(", "Code Injection", "CWE-94", "exec() call detected"),
        (
            "render_template_string(",
            "XSS",
            "CWE-79",
            "Template injection risk",
        ),
    ]

    for line_num, line in enumerate(code.splitlines(), 1):
        for pattern, vuln_type, cwe, desc in patterns:
            if pattern in line:
                vulnerabilities.append(
                    VulnerabilityInfo(
                        type=vuln_type,
                        cwe=cwe,
                        severity="high" if "Injection" in vuln_type else "medium",
                        line=line_num,
                        description=desc,
                    )
                )

    # Detect taint sources
    source_patterns = ["request.args", "request.form", "input(", "sys.argv"]
    for pattern in source_patterns:
        if pattern in code:
            taint_sources.append(pattern)

    vuln_count = len(vulnerabilities)
    if vuln_count == 0:
        risk_level = "low"
    elif vuln_count <= 2:
        risk_level = "medium"
    else:
        risk_level = "high"

    return SecurityResult(
        success=True,
        has_vulnerabilities=vuln_count > 0,
        vulnerability_count=vuln_count,
        risk_level=risk_level,
        vulnerabilities=vulnerabilities,
        taint_sources=taint_sources,
    )


def _symbolic_execute_sync(code: str, max_paths: int = 10) -> SymbolicResult:
    """Synchronous implementation of symbolic_execute."""
    valid, error = _validate_code(code)
    if not valid:
        return SymbolicResult(
            success=False,
            paths_explored=0,
            error=error,
        )

    # Check cache first (symbolic execution is expensive!)
    cache = _get_cache()
    cache_config = {"max_paths": max_paths}
    if cache:
        cached = cache.get(code, "symbolic", cache_config)
        if cached is not None:
            logger.debug("Cache hit for symbolic_execute")
            if isinstance(cached, dict):
                # Reconstruct ExecutionPath objects
                if "paths" in cached:
                    cached["paths"] = [ExecutionPath(**p) for p in cached["paths"]]
                return SymbolicResult(**cached)
            return cached

    try:
        # Import here to avoid circular imports
        from code_scalpel.symbolic_execution_tools import SymbolicAnalyzer
        from code_scalpel.symbolic_execution_tools.engine import PathStatus

        analyzer = SymbolicAnalyzer(max_loop_iterations=max_paths)
        result = analyzer.analyze(code)

        paths = []
        all_constraints = []
        for i, path in enumerate(result.paths):
            # PathResult has: path_id, status, constraints, variables, model
            # Convert Z3 constraints to string conditions
            conditions = [str(c) for c in path.constraints] if path.constraints else []
            all_constraints.extend(conditions)

            paths.append(
                ExecutionPath(
                    path_id=path.path_id,
                    conditions=conditions,
                    final_state=path.variables or {},
                    reproduction_input=path.model or {},
                    is_reachable=path.status == PathStatus.FEASIBLE,
                )
            )

        # If symbolic execution didn't find variables or constraints,
        # supplement with AST-based analysis
        symbolic_vars = (
            list(result.all_variables.keys()) if result.all_variables else []
        )
        constraints_list = list(set(all_constraints))

        if not symbolic_vars or not constraints_list:
            basic = _basic_symbolic_analysis(code)
            if not symbolic_vars and basic.symbolic_variables:
                symbolic_vars = basic.symbolic_variables
            if not constraints_list and basic.constraints:
                constraints_list = basic.constraints
            # Also use basic paths if symbolic found nothing
            if not paths and basic.paths:
                paths = basic.paths

        symbolic_result = SymbolicResult(
            success=True,
            paths_explored=len(paths) if paths else result.total_paths,
            paths=paths,
            symbolic_variables=symbolic_vars,
            constraints=constraints_list,
        )

        # Cache successful result
        if cache:
            cache.set(code, "symbolic", symbolic_result.model_dump(), cache_config)

        return symbolic_result

    except ImportError:
        # Fallback to basic path analysis
        return _basic_symbolic_analysis(code)
    except Exception as e:
        # If symbolic execution fails (e.g., unsupported AST nodes like f-strings),
        # fall back to basic AST-based analysis instead of returning an error
        logger.warning(f"Symbolic execution failed, using basic analysis: {e}")
        return _basic_symbolic_analysis(code)


@mcp.tool()
async def symbolic_execute(code: str, max_paths: int = 10) -> SymbolicResult:
    """
    Perform symbolic execution on Python code.

    Use this tool to explore execution paths and find bugs that static analysis misses.
    It treats variables as symbolic values and uses a Z3 solver to find inputs that
    trigger specific paths.

    Args:
        code: Python source code to analyze
        max_paths: Maximum number of paths to explore (default: 10)

    Returns:
        Symbolic execution result with discovered paths, constraints, and reproduction inputs
    """
    return await asyncio.to_thread(_symbolic_execute_sync, code, max_paths)


def _basic_symbolic_analysis(code: str) -> SymbolicResult:
    """Fallback symbolic analysis using AST inspection."""
    try:
        tree = ast.parse(code)

        # Count branches
        branch_count = 0
        symbolic_vars = []
        conditions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                branch_count += 1
                conditions.append(ast.unparse(node.test))
            elif isinstance(node, ast.While):
                branch_count += 1
                conditions.append(f"while: {ast.unparse(node.test)}")
            elif isinstance(node, ast.For):
                branch_count += 1
                if isinstance(node.target, ast.Name):
                    symbolic_vars.append(node.target.id)
            elif isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    symbolic_vars.append(arg.arg)

        # Estimate paths (2^branches, capped)
        estimated_paths = min(2**branch_count, 10)

        paths = [
            ExecutionPath(
                path_id=i,
                conditions=conditions[: i + 1] if i < len(conditions) else conditions,
                final_state={},
                reproduction_input=None,
                is_reachable=True,
            )
            for i in range(estimated_paths)
        ]

        return SymbolicResult(
            success=True,
            paths_explored=estimated_paths,
            paths=paths,
            symbolic_variables=list(set(symbolic_vars)),
            constraints=conditions,
        )

    except Exception as e:
        return SymbolicResult(
            success=False,
            paths_explored=0,
            error=f"Basic analysis failed: {str(e)}",
        )


# ============================================================================
# TEST GENERATION
# ============================================================================


def _generate_tests_sync(
    code: str, function_name: str | None = None, framework: str = "pytest"
) -> TestGenerationResult:
    """Synchronous implementation of generate_unit_tests."""
    valid, error = _validate_code(code)
    if not valid:
        return TestGenerationResult(
            success=False,
            function_name=function_name or "unknown",
            test_count=0,
            error=error,
        )

    try:
        from code_scalpel.generators import TestGenerator

        generator = TestGenerator(framework=framework)
        result = generator.generate(code, function_name=function_name)

        test_cases = [
            GeneratedTestCase(
                path_id=tc.path_id,
                function_name=tc.function_name,
                inputs=tc.inputs,
                description=tc.description,
                path_conditions=tc.path_conditions,
            )
            for tc in result.test_cases
        ]

        return TestGenerationResult(
            success=True,
            function_name=result.function_name,
            test_count=len(test_cases),
            test_cases=test_cases,
            pytest_code=result.pytest_code,
            unittest_code=result.unittest_code,
        )

    except Exception as e:
        return TestGenerationResult(
            success=False,
            function_name=function_name or "unknown",
            test_count=0,
            error=f"Test generation failed: {str(e)}",
        )


@mcp.tool()
async def generate_unit_tests(
    code: str, function_name: str | None = None, framework: str = "pytest"
) -> TestGenerationResult:
    """
    Generate unit tests from code using symbolic execution.

    Use this tool to automatically create test cases that cover all execution paths
    in a function. Each test case includes concrete input values that trigger a
    specific path through the code.

    Args:
        code: Source code containing the function to test
        function_name: Name of function to generate tests for (auto-detected if None)
        framework: Test framework ("pytest" or "unittest")

    Returns:
        Test generation result with generated test code and test cases
    """
    return await asyncio.to_thread(_generate_tests_sync, code, function_name, framework)


# ============================================================================
# REFACTOR SIMULATION
# ============================================================================


def _simulate_refactor_sync(
    original_code: str,
    new_code: str | None = None,
    patch: str | None = None,
    strict_mode: bool = False,
) -> RefactorSimulationResult:
    """Synchronous implementation of simulate_refactor."""
    valid, error = _validate_code(original_code)
    if not valid:
        return RefactorSimulationResult(
            success=False,
            is_safe=False,
            status="error",
            error=f"Invalid original code: {error}",
        )

    if new_code is None and patch is None:
        return RefactorSimulationResult(
            success=False,
            is_safe=False,
            status="error",
            error="Must provide either 'new_code' or 'patch'",
        )

    try:
        from code_scalpel.generators import RefactorSimulator

        simulator = RefactorSimulator(strict_mode=strict_mode)
        result = simulator.simulate(
            original_code=original_code,
            new_code=new_code,
            patch=patch,
        )

        security_issues = [
            RefactorSecurityIssue(
                type=issue.type,
                severity=issue.severity,
                line=issue.line,
                description=issue.description,
                cwe=issue.cwe,
            )
            for issue in result.security_issues
        ]

        return RefactorSimulationResult(
            success=True,
            is_safe=result.is_safe,
            status=result.status.value,
            reason=result.reason,
            security_issues=security_issues,
            structural_changes=result.structural_changes,
            warnings=result.warnings,
        )

    except Exception as e:
        return RefactorSimulationResult(
            success=False,
            is_safe=False,
            status="error",
            error=f"Simulation failed: {str(e)}",
        )


@mcp.tool()
async def simulate_refactor(
    original_code: str,
    new_code: str | None = None,
    patch: str | None = None,
    strict_mode: bool = False,
) -> RefactorSimulationResult:
    """
    Simulate applying a code change and check for safety issues.

    Use this tool before applying AI-generated code changes to verify they don't
    introduce security vulnerabilities or break existing functionality.

    Provide either the new_code directly OR a unified diff patch.

    Args:
        original_code: The original source code
        new_code: The modified code to compare against (optional)
        patch: A unified diff patch to apply (optional)
        strict_mode: If True, treat warnings as unsafe

    Returns:
        Simulation result with safety verdict and any issues found
    """
    return await asyncio.to_thread(
        _simulate_refactor_sync, original_code, new_code, patch, strict_mode
    )


def _crawl_project_sync(
    root_path: str,
    exclude_dirs: list[str] | None = None,
    complexity_threshold: int = 10,
    include_report: bool = True,
) -> ProjectCrawlResult:
    """Synchronous implementation of crawl_project."""
    try:
        from code_scalpel.project_crawler import ProjectCrawler

        crawler = ProjectCrawler(
            root_path,
            exclude_dirs=frozenset(exclude_dirs) if exclude_dirs else None,
            complexity_threshold=complexity_threshold,
        )
        result = crawler.crawl()

        # Convert to Pydantic models
        def to_func_info(f) -> CrawlFunctionInfo:
            return CrawlFunctionInfo(
                name=f.qualified_name,
                lineno=f.lineno,
                complexity=f.complexity,
            )

        def to_class_info(c) -> CrawlClassInfo:
            return CrawlClassInfo(
                name=c.name,
                lineno=c.lineno,
                methods=[to_func_info(m) for m in c.methods],
                bases=c.bases,
            )

        def to_file_result(fr, root: str) -> CrawlFileResult:
            import os

            return CrawlFileResult(
                path=os.path.relpath(fr.path, root),
                status=fr.status,
                lines_of_code=fr.lines_of_code,
                functions=[to_func_info(f) for f in fr.functions],
                classes=[to_class_info(c) for c in fr.classes],
                imports=fr.imports,
                complexity_warnings=[to_func_info(f) for f in fr.complexity_warnings],
                error=fr.error,
            )

        summary = CrawlSummary(
            total_files=result.total_files,
            successful_files=len(result.files_analyzed),
            failed_files=len(result.files_with_errors),
            total_lines_of_code=result.total_lines_of_code,
            total_functions=result.total_functions,
            total_classes=result.total_classes,
            complexity_warnings=len(result.all_complexity_warnings),
        )

        files = [to_file_result(f, result.root_path) for f in result.files_analyzed]
        errors = [to_file_result(f, result.root_path) for f in result.files_with_errors]

        report = ""
        if include_report:
            report = crawler.generate_report(result)

        return ProjectCrawlResult(
            success=True,
            root_path=result.root_path,
            timestamp=result.timestamp,
            summary=summary,
            files=files,
            errors=errors,
            markdown_report=report,
        )

    except Exception as e:
        return ProjectCrawlResult(
            success=False,
            root_path=root_path,
            timestamp="",
            summary=CrawlSummary(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_lines_of_code=0,
                total_functions=0,
                total_classes=0,
                complexity_warnings=0,
            ),
            error=f"Crawl failed: {str(e)}",
        )


# --- Helper functions for extract_code (refactored for maintainability) ---


def _extraction_error(target_name: str, error: str) -> ContextualExtractionResult:
    """Create a standardized error result for extraction failures."""
    return ContextualExtractionResult(
        success=False,
        target_name=target_name,
        target_code="",
        context_code="",
        full_code="",
        error=error,
    )


def _create_extractor(
    file_path: str | None, code: str | None, target_name: str
) -> tuple["SurgicalExtractor | None", ContextualExtractionResult | None]:
    """
    Create a SurgicalExtractor from file_path or code.

    Returns (extractor, None) on success, (None, error_result) on failure.
    """
    from code_scalpel import SurgicalExtractor

    if file_path is None and code is None:
        return None, _extraction_error(
            target_name, "Must provide either 'file_path' or 'code' argument"
        )

    if file_path is not None:
        try:
            return SurgicalExtractor.from_file(file_path), None
        except FileNotFoundError:
            return None, _extraction_error(target_name, f"File not found: {file_path}")
        except ValueError as e:
            return None, _extraction_error(target_name, str(e))
    else:
        try:
            return SurgicalExtractor(code), None
        except (SyntaxError, ValueError) as e:
            return None, _extraction_error(
                target_name, f"Syntax error in code: {str(e)}"
            )


def _extract_method(extractor: "SurgicalExtractor", target_name: str):
    """Extract a method, handling the ClassName.method_name parsing."""
    if "." not in target_name:
        return None, _extraction_error(
            target_name, "Method name must be 'ClassName.method_name' format"
        )
    class_name, method_name = target_name.rsplit(".", 1)
    return extractor.get_method(class_name, method_name), None


def _perform_extraction(
    extractor: "SurgicalExtractor",
    target_type: str,
    target_name: str,
    include_context: bool,
    include_cross_file_deps: bool,
    context_depth: int,
    file_path: str | None,
):
    """
    Perform the actual extraction based on target type and options.

    Returns (result, cross_file_result, error_result).
    """
    from code_scalpel.surgical_extractor import CrossFileResolution

    cross_file_result: CrossFileResolution | None = None

    # CROSS-FILE RESOLUTION PATH
    if include_cross_file_deps and file_path is not None:
        if target_type in ("function", "class"):
            cross_file_result = extractor.resolve_cross_file_dependencies(
                target_name=target_name,
                target_type=target_type,
                max_depth=context_depth,
            )
            return cross_file_result.target, cross_file_result, None
        else:
            # Method - fall back to regular extraction
            result, error = _extract_method(extractor, target_name)
            return result, None, error

    # INTRA-FILE CONTEXT PATH
    if target_type == "function":
        if include_context:
            return (
                extractor.get_function_with_context(
                    target_name, max_depth=context_depth
                ),
                None,
                None,
            )
        return extractor.get_function(target_name), None, None

    if target_type == "class":
        if include_context:
            return (
                extractor.get_class_with_context(target_name, max_depth=context_depth),
                None,
                None,
            )
        return extractor.get_class(target_name), None, None

    if target_type == "method":
        result, error = _extract_method(extractor, target_name)
        return result, None, error

    return (
        None,
        None,
        _extraction_error(
            target_name,
            f"Unknown target_type: {target_type}. Use 'function', 'class', or 'method'.",
        ),
    )


def _process_cross_file_context(cross_file_result) -> tuple[str, list[str]]:
    """Process cross-file resolution results into context_code and context_items."""
    if cross_file_result is None or not cross_file_result.external_symbols:
        return "", []

    external_parts = []
    external_names = []
    for sym in cross_file_result.external_symbols:
        external_parts.append(f"# From {sym.source_file}")
        external_parts.append(sym.code)
        external_names.append(f"{sym.name} ({sym.source_file})")

    context_code = "\n\n".join(external_parts)

    # Add unresolved imports as a comment
    if cross_file_result.unresolved_imports:
        unresolved_comment = "# Unresolved imports: " + ", ".join(
            cross_file_result.unresolved_imports
        )
        context_code = unresolved_comment + "\n\n" + context_code

    return context_code, external_names


def _build_full_code(
    imports_needed: list[str], context_code: str, target_code: str
) -> str:
    """Build the combined full_code for LLM consumption."""
    parts = []
    if imports_needed:
        parts.append("\n".join(imports_needed))
    if context_code:
        parts.append(context_code)
    parts.append(target_code)
    return "\n\n".join(parts)


@mcp.tool()
async def extract_code(
    target_type: str,
    target_name: str,
    file_path: str | None = None,
    code: str | None = None,
    include_context: bool = False,
    context_depth: int = 1,
    include_cross_file_deps: bool = False,
    include_token_estimate: bool = True,
) -> ContextualExtractionResult:
    """
    Surgically extract specific code elements (functions, classes, methods).

    **TOKEN-EFFICIENT MODE (RECOMMENDED):**
    Provide `file_path` - the server reads the file directly. The Agent
    never sees the full file content, saving potentially thousands of tokens.

    **CROSS-FILE DEPENDENCIES:**
    Set `include_cross_file_deps=True` to automatically resolve imports.
    If your function uses `TaxRate` from `models.py`, this will extract
    `TaxRate` from `models.py` and include it in the response.

    **LEGACY MODE:**
    Provide `code` as a string - for when you already have code in context.

    Args:
        target_type: Type of element - "function", "class", or "method".
        target_name: Name of the element. For methods, use "ClassName.method_name".
        file_path: Path to the source file (TOKEN SAVER - server reads file).
        code: Source code string (fallback if file_path not provided).
        include_context: If True, also extract intra-file dependencies.
        context_depth: How deep to traverse dependencies (1=direct, 2=transitive).
        include_cross_file_deps: If True, resolve imports from external files.
        include_token_estimate: If True, include estimated token count.

    Returns:
        ContextualExtractionResult with extracted code and metadata.

    Example (Efficient - Agent sends ~50 tokens, receives ~200):
        extract_code(
            file_path="/project/src/utils.py",
            target_type="function",
            target_name="calculate_tax"
        )

    Example (With cross-file dependencies):
        extract_code(
            file_path="/project/src/services/order.py",
            target_type="function",
            target_name="process_order",
            include_cross_file_deps=True
        )
    """
    from code_scalpel.surgical_extractor import ContextualExtraction, ExtractionResult

    # Step 1: Create extractor
    extractor, error = _create_extractor(file_path, code, target_name)
    if error:
        return error

    try:
        # Step 2: Perform extraction
        result, cross_file_result, error = _perform_extraction(
            extractor,
            target_type,
            target_name,
            include_context,
            include_cross_file_deps,
            context_depth,
            file_path,
        )
        if error:
            return error

        # Step 3: Handle None result
        if result is None:
            return _extraction_error(
                target_name,
                f"{target_type.capitalize()} '{target_name}' not found in code",
            )

        # Step 4: Process result based on type
        if isinstance(result, ExtractionResult):
            if not result.success:
                return _extraction_error(
                    target_name,
                    result.error
                    or f"{target_type.capitalize()} '{target_name}' not found",
                )
            target_code = result.code
            total_lines = (
                result.line_end - result.line_start + 1 if result.line_end > 0 else 0
            )
            line_start = result.line_start
            line_end = result.line_end
            imports_needed = result.imports_needed

            # Handle cross-file context
            context_code, context_items = _process_cross_file_context(cross_file_result)

        elif isinstance(result, ContextualExtraction):
            if not result.target.success:
                return _extraction_error(
                    target_name,
                    result.target.error
                    or f"{target_type.capitalize()} '{target_name}' not found",
                )
            target_code = result.target.code
            context_items = result.context_items
            context_code = result.context_code
            total_lines = result.total_lines
            line_start = result.target.line_start
            line_end = result.target.line_end
            imports_needed = result.target.imports_needed
        else:
            return _extraction_error(
                target_name, f"Unexpected result type: {type(result).__name__}"
            )

        # Step 5: Build final response
        full_code = _build_full_code(imports_needed, context_code, target_code)
        token_estimate = len(full_code) // 4 if include_token_estimate else 0

        return ContextualExtractionResult(
            success=True,
            target_name=target_name,
            target_code=target_code,
            context_code=context_code,
            full_code=full_code,
            context_items=context_items,
            total_lines=total_lines,
            line_start=line_start,
            line_end=line_end,
            token_estimate=token_estimate,
        )

    except Exception as e:
        return _extraction_error(target_name, f"Extraction failed: {str(e)}")


@mcp.tool()
async def update_symbol(
    file_path: str,
    target_type: str,
    target_name: str,
    new_code: str,
    create_backup: bool = True,
) -> PatchResultModel:
    """
        Surgically replace a function, class, or method in a file with new code.

        This is the SAFE way to modify code - you provide only the new symbol,
        and the server handles:
        - Locating the exact symbol boundaries (including decorators)
        - Validating the replacement code syntax
        - Preserving all surrounding code exactly
        - Creating a backup before modification
        - Atomic write (prevents partial writes)

        Args:
            file_path: Path to the Python source file to modify.
            target_type: Type of element - "function", "class", or "method".
            target_name: Name of the element. For methods, use "ClassName.method_name".
            new_code: The complete new definition (including def/class line and body).
            create_backup: If True (default), create a .bak file before modifying.

        Returns:
            PatchResultModel with success status, line changes, and backup path.

        Example (Fix a function):
            update_symbol(
                file_path="/project/src/utils.py",
                target_type="function",
                target_name="calculate_tax",
                new_code='''def calculate_tax(amount, rate=0.1):
        \"\"\"Calculate tax with proper rounding.\"\"\"
        return round(amount * rate, 2)
    '''
            )

        Example (Update a method):
            update_symbol(
                file_path="/project/src/models.py",
                target_type="method",
                target_name="User.validate_email",
                new_code='''def validate_email(self, email):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    '''
            )

        Safety Features:
            - Backup created at {file_path}.bak (unless create_backup=False)
            - Syntax validation before any file modification
            - Atomic write prevents corruption on crash
            - Original indentation preserved
    """
    from code_scalpel.surgical_patcher import SurgicalPatcher

    # Validate inputs
    if not file_path:
        return PatchResultModel(
            success=False,
            file_path="",
            target_name=target_name,
            target_type=target_type,
            error="file_path is required",
        )

    if not new_code or not new_code.strip():
        return PatchResultModel(
            success=False,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            error="new_code cannot be empty",
        )

    if target_type not in ("function", "class", "method"):
        return PatchResultModel(
            success=False,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            error=f"Invalid target_type: {target_type}. Use 'function', 'class', or 'method'.",
        )

    # Load the file
    try:
        patcher = SurgicalPatcher.from_file(file_path)
    except FileNotFoundError:
        return PatchResultModel(
            success=False,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            error=f"File not found: {file_path}",
        )
    except ValueError as e:
        return PatchResultModel(
            success=False,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            error=str(e),
        )

    # Apply the patch based on target type
    try:
        if target_type == "function":
            result = patcher.update_function(target_name, new_code)
        elif target_type == "class":
            result = patcher.update_class(target_name, new_code)
        elif target_type == "method":
            if "." not in target_name:
                return PatchResultModel(
                    success=False,
                    file_path=file_path,
                    target_name=target_name,
                    target_type=target_type,
                    error="Method name must be 'ClassName.method_name' format",
                )
            class_name, method_name = target_name.rsplit(".", 1)
            result = patcher.update_method(class_name, method_name, new_code)
        else:
            # Should not reach here due to validation above
            return PatchResultModel(
                success=False,
                file_path=file_path,
                target_name=target_name,
                target_type=target_type,
                error=f"Unknown target_type: {target_type}",
            )

        if not result.success:
            return PatchResultModel(
                success=False,
                file_path=file_path,
                target_name=target_name,
                target_type=target_type,
                error=result.error,
            )

        # Save the changes
        backup_path = patcher.save(backup=create_backup)

        return PatchResultModel(
            success=True,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            lines_before=result.lines_before,
            lines_after=result.lines_after,
            lines_delta=result.lines_delta,
            backup_path=backup_path,
        )

    except Exception as e:
        return PatchResultModel(
            success=False,
            file_path=file_path,
            target_name=target_name,
            target_type=target_type,
            error=f"Patch failed: {str(e)}",
        )


@mcp.tool()
async def crawl_project(
    root_path: str | None = None,
    exclude_dirs: list[str] | None = None,
    complexity_threshold: int = 10,
    include_report: bool = True,
) -> ProjectCrawlResult:
    """
    Crawl an entire project directory and analyze all Python files.

    Use this tool to get a comprehensive overview of a project's structure,
    complexity hotspots, and code metrics before diving into specific files.

    Args:
        root_path: Path to project root (defaults to current working directory)
        exclude_dirs: Additional directories to exclude (common ones already excluded)
        complexity_threshold: Complexity score that triggers a warning (default: 10)
        include_report: Include a markdown report in the response (default: True)

    Returns:
        Project crawl result with file analysis, summary statistics, and optional report
    """
    if root_path is None:
        root_path = str(PROJECT_ROOT)

    return await asyncio.to_thread(
        _crawl_project_sync,
        root_path,
        exclude_dirs,
        complexity_threshold,
        include_report,
    )


# ============================================================================
# RESOURCES
# ============================================================================


@mcp.resource("scalpel://project/call-graph")
def get_project_call_graph() -> str:
    """
    Get the project-wide call graph.

    Returns a JSON adjacency list:
    {
        "file.py:caller_function": ["target_function", "other_file.py:target_function"]
    }

    Use this to trace function calls across files and understand dependencies.
    """
    import json
    from code_scalpel.ast_tools.call_graph import CallGraphBuilder

    builder = CallGraphBuilder(PROJECT_ROOT)
    graph = builder.build()
    return json.dumps(graph, indent=2)


@mcp.resource("scalpel://project/dependencies")
def get_project_dependencies() -> str:
    """
    Returns a list of project dependencies detected in configuration files.
    Use this to verify if libraries used in generated code actually exist in the project.
    """
    import json
    from code_scalpel.ast_tools.dependency_parser import DependencyParser

    parser = DependencyParser(str(PROJECT_ROOT))
    deps = parser.get_dependencies()
    return json.dumps(deps, indent=2)


@mcp.resource("scalpel://project/structure")
def get_project_structure() -> str:
    """
    Get the project directory structure as a JSON tree.

    Use this resource to understand the file layout of the project.
    It respects .gitignore if possible (simple implementation for now).
    """

    def build_tree(path: Path) -> dict[str, Any]:
        tree = {"name": path.name, "type": "directory", "children": []}
        try:
            # Sort directories first, then files
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for item in items:
                # Skip hidden files/dirs and common ignore patterns
                if item.name.startswith(".") or item.name in [
                    "__pycache__",
                    "venv",
                    "node_modules",
                    "dist",
                    "build",
                ]:
                    continue

                if item.is_dir():
                    tree["children"].append(build_tree(item))
                else:
                    tree["children"].append({"name": item.name, "type": "file"})
        except PermissionError:
            pass
        return tree

    import json

    return json.dumps(build_tree(PROJECT_ROOT), indent=2)


@mcp.resource("scalpel://version")
def get_version() -> str:
    """Get Code Scalpel version information."""
    return f"""Code Scalpel v{__version__}

A precision toolkit for AI-driven code analysis.

Features:
- AST Analysis: Parse and analyze code structure
- Security Scanning: Taint-based vulnerability detection
- Symbolic Execution: Path exploration with Z3 solver

Supported Languages:
- Python (full support)
- JavaScript/TypeScript (planned v0.4.0)
"""


@mcp.resource("scalpel://capabilities")
def get_capabilities() -> str:
    """Get information about Code Scalpel's capabilities."""
    return """# Code Scalpel Capabilities

## Tools

### analyze_code
Parses Python code and extracts:
- Function definitions (sync and async)
- Class definitions
- Import statements
- Cyclomatic complexity
- Lines of code

### security_scan
Detects vulnerabilities:
- SQL Injection (CWE-89)
- Cross-Site Scripting (CWE-79)
- Command Injection (CWE-78)
- Path Traversal (CWE-22)
- Code Injection (CWE-94)

Uses taint analysis to track data flow from sources to sinks.

### symbolic_execute
Explores execution paths:
- Treats function arguments as symbolic
- Uses Z3 SMT solver for constraint solving
- Identifies reachable/unreachable paths
- Reports path conditions

## Security Notes
- Code is PARSED, never executed
- Maximum code size: 100KB
- No filesystem access from analyzed code
- No network access from analyzed code
"""


# ============================================================================
# PROMPTS
# ============================================================================


@mcp.prompt(title="Code Review")
def code_review_prompt(code: str) -> str:
    """Generate a comprehensive code review prompt."""
    return f"""Please analyze the following Python code and provide:

1. **Structure Analysis**: Identify functions, classes, and imports
2. **Security Review**: Check for potential vulnerabilities
3. **Quality Assessment**: Evaluate code quality and suggest improvements
4. **Edge Cases**: Identify potential edge cases and error conditions

Use the available Code Scalpel tools to gather detailed analysis:
- analyze_code: For structure and complexity
- security_scan: For vulnerability detection
- symbolic_execute: For path analysis

Code to review:
```python
{code}
```

Provide actionable recommendations for improvement."""


@mcp.prompt(title="Security Audit")
def security_audit_prompt(code: str) -> str:
    """Generate a security-focused audit prompt."""
    return f"""Perform a security audit of the following Python code.

Focus on:
1. **Input Validation**: Are all inputs properly validated?
2. **Injection Risks**: SQL, command, code injection vulnerabilities
3. **Authentication/Authorization**: Proper access controls
4. **Data Exposure**: Sensitive data handling
5. **Dependencies**: Known vulnerable patterns

Use security_scan tool to detect vulnerabilities automatically.

Code to audit:
```python
{code}
```

Provide a risk assessment and remediation steps for each finding."""


# ============================================================================
# v1.4.0 MCP TOOLS - Enhanced AI Context
# ============================================================================


def _get_file_context_sync(file_path: str) -> FileContextResult:
    """Synchronous implementation of get_file_context."""
    try:
        path = Path(file_path)
        
        # Try to resolve the path
        if not path.is_absolute():
            # Try relative to PROJECT_ROOT
            candidate = PROJECT_ROOT / path
            if candidate.exists():
                path = candidate
            else:
                # Try current working directory
                candidate = Path.cwd() / path
                if candidate.exists():
                    path = candidate
        
        if not path.exists():
            return FileContextResult(
                success=False,
                file_path=file_path,
                line_count=0,
                error=f"File not found: {file_path}",
            )
        
        code = path.read_text(encoding="utf-8")
        lines = code.splitlines()
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return FileContextResult(
                success=False,
                file_path=str(path),
                line_count=len(lines),
                error=f"Syntax error at line {e.lineno}: {e.msg}",
            )
        
        functions = []
        classes = []
        imports = []
        exports = []
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Only top-level functions
                if hasattr(node, 'col_offset') and node.col_offset == 0:
                    functions.append(node.name)
                    complexity += _count_complexity_node(node)
            elif isinstance(node, ast.ClassDef):
                if hasattr(node, 'col_offset') and node.col_offset == 0:
                    classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.Assign):
                # Check for __all__ exports
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List | ast.Tuple):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports.append(elt.value)
        
        # Quick security check
        has_security_issues = False
        security_patterns = ["eval(", "exec(", "cursor.execute", "os.system(", "subprocess.call("]
        for pattern in security_patterns:
            if pattern in code:
                has_security_issues = True
                break
        
        # Generate summary based on content
        summary_parts = []
        if classes:
            summary_parts.append(f"{len(classes)} class(es)")
        if functions:
            summary_parts.append(f"{len(functions)} function(s)")
        if "flask" in code.lower() or "app.route" in code:
            summary_parts.append("Flask web application")
        elif "django" in code.lower():
            summary_parts.append("Django module")
        elif "test_" in path.name or "pytest" in code:
            summary_parts.append("Test module")
        
        summary = ", ".join(summary_parts) if summary_parts else "Python module"
        
        return FileContextResult(
            success=True,
            file_path=str(path),
            language="python",
            line_count=len(lines),
            functions=functions,
            classes=classes,
            imports=imports[:20],  # Limit to avoid token bloat
            exports=exports,
            complexity_score=complexity,
            has_security_issues=has_security_issues,
            summary=summary,
        )
        
    except Exception as e:
        return FileContextResult(
            success=False,
            file_path=file_path,
            line_count=0,
            error=f"Analysis failed: {str(e)}",
        )


def _count_complexity_node(node: ast.AST) -> int:
    """Count cyclomatic complexity for a single node."""
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


@mcp.tool()
async def get_file_context(file_path: str) -> FileContextResult:
    """
    Get a file overview without reading full content.
    
    [v1.4.0] Use this tool to quickly assess if a file is relevant to your task
    without consuming tokens on full content. Returns functions, classes, imports,
    complexity score, and security warnings.
    
    Why AI agents need this:
    - Quickly assess file relevance before extracting code
    - Understand file structure without token overhead
    - Make informed decisions about which functions to modify
    
    Args:
        file_path: Path to the Python file (absolute or relative to project root)
    
    Returns:
        FileContextResult with file overview and metadata
    """
    return await asyncio.to_thread(_get_file_context_sync, file_path)


def _get_symbol_references_sync(symbol_name: str, project_root: str | None = None) -> SymbolReferencesResult:
    """Synchronous implementation of get_symbol_references."""
    try:
        root = Path(project_root) if project_root else PROJECT_ROOT
        
        if not root.exists():
            return SymbolReferencesResult(
                success=False,
                symbol_name=symbol_name,
                error=f"Project root not found: {root}",
            )
        
        references: list[SymbolReference] = []
        definition_file = None
        definition_line = None
        
        # Walk through all Python files
        for py_file in root.rglob("*.py"):
            # Skip common non-source directories
            if any(part.startswith('.') or part in ('__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build') 
                   for part in py_file.parts):
                continue
            
            try:
                code = py_file.read_text(encoding="utf-8")
                lines = code.splitlines()
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    # Check for function/class definitions
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        if node.name == symbol_name:
                            rel_path = str(py_file.relative_to(root))
                            if definition_file is None:
                                definition_file = rel_path
                                definition_line = node.lineno
                            
                            context = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                            references.append(SymbolReference(
                                file=rel_path,
                                line=node.lineno,
                                column=node.col_offset,
                                context=context.strip(),
                                is_definition=True,
                            ))
                    
                    # Check for function calls
                    elif isinstance(node, ast.Call):
                        func = node.func
                        name = None
                        if isinstance(func, ast.Name):
                            name = func.id
                        elif isinstance(func, ast.Attribute):
                            name = func.attr
                        
                        if name == symbol_name:
                            rel_path = str(py_file.relative_to(root))
                            line_no = getattr(node, 'lineno', 0)
                            context = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                            references.append(SymbolReference(
                                file=rel_path,
                                line=line_no,
                                column=getattr(node, 'col_offset', 0),
                                context=context.strip(),
                                is_definition=False,
                            ))
                    
                    # Check for name references
                    elif isinstance(node, ast.Name) and node.id == symbol_name:
                        rel_path = str(py_file.relative_to(root))
                        line_no = getattr(node, 'lineno', 0)
                        context = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                        # Avoid duplicates from Call nodes
                        if not any(r.file == rel_path and r.line == line_no for r in references):
                            references.append(SymbolReference(
                                file=rel_path,
                                line=line_no,
                                column=getattr(node, 'col_offset', 0),
                                context=context.strip(),
                                is_definition=False,
                            ))
            
            except (SyntaxError, UnicodeDecodeError):
                # Skip files that can't be parsed
                continue
        
        # Remove duplicates and sort
        seen = set()
        unique_refs = []
        for ref in references:
            key = (ref.file, ref.line, ref.is_definition)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        unique_refs.sort(key=lambda r: (not r.is_definition, r.file, r.line))
        
        return SymbolReferencesResult(
            success=True,
            symbol_name=symbol_name,
            definition_file=definition_file,
            definition_line=definition_line,
            references=unique_refs[:100],  # Limit to prevent token overflow
            total_references=len(unique_refs),
        )
        
    except Exception as e:
        return SymbolReferencesResult(
            success=False,
            symbol_name=symbol_name,
            error=f"Search failed: {str(e)}",
        )


@mcp.tool()
async def get_symbol_references(
    symbol_name: str,
    project_root: str | None = None,
) -> SymbolReferencesResult:
    """
    Find all references to a symbol across the project.
    
    [v1.4.0] Use this tool before modifying a function, class, or variable to
    understand its usage across the codebase. Essential for safe refactoring.
    
    Why AI agents need this:
    - Safe refactoring: know all call sites before changing signatures
    - Impact analysis: understand blast radius of changes
    - No hallucination: real references, not guessed ones
    
    Args:
        symbol_name: Name of the function, class, or variable to search for
        project_root: Project root directory (default: server's project root)
    
    Returns:
        SymbolReferencesResult with definition location and all references
    """
    return await asyncio.to_thread(_get_symbol_references_sync, symbol_name, project_root)


# ============================================================================
# ENTRYPOINT
# ============================================================================


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
    allow_lan: bool = False,
    root_path: str | None = None,
):
    """
    Run the Code Scalpel MCP server.

    Args:
        transport: Transport type - "stdio" or "streamable-http"
        host: Host to bind to (HTTP only)
        port: Port to bind to (HTTP only)
        allow_lan: Allow connections from LAN (disables host validation)
        root_path: Project root directory (default: current directory)

    Security Note:
        By default, the HTTP transport only allows connections from localhost.
        Use --allow-lan to enable LAN access. This disables DNS rebinding
        protection and allows connections from any host. Only use on trusted
        networks.
    """
    global PROJECT_ROOT
    if root_path:
        PROJECT_ROOT = Path(root_path).resolve()
        if not PROJECT_ROOT.exists():
            print(
                f"Warning: Root path {PROJECT_ROOT} does not exist. Using current directory."
            )
            PROJECT_ROOT = Path.cwd()

    print(f"Code Scalpel MCP Server v{__version__}")
    print(f"Project Root: {PROJECT_ROOT}")

    if transport == "streamable-http" or transport == "sse":
        from mcp.server.transport_security import TransportSecuritySettings

        mcp.settings.host = host
        mcp.settings.port = port

        if allow_lan or host == "0.0.0.0":
            # Disable host validation for LAN access
            # WARNING: Only use on trusted networks!
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
                allowed_hosts=["*"],
                allowed_origins=["*"],
            )
            print("WARNING: LAN access enabled. Host validation disabled.")
            print("Only use on trusted networks!")

        mcp.run(transport=transport)
    else:
        mcp.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Scalpel MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (HTTP only, default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (HTTP only, default: 8080)",
    )
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="Allow LAN connections (disables host validation, use on trusted networks only)",
    )
    parser.add_argument(
        "--root",
        help="Project root directory for resources (default: current directory)",
    )

    args = parser.parse_args()
    run_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
        allow_lan=args.allow_lan,
        root_path=args.root,
    )
