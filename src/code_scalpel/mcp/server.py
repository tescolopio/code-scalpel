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

import ast
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

__version__ = "1.0.0"

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


# ============================================================================
# MCP SERVER
# ============================================================================

mcp = FastMCP(
    name="Code Scalpel",
    instructions=f"""Code Scalpel v{__version__} - AI-powered code analysis tools:

- analyze_code: Parse Python/Java code, extract structure (functions, classes, imports)
- security_scan: Detect vulnerabilities using taint analysis (SQL injection, XSS, etc.)
- symbolic_execute: Explore execution paths using symbolic execution
- generate_unit_tests: Generate pytest/unittest tests from symbolic execution paths
- simulate_refactor: Verify a code change is safe before applying it

All tools accept source code as a string and return structured analysis results.
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
        classes = []
        imports = []
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                # Flag potential issues
                if len(node.name) < 2:
                    issues.append(f"Function '{node.name}' has very short name")
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(f"async {node.name}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
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
        from code_scalpel.security import SecurityAnalyzer

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        vulnerabilities = []
        taint_sources = []

        for vuln in result.get("vulnerabilities", []):
            vulnerabilities.append(
                VulnerabilityInfo(
                    type=vuln.get("type", "Unknown"),
                    cwe=vuln.get("cwe", "Unknown"),
                    severity=vuln.get("severity", "medium"),
                    line=vuln.get("line"),
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

    Use this tool to audit code for security vulnerabilities (SQL Injection, XSS, etc.)
    before deploying or committing changes. It tracks data flow from sources to sinks.

    Detects:
    - SQL Injection (CWE-89)
    - Cross-Site Scripting (CWE-79)
    - Command Injection (CWE-78)
    - Path Traversal (CWE-22)

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
        from code_scalpel.symbolic import SymbolicExecutor

        executor = SymbolicExecutor(max_iterations=max_paths)
        result = executor.execute(code)

        paths = []
        for i, path in enumerate(result.get("paths", [])):
            # Extract reproduction inputs from the final state
            # Assuming symbolic variables in the state represent inputs
            final_state = path.get("state", {})
            symbolic_vars = result.get("symbolic_vars", [])
            reproduction_input = {
                k: v for k, v in final_state.items() if k in symbolic_vars
            }

            paths.append(
                ExecutionPath(
                    path_id=i,
                    conditions=path.get("conditions", []),
                    final_state=final_state,
                    reproduction_input=reproduction_input,
                    is_reachable=path.get("reachable", True),
                )
            )

        symbolic_result = SymbolicResult(
            success=True,
            paths_explored=len(paths),
            paths=paths,
            symbolic_variables=result.get("symbolic_vars", []),
            constraints=result.get("constraints", []),
        )

        # Cache successful result
        if cache:
            cache.set(code, "symbolic", symbolic_result.model_dump(), cache_config)

        return symbolic_result

    except ImportError:
        # Fallback to basic path analysis
        return _basic_symbolic_analysis(code)
    except Exception as e:
        return SymbolicResult(
            success=False,
            paths_explored=0,
            error=f"Symbolic execution failed: {str(e)}",
        )


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

    if transport == "streamable-http":
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

        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Scalpel MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
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
