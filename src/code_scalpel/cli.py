"""
Code Scalpel CLI - Command-line interface for code analysis.

Usage:
    code-scalpel analyze <file>           Analyze a Python file
    code-scalpel analyze --code "..."     Analyze code string
    code-scalpel scan <file>              Security vulnerability scan
    code-scalpel mcp                      Start MCP server (for AI clients)
    code-scalpel server [--port PORT]     Start REST API server (legacy)
    code-scalpel version                  Show version
"""

import argparse
import json
import sys
from pathlib import Path


def analyze_file(filepath: str, output_format: str = "text") -> int:
    """Analyze a Python file and print results."""

    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1

    if path.suffix != ".py":
        print(f"Warning: File does not have .py extension: {filepath}", file=sys.stderr)

    try:
        code = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    return analyze_code(code, output_format, filepath)


def analyze_code(
    code: str, output_format: str = "text", source: str = "<string>"
) -> int:
    """Analyze code string and print results."""
    from .code_analyzer import AnalysisLevel, CodeAnalyzer

    analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)

    try:
        result = analyzer.analyze(code)
    except Exception as e:
        print(f"Error analyzing code: {e}", file=sys.stderr)
        return 1

    if output_format == "json":
        output = {
            "source": source,
            "success": not result.errors,
            "metrics": {
                "lines_of_code": result.metrics.lines_of_code,
                "num_functions": result.metrics.num_functions,
                "num_classes": result.metrics.num_classes,
                "cyclomatic_complexity": result.metrics.cyclomatic_complexity,
                "analysis_time_seconds": result.metrics.analysis_time_seconds,
            },
            "dead_code": [
                {
                    "name": dc.name,
                    "type": dc.code_type,
                    "line_start": dc.line_start,
                    "line_end": dc.line_end,
                    "reason": dc.reason,
                }
                for dc in result.dead_code
            ],
            "security_issues": result.security_issues,
            "suggestions": [
                {
                    "type": s.refactor_type,
                    "description": s.description,
                    "priority": s.priority,
                }
                for s in result.refactor_suggestions
            ],
            "errors": result.errors,
        }
        print(json.dumps(output, indent=2))
    else:
        # Text format
        print(f"\nCode Scalpel Analysis: {source}")
        print("=" * 60)

        print("\nMetrics:")
        print(f"   Lines of code: {result.metrics.lines_of_code}")
        print(f"   Functions: {result.metrics.num_functions}")
        print(f"   Classes: {result.metrics.num_classes}")
        print(f"   Cyclomatic complexity: {result.metrics.cyclomatic_complexity}")
        print(f"   Analysis time: {result.metrics.analysis_time_seconds:.3f}s")

        if result.dead_code:
            print(f"\nDead Code Detected ({len(result.dead_code)} items):")
            for dc in result.dead_code:
                print(
                    f"   - {dc.code_type} '{dc.name}' (lines {dc.line_start}-{dc.line_end})"
                )
                print(f"     Reason: {dc.reason}")

        if result.security_issues:
            print(f"\n[WARNING] Security Issues ({len(result.security_issues)}):")
            for issue in result.security_issues:
                print(
                    f"   - {issue.get('type', 'Unknown')}: {issue.get('description', 'No description')}"
                )

        if result.refactor_suggestions:
            print(f"\nSuggestions ({len(result.refactor_suggestions)}):")
            for s in result.refactor_suggestions[:5]:  # Show top 5
                print(f"   - [{s.refactor_type}] {s.description}")

        if result.errors:
            print("\n[ERROR] Errors:")
            for err in result.errors:
                print(f"   - {err}")

        print()

    return 0 if not result.errors else 1


def scan_security(filepath: str, output_format: str = "text") -> int:
    """Scan a file for security vulnerabilities using taint analysis."""
    from .symbolic_execution_tools import analyze_security

    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1

    try:
        code = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    try:
        result = analyze_security(code)
    except Exception as e:
        print(f"Error during security analysis: {e}", file=sys.stderr)
        return 1

    if output_format == "json":
        output = {
            "source": str(filepath),
            "has_vulnerabilities": result.has_vulnerabilities,
            "vulnerability_count": result.vulnerability_count,
            "vulnerabilities": [
                {
                    "type": v.vulnerability_type,
                    "cwe": v.cwe_id,
                    "source": v.taint_source.name,
                    "sink": v.sink_type.name,
                    "line": v.sink_location[0] if v.sink_location else None,
                    "taint_path": v.taint_path,
                }
                for v in result.vulnerabilities
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nCode Scalpel Security Scan: {filepath}")
        print("=" * 60)

        if not result.has_vulnerabilities:
            print("\n[OK] No vulnerabilities detected.")
        else:
            print(f"\n[WARNING] Found {result.vulnerability_count} vulnerability(ies):\n")
            for i, v in enumerate(result.vulnerabilities, 1):
                print(f"  {i}. {v.vulnerability_type} ({v.cwe_id})")
                print(f"     Source: {v.taint_source.name}")
                print(f"     Sink: {v.sink_type.name}")
                if v.sink_location:
                    print(f"     Line: {v.sink_location[0]}")
                print(f"     Taint Path: {' -> '.join(v.taint_path)}")
                print()

        print(result.summary())

    return 0 if not result.has_vulnerabilities else 2


def scan_code_security(code: str, output_format: str = "text") -> int:
    """Scan code string for security vulnerabilities."""
    from .symbolic_execution_tools import analyze_security

    try:
        result = analyze_security(code)
    except Exception as e:
        print(f"Error during security analysis: {e}", file=sys.stderr)
        return 1

    if output_format == "json":
        output = {
            "source": "<string>",
            "has_vulnerabilities": result.has_vulnerabilities,
            "vulnerability_count": result.vulnerability_count,
            "vulnerabilities": [
                {
                    "type": v.vulnerability_type,
                    "cwe": v.cwe_id,
                    "source": v.taint_source.name,
                    "sink": v.sink_type.name,
                    "line": v.sink_location[0] if v.sink_location else None,
                    "taint_path": v.taint_path,
                }
                for v in result.vulnerabilities
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print("\nCode Scalpel Security Scan: <string>")
        print("=" * 60)

        if not result.has_vulnerabilities:
            print("\n[OK] No vulnerabilities detected.")
        else:
            print(f"\n[WARNING] Found {result.vulnerability_count} vulnerability(ies):\n")
            for i, v in enumerate(result.vulnerabilities, 1):
                print(f"  {i}. {v.vulnerability_type} ({v.cwe_id})")
                print(f"     Source: {v.taint_source.name}")
                print(f"     Sink: {v.sink_type.name}")
                if v.sink_location:
                    print(f"     Line: {v.sink_location[0]}")
                print(f"     Taint Path: {' -> '.join(v.taint_path)}")
                print()

        print(result.summary())

    return 0 if not result.has_vulnerabilities else 2


def start_server(host: str = "0.0.0.0", port: int = 5000) -> int:
    """Start the REST API server (legacy, for non-MCP clients)."""
    from .integrations.rest_api_server import run_server

    print(f"Starting Code Scalpel REST API Server on {host}:{port}")
    print(f"   Health check: http://{host}:{port}/health")
    print(f"   Analyze endpoint: POST http://{host}:{port}/analyze")
    print(f"   Refactor endpoint: POST http://{host}:{port}/refactor")
    print(f"   Security endpoint: POST http://{host}:{port}/security")
    print("\nNote: For MCP-compliant clients (Claude Desktop, Cursor), use 'code-scalpel mcp' instead.")
    print("Press Ctrl+C to stop the server.\n")

    try:
        run_server(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")

    return 0


def start_mcp_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8080) -> int:
    """Start the MCP-compliant server (for AI clients like Claude Desktop, Cursor)."""
    from .mcp.server import run_server

    if transport == "stdio":
        print("Starting Code Scalpel MCP Server (stdio transport)")
        print("   This server communicates via stdin/stdout.")
        print("   Add to your Claude Desktop config or use with MCP Inspector.")
        print("\nPress Ctrl+C to stop.\n")
    else:
        print(f"Starting Code Scalpel MCP Server (HTTP transport) on {host}:{port}")
        print(f"   MCP endpoint: http://{host}:{port}/mcp")
        print("\nPress Ctrl+C to stop.\n")

    try:
        run_server(transport=transport, host=host, port=port)
    except KeyboardInterrupt:
        print("\nMCP Server stopped.")

    return 0


def main() -> int:
    """Main CLI entry point."""
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="code-scalpel",
        description="AI Agent toolkit for code analysis using ASTs, PDGs, and Symbolic Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  code-scalpel analyze myfile.py              Analyze a Python file
  code-scalpel analyze --code "def f(): pass" Analyze code string
  code-scalpel analyze myfile.py --json       Output as JSON
  code-scalpel scan myfile.py                 Security vulnerability scan
  code-scalpel scan myfile.py --json          Security scan with JSON output
  code-scalpel mcp                            Start MCP server (stdio, for Claude Desktop)
  code-scalpel mcp --http --port 8080         Start MCP server (HTTP transport)
  code-scalpel server --port 5000             Start REST API server (legacy)
  code-scalpel version                        Show version info

For more information, visit: https://github.com/tescolopio/code-scalpel
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Python code")
    analyze_parser.add_argument("file", nargs="?", help="Python file to analyze")
    analyze_parser.add_argument("--code", "-c", help="Code string to analyze")
    analyze_parser.add_argument(
        "--json", "-j", action="store_true", help="Output as JSON"
    )

    # Scan command (Security Analysis)
    scan_parser = subparsers.add_parser(
        "scan", help="Scan for security vulnerabilities (SQLi, XSS, etc.)"
    )
    scan_parser.add_argument("file", nargs="?", help="Python file to scan")
    scan_parser.add_argument("--code", "-c", help="Code string to scan")
    scan_parser.add_argument(
        "--json", "-j", action="store_true", help="Output as JSON"
    )

    # Server command (REST API - legacy)
    server_parser = subparsers.add_parser("server", help="Start REST API server (legacy)")
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", "-p", type=int, default=5000, help="Port to bind to (default: 5000)"
    )

    # MCP command (Model Context Protocol - recommended)
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server (for Claude Desktop, Cursor)")
    mcp_parser.add_argument(
        "--http", action="store_true", help="Use HTTP transport instead of stdio"
    )
    mcp_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to for HTTP (default: 127.0.0.1)"
    )
    mcp_parser.add_argument(
        "--port", "-p", type=int, default=8080, help="Port for HTTP transport (default: 8080)"
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "analyze":
        output_format = "json" if args.json else "text"
        if args.code:
            return analyze_code(args.code, output_format)
        elif args.file:
            return analyze_file(args.file, output_format)
        else:
            analyze_parser.print_help()
            return 1

    elif args.command == "scan":
        output_format = "json" if args.json else "text"
        if args.code:
            return scan_code_security(args.code, output_format)
        elif args.file:
            return scan_security(args.file, output_format)
        else:
            scan_parser.print_help()
            return 1

    elif args.command == "server":
        return start_server(args.host, args.port)

    elif args.command == "mcp":
        transport = "streamable-http" if args.http else "stdio"
        return start_mcp_server(transport, args.host, args.port)

    elif args.command == "version":
        print(f"Code Scalpel v{__version__}")
        print(f"Python {sys.version}")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
