"""
Code Scalpel CLI - Command-line interface for code analysis.

Usage:
    code-scalpel analyze <file>           Analyze a Python file
    code-scalpel analyze --code "..."     Analyze code string
    code-scalpel server [--port PORT]     Start MCP server
    code-scalpel version                  Show version
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def analyze_file(filepath: str, output_format: str = "text") -> int:
    """Analyze a Python file and print results."""
    from .code_analyzer import CodeAnalyzer, AnalysisLevel
    
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 1
    
    if not path.suffix == ".py":
        print(f"Warning: File does not have .py extension: {filepath}", file=sys.stderr)
    
    try:
        code = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1
    
    return analyze_code(code, output_format, filepath)


def analyze_code(code: str, output_format: str = "text", source: str = "<string>") -> int:
    """Analyze code string and print results."""
    from .code_analyzer import CodeAnalyzer, AnalysisLevel
    
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
        print(f"\n📊 Code Scalpel Analysis: {source}")
        print("=" * 60)
        
        print(f"\n📈 Metrics:")
        print(f"   Lines of code: {result.metrics.lines_of_code}")
        print(f"   Functions: {result.metrics.num_functions}")
        print(f"   Classes: {result.metrics.num_classes}")
        print(f"   Cyclomatic complexity: {result.metrics.cyclomatic_complexity}")
        print(f"   Analysis time: {result.metrics.analysis_time_seconds:.3f}s")
        
        if result.dead_code:
            print(f"\n🔍 Dead Code Detected ({len(result.dead_code)} items):")
            for dc in result.dead_code:
                print(f"   - {dc.code_type} '{dc.name}' (lines {dc.line_start}-{dc.line_end})")
                print(f"     Reason: {dc.reason}")
        
        if result.security_issues:
            print(f"\n⚠️  Security Issues ({len(result.security_issues)}):")
            for issue in result.security_issues:
                print(f"   - {issue.get('type', 'Unknown')}: {issue.get('description', 'No description')}")
        
        if result.refactor_suggestions:
            print(f"\n💡 Suggestions ({len(result.refactor_suggestions)}):")
            for s in result.refactor_suggestions[:5]:  # Show top 5
                print(f"   - [{s.refactor_type}] {s.description}")
        
        if result.errors:
            print(f"\n❌ Errors:")
            for err in result.errors:
                print(f"   - {err}")
        
        print()
    
    return 0 if not result.errors else 1


def start_server(host: str = "0.0.0.0", port: int = 8080) -> int:
    """Start the MCP server."""
    from .integrations.mcp_server import run_server
    
    print(f"🚀 Starting Code Scalpel MCP Server on {host}:{port}")
    print(f"   Health check: http://{host}:{port}/health")
    print(f"   Analyze endpoint: POST http://{host}:{port}/analyze")
    print(f"   Refactor endpoint: POST http://{host}:{port}/refactor")
    print(f"   Security endpoint: POST http://{host}:{port}/security")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        run_server(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    
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
  code-scalpel server --port 8080             Start MCP server
  code-scalpel version                        Show version info

For more information, visit: https://github.com/tescolopio/code-scalpel
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Python code")
    analyze_parser.add_argument("file", nargs="?", help="Python file to analyze")
    analyze_parser.add_argument("--code", "-c", help="Code string to analyze")
    analyze_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start MCP server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    server_parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind to (default: 8080)")
    
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
    
    elif args.command == "server":
        return start_server(args.host, args.port)
    
    elif args.command == "version":
        print(f"Code Scalpel v{__version__}")
        print(f"Python {sys.version}")
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
