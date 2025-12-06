"""
MCP Server - Model Context Protocol server for Code Scalpel.

This module provides a Flask-based MCP server that exposes Code Scalpel's
analysis capabilities via HTTP endpoints for agent queries.

v0.3.1: Added taint-based security scanning and symbolic execution endpoints.
"""

import time
from dataclasses import dataclass
from typing import Optional

from flask import Flask, Response, jsonify, request

# Version updated to match package
__version__ = "0.3.1"


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server."""

    # SECURITY: Default to localhost only. Use --host 0.0.0.0 explicitly for network access.
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    cache_enabled: bool = True
    max_code_size: int = 100000  # Maximum code size in characters


def create_app(config: Optional[MCPServerConfig] = None) -> Flask:
    """
    Create and configure the Flask MCP server application.

    Args:
        config: Optional server configuration.

    Returns:
        Configured Flask application.
    """
    if config is None:
        config = MCPServerConfig()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_code_size

    # Lazy import to avoid circular dependencies
    try:
        from .crewai import CrewAIScalpel
    except ImportError:
        from integrations.crewai import CrewAIScalpel
    scalpel = CrewAIScalpel(cache_enabled=config.cache_enabled)

    @app.route("/health", methods=["GET"])
    def health_check() -> Response:
        """Health check endpoint."""
        return jsonify(
            {"status": "healthy", "service": "code-scalpel-mcp", "version": __version__}
        )

    @app.route("/analyze", methods=["POST"])
    def analyze_code() -> Response:
        """
        Analyze Python code and return analysis results.

        Request body:
            {
                "code": "string",  # Python code to analyze
                "options": {       # Optional analysis options
                    "include_security": true,
                    "include_style": true
                }
            }

        Response:
            {
                "success": bool,
                "analysis": {...},
                "issues": [...],
                "suggestions": [...],
                "processing_time_ms": float,
                "error": "string" (optional)
            }
        """
        start_time = time.time()

        # Parse request body
        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Request body is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        code = data.get("code")
        if not code:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code field is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        if not isinstance(code, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code must be a string",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        if len(code) > config.max_code_size:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Code exceeds maximum size of {config.max_code_size} characters",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        try:
            # Perform analysis
            result = scalpel.analyze(code)

            response = {
                "success": result.success,
                "analysis": result.analysis,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "processing_time_ms": _elapsed_ms(start_time),
            }

            if result.error:
                response["error"] = result.error

            return jsonify(response)

        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Internal error: {str(e)}",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                500,
            )

    @app.route("/refactor", methods=["POST"])
    def refactor_code() -> Response:
        """
        Refactor Python code based on analysis.

        Request body:
            {
                "code": "string",           # Python code to refactor
                "task": "string" (optional) # Refactoring task description
            }

        Response:
            {
                "success": bool,
                "original_code": "string",
                "refactored_code": "string" (optional),
                "analysis": {...},
                "suggestions": [...],
                "processing_time_ms": float,
                "error": "string" (optional)
            }
        """
        start_time = time.time()

        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Request body is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        code = data.get("code")
        if not code:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code field is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        if not isinstance(code, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code must be a string",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        task = data.get("task", "improve code quality")

        try:
            result = scalpel.refactor(code, task)

            response = {
                "success": result.success,
                "original_code": result.original_code,
                "refactored_code": result.refactored_code,
                "analysis": result.analysis,
                "suggestions": result.suggestions,
                "processing_time_ms": _elapsed_ms(start_time),
            }

            if result.error:
                response["error"] = result.error

            return jsonify(response)

        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Internal error: {str(e)}",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                500,
            )

    @app.route("/security", methods=["POST"])
    def security_scan() -> Response:
        """
        Perform security-focused analysis on Python code.

        Request body:
            {
                "code": "string"  # Python code to analyze
            }

        Response:
            {
                "success": bool,
                "issues": [...],
                "risk_level": "string",
                "recommendations": [...],
                "processing_time_ms": float,
                "error": "string" (optional)
            }
        """
        start_time = time.time()

        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Request body is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        code = data.get("code")
        if not code:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code field is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        if not isinstance(code, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code must be a string",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        try:
            result = scalpel.analyze_security(code)
            result["processing_time_ms"] = _elapsed_ms(start_time)
            return jsonify(result)

        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Internal error: {str(e)}",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                500,
            )

    @app.route("/symbolic", methods=["POST"])
    def symbolic_analysis() -> Response:
        """
        Perform symbolic execution analysis on Python code.
        
        v0.3.0+: Uses Z3-powered symbolic execution to enumerate paths.

        Request body:
            {
                "code": "string"  # Python code to analyze
            }

        Response:
            {
                "success": bool,
                "total_paths": int,
                "feasible_paths": int,
                "paths": [...],
                "processing_time_ms": float,
                "error": "string" (optional)
            }
        """
        start_time = time.time()

        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Request body is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        code = data.get("code")
        if not code:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code field is required",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        if not isinstance(code, str):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Code must be a string",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                400,
            )

        try:
            result = scalpel.analyze_symbolic(code)
            result["processing_time_ms"] = _elapsed_ms(start_time)
            return jsonify(result)

        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Internal error: {str(e)}",
                        "processing_time_ms": _elapsed_ms(start_time),
                    }
                ),
                500,
            )

    @app.route("/tools", methods=["GET"])
    def list_tools() -> Response:
        """
        List all available MCP tools/endpoints.
        
        This endpoint helps agents discover what capabilities are available.
        """
        return jsonify({
            "service": "code-scalpel-mcp",
            "version": __version__,
            "tools": [
                {
                    "name": "analyze",
                    "endpoint": "/analyze",
                    "method": "POST",
                    "description": "Analyze Python code for style and security issues",
                },
                {
                    "name": "security",
                    "endpoint": "/security", 
                    "method": "POST",
                    "description": "Taint-based security scan (SQLi, XSS, Command Injection, Path Traversal)",
                },
                {
                    "name": "symbolic",
                    "endpoint": "/symbolic",
                    "method": "POST", 
                    "description": "Symbolic execution to enumerate paths and generate test inputs",
                },
                {
                    "name": "refactor",
                    "endpoint": "/refactor",
                    "method": "POST",
                    "description": "Refactor Python code based on analysis",
                },
            ]
        })

    return app


def _elapsed_ms(start_time: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (time.time() - start_time) * 1000


def run_server(host: str = "127.0.0.1", port: int = 8080, debug: bool = False) -> None:
    """
    Run the MCP server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        debug: Whether to run in debug mode. WARNING: Debug mode should
               never be enabled in production as it can allow arbitrary
               code execution.
    """
    import os
    import warnings

    # Warn if debug mode is enabled in production
    if debug and os.environ.get("FLASK_ENV") == "production":
        warnings.warn(
            "Debug mode should not be enabled in production. "
            "Set FLASK_ENV to 'development' or disable debug mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        debug = False  # Force disable debug in production

    config = MCPServerConfig(host=host, port=port, debug=debug)
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)


# Allow running directly as a script (development only)
if __name__ == "__main__":
    import os

    # Only enable debug mode in development
    is_development = os.environ.get("FLASK_ENV", "development") == "development"
    run_server(debug=is_development)
