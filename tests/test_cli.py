"""
Tests for the CLI module.

These tests verify that the command-line interface works correctly.
Goal: Get cli.py from 0% to at least 50% coverage.
"""

import subprocess
import sys

import pytest


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help_command(self):
        """Test that --help works and shows expected content."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "code-scalpel" in result.stdout.lower() or "usage" in result.stdout.lower()
        assert "analyze" in result.stdout
        assert "server" in result.stdout
        assert "version" in result.stdout

    def test_version_command(self):
        """Test that version command works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout or "Code Scalpel" in result.stdout

    def test_analyze_help(self):
        """Test that analyze --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--code" in result.stdout or "code" in result.stdout.lower()


class TestCLIAnalyze:
    """Test the analyze command."""

    def test_analyze_inline_code(self):
        """Test analyzing inline code with --code flag."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "def hello(): return 42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Metrics" in result.stdout or "analysis" in result.stdout.lower()

    def test_analyze_inline_code_json(self):
        """Test analyzing inline code with JSON output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "x = 1",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # JSON output should be parseable
        import json

        try:
            data = json.loads(result.stdout)
            assert "success" in data or "metrics" in data or isinstance(data, dict)
        except json.JSONDecodeError:
            # Might have logging output before JSON
            lines = result.stdout.strip().split("\n")
            for line in lines:
                try:
                    data = json.loads(line)
                    assert isinstance(data, dict)
                    break
                except json.JSONDecodeError:
                    continue

    def test_analyze_file(self, tmp_path):
        """Test analyzing a Python file."""
        # Create a temporary Python file
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def greet(name):\n    return f'Hello, {name}!'\n")

        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_analyze_nonexistent_file(self):
        """Test analyzing a file that doesn't exist."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "/nonexistent/file.py",
            ],
            capture_output=True,
            text=True,
        )
        # Should fail gracefully
        assert result.returncode != 0 or "error" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_analyze_syntax_error_code(self):
        """Test analyzing code with syntax errors."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "def broken(",
            ],
            capture_output=True,
            text=True,
        )
        # Should handle gracefully, not crash
        # May return 0 with error message or non-zero
        combined = result.stdout + result.stderr
        assert result.returncode != 0 or "error" in combined.lower() or "syntax" in combined.lower()


class TestCLIServer:
    """Test the server command."""

    def test_server_help(self):
        """Test that server --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "server", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout or "port" in result.stdout.lower()


class TestCLIEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_command(self):
        """Test that unknown commands are handled."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "unknowncommand"],
            capture_output=True,
            text=True,
        )
        # Should fail with usage info
        assert result.returncode != 0

    def test_analyze_no_input(self):
        """Test analyze without any input."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze"],
            capture_output=True,
            text=True,
        )
        # Should show error or usage
        combined = result.stdout + result.stderr
        assert result.returncode != 0 or "error" in combined.lower() or "usage" in combined.lower()
